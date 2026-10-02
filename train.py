import argparse
from collections import Counter
import json
from pathlib import Path
import random
import warnings

from data import load_dataset, prepare_reference_data, reference_signature, file_sha256
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models import build_model, choose_device, molecular_features, fit_router


class SpectrumDataset(Dataset):

    def __init__(self, data, indices, config):
        self.smiles = np.asarray(data['smiles'])[indices].astype(str)
        self.keys = np.asarray(data['group_key'])[indices].astype(str)
        self.spectrum = np.asarray(data['spectrum'], dtype=np.float32)[indices]
        field = config.get('domain_field', 'is_drug')
        values = np.asarray(data[field])[indices]
        self.drug = values == 1 if field == 'is_nps' else values.astype(bool)
        domain_keys = set(self.keys[self.drug])
        self.drug = np.asarray([key in domain_keys for key in self.keys], dtype=bool)
        self.fingerprints, self.bits, calculated_mass = molecular_features(self.smiles, config.get('fingerprint_bits', 2048), config.get('fingerprint_radius', 2))
        self.mass = np.asarray(data['exact_mass'], dtype=np.float32)[indices]
        if not np.isfinite(self.mass).all() or (self.mass <= 0).any():
            raise ValueError('Dataset has invalid exact_mass. Re-run data preparation.')
        if (np.abs(self.mass - calculated_mass) > 0.15).any():
            raise ValueError('Dataset mass differs from the supplied SMILES by >0.15 Da. Check salts / derivatization.')

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        return {'smiles': self.smiles[index], 'group_key': self.keys[index], 'spectrum': self.spectrum[index], 'fingerprints': self.fingerprints[index], 'bits': self.bits[index], 'mass': self.mass[index], 'drug': self.drug[index]}

def prepare_batch(batch, model, config, device):
    if config['model_type'] == 'neims_reimpl':
        inputs = {'fingerprints': batch['fingerprints'].to(device)}
    else:
        inputs = {key: value.to(device) for key, value in model.tokenize(batch['smiles']).items()}
    return (inputs, batch['mass'].to(device), batch['spectrum'].to(device))

def spectral_cosine(predicted_sqrt, raw_target):
    return F.cosine_similarity(predicted_sqrt.float(), raw_target.float().clamp_min(0).sqrt(), dim=1, eps=1e-08)

def loader_for_epoch(dataset, config, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    counts = Counter(dataset.keys)
    weights = np.asarray([1.0 / counts[key] for key in dataset.keys], dtype=np.float64)
    if config.get('domain_sampling', False):
        drug = dataset.drug
        if not drug.any():
            raise ValueError('No drug-related TRAIN molecules. Add SWGDRUG / reviewed labels before adaptation.')
        if not (~drug).any():
            warnings.warn('Training data are all drug-related; domain sampling has no additional effect.')
        else:
            fraction = config.get('drug_fraction', 0.5)
            if not 0 < fraction < 1:
                raise ValueError('drug_fraction must be strictly between 0 and 1')
            weights[drug] *= fraction / weights[drug].sum()
            weights[~drug] *= (1 - fraction) / weights[~drug].sum()
    sampler = WeightedRandomSampler(torch.as_tensor(weights), len(dataset), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=config.get('batch_size', 32), sampler=sampler, num_workers=0, pin_memory=False, generator=generator)

@torch.inference_mode()
def validate(model, dataset, config, device):
    model.eval()
    records = {}
    loader = DataLoader(dataset, batch_size=config.get('eval_batch_size', config.get('batch_size', 32)), shuffle=False)
    for batch in loader:
        inputs, mass, target = prepare_batch(batch, model, config, device)
        cosine = spectral_cosine(model(inputs, mass), target).cpu().numpy()
        for key, value, drug in zip(batch['group_key'], cosine, batch['drug'].tolist()):
            record = records.setdefault(key, {'values': [], 'drug': False})
            record['values'].append(float(value))
            record['drug'] = record['drug'] or bool(drug)
    scores = [float(np.mean(row['values'])) for row in records.values()]
    drug_scores = [float(np.mean(row['values'])) for row in records.values() if row['drug']]
    return {'val_cosine': float(np.mean(scores)), 'val_drug_cosine': float(np.mean(drug_scores)) if drug_scores else None, 'val_molecules': len(scores), 'val_drug_molecules': len(drug_scores)}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def train_smart(args, config, source, device, output):
    settings = dict(config['smart'])
    if args.epochs is not None:
        settings['epochs'] = args.epochs
    reference = prepare_reference_data(source)
    inner = dict(reference, neighbors=reference['neighbors_inner'], similarities=reference['similarities_inner'])
    selection = fit_router(inner, reference['inner_fit_indices'], reference['inner_select_indices'],
                           epochs=settings['epochs'], device=device, seed=args.seed, mode=args.model,
                           batch_size=settings['batch_size'], stop_patience=settings['patience'],
                           select=True, min_similarity=settings['minimum_similarity'],
                           log_callback=lambda row: print(json.dumps(dict(phase='selection', **row)), flush=True))
    fit = fit_router(reference, reference['train_indices'], np.empty(0, dtype=np.int64),
                     epochs=selection['best_epoch'], device=device, seed=args.seed, mode=args.model,
                     batch_size=settings['batch_size'], select=False,
                     min_similarity=settings['minimum_similarity'],
                     log_callback=lambda row: print(json.dumps(dict(phase='refit', **row)), flush=True))
    if selection['initial_state_sha256'] != fit['initial_state_sha256']:
        raise RuntimeError('Selection and refit initial states differ')
    payload = {'model_state': {k: v.detach().cpu() for k, v in fit['model'].state_dict().items()},
               'model_type': args.model, 'spectrum_dim': config['spectrum_dim'], 'seed': args.seed,
               'config': settings, 'parameters': fit['parameters'], 'epochs': selection['best_epoch'],
               'train_keys': source['candidate_keys'][reference['train_indices']].tolist(),
               'reference_signature': reference_signature(source, reference),
               'input_sha256': file_sha256(args.data)}
    torch.save(payload, output / 'model.pt')
    write_json(output / 'training.json', {
        'model': args.model, 'seed': args.seed, 'selected_epochs': selection['best_epoch'],
        'best_inner_cosine': selection['best_inner_cosine'], 'parameters': fit['parameters'],
        'reference_audit': reference['audit'], 'selection_history': selection['history'],
        'refit_history': fit['history'], 'input_sha256': payload['input_sha256'],
        'reference_signature': payload['reference_signature'],
        'selection_uses_external_validation_or_test': False,
        'initial_state_sha256': fit['initial_state_sha256']})


def train_upstream(args, config, source, device, output):
    settings = dict(config[args.model])
    settings['spectrum_dim'] = config['spectrum_dim']
    if args.epochs is not None:
        settings['epochs'] = args.epochs
    if args.model_dir:
        settings['model_dir'] = str(args.model_dir)
    if args.model == 'adapt' and not args.init_from:
        raise ValueError('Domain adaptation requires --init-from with a trained EI checkpoint')
    split = source['split']
    groups = {name: set(source['keys'][split == name]) for name in ('train', 'val', 'test')}
    if not groups['train'] or not groups['val']:
        raise ValueError('Upstream training requires separate train and validation structures')
    _, _, exact_mass = molecular_features(source['smiles'])
    arrays = {'smiles': source['smiles'], 'group_key': source['keys'], 'spectrum': source['spectra'],
              'exact_mass': exact_mass, 'is_drug': source['data']['is_drug']}
    training = SpectrumDataset(arrays, np.flatnonzero(split == 'train'), settings)
    validation = SpectrumDataset(arrays, np.flatnonzero(split == 'val'), settings)
    selection_metric = settings.get('selection_metric', 'val_cosine')
    if selection_metric == 'val_drug_cosine' and not validation.drug.any():
        raise ValueError('Domain adaptation requires source-flagged validation structures')
    model = build_model(settings).to(device)
    inherited_keys = set()
    if args.init_from:
        prior = torch.load(args.init_from, map_location='cpu', weights_only=False)
        if 'train_keys' not in prior or prior['config']['model_type'] != settings['model_type']:
            raise ValueError('Initialization checkpoint lacks compatible model or training identities')
        inherited_keys = set(prior['train_keys'])
        if inherited_keys & (groups['val'] | groups['test']):
            raise ValueError('Initialization checkpoint has seen current held-out structures')
        model.load_state_dict(prior['model_state'], strict=True)
    if settings['model_type'] == 'molformer':
        parameter_groups = [
            {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': settings['lora_lr']},
            {'params': list(model.head.parameters()), 'lr': settings['head_lr']}]
    else:
        parameter_groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': settings['learning_rate']}]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=settings['weight_decay'])
    history, best, best_epoch, stale = [], -float('inf'), 0, 0
    accumulation = int(settings.get('gradient_accumulation', 1))
    if int(settings['epochs']) < 1 or accumulation < 1:
        raise ValueError('Epochs and gradient accumulation must be positive')
    for epoch in range(1, int(settings['epochs']) + 1):
        model.train()
        loader = loader_for_epoch(training, settings, args.seed, epoch)
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        for batch_index, batch in enumerate(loader):
            inputs, mass, target = prepare_batch(batch, model, settings, device)
            prediction = model(inputs, mass)
            loss = 1 - spectral_cosine(prediction, target).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite spectral loss')
            start = (batch_index // accumulation) * accumulation
            (loss / min(accumulation, len(loader) - start)).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(mass)
            count += len(mass)
        metrics = validate(model, validation, settings, device)
        score = metrics[selection_metric]
        record = dict(epoch=epoch, train_loss=total / count, **metrics)
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best + settings.get('min_delta', 1e-5):
            best, best_epoch, stale = score, epoch, 0
            torch.save({'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'config': settings, 'train_keys': sorted(groups['train'] | inherited_keys),
                        'seed': args.seed, 'epoch': epoch, 'input_sha256': file_sha256(args.data)}, output / 'model.pt')
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    write_json(output / 'training.json', {'model': args.model, 'seed': args.seed,
                                          'best_epoch': best_epoch, 'selection_metric': selection_metric,
                                          'best_score': best, 'config': settings, 'history': history,
                                          'input_sha256': file_sha256(args.data)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/example.csv'))
    parser.add_argument('--config', type=Path, default=Path('config.json'))
    parser.add_argument('--model', choices=('smart', 'reweight', 'molformer', 'neims', 'adapt'), default='smart')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--init-from', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['spectrum_dim'] < 2 or config['threads'] < 1:
        raise ValueError('Invalid spectrum dimension or thread count')
    fixed = {'alpha': .25, 'minimum_similarity': .35, 'neighbors': 3, 'inner_partition_seed': 42}
    if any(config['smart'].get(k) != v for k, v in fixed.items()):
        raise ValueError('This implementation uses the documented fixed reference protocol')
    if args.model in ('smart', 'reweight') and args.init_from:
        raise ValueError('Router fitting starts from a fresh initialization')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(config['threads'])
    device = choose_device(args.device)
    source = load_dataset(args.data, dim=config['spectrum_dim'])
    output = args.output or Path('outputs') / args.model
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory to preserve existing runs')
    output.mkdir(parents=True, exist_ok=True)
    if args.model in ('smart', 'reweight'):
        train_smart(args, config, source, str(device), output)
    else:
        train_upstream(args, config, source, device, output)
    print('Saved:', output / 'model.pt', flush=True)


if __name__ == '__main__':
    main()
