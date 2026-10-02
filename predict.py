import argparse
import json
from pathlib import Path

from data import load_dataset, prepare_reference_data, read_predictions, save_predictions, reference_signature
import numpy as np
import torch

from models import AnalogRouter, load_predictor, predict_router, static_prediction, blend_spectra, normalize_spectra, choose_device


def check_upstream_training(source, keys):
    keys = np.asarray(keys, dtype=str)
    if keys.ndim != 1:
        raise ValueError('Upstream training identities must be a one-dimensional array')
    heldout = set(source['candidate_keys'][source['candidate_split'] != 'train'])
    if heldout.intersection(keys.tolist()):
        raise ValueError('Upstream predictor has seen current held-out structures')
    return sorted(set(keys.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/example.csv'))
    parser.add_argument('--config', type=Path, default=Path('config.json'))
    parser.add_argument('--method', choices=('base', 'copy', 'hybrid', 'reweight', 'smart'), default='smart')
    parser.add_argument('--checkpoint', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--base-predictions', type=Path)
    group.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--model-dir', type=Path, default=Path('models/molformer'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output', type=Path, default=Path('outputs/predictions.npz'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(int(config['threads']))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = choose_device(args.device)
    source = load_dataset(args.data, dim=config['spectrum_dim'], require_spectra=False)
    upstream_keys = []
    provenance = 'unknown'
    demonstration_only = False
    if args.base_checkpoint:
        predictor = load_predictor(args.base_checkpoint, device=str(device), model_dir_override=args.model_dir)
        upstream_keys = check_upstream_training(source, sorted(predictor.train_keys))
        provenance = 'recorded'
        base = predictor(source['candidate_smiles'].tolist(), batch_size=32)
        if base.shape != (len(source['candidate_keys']), config['spectrum_dim']):
            raise ValueError('Upstream prediction dimension differs from the dataset')
        del predictor
    else:
        if args.base_predictions is None and args.data.resolve() != Path('data/example.csv').resolve():
            raise ValueError('Supply --base-checkpoint or --base-predictions for your own data')
        base_path = args.base_predictions or Path('data/example_predictions.npz')
        base = read_predictions(base_path, source)
        with np.load(base_path, allow_pickle=False) as archive:
            if 'upstream_train_keys' in archive:
                upstream_keys = check_upstream_training(source, archive['upstream_train_keys'])
                provenance = 'recorded'
            if 'upstream_provenance' in archive and str(archive['upstream_provenance'].item()) == 'unknown':
                provenance = 'unknown'
            if 'demonstration_only' in archive:
                demonstration_only = bool(archive['demonstration_only'].item())
    covered = np.zeros(len(base), dtype=bool)
    seed = -1
    signature = ''
    if args.method == 'base':
        prediction = normalize_spectra(base)
    else:
        reference = prepare_reference_data(source)
        signature = reference_signature(source, reference)
        if args.method in ('copy', 'hybrid'):
            extra, covered = static_prediction(reference, args.method)
            alpha = float(config['smart']['alpha'])
        else:
            path = args.checkpoint or Path('outputs') / args.method / 'model.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['model_type'] != args.method or saved['spectrum_dim'] != config['spectrum_dim']:
                raise ValueError('Router checkpoint type or dimension mismatch')
            if signature != saved['reference_signature']:
                raise ValueError('Training reference structures or spectra changed since fitting')
            if saved['train_keys'] != source['candidate_keys'][reference['train_indices']].tolist():
                raise ValueError('Training reference identity mismatch')
            model = AnalogRouter(config['spectrum_dim'], reference['features'].shape[1])
            model.load_state_dict(saved['model_state'], strict=True)
            settings = saved['config']
            extra, covered = predict_router(model, reference, np.arange(len(base)), str(device), args.method,
                                             settings['batch_size'], settings['minimum_similarity'])
            alpha = float(settings['alpha'])
            seed = int(saved['seed'])
        prediction = blend_spectra(base, extra, covered, alpha=alpha)
    if args.output.exists():
        raise ValueError('Prediction output exists; use a new --output')
    save_predictions(args.output, source, prediction, method=np.asarray(args.method), seed=np.asarray(seed),
                     reference_covered=covered, reference_signature=np.asarray(signature),
                     candidate_split=source['candidate_split'],
                     upstream_provenance=np.asarray(provenance),
                     demonstration_only=np.asarray(demonstration_only),
                     **({'upstream_train_keys': np.asarray(upstream_keys, dtype=str)} if provenance == 'recorded' else {}))
    print(json.dumps({'method': args.method, 'candidates': len(base), 'reference_covered': int(covered.sum()),
                      'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
