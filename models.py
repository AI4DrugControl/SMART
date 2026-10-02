from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

def choose_device(value='auto'):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable.')
    return device

def molecular_features(smiles, n_bits=2048, radius=2):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    count_rows, bit_rows, mass_rows = ([], [], [])
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f'Invalid SMILES: {value!r}')
        counts = np.zeros(n_bits, dtype=np.float32)
        bits = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetCountFingerprint(molecule), counts)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), bits)
        count_rows.append(counts)
        bit_rows.append(bits)
        mass_rows.append(Descriptors.ExactMolWt(molecule))
    if not count_rows:
        return (np.empty((0, n_bits), np.float32), np.empty((0, n_bits), np.float32), np.empty(0, np.float32))
    return (np.stack(count_rows), np.stack(bit_rows), np.asarray(mass_rows, dtype=np.float32))

class ResidualBlock(nn.Module):

    def __init__(self, width, dropout):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, inputs):
        return inputs + self.block(inputs)

class BidirectionalSpectrumHead(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=512, blocks=2, dropout=0.1, isotope_margin=4):
        super().__init__()
        self.output_dim = int(output_dim)
        self.isotope_margin = int(isotope_margin)
        self.body = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), *[ResidualBlock(hidden_dim, dropout) for _ in range(blocks)], nn.LayerNorm(hidden_dim))
        self.forward_head = nn.Linear(hidden_dim, output_dim)
        self.reverse_head = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.register_buffer('bins', torch.arange(output_dim), persistent=False)

    def forward(self, embeddings, exact_mass):
        hidden = self.body(embeddings)
        forward = F.softplus(self.forward_head(hidden))
        reverse = F.softplus(self.reverse_head(hidden))
        center = torch.floor(exact_mass.float() + 0.5).long()
        reverse_index = center[:, None] + self.isotope_margin - self.bins[None, :]
        reverse_valid = (reverse_index >= 0) & (reverse_index < self.output_dim)
        aligned = reverse.gather(1, reverse_index.clamp(0, self.output_dim - 1)) * reverse_valid
        gate = torch.sigmoid(self.gate(hidden))
        prediction = gate * forward + (1.0 - gate) * aligned
        valid = (self.bins[None, :] > 0) & (self.bins[None, :] <= center[:, None] + self.isotope_margin)
        return prediction * valid

class SafeBatchNorm(nn.BatchNorm1d):

    def forward(self, values):
        if self.training and values.shape[0] == 1:
            return F.batch_norm(values, self.running_mean, self.running_var, self.weight, self.bias, training=False, momentum=self.momentum, eps=self.eps)
        return super().forward(values)

class NEIMSResidualBlock(nn.Module):

    def __init__(self, width, bottleneck_factor, dropout):
        super().__init__()
        inner = max(1, int(width * bottleneck_factor))
        self.block = nn.Sequential(SafeBatchNorm(width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, inner), SafeBatchNorm(inner), nn.ReLU(), nn.Linear(inner, width))

    def forward(self, values):
        return values + self.block(values)

class NEIMSReimplementation(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.get('hidden_dim', 2000)
        dimension = config['spectrum_dim']
        self.isotope_margin = config.get('isotope_margin', 5)
        self.input = nn.Sequential(SafeBatchNorm(config.get('fingerprint_bits', 4096)), nn.Linear(config.get('fingerprint_bits', 4096), width), nn.ReLU())
        self.blocks = nn.Sequential(*[NEIMSResidualBlock(width, config.get('bottleneck_factor', 0.5), config.get('dropout', 0.25)) for _ in range(config.get('residual_blocks', 7))], SafeBatchNorm(width), nn.ReLU())
        self.forward_head = nn.Linear(width, dimension)
        self.reverse_head = nn.Linear(width, dimension)
        self.gate = nn.Linear(width, dimension)
        self.register_buffer('bins', torch.arange(dimension), persistent=False)

    def forward(self, inputs, exact_mass):
        features = self.blocks(self.input(inputs['fingerprints']))
        direct = self.forward_head(features)
        reverse = self.reverse_head(features)
        ceiling = torch.floor(exact_mass.float() + 0.5).long() + self.isotope_margin
        source = ceiling[:, None] - self.bins[None, :]
        valid = (source >= 0) & (source < len(self.bins)) & (self.bins[None, :] > 0)
        aligned = reverse.gather(1, source.clamp(0, len(self.bins) - 1)) * valid
        gate = torch.sigmoid(self.gate(features))
        direct_valid = (self.bins[None, :] > 0) & (self.bins[None, :] <= ceiling[:, None])
        return F.relu(gate * direct + (1 - gate) * aligned) * direct_valid

class MolformerSpectrumModel(nn.Module):

    def __init__(self, config, model_dir_override=None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.config = config
        model_dir = Path(model_dir_override or config.get('model_dir', 'models/molformer'))
        if not model_dir.is_dir() or not (model_dir / 'config.json').is_file():
            raise FileNotFoundError(f'MoLFormer files missing in {model_dir}.')
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        backbone = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True, deterministic_eval=True)
        expected = config.get('lora_targets', ['query', 'value'])
        observed = {name.rsplit('.', 1)[-1] for name, module in backbone.named_modules() if isinstance(module, nn.Linear)}
        if not set(expected).issubset(observed):
            raise RuntimeError(f'LoRA modules {expected} not found. Available linear leaf names: {sorted(observed)}')
        lora = LoraConfig(r=config.get('lora_r', 8), lora_alpha=config.get('lora_alpha', 16), lora_dropout=config.get('lora_dropout', 0.05), target_modules=expected, bias='none')
        self.encoder = get_peft_model(backbone, lora)
        self.head = BidirectionalSpectrumHead(backbone.config.hidden_size, config['spectrum_dim'], config.get('hidden_dim', 512), config.get('residual_blocks', 2), config.get('dropout', 0.1), config.get('isotope_margin', 4))

    def tokenize(self, smiles):
        encoded = self.tokenizer(list(smiles), padding=True, truncation=False, return_tensors='pt')
        limit = self.config.get('max_tokens', 256)
        if encoded['input_ids'].shape[1] > limit:
            raise ValueError(f'SMILES exceeds max_tokens={limit}; review long structures or raise max_tokens explicitly.')
        return {key: value for key, value in encoded.items() if key in {'input_ids', 'attention_mask'}}

    def forward(self, inputs, exact_mass):
        encoded = self.encoder(**inputs, return_dict=True)
        mask = inputs['attention_mask'].unsqueeze(-1).to(encoded.last_hidden_state.dtype)
        pooled = (encoded.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled, exact_mass)

def build_model(config, model_dir_override=None):
    if config['model_type'] == 'neims_reimpl':
        return NEIMSReimplementation(config)
    if config['model_type'] == 'molformer':
        return MolformerSpectrumModel(config, model_dir_override)
    raise ValueError(f"Unknown model_type: {config['model_type']}")

def load_checkpoint(path, map_location='cpu'):
    return torch.load(path, map_location=map_location, weights_only=False)

def raw_intensity(sqrt_prediction):
    raw = sqrt_prediction.float().clamp_min(0).square()
    return raw / raw.amax(dim=-1, keepdim=True).clamp_min(1e-12)

class SpectrumPredictor:

    def __init__(self, checkpoint, device='auto', model_dir_override=None):
        self.device = choose_device(device)
        saved = load_checkpoint(checkpoint)
        self.config = saved['config']
        self.model = build_model(self.config, model_dir_override)
        self.model.load_state_dict(saved['model_state'], strict=True)
        self.model.to(self.device).eval()
        if 'train_keys' not in saved:
            raise ValueError('Checkpoint lacks training identity audit (train_keys)')
        self.train_keys = set(saved['train_keys'])
        self.metadata = {key: value for key, value in saved.items() if key not in {'model_state', 'optimizer_state', 'rng_state', 'train_keys'}}

    @torch.inference_mode()
    def __call__(self, smiles, exact_mass=None, batch_size=32):
        smiles = list(smiles)
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not smiles:
            return np.empty((0, self.config['spectrum_dim']), dtype=np.float32)
        supplied_mass = None if exact_mass is None else np.asarray(exact_mass, dtype=np.float32)
        if supplied_mass is not None and supplied_mass.shape != (len(smiles),):
            raise ValueError('exact_mass must have one value per SMILES')
        results = []
        for start in range(0, len(smiles), batch_size):
            batch_smiles = smiles[start:start + batch_size]
            counts, _, calculated = molecular_features(batch_smiles, self.config.get('fingerprint_bits', 2048), self.config.get('fingerprint_radius', 2))
            mass = calculated if supplied_mass is None else supplied_mass[start:start + batch_size]
            if not np.isfinite(mass).all() or (mass <= 0).any():
                raise ValueError('Non-finite or nonpositive exact_mass')
            if self.config['model_type'] == 'neims_reimpl':
                inputs = {'fingerprints': torch.from_numpy(counts)}
            else:
                inputs = self.model.tokenize(batch_smiles)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            values = self.model(inputs, torch.as_tensor(mass, device=self.device))
            results.append(raw_intensity(values).cpu().numpy())
        return np.concatenate(results)

def load_predictor(checkpoint, device='auto', model_dir_override=None):
    return SpectrumPredictor(checkpoint, device, model_dir_override)

class AnalogRouter(nn.Module):

    def __init__(self, dim, fp_dim=2048, hidden=64):
        super().__init__()
        self.dim = int(dim)
        if self.dim < 2 or int(fp_dim) < 1 or int(hidden) < 1:
            raise ValueError('Invalid spectrum/fingerprint/hidden dimension')
        self.encoder = nn.Sequential(nn.Linear(int(fp_dim), int(hidden)), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(3 * int(hidden) + 4, int(hidden)), nn.SiLU())
        self.router = nn.Sequential(nn.Linear(int(hidden) + 3, 32), nn.SiLU(), nn.Linear(32, 3))
        self.register_buffer('mz', torch.arange(self.dim, dtype=torch.float32), persistent=False)

    def forward(self, fpq, fpr, ref_l1, massq, massr, sim, mode='smart', return_valid=False):
        if mode not in ('smart', 'reweight'):
            raise ValueError('Unknown router mode: ' + str(mode))
        if ref_l1.ndim != 2 or ref_l1.shape[1] != self.dim:
            raise ValueError('ref_l1 must have shape [batch, spectrum dimension]')
        batch = ref_l1.shape[0]
        massq, massr, sim = (massq.reshape(batch), massr.reshape(batch), sim.reshape(batch))
        eq, er = (self.encoder(fpq), self.encoder(fpr))
        globals_ = torch.stack((sim, massq / self.dim, massr / self.dim, (massq - massr) / self.dim), dim=1)
        context = self.context(torch.cat((eq, er, eq - er, globals_), dim=1))
        mz = self.mz.to(dtype=ref_l1.dtype).expand(batch, -1)
        ref = ref_l1.clamp_min(0)
        ref = ref / ref.sum(1, keepdim=True).clamp_min(1e-12)
        peak_features = torch.stack((mz / self.dim, (massr[:, None] - mz) / self.dim, torch.sqrt(ref.clamp_min(0))), dim=2)
        logits = self.router(torch.cat((context[:, None, :].expand(-1, self.dim, -1), peak_features), dim=2))
        log_weights = F.log_softmax(logits, dim=2)
        original = self.mz.to(torch.long).expand(batch, -1)
        delta = torch.round(massq).to(torch.long) - torch.round(massr).to(torch.long)
        shifted = original + delta[:, None] if mode == 'smart' else original
        upper = torch.minimum(torch.round(massq).to(torch.long) + 3, torch.full_like(delta, self.dim - 1))
        source_valid = original >= 1
        copy_valid = source_valid & (original <= upper[:, None])
        shift_valid = source_valid & (shifted >= 1) & (shifted <= upper[:, None]) & (shifted < self.dim)
        positive = ref > 0
        safe_ref = torch.where(positive, ref, torch.ones_like(ref))
        log_ref = torch.log(safe_ref)
        retained = torch.cat((copy_valid & positive, shift_valid & positive), dim=1)
        log_mass = torch.cat((log_ref + log_weights[:, :, 0], log_ref + log_weights[:, :, 1]), dim=1)
        log_mass = log_mass.masked_fill(~retained, -float('inf'))
        valid = retained.any(dim=1)
        safe_log_mass = torch.where(valid[:, None], log_mass, torch.zeros_like(log_mass))
        centered_log_mass = safe_log_mass - safe_log_mass.amax(dim=1, keepdim=True)
        log_total = torch.logsumexp(centered_log_mass, dim=1, keepdim=True)
        normalized_mass = torch.where(retained, torch.exp(centered_log_mass - log_total), torch.zeros_like(safe_log_mass))
        copy_mass, shift_mass = normalized_mass.split(self.dim, dim=1)
        output = torch.zeros_like(ref)
        output.scatter_add_(1, original, copy_mass)
        output.scatter_add_(1, shifted.clamp(0, self.dim - 1), shift_mass)
        return (output, valid) if return_valid else output

def router_cosine(prediction, target):
    p = torch.where(prediction > 0, torch.sqrt(prediction.clamp_min(1e-12)), torch.zeros_like(prediction))
    t = torch.sqrt(target.clamp_min(0))
    cosine = (F.normalize(p, p=2, dim=1, eps=1e-12) * F.normalize(t, p=2, dim=1, eps=1e-12)).sum(1)
    nonempty = (prediction.sum(1) > 1e-12) & (target.sum(1) > 1e-12)
    return torch.where(nonempty, cosine.clamp(0, 1), torch.zeros_like(cosine))

def _router_data(data):
    required = ('features', 'masses', 'spectra', 'neighbors', 'similarities')
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError('Missing router data fields: ' + ', '.join(missing))
    features = np.asarray(data['features'], dtype=np.float32)
    masses = np.asarray(data['masses'], dtype=np.float32)
    spectra = np.asarray(data['spectra'], dtype=np.float32)
    neighbors = np.asarray(data['neighbors'], dtype=np.int64)
    similarities = np.asarray(data['similarities'], dtype=np.float32)
    n = len(features)
    if features.ndim != 2 or features.shape[1] < 1 or masses.shape != (n,):
        raise ValueError('Invalid fingerprint/mass shapes')
    if spectra.ndim != 2 or spectra.shape[0] != n or spectra.shape[1] < 2:
        raise ValueError('Invalid spectrum shape')
    if neighbors.ndim != 2 or neighbors.shape[0] != n or similarities.shape != neighbors.shape:
        raise ValueError('Invalid neighbour/similarity shapes')
    if not all((np.isfinite(x).all() for x in (features, masses, spectra, similarities))):
        raise ValueError('Router data contain non-finite values')
    if (spectra < 0).any() or (masses <= 0).any():
        raise ValueError('Spectra must be nonnegative and masses positive')
    if (neighbors < -1).any() or (neighbors >= n).any():
        raise ValueError('Neighbour indices must be -1 or valid molecular indices')
    if (similarities < 0).any() or (similarities > 1 + 1e-06).any():
        raise ValueError('Similarities must lie in [0, 1]')
    return dict(features=features, masses=masses, spectra=spectra, neighbors=neighbors, similarities=similarities)

def _indices(indices, n, label):
    x = np.asarray(indices, dtype=np.int64).reshape(-1)
    if (x < 0).any() or (x >= n).any() or len(np.unique(x)) != len(x):
        raise ValueError(label + ' must contain distinct valid molecular indices')
    return x

def _eligible(data, indices, min_similarity):
    refs = data['neighbors'][indices]
    sims = data['similarities'][indices]
    eligible = (refs >= 0) & (sims >= float(min_similarity))
    eligible &= refs != indices[:, None]
    safe = np.maximum(refs, 0)
    eligible &= data['spectra'][safe].sum(2) > 1e-12
    return eligible

def _forward_pairs(model, data, queries, refs, sims, device, mode):
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    return model(tensor(data['features'][queries]), tensor(data['features'][refs]), tensor(data['spectra'][refs]), tensor(data['masses'][queries]), tensor(data['masses'][refs]), tensor(sims), mode=mode, return_valid=True)

def predict_router(model, data, query_indices, device='cpu', mode='smart', batch_size=64, min_similarity=0.35):
    data = _router_data(data)
    queries = _indices(query_indices, len(data['features']), 'query_indices')
    if int(batch_size) < 1:
        raise ValueError('batch_size must be positive')
    eligible = _eligible(data, queries, min_similarity)
    where_q, where_k = np.nonzero(eligible)
    result = np.zeros((len(queries), data['spectra'].shape[1]), dtype=np.float32)
    weight_sum = np.zeros(len(queries), dtype=np.float64)
    previous_training = model.training
    model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(where_q), int(batch_size)):
                positions = where_q[start:start + int(batch_size)]
                columns = where_k[start:start + int(batch_size)]
                q = queries[positions]
                r = data['neighbors'][q, columns]
                sims = data['similarities'][q, columns]
                p, valid = _forward_pairs(model, data, q, r, sims, device, mode)
                p = p.cpu().numpy()
                weights = np.power(sims.astype(np.float64), 4) * valid.cpu().numpy()
                np.add.at(result, positions, p * weights[:, None].astype(np.float32))
                np.add.at(weight_sum, positions, weights)
    finally:
        model.train(previous_training)
    covered = weight_sum > 1e-12
    result[covered] /= weight_sum[covered, None].astype(np.float32)
    totals = result.sum(1, dtype=np.float64)
    covered &= totals > 1e-12
    result[covered] /= totals[covered, None].astype(np.float32)
    result[~covered] = 0
    return (result, covered)

def _router_state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()

def fit_router(data, train_indices, eval_indices, epochs, device='cpu', seed=42, mode='smart', batch_size=64, stop_patience=6, select=True, min_similarity=0.35, log_callback=None):
    data = _router_data(data)
    n = len(data['features'])
    train_indices = _indices(train_indices, n, 'train_indices')
    eval_indices = _indices(eval_indices, n, 'eval_indices')
    if not len(train_indices) or int(epochs) < 1 or int(batch_size) < 1:
        raise ValueError('Need nonempty training identities, positive epochs and batch size')
    if mode not in ('smart', 'reweight'):
        raise ValueError('Unknown router mode: ' + str(mode))
    if np.intersect1d(train_indices, eval_indices).size:
        raise ValueError('Inner training and selection identities overlap')
    if select and (not len(eval_indices) or int(stop_patience) < 1):
        raise ValueError('Selection requires held-out identities and positive patience')
    if not select and len(eval_indices):
        raise ValueError('Exact-epoch refit must receive an empty eval_indices array')
    allowed = np.zeros(n, dtype=bool)
    allowed[train_indices] = True
    for label, indices in (('training', train_indices), ('inner selection', eval_indices)):
        refs = data['neighbors'][indices]
        valid = (refs >= 0) & (data['similarities'][indices] >= float(min_similarity))
        if np.any(valid & (refs == indices[:, None])):
            raise ValueError(label + ' contains a self reference')
        if np.any(valid & ~allowed[np.maximum(refs, 0)]):
            raise ValueError(label + ' reference bank contains a held-out identity')
        if len(indices) and np.any(data['spectra'][indices].sum(1) <= 1e-12):
            raise ValueError(label + ' contains an empty target spectrum')
    eligible = _eligible(data, train_indices, min_similarity)
    counts = eligible.sum(1)
    fit_indices = train_indices[counts > 0]
    fit_columns = [np.flatnonzero(x) for x in eligible[counts > 0]]
    if not len(fit_indices):
        raise ValueError('No training molecule has an eligible nonempty reference')
    eval_eligible = _eligible(data, eval_indices, min_similarity).any(1)
    selection_indices = eval_indices[eval_eligible]
    if select and (not len(selection_indices)):
        raise ValueError('No inner selection molecule has an eligible reference')
    torch.manual_seed(int(seed))
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))
    model = AnalogRouter(data['spectra'].shape[1], data['features'].shape[1]).to(device)
    initial_hash = _router_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    history = []
    best_state = None
    best_epoch, stale = (0, 0)
    best_value = -float('inf')
    pair_hash = hashlib.sha256()
    for epoch in range(1, int(epochs) + 1):
        started = time.monotonic()
        model.train()
        chosen = np.array([columns[int(rng.integers(len(columns)))] for columns in fit_columns], dtype=np.int64)
        order = rng.permutation(len(fit_indices))
        pair_hash.update(np.column_stack((fit_indices[order], data['neighbors'][fit_indices[order], chosen[order]])).astype('<i8').tobytes())
        loss_total, trained, degenerate, batches = (0.0, 0, 0, 0)
        for start in range(0, len(order), int(batch_size)):
            positions = order[start:start + int(batch_size)]
            q, columns = (fit_indices[positions], chosen[positions])
            r = data['neighbors'][q, columns]
            sims = data['similarities'][q, columns]
            optimizer.zero_grad(set_to_none=True)
            pred, valid = _forward_pairs(model, data, q, r, sims, device, mode)
            nvalid = int(valid.sum().item())
            degenerate += len(q) - nvalid
            if nvalid == 0:
                continue
            target = torch.as_tensor(data['spectra'][q], dtype=torch.float32, device=device)
            loss = (1.0 - router_cosine(pred[valid], target[valid])).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite router training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_total += float(loss.detach()) * nvalid
            trained += nvalid
            batches += 1
        if trained == 0:
            raise ValueError('All training routes are degenerate; no optimizer update was possible')
        row = dict(epoch=epoch, mode=mode, train_loss=loss_total / trained, train_molecules=trained, training_batches=batches, degenerate_training_pairs=degenerate)
        if select:
            predictions, valid = predict_router(model, data, selection_indices, device=device, mode=mode, batch_size=batch_size, min_similarity=min_similarity)
            with torch.no_grad():
                values = router_cosine(torch.from_numpy(predictions), torch.from_numpy(data['spectra'][selection_indices])).numpy()
            value = float(values.mean())
            row.update(inner_cosine=value, inner_molecules=len(selection_indices), degenerate_inner_predictions=int((~valid).sum()))
            if value > best_value + 1e-08:
                best_value, best_epoch, stale = (value, epoch, 0)
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            else:
                stale += 1
        else:
            best_epoch = epoch
        row['seconds'] = time.monotonic() - started
        history.append(row)
        if log_callback is not None:
            log_callback(dict(row))
        if select and stale >= int(stop_patience):
            break
    if select:
        model.load_state_dict(best_state)
    return dict(model=model, history=history, best_epoch=int(best_epoch), best_inner_cosine=float(best_value) if select else None, train_molecules=len(train_indices), train_covered_molecules=len(fit_indices), inner_molecules=len(eval_indices), inner_covered_molecules=len(selection_indices), initial_state_sha256=initial_hash, sampled_pair_sequence_sha256=pair_hash.hexdigest(), parameters=sum((p.numel() for p in model.parameters())), mode=mode, seed=int(seed))

def normalize_spectra(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Nonfinite or negative predicted/reference intensity')
    sums = x.sum(axis=-1, keepdims=True)
    return np.divide(x, sums, out=np.zeros_like(x), where=sums > 0)

def static_prediction(data, kind):
    if kind not in ('copy', 'hybrid'):
        raise ValueError(kind)
    refs = data['neighbors']
    sims = data['similarities']
    n, dim = data['spectra'].shape
    result = np.zeros((n, dim), np.float32)
    weights = np.zeros(n, np.float64)
    grid = np.arange(dim)
    massbins = np.rint(data['masses']).astype(int)
    for q in range(n):
        for j in range(refs.shape[1]):
            r = int(refs[q, j])
            if r < 0 or sims[q, j] < np.float32(0.35):
                continue
            ref = data['spectra'][r]
            valid_copy = (grid > 0) & (grid <= massbins[q] + 3)
            out = np.zeros(dim, np.float32)
            out[valid_copy] += ref[valid_copy] * (1.0 if kind == 'copy' else 0.5)
            if kind == 'hybrid':
                dst = grid + massbins[q] - massbins[r]
                valid_shift = (grid > 0) & (dst > 0) & (dst < dim) & (dst <= massbins[q] + 3)
                np.add.at(out, dst[valid_shift], 0.5 * ref[valid_shift])
            if out.sum() > 0:
                weight = float(sims[q, j]) ** 4
                result[q] += weight * normalize_spectra(out)
                weights[q] += weight
    return (normalize_spectra(result), weights > 0)

def blend_spectra(base, extra, covered, alpha=0.25):
    base, extra = (normalize_spectra(base), normalize_spectra(extra))
    if extra.shape != base.shape or np.asarray(covered).shape != (len(base),):
        raise ValueError('Blend dimensions differ')
    if np.any(np.asarray(covered) & (extra.sum(axis=1) <= 0)):
        raise ValueError('Covered reference prediction is empty')
    out = base.copy()
    out[covered] = (1 - alpha) * base[covered] + alpha * extra[covered]
    return out
