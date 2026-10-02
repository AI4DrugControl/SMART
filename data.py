import csv
import hashlib
import math
import os
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    try:
        if int(os.environ.get(_name, '4')) < 1:
            os.environ[_name] = '4'
    except ValueError:
        os.environ[_name] = '4'
    os.environ.setdefault(_name, '4')

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reference_signature(source, reference):
    digest = hashlib.sha256()
    indices = sorted(reference['train_indices'], key=lambda index: str(source['candidate_keys'][index]))
    for index in indices:
        for field in ('candidate_keys', 'candidate_smiles'):
            token = str(source[field][index]).encode('utf-8')
            digest.update(len(token).to_bytes(8, 'little'))
            digest.update(token)
        for field in ('spectra', 'masses', 'features'):
            value = np.ascontiguousarray(reference[field][index], dtype='<f4')
            digest.update(value.size.to_bytes(8, 'little'))
            digest.update(value.tobytes())
    return digest.hexdigest()


def parse_spectrum(value, dim=751):
    if int(dim) != dim or dim < 2:
        raise ValueError('Spectrum dimension must be an integer of at least two')
    spectrum = np.zeros(int(dim), dtype=np.float64)
    for peak in str(value).split(';'):
        if not peak.strip():
            continue
        fields = peak.replace(':', ' ').split()
        if len(fields) != 2:
            raise ValueError('Each peak must contain one mass and one intensity')
        try:
            mass, intensity = map(float, fields)
        except ValueError as error:
            raise ValueError('Peak masses and intensities must be numeric') from error
        if not math.isfinite(mass) or not math.isfinite(intensity) or mass < 0 or intensity < 0:
            raise ValueError('Peak masses and intensities must be finite and nonnegative')
        index = math.floor(mass + 0.5)
        if index >= dim:
            raise ValueError('Peak mass exceeds the spectrum dimension')
        spectrum[index] += intensity
    maximum = spectrum.max()
    if maximum <= 0 or not np.isfinite(spectrum).all():
        raise ValueError('Spectrum must contain finite positive intensity')
    return (spectrum / maximum).astype(np.float32)


def molecular_identity(smiles):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError('Invalid SMILES: ' + str(smiles))
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError('Multicomponent structures are unsupported')
    if Chem.GetFormalCharge(molecule) or any(atom.GetIsotope() or not atom.GetAtomicNum() for atom in molecule.GetAtoms()):
        raise ValueError('Structures must be neutral, unlabeled, and contain no wildcard atoms')
    Chem.RemoveStereochemistry(molecule)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    key = Chem.MolToInchiKey(molecule).split('-')[0]
    if len(key) != 14:
        raise ValueError('Cannot generate a molecular identity for: ' + str(smiles))
    return key, canonical


def parse_flag(value):
    token = str(value).strip().lower()
    if token in ('', '0', 'false', 'no'):
        return False
    if token in ('1', 'true', 'yes'):
        return True
    raise ValueError('is_swgdrug must be true or false')


def load_dataset(path, dim=751, require_spectra=True):
    required = {'record_id', 'name', 'smiles', 'split', 'spectrum', 'electron_energy_ev', 'source'}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError('Dataset is missing columns: ' + ', '.join(sorted(required - set(reader.fieldnames or []))))
        records = list(reader)
    if not records:
        raise ValueError('Dataset has no records')
    spectra, keys, smiles, split, metadata, flags = [], [], [], [], [], []
    first_rows, identity_splits, record_owners = {}, {}, {}
    for index, record in enumerate(records):
        try:
            if None in record or any(value is None for value in record.values()):
                raise ValueError('CSV row has an unexpected number of fields')
            key, canonical = molecular_identity(record['smiles'])
            label = record['split'].strip().lower()
            if label not in {'train', 'val', 'test'}:
                raise ValueError('split must be train, val, or test')
            if key in identity_splits and identity_splits[key] != label:
                raise ValueError('Molecular identity appears in multiple splits: ' + key)
            identity_splits[key] = label
            first_rows.setdefault(key, index)
            identifiers = record_ids(record['record_id'])
            if not identifiers:
                raise ValueError('record_id must be nonempty')
            for identifier in identifiers:
                if identifier in record_owners and record_owners[identifier] != key:
                    raise ValueError('A record ID refers to multiple molecular identities')
                record_owners[identifier] = key
            if not require_spectra and label != 'train' and not record['spectrum'].strip():
                spectra.append(np.zeros(int(dim), dtype=np.float32))
            else:
                spectra.append(parse_spectrum(record['spectrum'], dim))
            keys.append(key)
            smiles.append(canonical)
            split.append(label)
            metadata.append({'electron_energy_ev': parse_energy(record['electron_energy_ev'])})
            flags.append(parse_flag(record.get('is_swgdrug', 'swgdrug' in record['source'].lower())))
        except ValueError as error:
            raise ValueError('CSV row ' + str(index + 2) + ': ' + str(error)) from error
    keys, smiles, split = (np.asarray(value, dtype=str) for value in (keys, smiles, split))
    candidate_keys = np.asarray(sorted(first_rows), dtype=str)
    candidate_rows = np.asarray([first_rows[key] for key in candidate_keys], dtype=np.int64)
    return {'spectra': np.asarray(spectra, dtype=np.float32), 'keys': keys, 'smiles': smiles,
            'split': split, 'records': records, 'metadata': metadata, 'candidate_keys': candidate_keys,
            'candidate_smiles': smiles[candidate_rows], 'candidate_rows': candidate_rows,
            'candidate_split': split[candidate_rows], 'data': {
                'record_id': [record['record_id'] for record in records],
                'source': [record['source'] for record in records], 'is_drug': np.asarray(flags, dtype=bool)}}


def validate_predictions(spectrum, source):
    spectrum = np.asarray(spectrum, dtype=np.float32)
    expected = (len(source['candidate_keys']), source['spectra'].shape[1])
    if spectrum.shape != expected:
        raise ValueError('Prediction shape must be ' + str(expected))
    if not np.isfinite(spectrum).all() or (spectrum < 0).any():
        raise ValueError('Predictions must be finite and nonnegative')
    return spectrum


def read_predictions(path, source):
    with np.load(path, allow_pickle=False) as archive:
        if not {'group_key', 'smiles', 'spectrum'}.issubset(archive.files):
            raise ValueError('Prediction cache requires group_key, smiles, and spectrum')
        keys = np.asarray(archive['group_key'], dtype=str)
        smiles = np.asarray(archive['smiles'], dtype=str)
        if not np.array_equal(keys, np.asarray(source['candidate_keys'], dtype=str)):
            raise ValueError('Prediction group keys do not exactly match candidate order')
        if not np.array_equal(smiles, np.asarray(source['candidate_smiles'], dtype=str)):
            raise ValueError('Prediction SMILES do not exactly match candidate order')
        if 'candidate_split' in archive and not np.array_equal(archive['candidate_split'], source['candidate_split']):
            raise ValueError('Data splits changed since predictions were generated')
        return validate_predictions(archive['spectrum'], source)


def save_predictions(path, source, spectrum, **extras):
    if {'group_key', 'smiles', 'spectrum'} & extras.keys():
        raise ValueError('Prediction metadata cannot replace required arrays')
    spectrum = validate_predictions(spectrum, source)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        np.savez_compressed(handle, group_key=np.asarray(source['candidate_keys'], dtype=str),
                            smiles=np.asarray(source['candidate_smiles'], dtype=str), spectrum=spectrum, **extras)


def parse_energy(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError):
        return None


def record_ids(value):
    return {x.strip() for x in str(value).split('|') if x.strip() and x.strip().lower() not in {'none', 'nan', 'null'}}


def inner_split(train_indices, keys, drugflags, seed=42, connectivity=None):
    indices = np.asarray(train_indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError('Reference preparation needs at least two training molecule identities')
    identities = {}
    for index in indices:
        identity = str(keys[index]) if connectivity is None else str(connectivity[index])
        identities.setdefault(identity, []).append(int(index))
    if len(identities) < 2:
        raise ValueError('Reference preparation needs two distinct training connectivities')
    identity_keys = {identity: min((str(keys[i]) for i in members)) for identity, members in identities.items()}
    identity_flags = {identity: any((bool(drugflags[i]) for i in members)) for identity, members in identities.items()}

    def stable_order(identity):
        token = '%s|%s' % (seed, identity_keys[identity])
        return (hashlib.sha256(token.encode('utf-8')).hexdigest(), identity_keys[identity])
    selection = []
    for flag in (False, True):
        group = sorted((identity for identity in identities if identity_flags[identity] == flag), key=stable_order)
        if len(group) >= 2:
            count = min(len(group) - 1, max(1, int(round(0.1 * len(group)))))
            selection.extend((i for identity in group[:count] for i in identities[identity]))
    if not selection:
        selection = list(identities[sorted(identities, key=stable_order)[0]])
    selected = set(selection)
    fit = np.asarray([int(i) for i in indices if int(i) not in selected], dtype=np.int64)
    select = np.asarray(sorted(selected), dtype=np.int64)
    if not len(fit) or not len(select) or set(fit) & set(select):
        raise ValueError('Invalid inner molecular partition')
    return (fit, select)


def consensus(rows, intensity_rows, record_ids, energies):
    rows = list(map(int, rows))
    matrix = np.asarray(intensity_rows, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(rows):
        raise ValueError('Reference rows and intensity matrix differ')
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix.sum(axis=1) <= 0).any():
        raise ValueError('Invalid training-reference intensity')
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = (find(i), find(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    seen_ids, seen_spectra = ({}, {})
    for local, row in enumerate(rows):
        digest = hashlib.sha256(np.round(matrix[local], 6).tobytes()).hexdigest()
        if digest in seen_spectra:
            union(local, seen_spectra[digest])
        else:
            seen_spectra[digest] = local
        for rid in record_ids[row]:
            if rid in seen_ids:
                union(local, seen_ids[rid])
            else:
                seen_ids[rid] = local
    components = {}
    for local in range(len(rows)):
        components.setdefault(find(local), []).append(local)

    def priority(local):
        energy = energies[rows[local]]
        category = 0 if energy is not None and abs(energy - 70.0) <= 0.5 else 1 if energy is None else 2
        return (category, abs(energy - 70.0) if energy is not None else 0.0, rows[local])
    representatives = [min(members, key=priority) for members in components.values()]
    known70 = [j for j in representatives if energies[rows[j]] is not None and abs(energies[rows[j]] - 70.0) <= 0.5]
    unknown = [j for j in representatives if energies[rows[j]] is None]
    if known70:
        selected, policy = (known70, 'known70_preferred')
    elif unknown:
        selected, policy = (unknown, 'unknown_energy_fallback')
    else:
        closest = min((energies[rows[j]] for j in representatives), key=lambda e: (abs(e - 70.0), e))
        selected = [j for j in representatives if abs(energies[rows[j]] - closest) <= 0.5]
        policy = 'closest_known_energy_fallback'
    consensus = matrix[selected].mean(axis=0)
    consensus /= consensus.sum()
    selected_set = set(selected)
    representative_for = {}
    for members in components.values():
        rep = min(members, key=priority)
        for local in members:
            representative_for[local] = rep
    row_audit = []
    for local, row in enumerate(rows):
        rep = representative_for[local]
        row_audit.append({'record_index': row, 'electron_energy_ev': energies[row], 'component_representative_record': rows[rep], 'is_component_representative': int(local == rep), 'selected_for_consensus': int(local in selected_set), 'consensus_record_weight': 1.0 / len(selected) if local in selected_set else 0.0, 'condition_policy': policy})
    details = {'raw_records': len(rows), 'deduplicated_components': len(components), 'selected_records': len(selected), 'condition_policy': policy, 'selected_record_indices': [rows[j] for j in selected], 'selected_energy_values': sorted({energies[rows[j]] for j in selected if energies[rows[j]] is not None}), 'unknown_energy_selected': any((energies[rows[j]] is None for j in selected))}
    return (consensus.astype(np.float32), row_audit, details)


def choose_neighbors(similarity, reference_indices, target_key, candidate_keys, target_connectivity, connectivity, minimum=0.35, count=3):
    similarity = np.asarray(similarity, dtype=np.float64)
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    if similarity.shape != reference_indices.shape:
        raise ValueError('Neighbor score/reference dimensions differ')
    eligible = np.flatnonzero(similarity >= minimum)
    keep = [int(j) for j in eligible if str(candidate_keys[reference_indices[j]]) != str(target_key) and str(connectivity[reference_indices[j]]) != str(target_connectivity)]
    keep.sort(key=lambda j: (-float(similarity[j]), str(candidate_keys[reference_indices[j]])))
    indices = np.full(count, -1, dtype=np.int64)
    values = np.zeros(count, dtype=np.float32)
    for slot, position in enumerate(keep[:count]):
        indices[slot] = reference_indices[position]
        values[slot] = similarity[position]
    return (indices, values, len(keep))


def prepare_reference_data(source):
    metadata = source['metadata']
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    keys = np.asarray(source['keys'], dtype=str)
    split = np.asarray(source['split'], dtype=str)
    candidate_keys = np.asarray(source['candidate_keys'], dtype=str)
    candidate_smiles = np.asarray(source['candidate_smiles'], dtype=str)
    candidate_rows = np.asarray(source['candidate_rows'], dtype=np.int64)
    candidate_split = np.asarray(source['candidate_split'], dtype=str)
    n = len(candidate_keys)
    if len(set(candidate_keys)) != n or len(metadata) != len(keys):
        raise ValueError('Malformed candidate identity or metadata alignment')
    if not np.array_equal(keys[candidate_rows], candidate_keys) or not np.array_equal(split[candidate_rows], candidate_split):
        raise ValueError('Candidate/record identity alignment differs')
    if len(candidate_smiles) != n or len(candidate_split) != n:
        raise ValueError('Candidate metadata shape differs')
    by_key = {str(key): [] for key in candidate_keys}
    for row, key in enumerate(keys):
        if str(key) not in by_key:
            raise ValueError('Record identity missing from full candidate library')
        by_key[str(key)].append(row)
    for key, rows in by_key.items():
        if not rows or len(set(split[rows])) != 1:
            raise ValueError('Identity spans multiple data splits: ' + key)
    train_indices = np.flatnonzero(candidate_split == 'train').astype(np.int64)
    if len(train_indices) < 2:
        raise ValueError('Insufficient training molecule identities')
    record_values = source.get('data', {}).get('record_id', [''] * len(keys))
    if len(record_values) != len(keys):
        raise ValueError('Record ID dimension differs')
    record_sources = source.get('data', {}).get('source', [''] * len(keys))
    if len(record_sources) != len(keys):
        raise ValueError('Record source dimension differs')
    ids_by_row = [record_ids(value) for value in record_values]
    owner, cross_identity = ({}, [])
    for row, ids in enumerate(ids_by_row):
        for rid in ids:
            key = str(keys[row])
            if rid in owner and owner[rid] != key:
                cross_identity.append({'record_id': rid, 'first_key': owner[rid], 'other_key': key})
            else:
                owner[rid] = key
    if cross_identity:
        raise ValueError('Shared record IDs cross molecular identities')
    flags = np.asarray(source.get('data', {}).get('is_drug', np.zeros(len(keys), bool)), dtype=bool)
    if flags.shape != (len(keys),):
        raise ValueError('Source-coverage flags have invalid shape')
    drugflags = np.asarray([bool(flags[by_key[str(key)]].any()) for key in candidate_keys], dtype=bool)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    features = np.zeros((n, 2048), dtype=np.float32)
    masses = np.empty(n, dtype=np.float32)
    connectivity, fingerprints = ([], [])
    for index, smi in enumerate(candidate_smiles):
        molecule = Chem.MolFromSmiles(str(smi))
        if molecule is None:
            raise ValueError('RDKit cannot read candidate: ' + str(smi))
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        connectivity.append(canonical)
        fp = generator.GetFingerprint(molecule)
        fingerprints.append(fp)
        DataStructs.ConvertToNumpyArray(fp, features[index])
        masses[index] = Descriptors.ExactMolWt(molecule)
    connectivity = np.asarray(connectivity, dtype=str)
    connectivity_splits = {}
    for identity, label in zip(connectivity, candidate_split):
        connectivity_splits.setdefault(str(identity), set()).add(str(label))
    if any((len(labels) != 1 for labels in connectivity_splits.values())):
        raise ValueError('Canonical connectivity appears in multiple original splits')
    inner_fit, inner_select = inner_split(train_indices, candidate_keys, drugflags, connectivity=connectivity)
    intensity_source = source['spectra']
    if len(intensity_source.shape) != 2 or intensity_source.shape[0] != len(keys):
        raise ValueError('Source spectral array has invalid dimensions')
    spectra = np.zeros((n, intensity_source.shape[1]), dtype=np.float32)
    energies = {}
    for row in np.flatnonzero(split == 'train'):
        energies[int(row)] = parse_energy(metadata[int(row)].get('electron_energy_ev'))
    manifest, bank_records, policies = ([], {}, {})
    for index in train_indices:
        key = str(candidate_keys[index])
        rows = by_key[key]
        if any((split[row] != 'train' for row in rows)):
            raise ValueError('Nontraining row entered reference bank')
        reference, row_audit, details = consensus(rows, intensity_source[rows], ids_by_row, energies)
        spectra[index] = reference
        bank_records[key] = details
        policies[details['condition_policy']] = policies.get(details['condition_policy'], 0) + 1
        for row in row_audit:
            row.update(candidate_index=int(index), group_key=key, split='train', source_flag=int(drugflags[index]), record_id=str(record_values[row['record_index']]), record_source=str(record_sources[row['record_index']]))
            manifest.append(row)
    neighbors_full = np.full((n, 3), -1, dtype=np.int64)
    similarities_full = np.zeros((n, 3), dtype=np.float32)
    neighbors_inner = np.full((n, 3), -1, dtype=np.int64)
    similarities_inner = np.zeros((n, 3), dtype=np.float32)
    counts_full, counts_inner = (np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))
    reference_fps = [fingerprints[int(i)] for i in train_indices]
    fit_mask = np.isin(train_indices, inner_fit)
    neighbor_rows = []
    for index in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[index], reference_fps), dtype=np.float64)
        for name, references, similarity in [('full', train_indices, sims), ('inner', train_indices[fit_mask], sims[fit_mask])]:
            selected, values, available = choose_neighbors(similarity, references, candidate_keys[index], candidate_keys, connectivity[index], connectivity)
            if name == 'full':
                neighbors_full[index], similarities_full[index], counts_full[index] = (selected, values, available)
            else:
                neighbors_inner[index], similarities_inner[index], counts_inner[index] = (selected, values, available)
            valid_slots = np.flatnonzero(selected >= 0)
            if not len(valid_slots):
                neighbor_rows.append({'bank': name, 'candidate_index': index, 'candidate_key': str(candidate_keys[index]), 'candidate_split': str(candidate_split[index]), 'neighbor_slot': -1, 'reference_index': -1, 'reference_key': '', 'reference_split': '', 'similarity': 0.0, 'available_count': available})
            for slot in valid_slots:
                ref = int(selected[slot])
                if candidate_split[ref] != 'train' or candidate_keys[ref] == candidate_keys[index] or connectivity[ref] == connectivity[index]:
                    raise AssertionError('Invalid reference-bank neighbor')
                neighbor_rows.append({'bank': name, 'candidate_index': index, 'candidate_key': str(candidate_keys[index]), 'candidate_split': str(candidate_split[index]), 'neighbor_slot': int(slot), 'reference_index': ref, 'reference_key': str(candidate_keys[ref]), 'reference_split': str(candidate_split[ref]), 'similarity': float(values[slot]), 'available_count': available})
    inner_select_set = set(inner_select.tolist())
    partition_rows = [{'candidate_index': int(i), 'group_key': str(candidate_keys[i]), 'source_flag': int(drugflags[i]), 'partition': 'inner_select' if i in inner_select_set else 'inner_fit'} for i in train_indices]
    audit = {'reference_molecules': int(len(train_indices)), 'raw_reference_records': int(len(manifest)), 'deduplicated_reference_components': int(sum((v['deduplicated_components'] for v in bank_records.values()))), 'selected_reference_records': int(sum((v['selected_records'] for v in bank_records.values()))), 'condition_policy_counts': policies, 'inner_fit_molecules': int(len(inner_fit)), 'inner_select_molecules': int(len(inner_select)), 'inner_seed': 42, 'inner_split_rule': 'Group canonical connectivity; stratify source flag; sort SHA256(seed|min_group_key); select rounded 10% of connectivities, at least one per nonsingleton stratum.', 'neighbors': 3, 'minimum_morgan_tanimoto': 0.35, 'fingerprint': 'Morgan radius=2 bits=2048 no chirality', 'full_candidates_with_reference': int((counts_full > 0).sum()), 'inner_candidates_with_reference': int((counts_inner > 0).sum()), 'cross_identity_shared_record_ids': 0, 'exact_connectivity_duplicate_candidate_count': int(n - len(set(connectivity))), 'nontrain_intensity_rows_read': 0, 'baseline_predictions_accessed': False, 'reference_self_exclusion': 'Every candidate excludes same group_key and canonical nonisomeric SMILES.', 'condition_policy': 'Prefer known 70 +/-0.5 eV; else unknown; else closest known energy to 70 and records within 0.5 eV of that energy.', 'duplicate_policy': 'Connected components of shared record IDs OR equal rounded-six-decimal L1 spectrum; one representative per component.', 'energy_warning': 'Unknown energy is not assumed to be 70 eV; distinct database records do not establish independent acquisition.', 'source_flag_warning': 'is_drug is SWGDRUG-source coverage, not confirmed drug or NPS status.', 'inner_selection_limit': 'Base predictor training is external; independence from inner selection must be checked for the supplied predictions.'}
    return {'features': features, 'masses': masses, 'spectra': spectra, 'drugflags': drugflags, 'train_indices': train_indices, 'inner_fit_indices': inner_fit, 'inner_select_indices': inner_select, 'neighbors_full': neighbors_full, 'similarities_full': similarities_full, 'neighbors_inner': neighbors_inner, 'similarities_inner': similarities_inner, 'neighbor_counts_full': counts_full, 'neighbor_counts_inner': counts_inner, 'bank_records': bank_records, 'audit': audit, 'neighbors': neighbors_full, 'similarities': similarities_full, 'bank_manifest': manifest, 'reference_neighbors': neighbor_rows, 'inner_partition': partition_rows}
