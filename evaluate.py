import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data import load_dataset, prepare_reference_data, read_predictions, reference_signature


METRICS = ("cosine", "weighted_cosine", "top1", "top5", "top10", "mrr")
TIE_ATOL = 1e-7


def features(spectra, weighted=False):
    values = np.asarray(spectra, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Spectra must be finite, nonnegative matrices")
    result = np.sqrt(values)
    if weighted:
        result *= np.arange(values.shape[1], dtype=np.float32)[None, :]
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    return np.divide(result, norm, out=np.zeros_like(result), where=norm > 0)


def cosine(observed, predicted, weighted=False):
    if np.shape(observed) != np.shape(predicted):
        raise ValueError("Observed and predicted spectrum shapes differ")
    return np.clip(np.sum(features(observed, weighted) * features(predicted, weighted), axis=1), 0, 1)


def retrieval(predicted, observed, correct, batch_size=128):
    predicted = np.asarray(predicted, dtype=np.float32)
    observed = np.asarray(observed, dtype=np.float32)
    correct = np.asarray(correct)
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("Retrieval batch size must be a positive integer")
    if predicted.ndim != 2 or observed.ndim != 2 or predicted.shape[1] != observed.shape[1]:
        raise ValueError("Candidate and query spectrum dimensions differ")
    if not len(predicted) or not len(observed) or not predicted.shape[1]:
        raise ValueError("Candidate and query spectra must be nonempty")
    if correct.shape != (len(observed),) or correct.dtype.kind not in "iu":
        raise ValueError("Correct candidate indices must be one integer per query")
    if (correct < 0).any() or (correct >= len(predicted)).any():
        raise ValueError("Correct candidate index is outside the library")
    candidates = features(predicted)
    queries = features(observed)
    ranks = np.empty(len(observed), dtype=np.int64)
    valid = (candidates[correct].sum(axis=1) > 0) & (queries.sum(axis=1) > 0)
    for start in range(0, len(observed), batch_size):
        stop = min(start + batch_size, len(observed))
        scores = np.clip(queries[start:stop] @ candidates.T, 0, 1)
        target = scores[np.arange(stop - start), correct[start:stop]]
        ranks[start:stop] = np.sum(scores >= target[:, None] - TIE_ATOL, axis=1)
    return {
        "cosine": cosine(observed, predicted[correct]),
        "weighted_cosine": cosine(observed, predicted[correct], True),
        "ranks": ranks,
        "valid": valid,
        "top1": ((ranks <= 1) & valid).astype(float),
        "top5": ((ranks <= 5) & valid).astype(float),
        "top10": ((ranks <= 10) & valid).astype(float),
        "mrr": np.where(valid, 1.0 / ranks, 0.0),
    }


def aggregate(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["group_key"], []).append(record)
    molecules = []
    for key, rows in sorted(grouped.items()):
        molecules.append({
            "group_key": key,
            "smiles": rows[0]["smiles"],
            "n_records": len(rows),
            "is_swgdrug_covered": any(row["is_swgdrug_covered"] for row in rows),
            "prediction_nonempty": all(row["prediction_nonempty"] for row in rows),
            **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
        })
    return molecules


def summarize(records, molecules):
    return {
        "n_records": len(records),
        "n_molecules": len(molecules),
        "empty_prediction_records": sum(not row["prediction_nonempty"] for row in records),
        "empty_prediction_molecules": sum(not row["prediction_nonempty"] for row in molecules),
        "macro": {metric: float(np.mean([row[metric] for row in molecules])) for metric in METRICS},
        "record_mean": {metric: float(np.mean([row[metric] for row in records])) for metric in METRICS},
    }


def evaluate(source, predicted, split="test", library="fixed-measured", batch_size=128, expected_reference_signature=""):
    if split not in {"train", "val", "test"} or library not in {"fixed-measured", "predicted"}:
        raise ValueError("Unknown split or candidate library policy")
    if split == "train" and library == "fixed-measured":
        raise ValueError("Fixed-measured retrieval allows val/test queries only; train queries would self-match")
    keys = np.asarray(source["candidate_keys"], dtype=str)
    smiles = np.asarray(source["candidate_smiles"], dtype=str)
    candidate_split = np.asarray(source["candidate_split"], dtype=str)
    predicted = np.asarray(predicted, dtype=np.float32)
    observed = np.asarray(source["spectra"], dtype=np.float32)
    record_keys = np.asarray(source["keys"], dtype=str)
    record_split = np.asarray(source["split"], dtype=str)
    if keys.ndim != 1 or not len(keys) or len(set(keys)) != len(keys) or np.any(keys == ""):
        raise ValueError("The full candidate library must contain unique, nonempty structure keys")
    if smiles.shape != keys.shape or candidate_split.shape != keys.shape:
        raise ValueError("Candidate metadata dimensions differ")
    if not np.isin(candidate_split, ["train", "val", "test"]).all():
        raise ValueError("Candidate splits must be train, val or test")
    if observed.ndim != 2 or predicted.shape != (len(keys), observed.shape[1]):
        raise ValueError("Prediction shape does not cover the complete candidate library")
    if record_keys.shape != (len(observed),) or record_split.shape != record_keys.shape:
        raise ValueError("Record metadata dimensions differ")
    features(predicted)
    query_rows = np.flatnonzero(record_split == split)
    if not len(query_rows):
        raise ValueError(f"No query records in split {split!r}")
    lookup = {key: index for index, key in enumerate(keys)}
    if any(key not in lookup for key in record_keys):
        raise ValueError("A measured structure is absent from the full candidate library")
    correct = np.asarray([lookup[key] for key in record_keys[query_rows]], dtype=np.int64)
    if not np.all(candidate_split[correct] == split):
        raise ValueError("Record and candidate split assignments disagree")
    candidates = predicted.copy()
    known = candidate_split == "train"
    reference = None
    if library == "fixed-measured" or expected_reference_signature:
        reference = prepare_reference_data(source)
    if expected_reference_signature and reference_signature(source, reference) != expected_reference_signature:
        raise ValueError("Training references changed since predictions were generated")
    if library == "fixed-measured":
        bank = np.asarray(reference["spectra"], dtype=np.float32)
        if bank.shape != predicted.shape or not known.any():
            raise ValueError("A complete training-only measured reference bank is required")
        features(bank)
        if np.any(bank[~known] != 0) or np.any(bank[known].sum(axis=1) <= 0):
            raise ValueError("Reference spectra must be nonempty for train candidates and zero elsewhere")
        candidates[known] = bank[known]
    values = retrieval(candidates, observed[query_rows], correct, batch_size)
    values["cosine"] = cosine(observed[query_rows], predicted[correct])
    values["weighted_cosine"] = cosine(observed[query_rows], predicted[correct], True)
    metadata = source.get("data", {})
    sources = np.asarray(metadata.get("source", [row.get("source", "") for row in source["records"]]), dtype=str)
    record_ids = np.asarray(metadata.get("record_id", [row.get("record_id", row.get("id", "")) for row in source["records"]]), dtype=str)
    flags = np.asarray(metadata.get("is_drug", ["SWGDRUG" in value.upper() for value in sources]), dtype=bool)
    if any(value.shape != (len(observed),) for value in (sources, record_ids, flags)):
        raise ValueError("Record source metadata dimensions differ")
    records = []
    for position, row in enumerate(query_rows):
        index = correct[position]
        records.append({
            "record_index": int(row), "record_id": str(record_ids[row]),
            "group_key": str(keys[index]), "smiles": str(smiles[index]), "source": str(sources[row]),
            "is_swgdrug_covered": bool(flags[row]),
            "prediction_nonempty": bool(predicted[index].sum() > 0),
            "retrieval_valid": bool(values["valid"][position]),
            "rank_worst_tie": int(values["ranks"][position]),
            **{metric: float(values[metric][position]) for metric in METRICS},
        })
    molecules = aggregate(records)
    summary = {"all": summarize(records, molecules)}
    swg_molecules = [row for row in molecules if row["is_swgdrug_covered"]]
    if swg_molecules:
        swg_keys = {row["group_key"] for row in swg_molecules}
        summary["swgdrug_covered"] = summarize([row for row in records if row["group_key"] in swg_keys], swg_molecules)
    identity = "\n".join(f"{key}\t{smi}" for key, smi in zip(keys, smiles))
    return {
        "split": split, "library_policy": library, "spectrum_dim": int(predicted.shape[1]),
        "interpretation": "Descriptive training-set evaluation" if split == "train" else "Held-out structure evaluation",
        "candidate_count": len(keys), "known_train_candidates": int(known.sum()),
        "known_measured_candidates": int(known.sum()) if library == "fixed-measured" else 0,
        "predicted_candidates": int((~known).sum()) if library == "fixed-measured" else len(keys),
        "empty_prediction_candidates": int((predicted.sum(axis=1) <= 0).sum()),
        "candidate_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "metric_protocol": {
            "cosine": "cos(sqrt(observed), sqrt(original prediction))",
            "weighted_cosine": "cos(mz * sqrt(observed), mz * sqrt(original prediction))",
            "retrieval_score": "sqrt_cosine", "first_bin_mz": 0, "bin_width_da": 1,
            "tie_policy": "worst rank", "tie_atol": TIE_ATOL,
            "empty_prediction": "failure: cosine, weighted cosine, top-k and reciprocal rank are zero",
            "aggregation": "mean records within each structure, then equal-weight mean over structures",
            "candidate_policy": "all unique structures in the input dataset; no mass filter",
            "swgdrug_cohort": "structures with any SWGDRUG-source record in the evaluated split; all their query records",
            "source_flag_meaning": "SWGDRUG source coverage does not establish confirmed NPS status",
        },
        "summary": summary, "per_molecule": molecules, "per_record": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate spectra and full-library structure retrieval")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data", default="data/example.csv")
    parser.add_argument("--predictions", default="outputs/predictions.npz")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--library", choices=("fixed-measured", "predicted"), default="fixed-measured")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source = load_dataset(args.data, dim=int(config.get("spectrum_dim", 751)))
    predicted = read_predictions(args.predictions, source)
    metadata = {}
    provenance = "unknown"
    demonstration_only = False
    signature = ""
    with np.load(args.predictions, allow_pickle=False) as archive:
        if "upstream_train_keys" in archive:
            training_keys = np.asarray(archive["upstream_train_keys"], dtype=str)
            if training_keys.ndim != 1:
                raise ValueError("Upstream training identities must be a one-dimensional array")
            evaluated_keys = set(source["candidate_keys"][source["candidate_split"] == args.split])
            if args.split != "train" and evaluated_keys.intersection(training_keys.tolist()):
                raise ValueError("Upstream predictor has seen structures in the evaluated held-out split")
            provenance = "recorded"
        if "upstream_provenance" in archive and str(archive["upstream_provenance"].item()) == "unknown":
            provenance = "unknown"
        if "demonstration_only" in archive:
            demonstration_only = bool(archive["demonstration_only"].item())
        if "reference_signature" in archive:
            signature = str(archive["reference_signature"].item())
        for field in ("method", "seed"):
            if field in archive and archive[field].size == 1:
                metadata[field] = archive[field].item()
    result = evaluate(source, predicted, args.split, args.library, args.batch_size, signature)
    result.update(metadata)
    result.update(upstream_provenance=provenance, demonstration_only=demonstration_only)
    if demonstration_only:
        result["interpretation"] = "Execution demonstration only; not an independent benchmark"
    elif provenance == "unknown" and args.split != "train":
        result["interpretation"] = "Split-based evaluation; upstream training provenance unavailable"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
