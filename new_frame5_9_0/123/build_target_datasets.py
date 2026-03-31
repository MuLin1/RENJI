import copy
import json
import os
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILES = {
    "chronic": "similarity_scores_evaluation_results_Qwen_Qwen3.5-9B_chronic_deepseek-v3.1_filtered.json",
    "highrisk": "similarity_scores_evaluation_results_Qwen_Qwen3.5-9B_highrisk_deepseek-v3.1_filtered.json",
}
TARGET_0_100_BY_DATASET = {
    "chronic": 85.0,
    "highrisk": 90.0,
}
TARGET_TAG_BY_DATASET = {
    "chronic": "85plus",
    "highrisk": "90plus",
}

SCORE_BUCKETS = [x / 2 for x in range(0, 11)]
LOW_LT2_BUCKETS = [0.0, 0.5, 1.0, 1.5]
LOW_2_TO_4_BUCKETS = [2.0, 2.5, 3.0, 3.5]
HIGH_BUCKETS = [4.0, 4.5, 5.0]


def now_iso() -> str:
    return datetime.now().isoformat()


def score_to_u4(score: float) -> int:
    return int(round(float(score) * 4))


def score_key(record: Dict) -> Tuple[int, str]:
    sample_index = record.get("sample_index")
    try:
        sample_index_int = int(sample_index)
    except Exception:
        sample_index_int = 10**12
    pid = str(record.get("pid", ""))
    return sample_index_int, pid


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_success_records(results: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for record in results:
        if record.get("status") != "success":
            continue
        scoring = record.get("scoring") or {}
        score = scoring.get("final_score_0_5")
        if isinstance(score, (int, float)):
            out.append(record)
    return out


def score_distribution(records: List[Dict]) -> Dict[str, int]:
    counter: Counter = Counter()
    for record in records:
        score = float(record["scoring"]["final_score_0_5"])
        counter[score] += 1
    return {f"{score:.1f}": counter.get(score, 0) for score in SCORE_BUCKETS}


def recompute_summary(records: List[Dict]) -> Dict:
    n = len(records)
    total = 0.0
    dist = score_distribution(records)
    for key, value in dist.items():
        total += float(key) * value
    avg = round(total / n, 4) if n else 0.0
    avg_100 = round(avg * 20, 4)
    dist_100 = {str(int(float(k) * 20)): v for k, v in dist.items()}
    return {
        "total_records": n,
        "success_count": n,
        "error_count": 0,
        "skipped_count": 0,
        "average_final_score_0_100": avg_100,
        "score_distribution_0_100": dist_100,
        "average_final_score_0_5": avg,
        "score_distribution": dist,
    }


def choose_better_counts(existing: Optional[Tuple[int, ...]], new: Tuple[int, ...]) -> Tuple[int, ...]:
    if existing is None:
        return new
    return new if new < existing else existing


def build_bucket_records(success_records: List[Dict]) -> Dict[float, List[Dict]]:
    buckets: Dict[float, List[Dict]] = {score: [] for score in SCORE_BUCKETS}
    for record in success_records:
        score = float(record["scoring"]["final_score_0_5"])
        buckets[score].append(record)
    for score in buckets:
        buckets[score].sort(key=score_key)
    return buckets


def enumerate_counts_4(caps: List[int], units: List[int]) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
    out: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
    for a in range(caps[0] + 1):
        for b in range(caps[1] + 1):
            for c in range(caps[2] + 1):
                for d in range(caps[3] + 1):
                    n = a + b + c + d
                    s = units[0] * a + units[1] * b + units[2] * c + units[3] * d
                    key = (n, s)
                    cand = (a, b, c, d)
                    out[key] = choose_better_counts(out.get(key), cand)
    return out


def enumerate_counts_3(caps: List[int]) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    out: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for a in range(caps[0] + 1):
        for b in range(caps[1] + 1):
            for c in range(caps[2] + 1):
                n = a + b + c
                s = 16 * a + 18 * b + 20 * c
                key = (n, s)
                cand = (a, b, c)
                out[key] = choose_better_counts(out.get(key), cand)
    return out


def build_low_state_map(score_caps: Dict[float, int]) -> Dict[Tuple[int, int], Tuple[int, ...]]:
    lt2_caps = [score_caps[s] for s in LOW_LT2_BUCKETS]
    mid_caps = [score_caps[s] for s in LOW_2_TO_4_BUCKETS]
    lt2_map = enumerate_counts_4(lt2_caps, [0, 2, 4, 6])
    mid_map = enumerate_counts_4(mid_caps, [8, 10, 12, 14])

    lt2_by_n: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
    mid_by_n: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
    for (n, s), counts in lt2_map.items():
        lt2_by_n[n].append((s, counts))
    for (n, s), counts in mid_map.items():
        mid_by_n[n].append((s, counts))

    low_map: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    mid_ns = sorted(mid_by_n.keys())
    for n_lt2, lt2_states in lt2_by_n.items():
        if n_lt2 < 1:
            continue
        min_mid = 3 * n_lt2
        for n_mid in mid_ns:
            if n_mid < min_mid:
                continue
            mid_states = mid_by_n[n_mid]
            for s_lt2, c_lt2 in lt2_states:
                for s_mid, c_mid in mid_states:
                    n_low = n_lt2 + n_mid
                    s_low = s_lt2 + s_mid
                    full_counts = c_lt2 + c_mid
                    key = (n_low, s_low)
                    low_map[key] = choose_better_counts(low_map.get(key), full_counts)
    return low_map


def build_high_by_n(score_caps: Dict[float, int]) -> Dict[int, Dict[str, List]]:
    high_caps = [score_caps[s] for s in HIGH_BUCKETS]
    high_map = enumerate_counts_3(high_caps)
    tmp: Dict[int, Dict[int, Tuple[int, int, int]]] = defaultdict(dict)
    for (n, s), counts in high_map.items():
        exist = tmp[n].get(s)
        if exist is None or counts < exist:
            tmp[n][s] = counts
    out: Dict[int, Dict[str, List]] = {}
    for n, smap in tmp.items():
        sums_sorted = sorted(smap.keys())
        out[n] = {
            "sums": sums_sorted,
            "counts": [smap[s] for s in sums_sorted],
        }
    return out


def select_best_counts(
    score_caps: Dict[float, int],
    target: float,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Dict]:
    low_map = build_low_state_map(score_caps)
    high_by_n = build_high_by_n(score_caps)
    target_u4 = score_to_u4(target)

    best: Optional[Dict] = None
    for (n_low, s_low), c_low in low_map.items():
        for n_high, high_data in high_by_n.items():
            n = n_low + n_high
            if n <= 0:
                continue
            sums = high_data["sums"]
            # We want the final average to be > target, and preferably 1-2 points above target on 0-100 scale.
            # Overshoot_0_100 = (sum_u4 - target_u4*n) * 5 / n
            req_min = target_u4 * n - s_low + 1  # strict > target
            if req_min > sums[-1]:
                continue

            # Try to hit overshoot in [1, 2] points (0-100 scale) by selecting a larger high-score sum if needed.
            delta_min_for_1 = (n + 4) // 5  # ceil(n/5)
            delta_max_for_2 = (2 * n) // 5  # floor(2n/5)
            min_sum_for_1 = target_u4 * n + delta_min_for_1 - s_low
            max_sum_for_2 = target_u4 * n + delta_max_for_2 - s_low

            # Effective range to search in sums.
            lo = max(req_min, min_sum_for_1)
            hi = max_sum_for_2

            chosen_idx: Optional[int] = None
            if lo <= hi:
                left_idx = bisect_left(sums, lo)
                right_idx = bisect_left(sums, hi + 1) - 1
                if left_idx <= right_idx:
                    desired_delta_u4 = int(round((1.5 * n) / 5.0))
                    desired_sum = target_u4 * n + desired_delta_u4 - s_low
                    mid_idx = bisect_left(sums, desired_sum)
                    mid_idx = max(left_idx, min(right_idx, mid_idx))
                    cand_idxs = {left_idx, right_idx, mid_idx}
                    if mid_idx - 1 >= left_idx:
                        cand_idxs.add(mid_idx - 1)
                    if mid_idx + 1 <= right_idx:
                        cand_idxs.add(mid_idx + 1)
                    chosen_idx = min(
                        cand_idxs,
                        key=lambda i: abs(((s_low + sums[i] - target_u4 * n) * 5.0 / n) - 1.5),
                    )

            # If we couldn't hit [1,2], prefer overshoot >= 1 (smallest possible), else fallback to smallest feasible.
            if chosen_idx is None:
                lo_ge_1 = max(req_min, min_sum_for_1)
                idx = bisect_left(sums, lo_ge_1)
                if idx < len(sums):
                    chosen_idx = idx
                else:
                    chosen_idx = bisect_left(sums, req_min)
                    if chosen_idx >= len(sums):
                        continue

            s_high = sums[chosen_idx]
            c_high = high_data["counts"][chosen_idx]
            sum_u4 = s_low + s_high
            delta_u4 = sum_u4 - target_u4 * n
            if delta_u4 <= 0:
                continue
            overshoot_0_100 = (delta_u4 * 5.0) / n
            cand = {
                "n": n,
                "sum_u4": sum_u4,
                "delta_u4": delta_u4,
                "overshoot_0_100": overshoot_0_100,
                "c_low": c_low,
                "c_high": c_high,
            }
            if best is None:
                best = cand
                continue
            # Preference:
            # 1) overshoot in [1, 2] points on 0-100 scale (user wants 1-2 points above)
            # 2) if none, choose minimal overshoot above 2
            # 3) else choose closest to 1 from below (largest overshoot < 1)
            def band_key(x: float) -> Tuple[int, float]:
                if 1.0 <= x <= 2.0:
                    return (0, abs(x - 1.5))
                if x > 2.0:
                    return (1, x)
                return (2, -x)

            cand_band = band_key(cand["overshoot_0_100"])
            best_band = band_key(best["overshoot_0_100"])
            if cand_band < best_band:
                best = cand
                continue
            if cand_band > best_band:
                continue
            # Same band: prefer larger n, then lexicographically smaller counts (more deterministic tie-break)
            if cand["n"] > best["n"]:
                best = cand
                continue
            if cand["n"] < best["n"]:
                continue
            if cand["c_low"] + cand["c_high"] < best["c_low"] + best["c_high"]:
                best = cand

    if best is None:
        raise RuntimeError("No feasible solution found under current constraints.")

    return best["c_low"], best["c_high"], best


def combine_counts(c_low: Tuple[int, ...], c_high: Tuple[int, ...]) -> Dict[float, int]:
    out = {}
    all_scores = LOW_LT2_BUCKETS + LOW_2_TO_4_BUCKETS + HIGH_BUCKETS
    all_counts = list(c_low) + list(c_high)
    for score, count in zip(all_scores, all_counts):
        out[score] = int(count)
    return out


def build_selected_records(bucket_records: Dict[float, List[Dict]], keep_counts: Dict[float, int]) -> List[Dict]:
    selected: List[Dict] = []
    for score in SCORE_BUCKETS:
        k = keep_counts.get(score, 0)
        if k > 0:
            selected.extend(bucket_records[score][:k])
    selected.sort(key=score_key)
    return selected


def build_output_payload(
    source_payload: Dict,
    source_success: List[Dict],
    selected_records: List[Dict],
    keep_counts: Dict[float, int],
    target: float,
    solve_meta: Dict,
    source_file: str,
) -> Dict:
    out_payload = {
        "metadata": copy.deepcopy(source_payload.get("metadata", {})),
        "results": selected_records,
    }
    out_meta = out_payload["metadata"]
    out_meta["summary"] = recompute_summary(selected_records)

    source_dist = score_distribution(source_success)
    selected_dist = score_distribution(selected_records)
    removed_dist = {}
    for score in SCORE_BUCKETS:
        key = f"{score:.1f}"
        removed_dist[key] = int(source_dist.get(key, 0) - selected_dist.get(key, 0))

    n = solve_meta["n"]
    avg = solve_meta["sum_u4"] / (4 * n) if n else 0.0
    avg_100 = avg * 20
    target_100 = target * 20
    lt2_count = sum(keep_counts[s] for s in LOW_LT2_BUCKETS)
    mid_count = sum(keep_counts[s] for s in LOW_2_TO_4_BUCKETS)
    lt4_count = lt2_count + mid_count

    out_meta["target_filtering"] = {
        "enabled": True,
        "source_file": os.path.basename(source_file),
        "scoring_scale": "0-100",
        "target_average_0_100": round(target_100, 4),
        "final_average_0_100": round(avg_100, 6),
        "overshoot_0_100": round(avg_100 - target_100, 6),
        "target_average_0_5": target,
        "final_average_0_5": round(avg, 6),
        "overshoot_0_5": round(avg - target, 6),
        "selected_count": n,
        "source_success_count": len(source_success),
        "removed_count": len(source_success) - n,
        "strictly_above_target": True,
        "score_distribution_selected": selected_dist,
        "score_distribution_removed": removed_dist,
        "score_distribution_selected_0_100": {str(int(float(k) * 20)): v for k, v in selected_dist.items()},
        "score_distribution_removed_0_100": {str(int(float(k) * 20)): v for k, v in removed_dist.items()},
        "selected_low_counts": {
            "lt2": lt2_count,
            "between_2_and_4": mid_count,
            "lt4_total": lt4_count,
        },
        "constraints": {
            "avg_gt_target": True,
            "lt2_min": 1,
            "between_2_and_4_at_least_3x_lt2": True,
            "lt4_not_all_removed": True,
        },
        "timestamp": now_iso(),
    }
    return out_payload


def process_one(dataset_name: str, input_path: str) -> str:
    target = TARGET_0_100_BY_DATASET[dataset_name] / 20.0
    target_tag = TARGET_TAG_BY_DATASET[dataset_name]

    payload = read_json(input_path)
    success_records = extract_success_records(payload.get("results", []))
    if not success_records:
        raise RuntimeError(f"{dataset_name}: no success records found in {input_path}")

    bucket_records = build_bucket_records(success_records)
    score_caps = {score: len(bucket_records[score]) for score in SCORE_BUCKETS}

    c_low, c_high, solve_meta = select_best_counts(score_caps, target)
    keep_counts = combine_counts(c_low, c_high)
    selected_records = build_selected_records(bucket_records, keep_counts)

    out_payload = build_output_payload(
        source_payload=payload,
        source_success=success_records,
        selected_records=selected_records,
        keep_counts=keep_counts,
        target=target,
        solve_meta=solve_meta,
        source_file=input_path,
    )

    base = os.path.basename(input_path)
    if base.endswith("_filtered.json"):
        base = base[: -len("_filtered.json")]
    else:
        base = os.path.splitext(base)[0]
    out_name = f"{base}_target{target_tag}_dataset.json"
    out_path = os.path.join(SCRIPT_DIR, out_name)
    write_json(out_path, out_payload)
    return out_path


def main() -> None:
    outputs = []
    for dataset_name, rel_path in INPUT_FILES.items():
        in_path = os.path.join(SCRIPT_DIR, rel_path)
        if not os.path.exists(in_path):
            raise FileNotFoundError(in_path)
        out_path = process_one(dataset_name, in_path)
        outputs.append((dataset_name, out_path))

    print("Done:")
    for dataset_name, out_path in outputs:
        print(f"- {dataset_name}: {out_path}")


if __name__ == "__main__":
    main()
