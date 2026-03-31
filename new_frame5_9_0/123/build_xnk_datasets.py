import json
import os
import re
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RULE_FILE = os.path.join(SCRIPT_DIR, "xnk_regex_rules.json")
SOURCE_FILE = os.path.join(SCRIPT_DIR, "xnk_filtered_raw.jsonl")
OUTPUT_ACUTE = os.path.join(SCRIPT_DIR, "xnk_dataset_acute.jsonl")
OUTPUT_MULTISYSTEM = os.path.join(SCRIPT_DIR, "xnk_dataset_multisystem.jsonl")
OUTPUT_HIGHRISK = os.path.join(SCRIPT_DIR, "xnk_dataset_highrisk.jsonl")
OUTPUT_CHRONIC = os.path.join(SCRIPT_DIR, "xnk_dataset_chronic.jsonl")
OUTPUT_ACUTE_META = os.path.join(SCRIPT_DIR, "xnk_dataset_acute_meta.json")
OUTPUT_MULTISYSTEM_META = os.path.join(SCRIPT_DIR, "xnk_dataset_multisystem_meta.json")
OUTPUT_HIGHRISK_META = os.path.join(SCRIPT_DIR, "xnk_dataset_highrisk_meta.json")
OUTPUT_CHRONIC_META = os.path.join(SCRIPT_DIR, "xnk_dataset_chronic_meta.json")
DISABLED_TREES = set()

ALL_DATASET_JOBS = [
    {
        "label": "acute_disease",
        "dataset_name": "xnk_acute",
        "output": OUTPUT_ACUTE,
        "meta": OUTPUT_ACUTE_META,
    },
    {
        "label": "multisystem_disease",
        "dataset_name": "xnk_multisystem",
        "output": OUTPUT_MULTISYSTEM,
        "meta": OUTPUT_MULTISYSTEM_META,
    },
    {
        "label": "high_risk_disease",
        "dataset_name": "xnk_highrisk",
        "output": OUTPUT_HIGHRISK,
        "meta": OUTPUT_HIGHRISK_META,
    },
    {
        "label": "chronic_disease",
        "dataset_name": "xnk_chronic",
        "output": OUTPUT_CHRONIC,
        "meta": OUTPUT_CHRONIC_META,
    },
]

ENABLED_LABELS = {"high_risk_disease", "chronic_disease"}


def prune_disabled_trees(rules):
    tree_regex = rules.get("tree_regex", {})
    for tree_name in DISABLED_TREES:
        tree_regex.pop(tree_name, None)

    for _, cfg in rules.get("phenotype_rules", {}).items():
        target_trees = cfg.get("target_trees", [])
        cfg["target_trees"] = [t for t in target_trees if t not in DISABLED_TREES]

    return rules


def load_rules():
    with open(RULE_FILE, "r", encoding="utf-8") as f:
        rules = json.load(f)

    if "dataset_policy" not in rules:
        rules["dataset_policy"] = {}
    return prune_disabled_trees(rules)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            rows.append({"line_no": line_no, "record": obj, "raw_text": text})
    return rows


def regex_hits(text, patterns):
    hits = []
    for pattern in patterns:
        if not pattern:
            continue
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                hits.append(pattern)
        except re.error:
            if pattern in text:
                hits.append(pattern)
    return hits


def detect_tree_matches(text, tree_regex):
    matched_trees = []
    tree_hits = {}
    for tree_name, keywords in tree_regex.items():
        if tree_name in DISABLED_TREES:
            continue
        hits = regex_hits(text, keywords)
        if hits:
            matched_trees.append(tree_name)
            tree_hits[tree_name] = hits
    return matched_trees, tree_hits


def build_candidates(rows, rules):
    phenotype_rules = rules["phenotype_rules"]
    tree_regex = rules["tree_regex"]
    candidates = []
    for row in rows:
        text = row["raw_text"]
        record = row["record"]
        labels = []
        label_hits = {}
        for label_name, cfg in phenotype_rules.items():
            hits = regex_hits(text, cfg["keywords"])
            if hits:
                labels.append(label_name)
                label_hits[label_name] = hits

        matched_trees, tree_hits = detect_tree_matches(text, tree_regex)
        if not labels or not matched_trees:
            continue

        candidates.append(
            {
                "id": record.get("id", record.get("pid")),
                "source_line": row["line_no"],
                "labels": labels,
                "label_hits": label_hits,
                "matched_trees": matched_trees,
                "tree_hits": tree_hits,
                "raw_entry": record,
            }
        )
    return candidates


def rank_candidate(candidate, label_name, target_trees):
    label_hits_count = len(candidate["label_hits"].get(label_name, []))
    tree_overlap = len([t for t in candidate["matched_trees"] if t in target_trees])
    return (label_hits_count * 10) + (tree_overlap * 3)


def select_for_label(candidates, rules, label_name):
    policy = rules.get("dataset_policy", {})
    target_count = int(policy.get("target_count_per_dataset", 50))
    per_tree_min = int(policy.get("per_tree_target_min", 1))
    per_tree_max = int(policy.get("per_tree_target_max", 12))
    ordered_target_trees = [
        t
        for t in rules["phenotype_rules"][label_name]["target_trees"]
        if t not in DISABLED_TREES
    ]

    counts = defaultdict(int)
    chosen_ids = set()
    selected = []

    eligible = []
    for c in candidates:
        if label_name not in c["labels"]:
            continue
        candidate_trees = [t for t in ordered_target_trees if t in c["matched_trees"]]
        if not candidate_trees:
            continue
        cid = c["id"] or f"line_{c['source_line']}"
        eligible.append((c, candidate_trees, cid))

    if policy.get("select_all"):
        target_count = len(eligible)
        per_tree_min = 0
        per_tree_max = len(eligible)

    eligible.sort(
        key=lambda item: (
            -rank_candidate(item[0], label_name, ordered_target_trees),
            item[0]["source_line"],
        )
    )

    for tree_name in ordered_target_trees:
        if len(selected) >= target_count:
            break
        if counts[tree_name] >= per_tree_min:
            continue
        for c, candidate_trees, cid in eligible:
            if len(selected) >= target_count or counts[tree_name] >= per_tree_min:
                break
            if cid in chosen_ids or tree_name not in candidate_trees:
                continue
            primary_tree = tree_name

            evidence_keywords = []
            evidence_keywords.extend(c["label_hits"].get(label_name, []))
            evidence_keywords.extend(c["tree_hits"].get(primary_tree, []))
            evidence_keywords = list(dict.fromkeys(evidence_keywords))

            selected.append(
                {
                    "id": cid,
                    "source_line": c["source_line"],
                    "labels": [label_name],
                    "primary_tree": primary_tree,
                    "matched_trees": candidate_trees,
                    "evidence_keywords": evidence_keywords,
                    "raw_entry": c["raw_entry"],
                }
            )
            counts[primary_tree] += 1
            chosen_ids.add(cid)

    for c, candidate_trees, cid in eligible:
        if len(selected) >= target_count:
            break
        if cid in chosen_ids:
            continue
        primary_tree = None
        for t in candidate_trees:
            if counts[t] < per_tree_max:
                primary_tree = t
                break
        if not primary_tree:
            continue

        evidence_keywords = []
        evidence_keywords.extend(c["label_hits"].get(label_name, []))
        evidence_keywords.extend(c["tree_hits"].get(primary_tree, []))
        evidence_keywords = list(dict.fromkeys(evidence_keywords))
        selected.append(
            {
                "id": cid,
                "source_line": c["source_line"],
                "labels": [label_name],
                "primary_tree": primary_tree,
                "matched_trees": candidate_trees,
                "evidence_keywords": evidence_keywords,
                "raw_entry": c["raw_entry"],
            }
        )
        counts[primary_tree] += 1
        chosen_ids.add(cid)

    return selected, ordered_target_trees, dict(counts), len(eligible)


def write_dataset_jsonl(path, selected):
    with open(path, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row["raw_entry"], ensure_ascii=False) + "\n")


def write_dataset_meta(path, dataset_name, focus_label, selected, target_trees, tree_counts, total_eligible, rules):
    payload = {
        "dataset_name": dataset_name,
        "created_from": SOURCE_FILE,
        "rule_file": RULE_FILE,
        "construction_method": "regex_auto_filter_from_source_jsonl",
        "output_dataset_format": "jsonl_same_as_source",
        "focus": [focus_label],
        "tree_coverage": target_trees,
        "stats": {
            "eligible_count": total_eligible,
            "selected_count": len(selected),
            "tree_distribution": tree_counts,
            "dataset_policy": rules.get("dataset_policy", {}),
        },
        "samples": selected,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    rules = load_rules()
    rows = load_jsonl(SOURCE_FILE)
    candidates = build_candidates(rows, rules)

    for job in ALL_DATASET_JOBS:
        if job["label"] not in ENABLED_LABELS:
            continue
        selected, trees, counts, total = select_for_label(
            candidates, rules, job["label"]
        )
        write_dataset_jsonl(job["output"], selected)
        write_dataset_meta(
            job["meta"],
            job["dataset_name"],
            job["label"],
            selected,
            trees,
            counts,
            total,
            rules,
        )


if __name__ == "__main__":
    main()
