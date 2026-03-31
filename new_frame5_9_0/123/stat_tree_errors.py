import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_records(data: dict):
    for round_item in data.get("round_results", []):
        for row in round_item.get("results", []):
            yield row


def aggregate(files: list[Path]) -> dict:
    wrong_by_predicted = Counter()
    wrong_by_ground_truth = Counter()
    total_wrong = 0
    total_samples = 0

    for file_path in files:
        data = load_json(file_path)
        for row in iter_records(data):
            total_samples += 1
            if row.get("is_correct") is False:
                total_wrong += 1
                wrong_by_predicted[str(row.get("predicted_tree", "未知"))] += 1
                wrong_by_ground_truth[str(row.get("ground_truth", "未知"))] += 1

    return {
        "total_samples": total_samples,
        "total_wrong": total_wrong,
        "wrong_by_predicted": wrong_by_predicted,
        "wrong_by_ground_truth": wrong_by_ground_truth,
    }


def print_counter(title: str, counter: Counter, top_n: int):
    print(title)
    if not counter:
        print("  无错误样本")
        return
    for i, (name, count) in enumerate(counter.most_common(top_n), start=1):
        print(f"  {i}. {name}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    files = [Path(p) for p in args.files]
    result = aggregate(files)

    print(f"样本总数: {result['total_samples']}")
    print(f"错误总数: {result['total_wrong']}")
    print_counter("按预测树统计（模型最常预测错成什么）:", result["wrong_by_predicted"], args.top)
    print_counter("按真实树统计（哪些树最常被判错）:", result["wrong_by_ground_truth"], args.top)


if __name__ == "__main__":
    main()
