#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import os
import csv
import argparse
from collections import defaultdict


SCENES = ["video", "live", "ad", "shop"]
METRICS = ["Accuracy", "F1"]


def compute_scores(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    binary = data.get("binary_metrics", {})

    scene_metrics = defaultdict(lambda: defaultdict(list))
    for task_name, task_metrics in binary.items():
        scene = task_name.split("_")[0]
        if scene not in SCENES:
            continue
        for m in METRICS:
            val = task_metrics.get(m)
            if val is not None:
                scene_metrics[scene][m].append(val)

    result = {}
    scene_accs = []
    scene_f1s = []

    for scene in SCENES:
        for m in METRICS:
            vals = scene_metrics[scene][m]
            if vals:
                score = sum(vals) / len(vals)
                result[f"{scene}_{m}"] = score
                if m == "Accuracy":
                    scene_accs.append(score)
                if m == "F1":
                    scene_f1s.append(score)
            else:
                result[f"{scene}_{m}"] = None

    result["avg_Accuracy"] = sum(scene_accs) / len(scene_accs) if scene_accs else None
    result["avg_F1"] = sum(scene_f1s) / len(scene_f1s) if scene_f1s else None

    return result


def extract_name(fp):
    parts = fp.split(os.sep)
    for i, p in enumerate(parts):
        if p.startswith("experiment_data.json"):
            if i + 1 < len(parts):
                return f"{p}/{parts[i + 1]}"
            return p
    if len(parts) >= 4:
        return f"{parts[-5]}/{parts[-4]}" if len(parts) >= 5 else parts[-4]
    return fp



def main():
    parser = argparse.ArgumentParser(description="Aggregate binary classification evaluation scores")
    parser.add_argument(
        "-o", "--output", default="z_8_show_score.csv", help="output CSV file"
    )
    parser.add_argument(
        "--with-name", default=0, help="include file path column in output"
    )
    parser.add_argument(
        "--results-dir", default="work_data/results", help="root directory to scan for evaluation reports"
    )
    args = parser.parse_args()

    file_list = []
    results_dir = args.results_dir
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if fname.endswith("_evaluation_report.json"):
                file_list.append(os.path.join(root, fname))
    file_list.sort()


    header_cols = ["name"]
    if args.with_name:
        header_cols.append("file")
    for scene in SCENES:
        for m in METRICS:
            header_cols.append(f"{scene}_{m}")
    header_cols.append("avg_Accuracy")
    header_cols.append("avg_F1")

    rows = []
    for fp in file_list:
        if not os.path.exists(fp):
            print(f"[WARN] file not found: {fp}", file=sys.stderr)
            continue
        try:
            scores = compute_scores(fp)
        except Exception as e:
            print(f"[ERROR] failed to process {fp}: {e}", file=sys.stderr)
            continue

        row = [extract_name(fp)]
        if args.with_name:
            row.append(fp)
        for scene in SCENES:
            for m in METRICS:
                v = scores[f"{scene}_{m}"]
                row.append(f"{v:.2f}" if v is not None else "NA")

        v_acc = scores["avg_Accuracy"]
        v_f1 = scores["avg_F1"]
        row.append(f"{v_acc:.2f}" if v_acc is not None else "NA")
        row.append(f"{v_f1:.2f}" if v_f1 is not None else "NA")

        rows.append(row)

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header_cols)
            writer.writerows(rows)
        print(f"written to {args.output}", file=sys.stderr)
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(header_cols)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
