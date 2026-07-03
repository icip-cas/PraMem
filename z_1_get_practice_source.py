import os
import pytz
import json
import tqdm
import random
import argparse
from config import *
from datetime import datetime
from typing import Dict, List
from prompt_builder import should_filter_action
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_user(args_tuple):
    user_entry, N_per_test, min_history_len, include_test_as_practice = args_tuple

    user_id = user_entry["user_id"]
    base_history = user_entry.get("base_history", [])
    test_time_all_actions = user_entry.get("test_time_all_actions", [])
    test_actions = user_entry.get("test_actions", [])

    practice_questions_dict = {}
    user_generated = 0
    user_skipped = 0
    user_filtered = 0

    for test_item in test_actions[0:1]:
        anchor_index = test_item["test_time_index"]
        test_action = test_time_all_actions[anchor_index]

        history_pool = base_history + test_time_all_actions[:anchor_index]
        if len(history_pool) < min_history_len:
            user_skipped += N_per_test
            continue

        candidates = []
        for idx in range(min_history_len, len(history_pool)):
            if is_valid_candidate(history_pool[idx]):
                candidates.append((idx, history_pool[idx]))
            else:
                user_filtered += 1

        if include_test_as_practice and is_valid_candidate(test_action):
            candidates.append((anchor_index, test_action))

        if not candidates:
            user_skipped += N_per_test
            continue

        sample_candidates = random.choices(candidates, weights=[(i + 1) ** 2 for i in range(len(candidates))], k=N_per_test)
        questions = []
        for idx, action in sample_candidates:
            questions.append({
                "action": action,
                "test_time_index": idx,
                "is_oracle": idx == anchor_index
            })
            user_generated += 1

        if questions:
            practice_questions_dict[str(anchor_index)] = questions

    result_entry = {
        "user_id": user_id,
        "user_profile": user_entry.get("user_profile", {}),
        "base_history": base_history,
        "test_time_all_actions": test_time_all_actions,
        "test_actions": test_actions,
        "practice_questions": practice_questions_dict,
        "stats": {
            "test_actions_count": len(test_actions),
            "practice_questions_generated": sum(len(v) for v in practice_questions_dict.values()),
        }
    }

    return result_entry, user_generated, user_skipped, user_filtered


def has_sufficient_context_for_prediction(action: Dict) -> bool:
    action_type = action.get("type", "")
    context = action.get("context", {})

    if action_type == "视频浏览":
        caption = context.get("caption", "")
        ocr = context.get("ocr_text", "")
        asr = context.get("asr_text", "")
        if not any([caption and caption.strip(), ocr and ocr.strip(), asr and asr.strip()]):
            return False

    elif action_type == "直播间":
        live_title = context.get("live_title", "")
        live_category = context.get("live_category", "")
        if not any([live_title and live_title.strip(), live_category and live_category.strip()]):
            return False

    return True

def get_action_type_set(action: Dict) -> set:
    return {a["type"] for a in action.get("action", []) if "type" in a}


def is_valid_candidate(action: Dict) -> bool:
    return has_sufficient_context_for_prediction(action) and not should_filter_action(action)


def generate_practice_questions(
    experiment_data_path: str,
    output_path: str,
    N_per_test: int,
    min_history_len: int = 5,
    seed: int = 42,
    force: bool = False,
    include_test_as_practice: bool = False,
):
    random.seed(seed)

    with open(experiment_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    users = data.get("users", [])

    new_metadata = metadata.copy()
    new_metadata.update({
        "practice_questions_config": {
            "N_per_test": N_per_test,
            "min_history_len": min_history_len,
            "seed": seed,
            "include_test_as_practice": include_test_as_practice,
        },
        "generated_at": datetime.now(pytz.timezone('Asia/Shanghai')).isoformat(),
        "source_dataset": experiment_data_path,
    })

    num_workers = min(40, len(users))
    task_args = [
        (user_entry, N_per_test, min_history_len, include_test_as_practice)
        for user_entry in users
    ]

    practice_users = []
    total_generated = 0
    total_skipped = 0
    total_filtered = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_user, arg): arg for arg in task_args}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result_entry, gen, skipped, filtered = future.result()
            practice_users.append(result_entry)
            total_generated += gen
            total_skipped += skipped
            total_filtered += filtered

    user_order = {u["user_id"]: i for i, u in enumerate(users)}
    practice_users.sort(key=lambda x: user_order[x["user_id"]])

    output_data = {
        "metadata": new_metadata,
        "users": practice_users,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path) and not force:
        print(f"File already exists: {output_path}, use --force to overwrite")
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nDone generating practice questions")
    print(f"  Total generated: {total_generated}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Filtered (insufficient context): {total_filtered}")


def main():
    parser = argparse.ArgumentParser(description="Generate practice question dataset")

    parser.add_argument("--force", type=int, default=1)
    parser.add_argument("--experiment_data", type=str, default="./work_data/experiment_data.json")
    parser.add_argument("--N_per_test", type=int, default=100)
    parser.add_argument("--min_history_len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_test_as_practice", type=int, default=0)

    args = parser.parse_args()

    print("=" * 50)
    print("Arguments:")
    print("=" * 50)
    for k, v in vars(args).items():
        print(f"{k:20s}: {v}")
    print("=" * 50)

    if args.include_test_as_practice:
        assert args.N_per_test == 1, f"args.N_per_test != 1"

    generate_practice_questions(
        experiment_data_path=args.experiment_data,
        output_path=f"{args.experiment_data}.oracle.json" if args.include_test_as_practice else f"{args.experiment_data}.practice.json",
        N_per_test=args.N_per_test,
        min_history_len=args.min_history_len,
        seed=args.seed,
        force=bool(args.force),
        include_test_as_practice=bool(args.include_test_as_practice),
    )


if __name__ == "__main__":
    main()
