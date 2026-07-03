import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from config import *


def quick_scan_user_file(json_file: Path) -> Optional[Tuple[str, int, Path]]:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
            for user_id, data in user_data.items():
                action_count = len(data.get("action_history", []))
                return (user_id, action_count, json_file)
    except Exception as e:
        print(f"  Warning: failed to scan file {json_file.name}: {e}")
        return None


def load_user_data(file_path: str, min_actions: int = 0, target_users: int = 0) -> Dict:
    path = Path(file_path)

    if path.is_dir():
        print(f"Loading user data from directory: {file_path}")
        json_files = list(path.glob("*.json"))
        total_files = len(json_files)
        print(f"Found {total_files} user files")

        if min_actions > 0:
            if target_users > 0:
                scan_target = min(target_users, total_files)
                print(f"  Early-stop strategy: stop after finding {scan_target} qualifying users")
            else:
                scan_target = total_files

            print(f"  Phase 1: quick scan (filtering users with >={min_actions} actions)...")
            user_info_list = []
            scanned_count = 0

            for json_file in json_files:
                info = quick_scan_user_file(json_file)
                scanned_count += 1

                if info:
                    user_id, action_count, file_path = info
                    if action_count >= min_actions:
                        user_info_list.append(info)

                        if target_users > 0 and len(user_info_list) >= scan_target:
                            print(f"    Scanned {scanned_count}/{total_files} files, found {len(user_info_list)} qualifying users (target reached, stopping scan)")
                            break

            if scanned_count == total_files:
                print(f"    Scanned all {total_files} files, found {len(user_info_list)} qualifying users")

            print(f"  Phase 2: loading {len(user_info_list)} qualifying users...")

            all_users = {}
            for user_id, action_count, file_path in user_info_list:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        user_data = json.load(f)
                        for uid, data in user_data.items():
                            all_users[uid] = data
                except Exception as e:
                    print(f"  Warning: failed to load file {file_path.name}: {e}")
                    continue
        else:
            all_users = {}
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        user_data = json.load(f)
                        for user_id, data in user_data.items():
                            all_users[user_id] = data
                except Exception as e:
                    print(f"  Warning: failed to load file {json_file.name}: {e}")
                    continue

        print(f"Successfully loaded {len(all_users)} users")
        return all_users

    else:
        print(f"Loading data: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} users")
        return data


def filter_users_by_month(all_users: Dict, target_month=None) -> Dict:
    if target_month is None:
        print(f"\nNo month filter applied, using all user data...")
        return all_users

    if isinstance(target_month, str):
        if ',' in target_month:
            target_months = [m.strip() for m in target_month.split(',')]
        else:
            target_months = [target_month]
    elif isinstance(target_month, list):
        target_months = target_month
    else:
        raise ValueError(f"target_month must be a string, list, or None, not {type(target_month)}")

    print(f"\nFiltering users with actions in {', '.join(target_months)}...")
    filtered_users = {}

    for user_id, user_data in all_users.items():
        actions = user_data.get("action_history", [])
        month_actions = [
            action for action in actions
            if any(action.get("timestamp", "").startswith(month) for month in target_months)
        ]

        if len(month_actions) >= 5:
            filtered_users[user_id] = {
                "user_profile": user_data["user_profile"],
                "action_history": sorted(
                    month_actions,
                    key=lambda x: x.get("timestamp", "")
                )
            }

    print(f"Found {len(filtered_users)} users with sufficient actions in {', '.join(target_months)}")
    return filtered_users


def save_evaluation_data(eval_data: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"Evaluation data saved to: {output_path}")
