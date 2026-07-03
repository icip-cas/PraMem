import os
import re
import json
import time
import tqdm
import random
import argparse
from typing import Optional
from utils.fix_json import safe_json_loads
from utils.use_my_api import LocalClient
from utils.prompts import (
    prompt_for_predict,
    prompt_for_reflect,
    prompt_for_review,
    prompt_for_generate_perturbations,
    prompt_for_check_perturbation,
    prompt_for_generate_virtual_scenes,
    prompt_for_check_generalization,
)


class SelfPractice:

    def __init__(
        self,
        user_id: str,
        test_id: str,
        practice_data_dir: str = "./work_data/practice_data",
        exp_memory_dir: str = "./work_data/exp_memory",
        practice_progress_dir: str = "./work_data/practice_progress",
        review_interval: int = 5,
        model_name: str = "DeepSeek-V3",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        total_rounds: int = 50,
        num_perturbations: int = 3,
        detail_truncate_len: int = 512,
        num_virtual_scenes: int = 9,
        choices_per_question: int = 4,
        use_check_generalization: bool = True,
        use_check_perturbation: bool = True,
    ):
        self.user_id = user_id
        self.test_id = test_id

        self.practice_data_dir = practice_data_dir
        self.exp_memory_dir = exp_memory_dir
        self.practice_progress_dir = practice_progress_dir

        self.review_interval = review_interval
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.total_rounds = total_rounds
        self.num_perturbations = num_perturbations
        self.detail_truncate_len = detail_truncate_len
        self.num_virtual_scenes = num_virtual_scenes
        self.choices_per_question = choices_per_question

        self.use_check_generalization = use_check_generalization
        self.use_check_perturbation = use_check_perturbation

        self.client = LocalClient(model_name="gpt-oss-120b", base_url=model_name)

    def call_llm(self, prompt: str, max_retries: int = 2) -> str:
        for attempt in range(max_retries):
            try:
                content, _ = self.client.call_openai(prompt=prompt, max_completion_tokens=self.max_tokens)
                return content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5
                    print(f"[retry] attempt {attempt+1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"[failed] all {max_retries} retries exhausted: {e}")
                    raise

    def load_practice_data(self) -> list[dict]:
        path = os.path.join(self.practice_data_dir, f"{self.user_id}_{self.test_id}.jsonl")
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def load_exp_memory(self) -> dict:
        return {"user_exp": [], "model_exp": [], "proposal": []}

    def save_exp_memory(self, memory: dict) -> None:
        os.makedirs(self.exp_memory_dir, exist_ok=True)
        path = os.path.join(self.exp_memory_dir, f"{self.user_id}_{self.test_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

    def load_practice_progress(self) -> dict:
        path = os.path.join(self.practice_progress_dir, f"{self.user_id}_{self.test_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_practice_progress(self, progress: dict) -> None:
        os.makedirs(self.practice_progress_dir, exist_ok=True)
        path = os.path.join(self.practice_progress_dir, f"{self.user_id}_{self.test_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def format_exp_memory(self, memory, with_proposal=0, use_ind_exp=1, use_cab_exp=1) -> str:
        lines = []
        lines.append("## Current Experiential Memory")

        lines.append("\n### Pattern Experience")
        if memory["user_exp"] and use_ind_exp:
            for i, exp in enumerate(memory["user_exp"], 1):
                lines.append(f"{i}. {exp}")
        else:
            lines.append("(none)")

        lines.append("\n### Calibration Experience")
        if memory["model_exp"] and use_cab_exp:
            for i, exp in enumerate(memory["model_exp"], 1):
                lines.append(f"{i}. {exp}")
        else:
            lines.append("(none)")

        if with_proposal:
            lines.append("\n### Unconfirmed Proposals")
            proposals = memory.get("proposal", [])
            if proposals:
                for i, proposal in enumerate(proposals, 1):
                    lines.append(f"{i}. {proposal}")
            else:
                lines.append("(none)")

        return "\n".join(lines)

    def restore_memory_from_progress(self, progress: dict, completed_rounds: list) -> dict | None:
        if not completed_rounds:
            return None

        last_round_idx = completed_rounds[-1]
        last_round_record = progress.get(f"round_{last_round_idx}")
        if last_round_record is None:
            return None

        memory_snapshot = last_round_record.get("memory_snapshot_after")
        if memory_snapshot is None:
            return None

        if all(k in memory_snapshot for k in ["user_exp", "model_exp", "proposal"]):
            return {
                "user_exp": list(memory_snapshot["user_exp"]),
                "model_exp": list(memory_snapshot["model_exp"]),
                "proposal": list(memory_snapshot["proposal"]),
            }
        return None

    def extract_and_truncate_history(self, simulated_input: str) -> tuple[str, str]:
        history_pattern = re.compile(
            r"(## 输入二[^\n]*\n.*?)(={3,}下面是你的第)",
            re.DOTALL
        )
        match = history_pattern.search(simulated_input)
        if not match:
            return "", simulated_input

        history_block = match.group(1)
        rest_start = match.start(2)
        rest_of_input = simulated_input[rest_start:]

        def truncate_detail(m):
            detail_content = m.group(1)
            if len(detail_content) > self.detail_truncate_len:
                detail_content = detail_content[:self.detail_truncate_len] + "..."
            return f"  详情：{detail_content}"

        truncated_history = re.sub(
            r"  详情：(.+?)(?=\n  (?:反应|场景|时间)|【行为|\Z)",
            truncate_detail,
            history_block,
            flags=re.DOTALL
        )

        return truncated_history, rest_of_input

    def replace_history_in_input(self, simulated_input: str, new_history_block: str) -> str:
        history_pattern = re.compile(
            r"(## 输入二[^\n]*\n.*?)(={3,}下面是你的第)",
            re.DOTALL
        )
        match = history_pattern.search(simulated_input)
        if not match:
            return simulated_input
        return simulated_input[:match.start(1)] + new_history_block + simulated_input[match.start(2):]

    def extract_current_scene(self, simulated_input: str) -> str:
        scene_pattern = re.compile(
            r"(场景详细信息如下：.*)",
            re.DOTALL
        )
        match = scene_pattern.search(simulated_input)
        if match:
            return match.group(1).strip()[:self.detail_truncate_len] + "..."
        return ""

    def step_predict(self, practice_item: dict, memory: dict) -> tuple[str, str]:
        exp_memory_text = self.format_exp_memory(memory)
        user_prompt = prompt_for_predict.format(
            exp_memory_text=exp_memory_text,
            practice_item_simulated_input=practice_item['simulated_input']
        )
        response = self.call_llm(user_prompt)
        return user_prompt, response

    def step_reflect_and_extract(
        self,
        practice_item: dict,
        memory: dict,
        predict_response: str,
        simulated_input_override: str = None,
    ) -> tuple[str, str, list[str]]:
        exp_memory_text = self.format_exp_memory(memory)
        sim_input = simulated_input_override if simulated_input_override is not None else practice_item['simulated_input']

        user_prompt = prompt_for_reflect.format(
            exp_memory_text=exp_memory_text,
            practice_item_simulated_input=sim_input,
            predict_response=predict_response,
            practice_item_simulated_label=practice_item['simulated_label']
        )

        response = self.call_llm(user_prompt)

        new_proposals = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("【提议】"):
                proposal_text = line[len("【提议】"):].strip()
                if proposal_text:
                    new_proposals.append(proposal_text)

        return user_prompt, response, new_proposals

    def generate_perturbations(self, history_text: str) -> list[dict]:
        user_prompt = prompt_for_generate_perturbations.format(
            history_text=history_text,
            num_perturbations=self.num_perturbations
        )
        response = self.call_llm(user_prompt)

        perturbations = []
        try:
            json_str = response.split("```json")[-1].split("```")[0].strip()
            parsed = safe_json_loads(json_str)
            if isinstance(parsed, list):
                for item in parsed:
                    if "perturbation_desc" in item and "perturbation_details" in item:
                        perturbations.append({
                            "perturbation_desc": item["perturbation_desc"],
                            "perturbation_details": item["perturbation_details"],
                        })
        except Exception as e:
            print(f"[warning] failed to parse perturbations: {e}")

        return perturbations

    def generate_virtual_scenes(self, real_scene_text: str) -> list[str]:
        user_prompt = prompt_for_generate_virtual_scenes.format(
            real_scene_text=real_scene_text,
            num_virtual_scenes=self.num_virtual_scenes,
        )
        response = self.call_llm(user_prompt)

        virtual_scenes = []
        try:
            json_str = response.split("```json")[-1].split("```")[0].strip()
            parsed = safe_json_loads(json_str)
            if isinstance(parsed, list):
                virtual_scenes = [str(s) for s in parsed]
        except Exception as e:
            print(f"[warning] failed to parse virtual scenes: {e}")

        return virtual_scenes

    def check_perturbation_sensitivity_batch(
        self,
        original_proposals: list[str],
        perturbation_details: str,
        perturbation_desc: str,
        history_text: str,
    ) -> tuple[list[bool], str]:
        original_proposals_text = "\n".join(
            f"{i}. {p}" for i, p in enumerate(original_proposals)
        )

        user_prompt = prompt_for_check_perturbation.format(
            history_text=history_text,
            perturbation_desc=perturbation_desc,
            perturbation_details=perturbation_details,
            original_proposals_text=original_proposals_text,
        )
        response = self.call_llm(user_prompt)

        is_sensitive_list = [False] * len(original_proposals)
        try:
            json_str = response.split("```json")[-1].split("```")[0].strip()
            parsed = safe_json_loads(json_str)
            if isinstance(parsed, list):
                for item in parsed:
                    idx = item.get("proposal_idx")
                    if isinstance(idx, int) and 0 <= idx < len(original_proposals):
                        is_sensitive_list[idx] = bool(item.get("is_sensitive", False))
        except Exception as e:
            print(f"    [check2] parse failed: {e}, defaulting all to not sensitive")

        return is_sensitive_list, response

    def check_generalization_batch(
        self,
        proposals: list[str],
        real_scene_text: str,
        virtual_scenes: list[str],
    ) -> tuple[list[bool], str]:
        num_distractors = self.choices_per_question - 1
        distractors = virtual_scenes[:num_distractors]

        options = [real_scene_text] + distractors
        options_text = "\n".join(
            f"{chr(ord('A') + i)}. {scene}" for i, scene in enumerate(options)
        )
        proposals_text = "\n".join(
            f"{i}. {p}" for i, p in enumerate(proposals)
        )

        user_prompt = prompt_for_check_generalization.format(
            options_text=options_text,
            proposals_text=proposals_text,
        )
        response = self.call_llm(user_prompt)

        generalization_passed_list = [True] * len(proposals)
        try:
            json_str = response.split("```json")[-1].split("```")[0].strip()
            parsed = safe_json_loads(json_str)
            if isinstance(parsed, list):
                for item in parsed:
                    idx = item.get("proposal_idx")
                    if isinstance(idx, int) and 0 <= idx < len(proposals):
                        generalization_passed_list[idx] = bool(item.get("generalization_passed", True))
        except Exception as e:
            print(f"    [check1] parse failed: {e}, defaulting all to passed")

        return generalization_passed_list, response

    def step_self_verify(
        self,
        practice_item: dict,
        memory: dict,
        predict_response: str,
        new_proposals: list[str],
    ) -> tuple[list[str], dict]:
        if not new_proposals:
            return [], {}

        original_simulated_input = practice_item['simulated_input']
        truncated_history, _ = self.extract_and_truncate_history(original_simulated_input)
        real_scene_text = self.extract_current_scene(original_simulated_input)

        verify_records = {}

        check1_passed_list = [True] * len(new_proposals)

        if self.use_check_generalization:
            if real_scene_text:
                virtual_scenes = self.generate_virtual_scenes(real_scene_text)
                print(f"  [check1] generated {len(virtual_scenes)} virtual scenes")
                if len(virtual_scenes) >= self.choices_per_question - 1:
                    check1_passed_list, check1_response = self.check_generalization_batch(
                        proposals=new_proposals,
                        real_scene_text=real_scene_text,
                        virtual_scenes=virtual_scenes,
                    )
                    verify_records["check1"] = {
                        "check1_virtual_scenes": virtual_scenes,
                        "check1_response": check1_response,
                        "check1_results": check1_passed_list,
                    }
                else:
                    print("    [check1] not enough virtual scenes, all passed by default")
            else:
                print("    [check1] could not extract current scene, all passed by default")
        else:
            print("  [check1] disabled (ablation), all passed by default")

        passed_check1 = [p for p, ok in zip(new_proposals, check1_passed_list) if ok]
        print(f"  [check1] {len(passed_check1)}/{len(new_proposals)} proposals passed")

        if not passed_check1:
            return [], verify_records

        check2_passed_list = [True] * len(passed_check1)

        if self.use_check_perturbation:
            if truncated_history:
                perturbations = self.generate_perturbations(truncated_history)
                print(f"  [check2] generated {len(perturbations)} perturbations")
                if perturbations:
                    pert = perturbations[0]
                    check2_passed_list, check2_response = self.check_perturbation_sensitivity_batch(
                        original_proposals=passed_check1,
                        perturbation_details=pert["perturbation_details"],
                        perturbation_desc=pert["perturbation_desc"],
                        history_text=truncated_history,
                    )
                    verify_records["check2"] = {
                        "check1_perturbation": {"perturbation_desc": pert["perturbation_desc"], "perturbation_details": pert["perturbation_details"]},
                        "check1_response": check2_response,
                        "check1_results": check2_passed_list,
                    }
                else:
                    print("    [check2] could not generate perturbations, all passed by default")
            else:
                print("    [check2] could not extract history, all passed by default")
        else:
            print("  [check2] disabled (ablation), all passed by default")

        verified_proposals = [p for p, ok in zip(passed_check1, check2_passed_list) if ok]
        print(f"  [check2] {len(verified_proposals)}/{len(passed_check1)} proposals passed")

        return verified_proposals, verify_records

    def step_review_memory(self, memory: dict, round_idx: int) -> tuple[str, str]:
        exp_memory_text = self.format_exp_memory(memory)

        if memory["proposal"]:
            proposal_text = "\n".join(f"{i}. {p}" for i, p in enumerate(memory["proposal"], 1))
        else:
            proposal_text = "(none)"

        user_prompt = prompt_for_review.format(
            exp_memory_text=exp_memory_text,
            proposal_text=proposal_text
        )

        response = self.call_llm(user_prompt)
        return user_prompt, response

    def parse_reviewed_memory(self, review_response: str, old_memory: dict) -> dict:
        try:
            json_str = review_response.split("```json")[-1].split("```")[0].strip()
            new_memory = safe_json_loads(json_str)
            if all(k in new_memory for k in ["user_exp", "model_exp", "proposal"]):
                return new_memory
        except Exception:
            print(review_response)
            pass

        print("[warning] failed to parse reviewed memory, keeping original")
        return old_memory


    def run(self) -> None:
        print(f"[start] user={self.user_id} test={self.test_id}")

        practice_items = self.load_practice_data()
        total_rounds = min(len(practice_items), self.total_rounds)
        print(f"[data] loaded {total_rounds} practice items")

        progress = self.load_practice_progress()
        completed_rounds = progress.get("completed_rounds", [])
        start_round = len(completed_rounds)

        if start_round > 0:
            print(f"[resume] {start_round} rounds completed, resuming from round {start_round}")
            restored_memory = self.restore_memory_from_progress(progress, completed_rounds)
            if restored_memory is not None:
                memory = restored_memory
                print(f"[resume] memory restored from progress")
            else:
                memory = self.load_exp_memory()
                print(f"[resume] could not restore memory, loading fresh")
        else:
            memory = self.load_exp_memory()

        print(f"[memory] user_exp={len(memory['user_exp'])} model_exp={len(memory['model_exp'])} proposal={len(memory['proposal'])}")

        for round_idx in tqdm.trange(start_round, total_rounds):
            practice_item = practice_items[round_idx]

            try:
                print(f"\n{'='*60}")
                print(f"[round {round_idx + 1}/{total_rounds}] idx={practice_item['idx']}")
                print(f"{'='*60}")

                round_record = {
                    "round_idx": round_idx,
                    "practice_idx": practice_item["idx"],
                    "memory_snapshot_before": {
                        "user_exp": list(memory["user_exp"]),
                        "model_exp": list(memory["model_exp"]),
                        "proposal": list(memory["proposal"]),
                    },
                }

                print(f"  [step1] predicting...")
                predict_user, predict_response = self.step_predict(practice_item, memory)
                round_record["step1_predict"] = {
                    "user_prompt": predict_user,
                    "response": predict_response,
                }
                print(f"  [step1] done, response_len={len(predict_response)}")

                print(f"  [step2] reflecting...")
                reflect_user, reflect_response, new_proposals = self.step_reflect_and_extract(practice_item, memory, predict_response)
                round_record["step2_reflect_and_extract"] = {
                    "user_prompt": reflect_user,
                    "response": reflect_response,
                    "new_proposals_before_verify": new_proposals,
                }
                print(f"  [step2] done, {len(new_proposals)} raw proposals")

                print(f"  [step2*] self-verifying {len(new_proposals)} proposals...")
                verified_proposals, verify_records = self.step_self_verify(practice_item, memory, predict_response, new_proposals)

                memory["proposal"].extend(verified_proposals)

                round_record["step2_5_self_verify"] = {
                    "new_proposals_before_verify": new_proposals,
                    "verified_proposals": verified_proposals,
                    "num_passed": len(verified_proposals),
                    "num_rejected": len(new_proposals) - len(verified_proposals),
                    "verify_records": verify_records,
                    "proposal_pool_after": list(memory["proposal"]),
                }
                print(f"  [step2*] {len(verified_proposals)}/{len(new_proposals)} passed, pool size={len(memory['proposal'])}")

                should_review = ((round_idx + 1) % self.review_interval == 0) or ((round_idx + 1) == total_rounds)
                round_record["review_triggered"] = should_review

                if should_review:
                    print(f"  [step3] reviewing memory at round {round_idx + 1}...")
                    review_user, review_response = self.step_review_memory(memory, round_idx)

                    old_memory_snapshot = {
                        "user_exp": list(memory["user_exp"]),
                        "model_exp": list(memory["model_exp"]),
                        "proposal": list(memory["proposal"]),
                    }
                    memory = self.parse_reviewed_memory(review_response, memory)

                    round_record["step3_review"] = {
                        "user_prompt": review_user,
                        "response": review_response,
                        "memory_before_review": old_memory_snapshot,
                        "memory_after_review": {
                            "user_exp": list(memory["user_exp"]),
                            "model_exp": list(memory["model_exp"]),
                            "proposal": list(memory["proposal"]),
                        },
                    }
                    print(f"  [step3] done: user_exp={len(memory['user_exp'])} model_exp={len(memory['model_exp'])} proposal={len(memory['proposal'])}")

                    self.save_exp_memory(memory)
                    print(f"  [saved] memory updated")
                else:
                    next_review_round = ((round_idx // self.review_interval) + 1) * self.review_interval
                    print(f"  [step3] skipped (next review at round {next_review_round})")

                round_record["memory_snapshot_after"] = {
                    "user_exp": list(memory["user_exp"]),
                    "model_exp": list(memory["model_exp"]),
                    "proposal": list(memory["proposal"]),
                }

                completed_rounds.append(round_idx)
                progress["completed_rounds"] = completed_rounds
                progress[f"round_{round_idx}"] = round_record

                self.save_practice_progress(progress)
                print(f"  [saved] round {round_idx + 1} progress saved")

            except Exception as e:
                print(f"[error] round {round_idx+1} failed, skipping: {e}")
                progress[f"round_{round_idx}_error"] = str(e)
                self.save_practice_progress(progress)
                continue

        self.save_exp_memory(memory)
        progress["status"] = "completed"
        progress["final_memory"] = {
            "user_exp": list(memory["user_exp"]),
            "model_exp": list(memory["model_exp"]),
            "proposal": list(memory["proposal"]),
        }
        self.save_practice_progress(progress)

        print(f"\n{'='*60}")
        print(f"[done] user={self.user_id} test={self.test_id}")
        print(f"[final] user_exp={len(memory['user_exp'])} model_exp={len(memory['model_exp'])} proposal={len(memory['proposal'])}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Practice experiential memory update pipeline")
    parser.add_argument("--user_id", type=str, default="YOUR_USER_ID")
    parser.add_argument("--test_id", type=str, default="0")
    parser.add_argument("--practice_data_dir", type=str, default="./work_data/practice_data/experiment_data.json.practice")
    parser.add_argument("--exp_memory_dir", type=str, default="./work_data/exp_memory/experiment_data.json.practice")
    parser.add_argument("--practice_progress_dir", type=str, default="./work_data/practice_progress/experiment_data.json.practice")
    parser.add_argument("--review_interval", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="http://your-model-endpoint/v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--total_rounds", type=int, default=80)
    parser.add_argument("--num_perturbations", type=int, default=1)
    parser.add_argument("--detail_truncate_len", type=int, default=128)
    parser.add_argument("--num_virtual_scenes", type=int, default=7)
    parser.add_argument("--choices_per_question", type=int, default=8)
    parser.add_argument("--use_check_generalization", type=int, default=1)
    parser.add_argument("--use_check_perturbation", type=int, default=1)

    args = parser.parse_args()

    print("=" * 50)
    print("Args:")
    print("=" * 50)
    for k, v in vars(args).items():
        print(f"{k:20s}: {v}")
    print("=" * 50)

    sp = SelfPractice(
        practice_data_dir=args.practice_data_dir,
        exp_memory_dir=args.exp_memory_dir,
        practice_progress_dir=args.practice_progress_dir,
        user_id=args.user_id,
        test_id=args.test_id,
        review_interval=args.review_interval,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        total_rounds=args.total_rounds,
        num_perturbations=args.num_perturbations,
        detail_truncate_len=args.detail_truncate_len,
        num_virtual_scenes=args.num_virtual_scenes,
        choices_per_question=args.choices_per_question,
        use_check_generalization=args.use_check_generalization,
        use_check_perturbation=args.use_check_perturbation,
    )
    sp.run()

