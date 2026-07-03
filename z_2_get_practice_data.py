import os
import json
import tqdm
import random
from prepare_experiment_data import has_sufficient_context_for_prediction
from prompt_builder import build_history_summary, get_all_questions_for_action, build_single_binary_prompt, build_single_continuous_prompt, build_single_text_prompt, build_movie_prompt


def get_simulated_input_single(user_profile, action_history, test_action, max_history_tokens):
    prompt_part1 = ""
    scenario_list = []
    question_list = []

    action_history = [action for action in action_history if has_sufficient_context_for_prediction(action)]

    for question_info in get_all_questions_for_action(test_action):
        question_type = question_info.get("type", "binary")

        if question_type == "binary":
            prompt_data = build_single_binary_prompt(
                user_profile,
                action_history,
                test_action,
                question_info,
                max_history_tokens=max_history_tokens,
            )
        elif question_type == "multi_class":
            prompt_data = build_movie_prompt(
                user_profile,
                action_history,
                test_action,
                question_info,
                max_history_tokens=max_history_tokens,
            )
        elif question_type == "continuous":
            prompt_data = build_single_continuous_prompt(
                user_profile,
                action_history,
                test_action,
                question_info,
                max_history_tokens=max_history_tokens,
            )
        elif question_type == "text":
            prompt_data = build_single_text_prompt(
                user_profile,
                action_history,
                test_action,
                question_info,
                max_history_tokens=max_history_tokens,
            )
        else:
            print("WARNING: unknown question type")
            continue

        prompt = prompt_data['prompt']

        if "## 输入三：当前测试场景" in prompt:
            part1_spliter = "## 输入三：当前测试场景"
        elif "## 输入三：当前客服对话场景" in prompt:
            part1_spliter = "## 输入三：当前客服对话场景"
        elif "## 预测任务" in prompt:
            part1_spliter = "## 预测任务"
        else:
            raise ValueError("No valid splitter found in prompt")

        if prompt_part1 == "":
            prompt_part1 = prompt.split(part1_spliter)[0].strip()
        else:
            assert prompt_part1 == prompt.split(part1_spliter)[0].strip()

        if question_type == "binary":
            scenario_list.append(prompt_data['scenario_desc'])
            question_list.append(prompt_data['yes_no_question'])
        elif question_type == "multi_class":
            scenario_list.append(prompt_data['scenario_desc'])
        elif question_type == "continuous":
            scenario_list.append(prompt_data['scenario_desc'])
            question_list.append(prompt_data['question_text'])
        elif question_type == "text":
            field = prompt_data['field']
            if field == "search_keyword":
                question_list.append('Predict what keyword this user would type into the search box now')
            elif field == "next_user_message":
                question_list.append(prompt_data['question_text'])
            else:
                scenario_list.append(prompt_data['scenario_desc'])
                question_list.append(prompt_data['question_text'])
        else:
            print("WARNING: unknown question type")
            continue

        scenario_list = list(set(scenario_list))
        prompt_part2 = "\n".join(scenario_list + question_list)

    return prompt_part1, prompt_part2

def get_simulated_input(user_profile, history_actions, target_actions, max_history_tokens):
    task_id = 1
    simulated_input_part1 = ""
    simulated_input_part2 = ""

    for action_id in range(len(target_actions)):
        prompt_part1, prompt_part2 = get_simulated_input_single(user_profile, history_actions, target_actions[action_id], max_history_tokens)

        if simulated_input_part1 == "":
            simulated_input_part1 = prompt_part1
        else:
            pass

        simulated_input_part2 += f"===== Practice task {task_id} =====\n{prompt_part2}\n\n"
        task_id += 1

    simulated_input = f"{simulated_input_part1}\n\n{simulated_input_part2}".strip()
    return simulated_input

def get_simulated_label(target_actions):
    summary = build_history_summary(target_actions, max_history_tokens=9999999)
    summary = "\n".join(summary.split('\n')[1:])
    summary = "The following are the answers to the practice tasks above. If a question is not explicitly answered, the default answer is No. " + summary
    simulated_label = summary.replace("【行为", "【Practice task answer")
    return simulated_label


if __name__ == "__main__":

    practice_source_name = "experiment_data.json.practice"
    practice_source = json.load(open(f"./work_data/{practice_source_name}.json"))

    output_dir_path = f"./work_data/practice_data/{practice_source_name}"
    os.makedirs(output_dir_path, exist_ok=True)

    print(practice_source_name)
    print(output_dir_path)

    max_history_tokens = 12000

    for user_data in tqdm.tqdm(practice_source['users'], desc="Processing users"):
        user_id = user_data['user_id']
        user_profile = user_data['user_profile']
        test_time_all_actions = user_data['test_time_all_actions']
        base_history = user_data['base_history']

        for ti, test_idx in enumerate(list(user_data['practice_questions'].keys())[0:1]):

            print(f"test_idx={test_idx}, {ti}/{len(list(user_data['practice_questions'].keys()))}")

            output_file = open(f"{output_dir_path}/{user_id}_0.jsonl", 'w')

            random.seed(1024)
            random.shuffle(user_data['practice_questions'][test_idx])
            for practice_idx in tqdm.trange(len(user_data['practice_questions'][test_idx]), desc="Practice items"):
                target_actions = [user_data['practice_questions'][test_idx][practice_idx]['action']]
                target_action_start_index = user_data['practice_questions'][test_idx][practice_idx]["test_time_index"]

                history_actions = base_history + test_time_all_actions[:target_action_start_index]

                try:
                    simulated_input = get_simulated_input(user_profile, history_actions, target_actions, max_history_tokens)
                    simulated_label = get_simulated_label(target_actions)

                    output_file.write(json.dumps({
                        "idx": f'{user_id}_{test_idx}_{practice_idx}',
                        "simulated_input": simulated_input,
                        "simulated_label": simulated_label,
                        "target_actions": target_actions
                    }, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"Error processing {user_id}_{test_idx}_{practice_idx}: {e}")
                    continue

            output_file.close()

    print("done")
