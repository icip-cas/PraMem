# PraMem

PraMem: Practice-Derived Experiential Memory for Long-Horizon Behavior Prediction

## 0. Environment

```
pip install openai tqdm jieba rank_bm25 pytz openpyxl
```

## 1. Data Preparation

Download `experiment_data.json` from [Google Drive](https://drive.google.com/file/d/1nvxxlVSlLDz5imLxFOWUU3u_2sdFTTsM/view?usp=drive_link) and place it at:

```
work_data/experiment_data.json
```

## 2. Build Practice Data

**Step 1**: Sample practice questions from each user's history.

```
python z_1_get_practice_source.py \
    --experiment_data ./work_data/experiment_data.json \
    --N_per_test 100
# Output: ./work_data/experiment_data.json.practice.json
```

**Step 2**: Build formatted practice inputs for each user.

```
python z_2_get_practice_data.py
# Output: ./work_data/practice_data/experiment_data.json.practice/{user_id}_0.jsonl
```

## 3. Build Experiential Memory

Run Self-Practice for all users. Edit `z_5_practice_with_self_check.sh` to set your model endpoints, then:

```
bash z_5_practice_with_self_check.sh
# Output: ./work_data/exp_memory/experiment_data.json.practice/{user_id}_0.json
#         ./work_data/practice_progress/experiment_data.json.practice/{user_id}_0.json
```

To run a single user:

```
python z_5_practice_with_self_check.py \
    --user_id <user_id> \
    --test_id 0 \
    --model_name http://<your-model-endpoint>/v1
```

## 4. Evaluate

Run evaluation with PraMem memory:

```
python evaluator.py \
    --USE_EXP_MEM 101 \
    --mem_name experiment_data.json.practice \
    --models gpt-oss-120b \
    --max-history-tokens 8000
# Output: ./work_data/results/
```

`--USE_EXP_MEM` options:

- `0` — no memory (baseline)
- `101` — PraMem
