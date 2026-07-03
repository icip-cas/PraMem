SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

practive_file_name="experiment_data.json.practice"
DATA_DIR="./work_data/practice_data/${practive_file_name}"
MEM_DIR="./work_data/exp_memory/${practive_file_name}"
PROGRESS_DIR="./work_data/practice_progress/${practive_file_name}"
LOG_DIR="./work_data/practice_logs/${practive_file_name}"
SCRIPT="python -u z_5_practice_with_self_check.py"

mkdir -p "$LOG_DIR"
mkdir -p "$MEM_DIR"
mkdir -p "$PROGRESS_DIR"

mapfile -t FILES < <(ls "$DATA_DIR"/*.jsonl 2>/dev/null | sort)

TOTAL=${#FILES[@]}
echo "Total files found: $TOTAL"

if [ $TOTAL -eq 0 ]; then
    echo "No jsonl files found!"
    exit 1
fi

mapfile -t FILES < <(printf '%s\n' "${FILES[@]}" | shuf)
echo "Files shuffled randomly."

MODELS=(
    "http://your-model-endpoint-1/v1"
    "http://your-model-endpoint-2/v1"
)

NUM_MODELS=${#MODELS[@]}

GROUP_SIZE=$(( (TOTAL + NUM_MODELS - 1) / NUM_MODELS ))
echo "Files per group (approx): $GROUP_SIZE"

for (( i=0; i<NUM_MODELS; i++ )); do
    MODEL=${MODELS[$i]}
    START=$(( i * GROUP_SIZE ))
    END=$(( START + GROUP_SIZE ))

    if [ $START -ge $TOTAL ]; then
        echo "Model $MODEL: no files assigned, skipping."
        continue
    fi
    if [ $END -gt $TOTAL ]; then
        END=$TOTAL
    fi

    GROUP_FILES=("${FILES[@]:$START:$GROUP_SIZE}")
    ACTUAL_COUNT=${#GROUP_FILES[@]}
    echo "Model $MODEL assigned $ACTUAL_COUNT files (index $START to $((END-1)))"

    (
        for FILE in "${GROUP_FILES[@]}"; do
            BASENAME=$(basename "$FILE" .jsonl)
            USER_ID=$(echo "$BASENAME" | cut -d'_' -f1)
            TEST_ID=$(echo "$BASENAME" | cut -d'_' -f2)
            LOG_FILE="$LOG_DIR/${USER_ID}_${TEST_ID}.log"

            $SCRIPT \
                --practice_data_dir "$DATA_DIR" \
                --exp_memory_dir "$MEM_DIR" \
                --practice_progress_dir "$PROGRESS_DIR" \
                --user_id "$USER_ID" \
                --test_id "$TEST_ID" \
                --model_name "$MODEL" \
                > "$LOG_FILE" 2>&1
        done
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$MODEL] All tasks finished."
    ) &

done
