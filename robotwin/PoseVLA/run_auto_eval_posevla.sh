#!/bin/bash
# Auto evaluation script for PoseVLA policy on RoboTwin platform
# Uses policy/PoseVLA/robotwin/PoseVLA deployment package.
# Run tasks across multiple GPUs

echo "Starting PoseVLA evaluation on RoboTwin at $(date)"

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${SCRIPT_DIR}}"
POLICY_NAME="PoseVLA"
POLICY_REPO_DIR="${ROBOTWIN_ROOT}/policy/${POLICY_NAME}"
POLICY_RUNTIME_DIR="${POLICY_REPO_DIR}/robotwin"
POLICY_DEPLOY_DIR="${POLICY_RUNTIME_DIR}/${POLICY_NAME}"
CONFIG_FILE="${POLICY_DEPLOY_DIR}/deploy_policy.yml"
CHECKPOINT_ID="19999"
CKPT_DIR_NAME="align_bs12_1_robotwin"
TASK_CONFIG="demo_randomized"  # or "demo_clean" demo_randomized
SEED=0
ACTION_TYPE='eep'  # PoseVLA Robotwin deployment expects eep by default.

# GPU configuration - using 8 GPUs
GPU_IDS=(0) # 1 2 3 4 5 6 7

# All 50 tasks
TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "click_alarmclock"
    "click_bell"
    "dump_bin_bigbin"
    "grab_roller"
    "handover_block"
    "handover_mic"
    "hanging_mug"
    "lift_pot"
    "move_can_pot"
    "move_pillbottle_pad"
    "move_playingcard_away"
    "move_stapler_pad"
    "open_laptop"
    "open_microwave"
    "pick_diverse_bottles"
    "pick_dual_bottles"
    "place_a2b_left"
    "place_a2b_right"
    "place_bread_basket"
    "place_bread_skillet"
    "place_burger_fries"
    "place_can_basket"
    "place_cans_plasticbox"
    "place_container_plate"
    "place_dual_shoes"
    "place_empty_cup"
    "place_fan"
    "place_mouse_pad"
    "place_object_basket"
    "place_object_scale"
    "place_object_stand"
    "place_phone_stand"
    "place_shoe"
    "press_stapler"
    "put_bottles_dustbin"
    "put_object_cabinet"
    "rotate_qrcode"
    "scan_object"
    "shake_bottle"
    "shake_bottle_horizontally"
    "stack_blocks_three"
    "stack_blocks_two"
    "stack_bowls_three"
    "stack_bowls_two"
    "stamp_seal"
    "turn_switch"
)

# ============================================================================
# Validation
# ============================================================================
if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$POLICY_RUNTIME_DIR" ]; then
    echo "Error: PoseVLA runtime dir not found: $POLICY_RUNTIME_DIR"
    exit 1
fi

cd "$ROBOTWIN_ROOT" || exit 1

# Set environment
export PYTHONPATH="${POLICY_RUNTIME_DIR}:${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1  # Force unbuffered output for real-time logs

# Create logs directory
LOG_DIR="${ROBOTWIN_ROOT}/eval_result/auto_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

echo -e "\n\033[33m=== Evaluation Configuration ===\033[0m"
echo "RoboTwin Root: $ROBOTWIN_ROOT"
echo "Policy: $POLICY_NAME"
echo "Policy Runtime: $POLICY_RUNTIME_DIR"
echo "Config: $CONFIG_FILE"
echo "Checkpoint ID: $CHECKPOINT_ID"
echo "Checkpoint Dir: $CKPT_DIR_NAME"
echo "Task Config: $TASK_CONFIG"
echo "Tasks: ${#TASKS[@]}"
echo "GPUs: ${GPU_IDS[*]}"
echo "Action Type: $ACTION_TYPE"
echo "Seed: $SEED"
echo "Log Dir: $LOG_DIR"
echo "================================"

# ============================================================================
# GPU Management Functions
# ============================================================================
declare -A gpu_pid

for gpu_id in "${GPU_IDS[@]}"; do
    gpu_pid[$gpu_id]=""
done

is_running() {
    [ -n "$1" ] && kill -0 "$1" 2>/dev/null
}

get_free_gpu() {
    while true; do
        for gpu_id in "${GPU_IDS[@]}"; do
            if ! is_running "${gpu_pid[$gpu_id]}"; then
                echo "$gpu_id"
                return 0
            fi
        done
        sleep 2
    done
}

show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local bar_length=50
    local filled=$((percent * bar_length / 100))
    
    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%$((bar_length - filled))s" | tr ' ' ' '
    printf "] %d%% (%d/%d)" "$percent" "$current" "$total"
}

# ============================================================================
# Launch Tasks
# ============================================================================
pids=()
completed=0
total=${#TASKS[@]}

echo -e "\n\033[32mLaunching evaluation tasks...\033[0m"

for task in "${TASKS[@]}"; do
    gpu_id=$(get_free_gpu)
    log_file="${LOG_DIR}/${task}.log"

    echo -e "\033[36m→ Task: $task | GPU: $gpu_id\033[0m"

    (
        # Isolate both CUDA and Vulkan GPU access
        export CUDA_VISIBLE_DEVICES=$gpu_id
        export MESA_VK_DEVICE_SELECT=$gpu_id  # For Mesa Vulkan driver
        export VK_DEVICE_INDEX=$gpu_id        # Generic Vulkan device selection
        
        python -u script/eval_policy.py \
            --config "${CONFIG_FILE}" \
            --overrides \
            --task_name "${task}" \
            --task_config "${TASK_CONFIG}" \
            --seed "${SEED}" \
            --checkpoint_id "${CHECKPOINT_ID}" \
            --ckpt_dir_name "${CKPT_DIR_NAME}" \
            --policy_name "${POLICY_NAME}" \
            --action_type "${ACTION_TYPE}" \
            > "$log_file" 2>&1
                    # --use_prior "true" \
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "✓ Task $task completed successfully" >> "$log_file"
        else
            echo "✗ Task $task failed with exit code $exit_code" >> "$log_file"
        fi
    ) &
    
    pid=$!
    gpu_pid[$gpu_id]=$pid
    pids+=($pid)
    
    # Delay to avoid SAPIEN Vulkan renderer initialization conflicts
    sleep 3
done

# ============================================================================
# Wait for Completion
# ============================================================================
echo -e "\n\033[33mWaiting for completion...\033[0m"

for pid in "${pids[@]}"; do
    wait "$pid"
    ((completed++))
    show_progress $completed $total
done

echo -e "\n\033[32m✓ All tasks completed!\033[0m"

# ============================================================================
# Generate Summary
# ============================================================================
summary="${LOG_DIR}/evaluation_summary.txt"

cat > "$summary" << EOF
PoseVLA Evaluation Summary
======================================
Date: $(date)
Host: $(hostname)
RoboTwin: $ROBOTWIN_ROOT
Policy: $POLICY_NAME
Policy Runtime: $POLICY_RUNTIME_DIR
Config: $CONFIG_FILE
Checkpoint ID: $CHECKPOINT_ID
Checkpoint Dir: $CKPT_DIR_NAME
Task Config: $TASK_CONFIG
Seed: $SEED
Total Tasks: $total
GPUs: ${GPU_IDS[*]}

Task Results:
-------------
EOF

success=0
failed=0

for task in "${TASKS[@]}"; do
    log_file="${LOG_DIR}/${task}.log"
    
    if [ ! -f "$log_file" ]; then
        echo "  ⚠️  $task: LOG NOT FOUND" >> "$summary"
        ((failed++))
    elif grep -q "completed successfully\|Episode.*completed" "$log_file" 2>/dev/null; then
        echo "  ✅ $task: SUCCESS" >> "$summary"
        ((success++))
    else
        echo "  ❌ $task: FAILED" >> "$summary"
        ((failed++))
    fi
done

cat >> "$summary" << EOF

Summary Statistics:
-------------------
✅ Successful: $success
❌ Failed: $failed
Total: $total
Success Rate: $(awk "BEGIN {printf \"%.1f\", $success * 100.0 / $total}")%

Logs: $LOG_DIR
EOF

echo -e "\n\033[36m=== Summary ===\033[0m"
cat "$summary"

if [ $failed -eq 0 ]; then
    echo -e "\033[32m🎉 All tasks passed!\033[0m"
    exit 0
else
    echo -e "\033[33m⚠️  Check logs for failures.\033[0m"
    exit 1
fi
