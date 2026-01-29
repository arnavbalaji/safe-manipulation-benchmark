#!/bin/bash

# Usage: ./run_eval_checkpoints.sh --checkpoint_dir <path> --output_name <name> --steps <step1> <step2> ... [options]
# Example: ./run_eval_checkpoints.sh --checkpoint_dir /path/to/checkpoints --output_name no_live --steps 16000 20000 26000 30000

# Default values
CHECKPOINT_DIR=""
OUTPUT_NAME=""
VOCAB_HDF5="/media/Data/safemanibench/playback_data/shelve_item/no_live_feedback.hdf5"
FILTER_MODE=None
N_EPISODES=30
MAX_STEPS=400
DEVICE="cuda"
EXECUTE_HORIZON=1
POLICY_INPUT_TYPE="seg"
ACTION_CHUNK_SIZE=8
NUM_SEG_VIEWS=3
ENV_HEALTH_THRESHOLD=95.0
VIDEO_CAMERA_TYPE="external"
VIDEO_CAMERA_NAME="external_sensor0"
VIDEO_FPS=30
SAVE_VIDEOS=true
NORMALIZE_ACTION=true

# Checkpoint steps to iterate over (can be overridden with --steps)
STEPS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --output_name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --vocab_hdf5)
            VOCAB_HDF5="$2"
            shift 2
            ;;
        --filter_mode)
            FILTER_MODE="$2"
            shift 2
            ;;
        --n_episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --execute_horizon)
            EXECUTE_HORIZON="$2"
            shift 2
            ;;
        --policy_input_type)
            POLICY_INPUT_TYPE="$2"
            shift 2
            ;;
        --action_chunk_size)
            ACTION_CHUNK_SIZE="$2"
            shift 2
            ;;
        --num_seg_views)
            NUM_SEG_VIEWS="$2"
            shift 2
            ;;
        --env_health_threshold)
            ENV_HEALTH_THRESHOLD="$2"
            shift 2
            ;;
        --video_camera_type)
            VIDEO_CAMERA_TYPE="$2"
            shift 2
            ;;
        --video_camera_name)
            VIDEO_CAMERA_NAME="$2"
            shift 2
            ;;
        --video_fps)
            VIDEO_FPS="$2"
            shift 2
            ;;
        --no_save_videos)
            SAVE_VIDEOS=false
            shift
            ;;
        --no_normalize_action)
            NORMALIZE_ACTION=false
            shift
            ;;
        --steps)
            shift
            # Collect all following arguments until we hit another flag or end
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                STEPS+=("$1")
                shift
            done
            ;;
        --help)
            echo "Usage: $0 --checkpoint_dir <path> --output_name <name> --steps <step1> <step2> ... [options]"
            echo ""
            echo "Required arguments:"
            echo "  --checkpoint_dir       Path to checkpoint directory"
            echo "  --output_name          Output name prefix for video directories"
            echo "  --steps                Space-separated list of checkpoint steps to evaluate"
            echo ""
            echo "Optional arguments:"
            echo "  --vocab_hdf5           Path to vocab HDF5 file (default: /media/Data/safemanibench/playback_data/shelve_item/no_live_feedback.hdf5)"
            echo "  --filter_mode          Filter mode used during training (default: episode)"
            echo "  --n_episodes           Number of episodes to run (default: 30)"
            echo "  --max_steps            Max steps per episode (default: 400)"
            echo "  --device               Device for policy (default: cuda)"
            echo "  --execute_horizon      Actions to execute before re-planning (default: 1)"
            echo "  --policy_input_type    Input type: seg, joint_pos_eef_pose, etc. (default: seg)"
            echo "  --action_chunk_size    Action chunk size (default: 8)"
            echo "  --num_seg_views        Number of segmentation views (default: 3)"
            echo "  --env_health_threshold Environment health threshold (default: 95.0)"
            echo "  --video_camera_type    Camera type for video (default: external)"
            echo "  --video_camera_name    Camera name for video (default: external_sensor0)"
            echo "  --video_fps            FPS for saved videos (default: 30)"
            echo "  --no_save_videos       Disable video saving"
            echo "  --no_normalize_action  Disable action normalization"
            echo ""
            echo "Example:"
            echo "  $0 --checkpoint_dir /path/to/checkpoints --output_name no_live --steps 16000 20000 26000 30000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: --checkpoint_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$OUTPUT_NAME" ]; then
    echo "Error: --output_name is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ ${#STEPS[@]} -eq 0 ]; then
    echo "Error: --steps is required (provide at least one checkpoint step)"
    echo "Example: --steps 16000 20000 26000 30000"
    echo "Use --help for usage information"
    exit 1
fi

# Build optional flags
OPTIONAL_FLAGS=""
if [ "$SAVE_VIDEOS" = true ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --save_videos"
fi
if [ "$NORMALIZE_ACTION" = true ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --normalize_action"
fi

echo "========================================"
echo "Running evaluation with:"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Output name: $OUTPUT_NAME"
echo "  Steps: ${STEPS[*]}"
echo "  Vocab HDF5: $VOCAB_HDF5"
echo "  Filter mode: $FILTER_MODE"
echo "  Episodes: $N_EPISODES"
echo "========================================"

for step in "${STEPS[@]}"; do
    echo ""
    echo "========================================"
    echo "Evaluating checkpoint: step_${step}.pth"
    echo "========================================"

    python eval_scripts/shelve_item_eval.py \
        --checkpoint "${CHECKPOINT_DIR}/step_${step}.pth" \
        --filter_mode "$FILTER_MODE" \
        --vocab_hdf5 "$VOCAB_HDF5" \
        --video_dir "resources/eval_results/shelve_item/${OUTPUT_NAME}_${step}" \
        --n_episodes "$N_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --device "$DEVICE" \
        --execute_horizon "$EXECUTE_HORIZON" \
        --policy_input_type "$POLICY_INPUT_TYPE" \
        --action_chunk_size "$ACTION_CHUNK_SIZE" \
        --num_seg_views "$NUM_SEG_VIEWS" \
        --env_health_threshold "$ENV_HEALTH_THRESHOLD" \
        --video_camera_type "$VIDEO_CAMERA_TYPE" \
        --video_camera_name "$VIDEO_CAMERA_NAME" \
        --video_fps "$VIDEO_FPS" \
        $OPTIONAL_FLAGS

    echo "Finished evaluating step_${step}.pth"
done

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "========================================"
