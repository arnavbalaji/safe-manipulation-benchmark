# Installation

1. conda create -n safemanibench python=3.10
2. conda activate safemanibench
3. git clone https://github.com/UT-Austin-RobIn/BEHAVIOR-1K.git
4. cd BEHAVIOR-1K
5. git checkout proj/safemanibench
6. Try: `HF_HUB_DISABLE_XET=1 ./setup.sh --omnigibson --bddl --dataset --primitives`
if curobo error, 
Try: `HF_HUB_DISABLE_XET=1 ./setup.sh --omnigibson --bddl --dataset`
7. cd .. && git clone https://github.com/arnavbalaji/safe-manipulation-benchmark.git
8. cd safe-manipulation-benchmark
9. pip install -e .
10. `OMNIGIBSON_HEADLESS=1 python unit_tests/nav_to_table.py`


# BEHAVIOR Related commands:

### Download raw demonstration hdf5
`hf download behavior-1k/2025-challenge-rawdata --repo-type dataset  --local-dir $HOME/behavior_dataset --include "task-0030/*"`

https://huggingface.co/datasets/behavior-1k/2025-challenge-rawdata/tree/main

### Download annotations
`hf download behavior-1k/2025-challenge-demos --repo-type dataset  --local-dir $HOME/behavior_dataset/annotations --include "annotations/task-0030/*"`

https://huggingface.co/datasets/behavior-1k/2025-challenge-demos/tree/main

### Rolling out Gr00t
python eval_scripts/behavior_eval.py --task_name make_microwave_popcorn --rollout_name rollout_0000_00402500 --playback


# Other commands:

### Data collection and evals
#### without live feedback
python data_collection_scripts/shelve_item.py --teleop --n_episodes 15 --collect_hdf5_path resources/teleop_data/trial_6.hdf5 
#### with live feedback
python data_collection_scripts/shelve_item.py --teleop --n_episodes 15 --collect_hdf5_path resources/teleop_data/trial_6.hdf5 --live_feedback 
#### just compute metrics
python data_collection_scripts/shelve_item.py --compute_metrics --playback_hdf5_path resources/playback_data/shelve_item/no_live_feedback.hdf5
#### eval
python eval_scripts/shelve_item_eval.py --checkpoint /home/arpit/test_projects/rl-flow-matching/checkpoints/shelve_item_no_live_feedback/step_10500.pth --vocab_hdf5 resources/playback_data/shelve_item/no_live_feedback.hdf5 --save_videos --video_dir resources/videos/shelve_item/eval_no_live_feedback

Firewood:
python eval_scripts/firewood_eval.py --checkpoint /home/arpit/test_projects/rl-flow-matching/checkpoints/add_firewood_trial_1/step_30000.pth --vocab_hdf5 /home/arpit/test_projects/safe-manipulation-benchmark/resources/playback_data/add_firewood/firewood_trial_1_playback.hdf5 --n_episodes 15 --video_dir resources/eval_results/firewood/no_live --max_steps 400 --seed 1


