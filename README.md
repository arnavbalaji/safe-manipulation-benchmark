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


