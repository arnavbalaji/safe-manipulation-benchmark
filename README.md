# Installation

1. conda create -n safemanibench python=3.10
2. conda activate safemanibench
3. git clone https://github.com/UT-Austin-RobIn/BEHAVIOR-1K.git
4. cd BEHAVIOR-1K
5. git checkout proj/safemanibench
6. HF_HUB_DISABLE_XET=1 ./setup.sh --omnigibson --bddl --dataset
7. cd .. && git clone https://github.com/arnavbalaji/safe-manipulation-benchmark.git
8. cd safe-manipulation-benchmark
9. pip install -e .
10. OMNIGIBSON_HEADLESS=1 python unit_tests/nav_to_table.py