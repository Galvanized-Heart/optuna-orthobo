# Runs every permutation of the following configs as hydra multirun
uv run scripts/run_benchmark.py -m \
sampler=orthobo,naive,vanilla \
benchmark=hartmann6,ackley10,levy16 \
seed=1,2,3,4,5 \
experiment.n_trials=200 \
experiment.n_startup_trials=10 \
experiment.mc_budget=64 \