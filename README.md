# Learning Fuel-Burn Strategies in a Simple Racing Game

## 1. Project overview

This project investigates how a fixed fuel budget should be allocated across a short race in order to maximise total distance travelled.

The race lasts for 10 rounds. The vehicle starts with 100 litres of fuel, and at most 30 litres can be burned in any one round. Burning fuel increases the vehicle's energy, which determines the distance travelled in that round. Some of the energy is then carried over into the next round.

The project compares several approaches on the same racing-fuel environment:

1. A manual human-designed baseline strategy.
2. A dynamic-programming benchmark.
3. A constrained optimisation baseline using SLSQP.
4. A Deep Q-Network reinforcement-learning agent.
5. A drag-sensitive extension where energy carry-over decreases when post-burn energy is high.

The main purpose is not to show that reinforcement learning outperforms the optimiser. Instead, the project tests whether a DQN agent can recover the same or near-same policy structure as validated optimisation benchmarks.

---

## 2. Folder structure

The supporting material folder contains the following main files and folders.

```text
project/
|
├── notebooks/
│   ├──baseline_optimisation.ipynb
│   ├──dqn_training.ipynb
│   ├──rf_clean_dqn/
│   ├──rf_dqn_logs/
│   ├──rf_drag_dqn/
│   ├──rf_drag_multiseed/
│   ├──drag_burn_schedules.png
│   ├──drag_continuous_vs_integer_benchmark.png
│   ├──drag_energy_comparison.png
│   ├──drag_energy_trajectory.png
│   ├── drag_multiseed_dqn_vs_integer_benchmark.png
│   ├──drag_retention_comparison.png
│   └──drag_retention_trajectory.png
|
└── README.md
```

Depending on how the folder is downloaded, the `README.md` file may appear inside the `project` folder or inside the `notebooks` folder. The notebooks and output folders should remain in the same relative structure because some cells save or load files from local paths.

---

## 3. Main notebooks

### 3.1 `baseline_optimisation.ipynb`

This notebook contains the simulator and optimisation-side analysis.

It includes:

- the base RacingFuelEnv simulator,
- the manual baseline strategy,
- the dynamic-programming benchmark,
- the fixed-friction SLSQP optimisation baseline,
- the drag-sensitive carry-over extension,
- the continuous SLSQP optimum under drag-sensitive dynamics,
- the rounded integer-feasible benchmark used for DQN comparison,
- the drag-sensitive comparison figures.

Important generated figures include:

```text
drag_continuous_vs_integer_benchmark.png
drag_burn_schedules.png
drag_energy_trajectory.png
drag_retention_trajectory.png
```

The most important drag-sensitive optimiser results are:

```text
Continuous SLSQP optimum return: 62.848043
Continuous SLSQP burn schedule:
[30.000, 20.883, 9.793, 9.793, 9.793, 9.793, 9.793, 0.152, 0.000, 0.000]

Rounded integer-feasible benchmark return: 62.847405
Rounded integer-feasible burn schedule:
[30, 21, 10, 9, 10, 10, 10, 0, 0, 0]
```

The rounded integer-feasible benchmark is used because the DQN action space uses one-litre discrete burn actions.

---

### 3.2 `dqn_training.ipynb`

This notebook contains the reinforcement-learning implementation and evaluation.

It includes:

- the Gymnasium-compatible DQN environment,
- fixed-friction DQN training and evaluation,
- best-checkpoint evaluation,
- drag-sensitive DQN evaluation,
- multi-seed DQN comparison,
- the final DQN-vs-benchmark figure.

Important generated figures include:

```text
drag_dqn_vs_integer_benchmark_burns.png
drag_energy_comparison.png
drag_retention_comparison.png
drag_multiseed_dqn_vs_integer_benchmark.png
```

The key final drag-sensitive DQN results are:

```text
Best multi-seed DQN return: 62.846689
Best multi-seed DQN burn schedule:
[30, 20, 10, 10, 10, 10, 10, 0, 0, 0]

Integer-feasible optimiser benchmark return: 62.847405
Absolute gap: 0.000716

Multi-seed DQN returns:
[62.846689, 62.844991, 62.844663]

Mean return: 62.845448
Sample standard deviation: 0.001087
```

---

## 4. Python dependencies

The notebooks were developed using Python in JupyterHub/JupyterLab.

The main Python packages used are:

```text
numpy
matplotlib
scipy
gymnasium
stable-baselines3
torch
```

A typical setup command is:

```bash
pip install numpy matplotlib scipy gymnasium stable-baselines3 torch
```

If running on the QMUL JupyterHub environment, many packages may already be installed. Some notebooks contain setup cells that can be run only if a package is missing.

---

## 5. How to run the project

### Recommended order

Run the notebooks in this order:

1. `baseline_optimisation.ipynb`
2. `dqn_training.ipynb`

This order is recommended because the optimisation notebook defines and explains the baselines that are later used for DQN comparison.

---

### 5.1 Running `baseline_optimisation.ipynb`

Open the notebook in JupyterLab and run the cells from top to bottom.

This notebook should run without needing saved model files. It calculates the manual baseline, DP benchmark, fixed-friction SLSQP solution, and drag-sensitive SLSQP solution directly.

The most important section is the drag-sensitive extension near the bottom of the notebook. It produces the continuous-vs-integer benchmark comparison and the drag-sensitive analysis figures.

Expected key output:

```text
Continuous optimum total distance: 62.848043
Integer benchmark total distance: 62.847405
Greedy total distance: 61.097609
Uniform total distance: 57.683472
```

---

### 5.2 Running `dqn_training.ipynb`

This notebook contains training and evaluation code for the DQN agent.

Some cells may take longer to run because they train reinforcement-learning agents. For review purposes, the notebook includes saved outputs and generated figures from completed runs.

The folders below contain saved training outputs and/or best model checkpoints:

```text
rf_clean_dqn/
rf_dqn_logs/
rf_drag_dqn/
rf_drag_multiseed/
```

If the examiner only wants to inspect the final results and figures, the notebook outputs are already included. If the examiner wants to rerun training, run the notebook from top to bottom, but note that RL training cells may take time.

For the final drag-sensitive multi-seed figure, the notebook contains a final reporting cell that recreates the figure from the completed multi-seed results. This avoids unnecessary retraining while still matching the reported results.

Expected key output:

```text
Best DQN return: 62.846689
Integer-feasible benchmark return: 62.847405
Gap: 0.000716
```

---

## 6. Notes on fixed-friction and drag-sensitive experiments

The project contains two versions of the environment.

### Fixed-friction model

The original baseline model uses a fixed retention coefficient of 0.90. Under this model, the optimal strategy is strongly front-loaded:

```text
[30, 30, 30, 10, 0, 0, 0, 0, 0, 0]
```

The best DQN checkpoint matched this fixed-friction optimum exactly.

### Drag-sensitive model

The drag-sensitive model makes the retention coefficient decrease as post-burn energy increases. This means that very high energy causes greater losses before the next round.

This creates a more interesting pacing problem. The continuous optimum becomes fractional and smoother, while the DQN is compared against a rounded integer-feasible benchmark because it can only choose one-litre actions.

---

## 7. Figure guide

The main figures used in the final report are generated by the notebooks.

### Main report figures

```text
Figure 13: drag_retention_trajectory.png
Figure 14: drag_burn_schedules.png
Figure 15: drag_multiseed_dqn_vs_integer_benchmark.png
```

### Appendix B supplementary figures

```text
Figure 16: drag_continuous_vs_integer_benchmark.png
Figure 17: drag_energy_trajectory.png
Figure 18: drag_energy_comparison.png
```

Older files such as `drag_dqn_vs_optimizer_burns.png` are retained only as development artefacts and are not used in the final report.

---

## 8. Reproducibility notes

The project uses deterministic evaluation for reported DQN returns. Random seeds are set in the notebooks where appropriate.

For reinforcement learning, the final model in memory is not always the best model seen during training. Therefore, the project reports best-checkpoint evaluation where appropriate. This is why saved model folders and evaluation callbacks are included in the supporting material.

The final drag-sensitive DQN comparison is based on the best multi-seed result from completed runs, with the reported mean and sample standard deviation across three seeds.

---

## 9. Important interpretation note

The drag-sensitive continuous SLSQP optimum and the rounded integer-feasible benchmark are not the same thing.

The continuous SLSQP optimum can use fractional fuel values. The DQN cannot do this because DQN uses a discrete action space with one-litre actions from 0 to 30 litres.

Therefore, the DQN should be compared against the rounded integer-feasible benchmark, not directly against the continuous optimum.

This is why the final comparison uses:

```text
Rounded integer-feasible benchmark:
[30, 21, 10, 9, 10, 10, 10, 0, 0, 0]

Best multi-seed DQN:
[30, 20, 10, 10, 10, 10, 10, 0, 0, 0]
```

---

## 10. Contact / author

Student: Raymond Mario Charles  
University Email Address: ec23082@qmul.ac.uk
Programme: BEng Computer Systems Engineering  
Project title: Learning Fuel-Burn Strategies in a Simple Racing Game  
Supervisor: Soren Riis 

