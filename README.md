Algorithm and hyperparameter optimization is a well studied topic in machine learning. Recent work relies on Bayesian optimization especially for solving the hyperparameter
optimization problem for a given machine learning algorithm. Suppose that prior tasks are available on which the algorithm and hyperparameter optimization has already been
dealt with either by domain experts or through extensive cross-validation trials. In this work, we propose to leverage outcomes from these tasks together with Bayesian optimization procedures to
warm-start algorithm selection and hyperparameter tuning for a new task.

## Implementation details

For this experiment, we use the same set of six functions for generating six different sets of data. Each function substitutes an algorithm behaviour for which we produce six similar data sets. 

For the model selection, we use three randomly selected configurations to initialize each model. Similarly, we fit GPs to each meta-learning dataset on 20 randomly selected configurations. Then, each optimization run is initialized with three randomly selected configurations, until a total of 20 runs is conducted. The optimization is repeated 120 times, each with a different random selection of configurations. This is repeated for each algorithm function, for a total of 720 optimization runs. For details, see PDF attached.

## Installation

Requires: [george](https://github.com/automl/george.git), [ROBO](https://github.com/automl/RoBO/blob/master/README.md).

## Usage
1. Run 'model-selection.py'; this will procude a 'json' file
2. Plot results with 'plot.py' (which plots the distance to the global minimum at each iteration, or [regret](https://www.ismll.uni-hildesheim.de/pub/pdfs/wistuba_et_al_ECML_2016.pdf))

