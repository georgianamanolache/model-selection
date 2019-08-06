<snippet>
  <content><![CDATA[
Model selection for machine learning with meta-data.
================================================

Algorithm and hyperparameter optimization is a well studied topic in machine learning. Recent work relies on Bayesian optimization especially for solving the hyperparameter
optimization problem for a given machine learning algorithm. Suppose that prior tasks are available on which the algorithm and hyperparameter optimization has already been
dealt with either by domain experts or through extensive cross-validation trials. In this work, we propose to leverage outcomes from these tasks together with Bayesian optimization procedures to
warm-start algorithm selection and hyperparameter tuning for a new task.

## Installation
------------
Requires:
[george](https://github.com/automl/george.git)
[ROBO](https://github.com/automl/RoBO/blob/master/README.md)

## Usage
------------
1. Run 'model-selection.py'; this will procude a 'json' file
2. Plot results with 'plot.py'
]]></content>
  <tabTrigger>readme</tabTrigger>
</snippet>
