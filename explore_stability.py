from ray import tune
from ray.tune import grid_search as gs
from numpy import arange as ran
from functions_MSF import MSF_trial

# Model parameter space. They need to be present in the yaml file. Thos are the set of parameters that will be
# explored and whose MSF will be computed. In this case, we are exploring the mu1im and mu0im parameters using
# the grid_search function from ray.tune.
parameter_space = {
                   'mu1im': gs(ran(-2, 2, 0.1)),
                   'mu0im': gs(ran(-3, 3, 0.1)),
                   }

meta_config = {'configuration_name': 'TEST.yml',
               'hyperparameters': parameter_space}

tuner = tune.run(
    tune.with_resources(MSF_trial,
                        resources={"cpu": 0.5}),
    verbose=1,
    num_samples=1,
    config=meta_config
)
