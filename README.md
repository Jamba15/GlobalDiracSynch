
# Stability Exploration and Analysis Repository

This repository contains a set of scripts and functions used for exploring and analyzing model stability through parameter tuning and visualization. The main components of this repository are the scripts `explore_stability.py` and `plot_results.py`, which utilize a set of utility functions defined in various Python files (`functions_*.py`). The goal of these scripts is to automate parameter exploration and visualize results based on configurations defined by the user.
Here's the markdown code reflecting the updated repository structure based on the image:

## Repository Structure

The repository is structured as follows:

```
GlobalDiracSync/
│
├── configurations/
│   ├── PaperResults.yml
│
├── MSF Analysis/
│   ├── StuartLandau_same_model/
│
├── explore_stability.py
├── plot_results.py
├── functions_MSF.py
├── functions_MSF_plot.py
├── functions_utility.py
├── requirements.txt
```
Each of these components serves a specific purpose:
```
- **`configurations/`**: Contains YAML files defining the parameter configurations used in the experiments and the 
  name of the folder where the results should be saved (`PaperResults.yml`).
- **`MSF Analysis/`**: A directory that includes subdirectories (e.g., `StuartLandau_same_model`) for organizing 
results and models.
- **`explore_stability.py`**: A script for exploring parameter stability using Ray Tune.
- **`plot_results.py`**: A script for plotting and analyzing the results based on configurations.
- **`functions_MSF.py`**: Contains utility functions used for Model Stability Function (MSF) computations.
- **`functions_MSF_plot.py`**: Includes plotting functions specific to MSF results.
- **`functions_utility.py`**: Additional utility functions for various operations.
- **`requirements.txt`**: Specifies the Python dependencies required for running the scripts in the repository. Install them using:

```

## Prerequisites

Install the requirements them using:
```bash
  pip install -r requirements.txt
```

## Usage

### 1. Exploring the Stability Space

The script `explore_stability.py` performs a grid search over specified parameter ranges using Ray Tune. The parameters being explored are `mu1im` and `mu0im`:

```python
from ray import tune
from ray.tune import grid_search as gs
from numpy import arange as ran
from functions_MSF import MSF_trial

# Model parameter space. These parameters are explored using grid_search.
parameter_space = {
    'mu1im': gs(ran(-2, 2, 0.1)),
    'mu0im': gs(ran(-3, 3, 0.1)),
}

meta_config = {
    'configuration_name': 'PaperResults.yml',
    'hyperparameters': parameter_space
}

tuner = tune.run(
    tune.with_resources(MSF_trial, resources={"cpu": 0.5}),
    verbose=1,
    num_samples=1,
    config=meta_config
)
```

- **`parameter_space`**: Defines the parameter ranges (`mu1im` and `mu0im`) to be explored.
- **`meta_config`**: Metadata configuration for the experiment, including the configuration file name and hyperparameters.
- **`tuner`**: Runs the parameter search using Ray Tune and the `MSF_trial` function defined in `functions_MSF.py`.
As soon as a certain score function is defined (such as the lenght of the MSF stability interval) and returned (see 
  `ray` docs) this can be adapted for parameters search in a larger space employing more advanced tools such as 
  HyperBand or ASHA (see `ray` docs).
### 2. Plotting the Results

The script `plot_results.py` loads the configuration and data generated from the stability exploration and visualizes it. It uses various filtering techniques and functions to produce the plots shown in the paper:

```python
from functions_MSF import *

limit_threads_to_N(14)

# Specify the configuration file
configuration_file = 'PaperResults.yml'

# Load the configuration file
test = LoadConfig(configuration_file)

# Load the dataframe using MSFDataHandler
handler = MSFDataHandler(LoadConfig(configuration_file))
df = handler.return_dataframe()

# Apply filters to the dataframe
df_alpha0_0 = df[df['alpha0'] == 0]
df_mu0im_range = df_alpha0_0[(df_alpha0_0['mu0im'] >= -2) & (df_alpha0_0['mu0im'] <= 2)]

# Plot the results based on filtered dataframe
plot_b_star_intervals(df_mu0im_range, start_with_0_01=True)
```

- **Configuration File**: The script reads from the configuration file (`PaperResults.yml`) to load experiment settings.
- **`MSFDataHandler`**: Handles data loading and filtering based on conditions set in the script. As can be seen it 
  can extract the csv files and make the dataset readily accessible.
- **Plotting Function**: The `plot_b_star_intervals` function visualizes the filtered results, showing the stability intervals specified in the paper.

### Configuration File

Ensure that the configuration file (`PaperResults.yml`) is properly set up with the necessary parameter ranges and other configurations required for your model. The configuration file should be structured in YAML format, specifying the parameters that `explore_stability.py` will explore.

## Customization

- To modify the parameter ranges or configuration, adjust the `parameter_space` dictionary and `meta_config` values in `explore_stability.py`.
- Update the filters in `plot_results.py` according to the specific criteria and data subsets you wish to visualize.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or issues, please contact the repository maintainer or open an issue on the repository page.
