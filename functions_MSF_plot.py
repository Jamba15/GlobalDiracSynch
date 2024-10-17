from joblib import Parallel, delayed
from functions_utility import *
import os
import matplotlib.pyplot as plt
import uuid
from matplotlib import colors as mcolors
from os.path import join
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm


class MSFDataHandler():
    def __init__(self, config: dict):
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.config = config
        # Check if the configuration dictionary contains the results folder path and the name of the experiment.
        # Raise error if not
        if 'results_folder' not in config['Global'].keys():
            raise ValueError('The configuration dictionary does not contain the results folder path (results_folder)')
        if 'experiment_name' not in config['Global'].keys():
            raise ValueError(
                'The configuration dictionary does not contain the name of the experiment (experiment_name)')

        self.results_folder = config['Global']['results_folder']
        self.experiment_name = config['Global']['experiment_name']
        self.results_path = os.path.join(current_path, self.results_folder, self.experiment_name)

        # Create the experiment folder if it does not exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.figures_path = os.path.join(self.results_path, 'Figures')

        # Create the figures folder if it does not exist
        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path)
        self.font_style = dict(fontsize=15, fontdict={'fontname': 'Serif'})

    def save_results_dictionary(self, results_dictionary: dict):
        """
        This function saves the results in a csv file in the experiment folder. The csv file contations the information in
        results_dictionary. It saves all the parameters in the configuration file (after hyperparameter substitution).
        :return: None
        """
        file_uuid = uuid.uuid4().hex
        # create a dictionary made by config[coupling_intensity] and results_dictionary
        results_dictionary = {**self.config['coupling_intensity'], **results_dictionary}
        results_df = pd.DataFrame([results_dictionary])
        results_df.to_csv(os.path.join(self.results_path, 'results_' + file_uuid + '.csv'), index=False)

    def remove_results(self, based_on: dict = None):
        """
        Removes files from the results folder based on filtering criteria or the entire folder if no criteria given.
        :param based_on: Dictionary with keys as DataFrame columns and values as conditions for filtering.
        :return: None
        """
        if not os.path.exists(self.results_path):
            print('No old results to remove')
            return

        if based_on is None:
            shutil.rmtree(self.results_path)
            print('Old results folder removed')
        else:
            files_to_remove = []
            for file in os.listdir(self.results_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(self.results_path, file)
                    results = pd.read_csv(file_path)
                    if any(not results.query(f'`{key}` {value}').empty for key, value in based_on.items()):
                        files_to_remove.append(file_path)

            for file_path in files_to_remove:
                os.remove(file_path)
                print(f'Removed file: {file_path}')

            if not os.listdir(self.results_path):
                shutil.rmtree(self.results_path)
                print('Old results folder removed')

    def return_dataframe(self):
        """This function returns a dataframe with the results of the experiment. It extracts al the csv files from the
        experiment folder and concatenates them in a single dataframe."""
        results = []
        for file in tqdm(os.listdir(self.results_path)):
            if file.endswith('.csv'):
                results.append(pd.read_csv(os.path.join(self.results_path, file)))

        return pd.concat(results)


# avoid single list elements in the dictionary
def no_single_list_elements(dictionary: dict):
    for key, value in dictionary.items():
        if isinstance(value, list) and len(value) == 1:
            dictionary[key] = value[0]
    return dictionary


def plot_b_star_intervals(df, start_with_0_01=False):
    """
    Plots a square with 'mu0im' on the X-axis and 'mu1im' on the Y-axis and places a dot at the corresponding
    location whose color depends on the size of the interval of 'b_star'. If 'b_star' has
    multiple intervals (non-constant step), it plots a black square. If 'b_star' is NaN,
    it plots a red dot. Intervals not starting with 0.01 are treated as unstable and plotted
    in orange with the label 'Unstable (around 0)'.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'mu1im', 'mu0im', and 'b_star' columns.
    start_with_0_01 (bool): If True, treats intervals not starting with 0.01 as unstable.
    """
    # Set font to modern serif (e.g., DejaVu Serif)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 13  # Increase font size to 14 points

    # Lists to store data for plotting
    x_constant = []
    y_constant = []
    interval_sizes = []

    x_not_constant = []
    y_not_constant = []

    x_nan = []
    y_nan = []

    x_unstable_around_0 = []
    y_unstable_around_0 = []

    for idx, row in df.iterrows():
        mu1im = row['mu1im']
        mu0im = row['mu0im']
        b_star_value = row['b_star']

        if pd.isna(b_star_value):
            x_nan.append(mu0im)  # Invert axes: mu0im on X-axis
            y_nan.append(mu1im)  # mu1im on Y-axis
            continue

        # Parse 'b_star' into a numpy array
        b_star_arr = parse_b_star(b_star_value)

        if len(b_star_arr) == 0:
            x_nan.append(mu0im)  # Invert axes: mu0im on X-axis
            y_nan.append(mu1im)  # mu1im on Y-axis
            continue

        # Check if the interval starts with 0.01 when the flag is set
        if start_with_0_01 and not np.isclose(b_star_arr[0], 0.01, atol=1e-6):
            # Treat as unstable (around 0), add to the corresponding lists
            x_unstable_around_0.append(mu0im)  # Invert axes: mu0im on X-axis
            y_unstable_around_0.append(mu1im)  # mu1im on Y-axis
            continue

        # Check if steps are constant
        is_constant_step = check_constant_step(b_star_arr)

        if not is_constant_step:
            x_not_constant.append(mu0im)  # Invert axes: mu0im on X-axis
            y_not_constant.append(mu1im)  # mu1im on Y-axis
        else:
            x_constant.append(mu0im)  # Invert axes: mu0im on X-axis
            y_constant.append(mu1im)  # mu1im on Y-axis
            interval_size = np.max(b_star_arr) - np.min(b_star_arr)
            interval_sizes.append(interval_size)

    plt.figure()
    ax = plt.gca()

    # Plot points where b_star is NaN in red
    if x_nan:
        plt.scatter(x_nan, y_nan, marker='s', color='red', s=50, label='Unstable')

    # Plot points where interval does not start at 0.01 in orange
    if x_unstable_around_0:
        plt.scatter(x_unstable_around_0, y_unstable_around_0, marker='s', color='red', s=50)

    # Plot points with non-constant steps as black squares
    if x_not_constant:
        plt.scatter(x_not_constant, y_not_constant, marker='s', color='black', s=100,
                    label='Stable Multiple Regions')

    # Plot points with constant steps, colored by interval size
    if x_constant:
        # Normalize interval sizes for colormap
        min_size = min(interval_sizes)
        max_size = max(interval_sizes)
        norm = mcolors.Normalize(vmin=min_size, vmax=max_size)
        cmap = plt.cm.viridis
        scatter = plt.scatter(x_constant, y_constant, marker='s', c=interval_sizes, cmap=cmap, s=50,
                              label='Stable', norm=norm)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Interval Size')

    plt.xlabel('$\mu_\Im^{(0)}$')  # Inverted X-axis label
    plt.ylabel('$\mu_\Im^{(1)}$')  # Inverted Y-axis label
    # plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def parse_b_star(b_star_value):
    """
    Parses the 'b_star' value into a numpy array.

    Parameters:
    b_star_value: Could be a string representation of an array, a single float, or a list/array.

    Returns:
    numpy.ndarray: Parsed array of 'b_star' values.
    """
    if pd.isna(b_star_value):
        return np.array([])
    elif isinstance(b_star_value, str):
        b_star_str = b_star_value.replace('[', '').replace(']', '').replace('\n', ' ')
        b_star_elements = b_star_str.strip().split()
        if b_star_elements:
            b_star_arr = np.array([float(val) for val in b_star_elements])
        else:
            b_star_arr = np.array([])
    elif isinstance(b_star_value, (float, int)):
        b_star_arr = np.array([b_star_value])
    elif isinstance(b_star_value, (list, np.ndarray)):
        b_star_arr = np.array(b_star_value)
    else:
        b_star_arr = np.array([])
    return b_star_arr


def check_constant_step(arr):
    """
    Checks if the steps in the array are constant.

    Parameters:
    arr (numpy.ndarray): Array of 'b_star' values.

    Returns:
    bool: True if steps are constant, False otherwise.
    """
    if len(arr) < 2:
        return True  # Assume constant if less than 2 elements
    diffs = np.diff(arr)
    return np.allclose(diffs, diffs[0], atol=1e-6)
def MSF_SL(config: dict, plot=False):
    params = no_single_list_elements(config['coupling_intensity'])

    # Calcolo di sigma1im se è None
    if params['sigma1im'] is None:
        params['sigma1im'] = params['sigma0im'] + params['beta1im'] * params['sigma1re'] / params['beta1re'] - params['beta0im'] * params['sigma0re'] / params['beta0re']

    # Calcolo di gamma
    gamma = np.sqrt((params['sigma1re'] * params['beta0re']) / (params['beta1re'] * params['sigma0re']))

    # Definire le matrici J^(a) e M^(a)
    def J_a(sigma_re, beta_im, beta_re):
        return np.array([
            [-2 * sigma_re, 0],
            [-2 * beta_im * sigma_re / beta_re, 0]
        ])

    def M_a(mu_re, mu_im):
        return np.array([
            [-mu_re, mu_im],
            [-mu_im, -mu_re]
        ])

    # Calcolo delle matrici per a=0
    J_0 = J_a(params['sigma0re'], params['beta0im'], params['beta0re'])
    M_0 = M_a(params['mu0re'], params['mu0im'])

    # Calcolo delle matrici per a=1
    J_1 = J_a(params['sigma1re'], params['beta1im'], params['beta1re'])
    M_1 = M_a(params['mu1re'], params['mu1im'])

    # Definire i parametri aggiuntivi
    alpha_0 = params['a0']
    alpha_1 = params['a1']

    def compute_max_real_eigenvalue(b_q):
        # Esegui le operazioni matematiche prima di usare np.hstack e np.vstack
        top_left = J_0 + b_q ** 2 * alpha_0 * M_0
        top_right = (1 - alpha_0) * gamma * b_q * M_0
        bottom_left = (1 - alpha_1) * b_q * M_1 / gamma
        bottom_right = J_1 + b_q ** 2 * alpha_1 * M_1

        # Definire la matrice J_q usando np.hstack e np.vstack
        J_q = np.vstack([
            np.hstack([top_left, top_right]),
            np.hstack([bottom_left, bottom_right])
        ])

        # Calcola il massimo della parte reale degli autovalori di J_q
        real_eigenvalues = np.real(np.linalg.eigvals(J_q))
        return b_q, np.max(real_eigenvalues)

    # Parallelizzare il ciclo su b_q
    b_q_values = np.arange(0, 20.01, 0.1)
    results = Parallel(n_jobs=-1)(delayed(compute_max_real_eigenvalue)(b_q) for b_q in b_q_values)

    b_star_values = []
    corresponding_negative_eigenvalues = []

    for b_q, max_real_eigen in results:
        if max_real_eigen < 0:
            b_star_values.append(b_q)
            corresponding_negative_eigenvalues.append(max_real_eigen)

    if b_star_values:
        b_star = b_star_values
    else:
        b_star = None
        corresponding_negative_eigenvalues = None

    if plot:
        b_q_array = np.array([result[0] for result in results])
        max_real_eigen_array = np.array([result[1] for result in results])

        plt.figure(figsize=(10, 6))
        plt.plot(b_q_array, max_real_eigen_array, label='Max Real Part of Eigenvalue')
        plt.axhline(0, color='red', linestyle='--', label='Zero Line')
        plt.xlabel('b_q')
        plt.ylabel('Max Real Part of Eigenvalue')
        plt.title('Max Real Part of Eigenvalue vs b_q')
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        'b_star': np.array(b_star),
        'corresponding_negative_eigenvalues': np.array(corresponding_negative_eigenvalues)
    }


if __name__ == '__main__':
    from ray import tune
    from ray.tune import grid_search as gs
    from numpy import arange as ran

    config = LoadConfig('teo_sim_toro.yml')
    MSF_SL(config, plot=True)
