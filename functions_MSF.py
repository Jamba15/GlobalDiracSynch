from functions_MSF_plot import *
import numpy as np
from joblib import Parallel, delayed
import os

def MSF_trial(meta_config: dict):
    config = LoadConfig(meta_config['configuration_name'],
                        hyper=meta_config.get('hyperparameters'))

    res_dict = MSF_SL(config)
    handler = MSFDataHandler(config)
    handler.save_results_dictionary(res_dict)

def limit_threads_to_N(N: int):
    """
    Limit the CPU threads to a specified number (N).

    Parameters:
    N (int): The number of logical cores to which the CPU affinity will be limited.

    Description:
    This function modifies the CPU affinity, limiting the number of logical cores
    the current process can run on. It first retrieves the list of available logical
    cores, then limits the affinity to the first N cores in that list.
    """
    # Retrieves the list of available logical CPU cores
    available_cpus = os.sched_getaffinity(0)

    # Limit the CPU affinity to the first N logical cores (or fewer if there are less than N)
    limited_cpus = list(available_cpus)[:N]

    # Set the CPU affinity to only the first N logical cores
    os.sched_setaffinity(0, limited_cpus)

    # Print a message indicating the cores the affinity has been limited to
    print(f"CPU affinity has been limited to these logical cores: {limited_cpus}")


def MSF_trial(meta_config: dict):
    """
    Run an MSF (Multi-Stage Framework) trial based on a given meta-configuration.

    Parameters:
    meta_config (dict): A dictionary containing configuration details for the MSF trial.

    Description:
    This function takes a dictionary `meta_config` with the necessary configuration
    information, loads the configuration using the `LoadConfig` function, runs an MSF
    trial through `MSF_SL`, and finally saves the results using `MSFDataHandler`.
    """
    # Load configuration based on the provided configuration name and hyperparameters
    config = LoadConfig(meta_config['configuration_name'],
                        hyper=meta_config.get('hyperparameters'))

    # Run the MSF trial and retrieve the results dictionary
    res_dict = MSF_SL(config)

    # Create an MSF data handler and save the results dictionary
    handler = MSFDataHandler(config)
    handler.save_results_dictionary(res_dict)


def no_single_list_elements(dictionary: dict):
    """
    Remove single-item lists from a dictionary by replacing them with the single element.

    Parameters:
    dictionary (dict): A dictionary where some values may be lists with a single element.

    Returns:
    dict: A dictionary where single-item lists are replaced by their single element.

    Description:
    This function iterates through the dictionary and checks if any values are lists
    containing a single item. If so, it replaces the list with the single element itself.
    """
    for key, value in dictionary.items():
        # If the value is a list with only one element, replace the list with that element
        if isinstance(value, list) and len(value) == 1:
            dictionary[key] = value[0]
    return dictionary

def MSF_SL(config: dict, plot=False, add_B=False, **kwargs):
    """
    Perform Multi-Stage Framework Stability Level (MSF_SL) calculations.

    Parameters:
    config (dict): Configuration dictionary containing system parameters.
    plot (bool): Whether to plot the results of the maximum real part of eigenvalues vs. b_q.
    add_B (bool): If True, also plot the MSF values of the singular values of matrix B (if provided in kwargs).
    **kwargs: Additional keyword arguments that can include matrix B.

    Returns:
    dict: A dictionary containing:
        'b_star' (array): The b_q values where the maximum real part of eigenvalues is negative.
        'corresponding_negative_eigenvalues' (array): The corresponding negative eigenvalues at those b_star values.
    """

    # Remove single-item lists from the configuration parameters
    params = no_single_list_elements(config['coupling_intensity'])

    # Calculate sigma1im if it's None
    if params['sigma1im'] is None:
        params['sigma1im'] = (params['sigma0im'] + params['beta1im'] * params['sigma1re'] / params['beta1re']
                              - params['beta0im'] * params['sigma0re'] / params['beta0re'])

    # Calculate gamma
    gamma = np.sqrt((params['sigma1re'] * params['beta0re']) / (params['beta1re'] * params['sigma0re']))

    # Define the J^(a) matrix
    def J_a(sigma_re, beta_im, beta_re):
        return np.array([
            [-2 * sigma_re, 0],
            [-2 * beta_im * sigma_re / beta_re, 0]
        ])

    # Define the M^(a) matrix
    def M_a(mu_re, mu_im):
        return np.array([
            [-mu_re, mu_im],
            [-mu_im, -mu_re]
        ])

    # Calculate the J and M matrices for a=0
    J_0 = J_a(params['sigma0re'], params['beta0im'], params['beta0re'])
    M_0 = M_a(params['mu0re'], params['mu0im'])

    # Calculate the J and M matrices for a=1
    J_1 = J_a(params['sigma1re'], params['beta1im'], params['beta1re'])
    M_1 = M_a(params['mu1re'], params['mu1im'])

    # Define additional parameters
    alpha_0 = params['alpha0']
    alpha_1 = params['alpha1']

    # Function to compute the maximum real part of eigenvalues
    def compute_max_real_eigenvalue(b_q):
        # Perform mathematical operations to create submatrices for the J_q matrix
        top_left = J_0 + b_q ** 2 * alpha_0 * M_0
        top_right = (1 - alpha_0) * gamma * b_q * M_0
        bottom_left = (1 - alpha_1) * b_q * M_1 / gamma
        bottom_right = J_1 + b_q ** 2 * alpha_1 * M_1

        # Define the J_q matrix by combining the submatrices
        J_q = np.vstack([
            np.hstack([top_left, top_right]),
            np.hstack([bottom_left, bottom_right])
        ])

        # Calculate the maximum real part of the eigenvalues of J_q
        real_eigenvalues = np.real(np.linalg.eigvals(J_q))
        return b_q, np.max(real_eigenvalues)

    # Parallelize the loop over b_q values
    b_q_values = np.arange(0, 5.01, 0.01)
    results = Parallel(n_jobs=-1)(delayed(compute_max_real_eigenvalue)(b_q) for b_q in b_q_values)

    # Initialize lists to store b_star values and corresponding negative eigenvalues
    b_star_values = []
    corresponding_negative_eigenvalues = []

    # Collect b_q values where the maximum real eigenvalue is negative
    for b_q, max_real_eigen in results:
        if max_real_eigen < 0:
            b_star_values.append(b_q)
            corresponding_negative_eigenvalues.append(max_real_eigen)

    # Set the final b_star and eigenvalues arrays
    if b_star_values:
        b_star = b_star_values
    else:
        b_star = None
        corresponding_negative_eigenvalues = None

    # Plot results if the plot flag is True
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

        # If add_B is True and matrix B is provided, plot the MSF values for the singular values of B
        if add_B and 'B' in kwargs:
            B = kwargs['B']
            singular_values = np.sqrt(np.linalg.svd(B, compute_uv=False))
            msf_values = [compute_max_real_eigenvalue(sv)[1] for sv in singular_values]
            plt.plot(singular_values, msf_values, 'bo', label='MSF Values of Singular Values of B')

        plt.show()

    # Return the results as a dictionary
    return {
        'b_star': np.array(b_star),
        'corresponding_negative_eigenvalues': np.array(corresponding_negative_eigenvalues)
    }

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

if __name__ == '__main__':
    limit_threads_to_N(14)
    # from ray import tune
    # from ray.tune import grid_search as gs
    # from numpy import arange as ran
    #
    # parameter_space = {
    #                    'mu1im': gs(ran(-2, 2, 0.1)),
    #                    'mu0im': gs(ran(-3, 3, 0.1)),
    #                    }
    #
    # meta_config = {'configuration_name': 'PaperResults.yml',
    #                'hyperparameters': parameter_space}
    #
    # tuner = tune.run(
    #     tune.with_resources(MSF_trial,
    #                         resources={"cpu": 0.5}),
    #     verbose=1,
    #     num_samples=1,
    #     config=meta_config
    # )

    test = LoadConfig('PaperResults.yml')

    handler = MSFDataHandler(LoadConfig('PaperResults.yml'))
    df = handler.return_dataframe()
    # extract the sub dataframe wiht b_star that is not none

    # Funzione per convertire la stringa in array numpy
    # def string_to_array(s):
    #     return np.fromstring(s.strip('[]\n'), sep=' ')


    # Convertire le stringhe in array numpy
    # df['b_star'] = df['b_star'].apply(string_to_array)

    # %%
    df_alpha0_0 = df[df['alpha0'] == 0]
    df_mu0im_range = df_alpha0_0[(df_alpha0_0['mu0im'] >= -2) & (df_alpha0_0['mu0im'] <= 2)]

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import pandas as pd


    plot_b_star_intervals(df_mu0im_range, start_with_0_01=True)

