import fnmatch
import os
from os.path import join, dirname, abspath
import yaml


class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

    # create constructor for !path tag
    def construct_path(self, node):
        return os.path.normpath(self.construct_scalar(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)


PrettySafeLoader.add_constructor(
    '!path',
    PrettySafeLoader.construct_path)


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def DictionaryReplace(dictionary: dict, sobstitute: dict) -> dict:
    """
    Sostituisce le chiavi di un dizionario con i valori di un altro dizionario. La sostituizione avviene in profondità
    e solo se la chiave è uguale
    :param dictionary: Il primo dizionario in cui la sostituzione deve avvenire in accordo alle chiavi di sobstitute
    :param sobstitute: Dizionario con le chiavi da sostituire
    :return: Il dizionario con i valori sostituiti in accordo alle chiavi di sobstitute
    """
    found = False
    for key, value in sobstitute.items():
        for original_key, original_value in dictionary.items():
            if isinstance(original_value, dict):
                dictionary[original_key] = DictionaryReplace(original_value, sobstitute)
            elif original_key == key:
                dictionary[original_key] = value
                found = True
    return dictionary


def LoadConfig(config_name, hyper=None):
    """
    This function loads the configuration file from the config folder. If hyper is not None, it will replace the
    hyperparameters in the configuration file with the ones in hyper
    :param config_name: The name of the configuration file
    :param hyper: The dictionary with the hyperparameters to replace in the configuration file
    :return: The configuration dictionary
    """
    # Check if hyper is a dictionary
    if hyper is not None:
        if not isinstance(hyper, dict):
            raise TypeError("Hyperparameters must be a dictionary")
        # Ensure that hyper values are lists, if not, convert them to lists
        for key, value in hyper.items():
            if not isinstance(value, list):
                hyper[key] = [value]

    configuration_file = find(config_name, join(dirname(abspath(__file__))))[0]
    with open(configuration_file, 'r') as c:
        configuration = yaml.load(c, Loader=PrettySafeLoader)

    if hyper is not None:
        new_hyper = {}
        for key, values in hyper.items():
            new_hyper[key] = values[0] if len(values) > 1 else values
        return DictionaryReplace(configuration, new_hyper)
    else:
        return configuration


def old_trials_remover(trial_info):
    # If the folder already exists and remove_old is True then remove all the files
    config = LoadConfig(trial_info['configuration_name'])
    if os.path.exists(config["save_trajectory"]["trajectory_folder"]) and config["save_trajectory"]["remove_old"]:
        folder = config["save_trajectory"]["trajectory_folder"]
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            os.remove(file)



