"""

Utility functions to load the desired experiment using allensdk and structure the imaging data to
(trials x features) dataset and (x trials) labels

"""

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import numpy as np

def load_cache(cache_dir="cache_dir"):
    """
    Initialize and return a cache directory for the project.
    :param cache_dir: desired cache directory to save data to
    :return: cache
    """
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    return cache

def get_ophys_table(cache):
    """
    Get the ophys table from the cache
    :param cache: returned by load_cache
    :return: pandas dataframe
    """
    return cache.get_ophys_experiment_table(as_df=True)

def get_ophys_table_with_cellcount(cache):
    """
    Get the ophys table from the cache and join the table with cell count from cell table
    :param cache: returned by load_cache
    :return:  pandas dataframe
    """
    ophys_table = cache.get_ophys_experiment_table(as_df=True)
    cell_table = cache.get_ophys_cells_table()
    cell_per_exp = cell_table.groupby(['ophys_experiment_id']).size().rename('cell_count')
    return ophys_table.join(cell_per_exp, how='left')

def filter_ophys_experiments(ophys_table, passive: bool=True, min_cell_count=100):
    """
    select only experiment with passive/active and cell count >= min_cell_count
    :param passive: (string) experiments in which animal only sees a visual stimuli without any behavioral task is passive
    :param min_cell_count: (integer) number of cells recorded in the visual task
    :return: filtered experiments as a pandas dataframe
    """
    return ophys_table[(ophys_table["passive"] == passive) & (ophys_table["cell_count"] >= min_cell_count)]

def load_experiment(cache, ophys_experiment_id=0):
    """
    Load an experiment from the ophys_table
    :param cache: returned by load_cache
    :param ophys_experiment_id: id of the ophys experiment in the ophys_experiment_table
    :return: download selected experiment if not already exist in the cache directory and return it
    """
    return cache.get_behavior_ophys_experiment(ophys_experiment_id=ophys_experiment_id)

def build_trial_dataset(experiment):
    """
    Extract only the timestamps where the experiment received a change in stimuli, not spontaneous or rest
    :param experiment: the downloaded ophys experiment from load_experiment

    :return:

    X_trials: numpy dataset of dffs with shape (trials x features)
    y: numpy array of labels with shape (x trials)
    """
    sp = experiment.stimulus_presentations
    sp = sp[sp['stimulus_block_name'].str.contains('change_detection')]
    dff = experiment.dff_traces
    time_stamps = np.array(experiment.ophys_timestamps)
    if 'dff' in dff.columns:
        dff_data = np.vstack(dff['dff'].values)
    else:
        raise RuntimeError("Error in extracting dff")
    trials, labels = [], []

    for _, trial in sp.iterrows():
        start, stop = trial['start_time'], trial['end_time']
        idx = np.where((time_stamps >= start) & (time_stamps <= stop))[0]
        if idx.size == 0:
            continue
        # average dff across time
        response = dff_data[:, idx].mean(axis=1)
        trials.append(response)
        labels.append(trial['image_name'])

    X_trials = np.vstack(trials)  # shape: (n_trials, n_cells)
    y = np.array(labels)

    return X_trials, y




