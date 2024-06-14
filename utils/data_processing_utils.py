import numpy as np
import pandas as pd
import pickle as pkl

from random import sample
from sklearn.model_selection import StratifiedShuffleSplit

def normalize_per_sample(
        df: pd.DataFrame,
        id_column: str):
    """
    Normalize a dataframe (column-wise)
    and identify empty samples.

    Keyword arguments:
    df -- dataframe with samples as columns
    id_column -- name of column to exclude

    Return: normalized dataframe and indices of
    empty samples
    """
    invalid_samples = []    
    for column in df.columns:
        # Sanity check: exclude the ID column
        if (not column == id_column):
            # Second condition to account for pseudocounts
            if df[column].sum() == 0 \
               or len(set(df[column])) == 1:
                print("Found empty samples")
                invalid_samples.append(column)             
            df[column] = df[column] / df[column].sum()
            
    return df, invalid_samples

def filter_out_low_abundance_features(
        df: pd.DataFrame,
        high_abundance_threshold: float,
        low_abundance_threshold: float,
        data_type: str,
        samples_threshold: float,
        zeros_threshold: float=0.95):
    """
    Filter out low-abundance features (columns)
    from a dataframe. Only keep features that are
    more than high_abundance_threshold in at least
    samples_threshold of samples. Discard features less
    than low_abundance_threshold in more than samples_threshold
    of samples. Additionally discard features with too many zeros.

    Keyword arguments:
    df -- dataframe with features as columns
    high_abundance_threshold -- threshold to consider samples highly abundant
    low_abundance_threshold -- threshold to consider samples lowly abundant
    data_type -- either mGx, mGx_pa, mGx_taxa, mTx, mPx, mBx or mixed
    samples_threshold -- fraction of samples to consider for abundance filtering
    zeros_threshold -- maximum allowed fraction of zeros for a feature

    Return: list of names of columns with low abundance
    """
    to_drop = set()
    num_samples = len(df)
    threshold = int(num_samples * samples_threshold)
    temp = low_abundance_threshold

    # Exclude these data types from low abundance filtering
    if data_type in {"mPx", "mGx_taxa"}:
        low_abundance_threshold = -1
    
    for feature in df.columns:
        if data_type == "mixed":
            if "mGx_taxa" in feature or "mPx" in feature:
                low_abundance_threshold = -1
            else:
                low_abundance_threshold = temp
            
        num_high_abundance_samples = (df[feature] > high_abundance_threshold)\
            .astype(int).sum()
        num_low_abundance_samples = (df[feature] < low_abundance_threshold)\
            .astype(int).sum()
        
        # Drop everything that is not highly abundant enough
        # or has very low abundance across samples
        if (num_high_abundance_samples < threshold):
            to_drop.add(feature)
        if low_abundance_threshold > -1:
            if (num_low_abundance_samples > threshold):
                to_drop.add(feature)

        # Drop features with too many zeros
        num_zeros = (df[feature] < 1e-9).astype(int).sum()
        if num_zeros > (num_samples * zeros_threshold):
            to_drop.add(feature)
            
    return to_drop

def map_to_ibd_diagnosis(
        x: int):
    """
    Map string to a number (class).
    Labelling: HC = 0, UC = 1, CD = 2

    Keyword arguments:
    x -- string to map

    Return: corresponding integer
    """
    if x == "Control" or x == "nonIBD": return 0
    if x == "UC": return 1
    return 2

# Helper function for separate_samples
def filter_participants(
        sample_names: list[str],
        participant_samples: dict()):
    # First, get rid of participants with no samples
    participant_subset = set()
    for participant in participant_samples.keys():
        has_samples = False
        
        for sample in participant_samples[participant]:
            if sample in sample_names:
                has_samples = True
                break

        if has_samples: participant_subset.add(participant)
    
    return {participant: participant_samples[participant]
            for participant in participant_subset}

# Helper function for separate_samples
def stratified_split(
        participant_samples_filtered: list[str],
        random_state: int,
        classes: dict(),
        test_size: float=0.2):
    # This is sorted to ensure reproducibility for one random seed
    participants = sorted(list(participant_samples_filtered.keys()))
    # Determine participants' classes
    y = []
    for participant in participants:
        # Just take the class of the first sample (they should all be the same)
        diagnosis = classes[participant_samples_filtered[participant][0]]
        y.append(diagnosis)

    split_object = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(split_object.split(participants, y))
    
    participants = np.array(participants)
    
    return participants[train_idx], participants[test_idx]


def separate_samples(
        sample_names: list[str],
        participant_samples: dict(),
        random_state: int,
        classes: dict,
        test_size: float=0.2):
    """
    Separate samples into training and testing,
    according to which study participant they come from.

    Keyword arguments:
    sample names -- list of sample names
    participant_samples -- dict of the form participant_id -> [list of samples]
    random_state -- seed to use for splitting
    classes -- dict of the form sample -> class
    test_size -- fraction of samples used for testing

    Return: corresponding integer
    """    
    participant_samples_filtered = filter_participants(sample_names,
                                                       participant_samples)
    
    train, test = stratified_split(participant_samples_filtered,
                                   random_state,
                                   classes,
                                   test_size)

    training_samples = set(np.concatenate([participant_samples_filtered[participant] 
                                           for participant in train]).ravel())
    training_samples = list(training_samples.intersection(sample_names))

    test_samples = set(np.concatenate([participant_samples_filtered[participant] 
                                       for participant in test]).ravel())
    test_samples = list(test_samples.intersection(sample_names))
    
    if len(test_samples) == 0:
        raise Exception("Empty test set.")

    return sorted(training_samples), sorted(test_samples)

# Helper function for build_dataset_dict
def to_dict(
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        classes: dict(),
        train_idx: list[int],
        val_idx: list[int]):
    samples = input_df.index.values.tolist()    
    diagnosis = [classes[sample] for sample in samples]
    
    return {'subject_ids': samples, 
            'met_ids':  output_df.columns.tolist(), 
            'met_group_ids': output_df.columns.tolist(), 
            'met_fea': output_df.to_numpy(), 
            'met_group_fea': output_df.to_numpy(), 
            'bac_ids': input_df.columns.tolist(), 
            'bac_group_ids': input_df.columns.tolist(), 
            'bac_fea': input_df.to_numpy(), 
            'bac_group_fea': input_df.to_numpy(), 
            'train_idx': train_idx, 
            'val_idx': val_idx, 
            'diagnosis': diagnosis}

def build_dataset_dict(
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        dataset_name: str,
        classes: dict(),
        train_idx: list[int],
        val_idx: list[int]):
    """
    Create .pkl file for SparseNED input.

    Keyword arguments:
    input_df -- input data
    output_df -- output data
    dataset_name -- name to save the file
    classes -- dict of the form sample -> class
    train_idx -- indices of training samples
    val_idx -- indices of validation samples

    Return: corresponding integer
    """
    assert (not input_df.isnull().values.any())
    assert (not output_df.isnull().values.any())
    
    assert (not input_df.isna().values.any())
    assert (not output_df.isna().values.any())
    
    input_df.sort_index(inplace=True)
    output_df.sort_index(inplace=True)
 
    # Check that the samples are the same and ordered
    assert input_df.index.values.tolist() == output_df.index.values.tolist()
    
    dataset_dict = to_dict(input_df,
                           output_df,
                           classes,
                           train_idx,
                           val_idx)
    
    with open(dataset_name + ".pkl", 'wb') as f:
        pkl.dump(dataset_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
