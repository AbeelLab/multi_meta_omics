import collections
import os
import re

import numpy as np
import pandas as pd

def prepare_data(data_file, separator="\t", model="rf"):
    df = pd.read_csv(data_file, sep=separator, index_col="ID")

    # This has to be done because R dataframes have weird column names
    df.columns = [x[1:] if x[0] == "X" else x for x in df.columns]
    df.columns = [re.sub('[^a-zA-Z0-9\n\.\_]', '.', x) for x in df.columns]
    df.sort_index(axis=1, inplace=True)

    # Make sure samples are in the correct order (we will deal with features later in the pre-processing)
    df.sort_index(inplace=True)

    return df

# Scale data for BiomeNED
def scale_and_center(a):
    # Apply centering and scaling as done in the original method
    mu = (a.max(axis=0) + a.min(axis=0))/2.0
    a = a - mu
    std = (a.max(axis=0) - a.min(axis=0))/2.0

    mask = std != 0.0
    a /= np.where(mask, std, 1)

    return a

def compute_spearman_top_k(real, predictions, k=50):
    # Take common columns between ground-truth features and predictions
    # We do this because some models filter out features that are not well-predicted
    common_features = list(set(predictions.columns).intersection(set(real.columns)))
    print("Found", len(common_features), "common features")
    
    #assert len(common_features) == len(predictions.columns)
    
    # Compute Spearman correlations between ground truth and predictions
    correlations = real[common_features].corrwith(predictions[common_features],
                                                  method='spearman')
    correlations = correlations.to_dict()
    
    # Make Counter object 
    correlations = collections.Counter({x: count for x, count in correlations.items() 
                                        if (count > 0 and not (np.isnan(count)))})
    
    # Average correlation for the k best predicted features, all non-NaN correlations
    return np.array([x[1] for x in correlations.most_common(k)]).mean(), correlations.most_common()
    

def evaluate_top_features(omics_to_omics_model: str,
                          model_input: str,
                          model_output: str,
                          seeds: list[int],
                          dp: str,
                          config,
                          k=0.25,
                          latent=False):
    # For each feature, store list of correlations
    feature_correlations = dict()

    if omics_to_omics_model == "mimenet":
        separator = ","
    else:
        separator = "\t"
    
    # Iterate over seeds
    for seed in seeds:
        # Load real data
        true_dir = config.input_data_root + dp + "/seed_" + str(seed)

        if omics_to_omics_model != "biomened":
            ground_truth_file = "%s/%s/test/to_%s/from_%s_output.txt" % (true_dir,
                                                                         omics_to_omics_model,
                                                                         model_output, 
                                                                         model_input)
            real = prepare_data(ground_truth_file,
                                separator = separator,
                                model=omics_to_omics_model)
            if omics_to_omics_model == "mimenet":
                real = real.transpose()
        else:
            if dp == "arcsin" or dp == "quantile":
                ground_truth_file = "%s/%s/test/to_%s/from_%s_output.txt" % (true_dir,
                                                                             "mimenet",
                                                                             model_output, 
                                                                             model_input)
                real = prepare_data(ground_truth_file,
                                    separator = ",",
                                    model="mimenet")
            else:
                ground_truth_file = "%s/%s/test/to_%s/from_%s_output.txt" % (true_dir,
                                                                             "melonnpan",
                                                                             model_output, 
                                                                             model_input)
                real = prepare_data(ground_truth_file,
                                    model="melonnpan")                
        
        
        # Load predicted data
        pred_dir = config.output_data_root + dp + "/seed_" + str(seed)
        pred_dir += "/" + omics_to_omics_model
        if not latent:
            pred_dir += "/output_" + model_input + "_to_" + model_output
        else:
            pred_dir += "/output_" + model_input + "_to_" + model_output + "_latent"
        predictions_file = pred_dir + "/test_predictions.txt"

        if not os.path.exists(predictions_file):
            print("Failed experiment:", predictions_file)
            continue
        
        predictions = prepare_data(predictions_file)

        if omics_to_omics_model == "biomened":
            predictions_np = scale_and_center(predictions.to_numpy())
            predictions = pd.DataFrame(predictions_np,
                                       index=predictions.index,
                                       columns = predictions.columns)

        _, correlations = compute_spearman_top_k(real,
                                                 predictions)

        for feature, corr in correlations:
            if feature not in feature_correlations:
                feature_correlations[feature] = [corr]
            else:
                 feature_correlations[feature].append(corr)

    means = {feature: np.mean(feature_correlations[feature])
             for feature in feature_correlations}
    stds = {feature: np.std(feature_correlations[feature])
             for feature in feature_correlations}

    # Compute top features based on mean correlation across seeds
    if k < 1:
        keep = int(k * len(means))
    else:
        keep = int(k)
        
    correlations = collections.Counter(means).most_common(keep)

    mean = np.mean([x[1] for x in correlations])
    features = [x[0] for x in correlations]
    std = np.mean([stds[feature] for feature in features])
    
    # Return mean and standard deviation of feature correlations for the top k
    return mean, std, features
