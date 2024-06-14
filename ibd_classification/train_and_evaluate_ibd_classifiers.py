import argparse
import collections
import os

import config
import pandas as pd
import numpy as np
import pickle as pkl

from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from utils import data_processing_utils
from utils import eval_utils

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--training_data", help="Path to file used for training.")
parser.add_argument("--test_data", help="Path to file used for testing.")
parser.add_argument("--out_dir", help="Directory to save results.")
parser.add_argument("--classes", help="Path to pickle file containing sample-to-class mapping.")
parser.add_argument("--participant_samples", help="Path to pickle file containing participant-to-samples mapping.")

parser.add_argument("--ibd_vs_noibd", help="Do binary classification of IBD vs. no IBD.", action="store_true")
parser.add_argument("--cd_vs_noibd", help="Do binary classification of CD vs. no IBD.", action="store_true")


parser.add_argument("-d", "--downsample", action="store_true",
                    help="If used, the training data will be downsampled for equal class proportions.")
parser.add_argument("-p", "--pca", action="store_true",
                    help="If used, PCA is applied to the input data.") 
parser.add_argument("-pv", "--pca_var", default=0.95,
                    help="Proportion of variance to keep when applying PCA.")
parser.add_argument("-i", "--num_iter", default=1, type=int,
                    help="Number of random search iterations for fine-tuning the IBD classifiers.")

parser.add_argument("-s", "--seed", type=int,
                    help="Random seed.")

args = vars(parser.parse_args())

pca_var = args['pca_var']
seed = args["seed"]

def make_splits(df, classes, participants_samples):
    cv_splits = []
    
    for i in range(5):
        samples = df.index.values.tolist()
        train_idx, val_idx = data_processing_utils.separate_samples(samples, 
                                                                    participant_samples, 
                                                                    test_size=0.2,
                                                                    classes=classes, 
                                                                    random_state=i)
        train_idx = [samples.index(x) for x in train_idx]
        val_idx = [samples.index(x) for x in val_idx]

        cv_splits.append((train_idx, val_idx))

    return cv_splits

def transform_and_train(classifier, param_grid, df, y, cv_splits, save_dir):
    search = RandomizedSearchCV(estimator=classifier(),
                                cv=cv_splits,
                                param_distributions=param_grid,
                                n_jobs = 8,
                                random_state=seed,
                                n_iter = args['num_iter'])

    # Apply centering and scaling
    # For tree-based classifiers this shouldn't make a difference.
    # But it can be useful for other classifiers.
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # PCA
    pca = None
    if args['pca']:
        pca = PCA(n_components=pca_var)
        scaled_data = pca.fit_transform(scaled_data)

    search.fit(scaled_data, y)

    os.makedirs(save_dir, exist_ok=True)
        
    dump(search, save_dir + "search.joblib")
    dump(scaler, save_dir + "scaler.joblib")

    # Save PCA object
    if args['pca']: dump(pca, save_dir + "pca.joblib")

    print("Saved model:", save_dir)

    return search, scaler, pca

def evaluate(search, scaler, pca, df_test, y_test, save_dir):
    eval_results = dict()

    # Scale and transform
    df_test = scaler.transform(df_test)
    if args['pca']: df_test = pca.transform(df_test)

    best_model = search.best_estimator_
    predictions = best_model.predict(df_test)

    eval_results["accuracy"] = accuracy_score(predictions, y_test)
    print("Accuracy", accuracy_score(predictions, y_test))
    eval_results["c_matrix"] = confusion_matrix(predictions, y_test)
    precision, recall, f1, _ = precision_recall_fscore_support(predictions,
                                                               y_test)
    eval_results["precision"] = precision
    eval_results["recall"] = recall
    eval_results["f1"] = f1

    # Save evaluation results
    with open(save_dir + "eval_results.pkl", "wb") as f:
        pkl.dump(eval_results, f)
    f.close()

def find_important_features(df_test, y_test, search, classifier_name, save_dir):
    best_model = search.best_estimator_
    
    importances = permutation_importance(best_model, df_test, y_test, n_repeats=10,
                                         random_state=100, n_jobs=8).importances_mean

    
    
    features_and_importances = list(zip(list(df_test.columns), importances))
    features_and_importances = sorted(features_and_importances, key=lambda x: x[1], reverse=True)

    # Save all features and importances
    with open(save_dir + "features_and_importances.pkl", "wb") as f:
        pkl.dump(features_and_importances, f)
    f.close()

    keep = int(0.25 * len(features_and_importances))
    important_features = [x[0] for x in features_and_importances[:keep]]

    # Save ordered list of important features
    with open(save_dir + "important_features.pkl", "wb") as f:
        pkl.dump(important_features, f)
    f.close()                 

classifiers = [RandomForestClassifier]
param_grids = [{'n_estimators': [64, 128, 256, 512, 1024],
                'max_depth': [16, 32, 64, 128, None],
                'min_samples_split': [2, 4, 8],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [seed]}]

# For file naming
classifier_names = ["rf"]

for i in range(len(classifiers[:1])):
    # Load data
    training_data = eval_utils.prepare_data(args["training_data"])
    test_data = eval_utils.prepare_data(args["test_data"])

    # Select the same features
    common_features = list(set(training_data.columns).intersection(test_data.columns))
    common_features = sorted(common_features)
    training_data = training_data[common_features]
    test_data = test_data[common_features]

    # Load classes
    with open(args["classes"], "rb") as f:
        classes = pkl.load(f)
    f.close()

    y = [classes[x] for x in training_data.index.values.tolist()]
    y_test = [classes[x] for x in test_data.index.values.tolist()]

    # Modify data for binary classification
    if args['ibd_vs_noibd']:
        y = [int(bool(classes[x]))
             for x in training_data.index.values.tolist()]
        y_test = [int(bool(classes[x]))
                  for x in test_data.index.values.tolist()]
    if args['cd_vs_noibd']:
        no_uc_samples_train = [sample
                               for sample in training_data.index.values.tolist()
                               if classes[sample] < 2]
        training_data = training_data.loc[no_uc_samples_train]
        y = [classes[x] for x in training_data.index.values.tolist()]

        no_uc_samples_test = [sample
                              for sample in test_data.index.values.tolist()
                              if classes[sample] < 2]
        test_data = test_data.loc[no_uc_samples_test]
        y_test = [classes[x] for x in test_data.index.values.tolist()]
        
    # Downsample to equal class proportions
    if args["downsample"]:
        undersampler = RandomUnderSampler(random_state=seed)
        training_data, y = undersampler.fit_resample(training_data, y)
        test_data, y_test = undersampler.fit_resample(test_data, y_test)

    print("Training", len(y))
    print("Testing", len(y_test))
    
    # Make validation splits
    if args["participant_samples"] is not None:
        with open(args["participant_samples"], "rb") as f:
            participant_samples = pkl.load(f)
        f.close()
    else:
        # Just map each sample to itself
        participant_samples = dict()
        for sample in training_data.index.values.tolist() + test_data.index.values.tolist():
            participant_samples[sample] = [sample]

    if not os.path.exists(args["out_dir"]):
        os.makedirs(args["out_dir"])
    else:
        statistics = {"dim": len(training_data.columns),
                      "num_train_samples": len(y),
                      "num_test_samples": len(y_test)}
        # Save evaluation results
        with open(args['out_dir'] + "dataset_stats.pkl", "wb") as f:
            pkl.dump(statistics, f)
        f.close()
        
        
    search, scaler, pca = transform_and_train(classifiers[i],
                                              param_grids[i],
                                              training_data,
                                              y,
                                              5,
                                              args["out_dir"])

    evaluate(search,
             scaler,
             pca,
             test_data,
             y_test,
             args["out_dir"])      
