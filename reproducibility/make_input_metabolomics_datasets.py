import argparse
import os

import config
import pandas as pd
import pickle as pkl

from utils import data_processing_utils

# Parse arguments for feature filtering
parser = argparse.ArgumentParser()
parser.add_argument("-lt",
                    "--low_abundance_threshold",
                    type=float)
parser.add_argument("-ht",
                    "--high_abundance_threshold",
                    type=float)
parser.add_argument("-st",
                    "--samples_threshold",
                    type=float)
parser.add_argument("-ft",
                    "--filtering_type",
                    type=str)
args = vars(parser.parse_args())

low_abundance_threshold = args["low_abundance_threshold"]
high_abundance_threshold = args["high_abundance_threshold"]
samples_threshold = args["samples_threshold"]
filtering_type = args["filtering_type"]

# ---> FRANZOSA 2019
print("Dataset f2019")
# Define file locations
mgx = pd.read_excel(config.f2019_root + "mgx_abundances.xlsx",
                    index_col="# Feature / Sample")
mbx = pd.read_excel(config.f2019_root + "metabolite_abundances.xlsx",
                    index_col="# Feature / Sample")
classes = pd.read_excel(config.f2019_root + "classes.xlsx",
                        index_col="# Feature / Sample")

# Map classes to integers and save them
class_dict = dict()
for column in classes.columns:
    class_label = classes.at["Diagnosis", column]
    class_dict[column] = data_processing_utils.map_to_ibd_diagnosis(class_label)
f = open(config.f2019_root + "classes.pkl", "wb")
pkl.dump(class_dict, f)
f.close()

# Pre-process data
mgx, empty_samples = data_processing_utils.normalize_per_sample(mgx, "# Feature / Sample")
mgx.drop(empty_samples, axis=1, inplace=True)
mgx.rename(mapper={"# Feature / Sample": "ID"},
           axis="columns",
           inplace=True)

#c18_metabolites = [x for x in mbx.index.values.tolist() if "C18-neg" in x]
#mbx = mbx.loc[c18_metabolites]
mbx, empty_samples = data_processing_utils.normalize_per_sample(mbx, "# Feature / Sample")
mbx.drop(empty_samples, axis=1, inplace=True)
mbx.rename(mapper={"# Feature / Sample": "ID"},
           axis="columns",
           inplace=True)

# Check that all samples are the same
assert set(mgx.columns) == set(mbx.columns)

# Set index columns
mgx.index.set_names(["ID"], inplace=True)
mbx.index.set_names(["ID"], inplace=True)

# Transpose
mgx = mgx.transpose()
mbx = mbx.transpose()

print("# features before filtering")
print("mGx", len(mgx.columns))
print("mBx", len(mbx.columns))

# Filter out low abundance features
# MGX
to_drop = data_processing_utils.filter_out_low_abundance_features(mgx, 
                                                                  high_abundance_threshold, 
                                                                  low_abundance_threshold,
                                                                  "mGx",
                                                                  samples_threshold)
mgx.drop(to_drop, inplace=True, axis=1)
# MBX
to_drop = data_processing_utils.filter_out_low_abundance_features(mbx, 
                                                                  high_abundance_threshold, 
                                                                  low_abundance_threshold,
                                                                  "mBx",
                                                                  samples_threshold)
mbx.drop(to_drop, inplace=True, axis=1)

# Train/test split: divide based on external validation cohort
train_samples = [x for x in mgx.index.values.tolist()
                 if "Validation" not in x]
test_samples = [x for x in mgx.index.values.tolist()
                 if "Validation" in x]

print("# features after filtering")
print("mGx", len(mgx.columns))
print("mBx", len(mbx.columns))

# Re-normalize
mgx = mgx.div(mgx.sum(axis=1), axis=0)
mbx = mbx.div(mbx.sum(axis=1), axis=0)

mgx_train = mgx.loc[train_samples]
mgx_test = mgx.loc[test_samples]
mbx_train = mbx.loc[train_samples]
mbx_test = mbx.loc[test_samples]

# Save normalized data
save_to_dir = config.f2019_input_data_root + "default/melonnpan/"
if not os.path.exists(save_to_dir): os.makedirs(save_to_dir)

mgx_train.to_csv(save_to_dir + "mgx_train_" + filtering_type + ".tsv",
                 sep="\t", index_label="ID")
mgx_test.to_csv(save_to_dir + "mgx_test_" + filtering_type + ".tsv",
                sep="\t", index_label="ID")
mbx_train.to_csv(save_to_dir + "mbx_train_" + filtering_type + ".tsv",
                 sep="\t", index_label="ID")
mbx_test.to_csv(save_to_dir + "mbx_test_" + filtering_type + ".tsv",
                sep="\t", index_label="ID")

# --> WANG 2020, YACHIDA 2019
def map_to_cancer(x):
    if x == "Healthy": return 0
    if "Stage" in x: return 1
    return 2

def map_to_esdr(x):
    if x == "Control": return 0
    return 1

datasets = {"w2020": config.w2020_root,
            "y2019": config.y2019_root}

class_mappings = {"w2020": map_to_esdr,
                  "y2019": map_to_cancer}

for dataset in datasets:
    print("Dataset", dataset)
    mgx = pd.read_csv(datasets[dataset] + "species.tsv",
                      index_col="Sample",
                      sep="\t").transpose()
    mbx = pd.read_csv(datasets[dataset] + "mtb.tsv",
                      index_col="Sample",
                      sep="\t").transpose()
    metadata = pd.read_csv(datasets[dataset] + "metadata.tsv",
                           sep="\t")

    class_dict = dict()
    for sample, diagnosis in zip(metadata["Sample"], metadata["Study.Group"]):
        class_dict[sample] = class_mappings[dataset](diagnosis)
    f = open(datasets[dataset] + "classes.pkl","wb")
    pkl.dump(class_dict, f)
    f.close()

    # For cancer study, remove samples from HS and MP (class 2)
    mgx = mgx[[sample for sample in class_dict
               if class_dict[sample] < 2]]
    mbx = mbx[[sample for sample in class_dict
               if class_dict[sample] < 2]]

    # Pre-process data
    mgx, empty_samples = data_processing_utils.normalize_per_sample(mgx, "Sample")
    mgx.drop(empty_samples, axis=1, inplace=True)
    mgx.rename(mapper={"Sample": "ID"},
               axis="columns",
               inplace=True)

    mbx, empty_samples = data_processing_utils.normalize_per_sample(mbx, "Sample")
    mbx.drop(empty_samples, axis=1, inplace=True)
    mbx.rename(mapper={"Sample": "ID"},
               axis="columns",
               inplace=True)

    # Check that all samples are the same
    assert set(mgx.columns) == set(mbx.columns)

    mgx.index.set_names(["ID"], inplace=True)
    mbx.index.set_names(["ID"], inplace=True)

    # Transpose
    mgx = mgx.transpose()
    mbx = mbx.transpose()

    print("# features before filtering")
    print("mGx", len(mgx.columns))
    print("mBx", len(mbx.columns))

    # MGX
    to_drop = data_processing_utils.filter_out_low_abundance_features(mgx, 
                                                                      high_abundance_threshold, 
                                                                      low_abundance_threshold,
                                                                      "mGx",
                                                                      samples_threshold)
    mgx.drop(to_drop, inplace=True, axis=1)
    # MBX
    to_drop = data_processing_utils.filter_out_low_abundance_features(mbx, 
                                                                      high_abundance_threshold, 
                                                                      low_abundance_threshold,
                                                                      "mBx",
                                                                      samples_threshold)
    mbx.drop(to_drop, inplace=True, axis=1)

    # Re-normalize
    mgx = mgx.div(mgx.sum(axis=1), axis=0)
    mbx = mbx.div(mbx.sum(axis=1), axis=0)

    # Train/test split: divide based on external validation cohort
    participant_samples = {sample: [sample]
                           for sample in class_dict}
    train_samples, test_samples = data_processing_utils.separate_samples(mbx.index.values.tolist(),
                                                                         participant_samples,
                                                                         42,
                                                                         class_dict,
                                                                         test_size=0.2)

    mgx_train = mgx.loc[train_samples]
    mgx_test = mgx.loc[test_samples]
    mbx_train = mbx.loc[train_samples]
    mbx_test = mbx.loc[test_samples]

    print("# features after filtering")
    print("mGx", len(mgx.columns))
    print("mBx", len(mbx.columns))

    # Save normalized data
    save_to_dir = datasets[dataset] + "input/default/melonnpan/"
    if not os.path.exists(save_to_dir): os.makedirs(save_to_dir)

    mgx_train.to_csv(save_to_dir + "mgx_train_" + filtering_type + ".tsv",
                     sep="\t", index_label="ID")
    mgx_test.to_csv(save_to_dir + "mgx_test_" + filtering_type + ".tsv",
                    sep="\t", index_label="ID")
    mbx_train.to_csv(save_to_dir + "mbx_train_" + filtering_type + ".tsv",
                     sep="\t", index_label="ID")
    mbx_test.to_csv(save_to_dir + "mbx_test_" + filtering_type + ".tsv",
                    sep="\t", index_label="ID")
