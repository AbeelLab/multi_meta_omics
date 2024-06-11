import argparse
import os

import config
import pandas as pd
import pickle as pkl

from utils import data_processing_utils

parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--seed",
                    type=int)
parser.add_argument("--impute_zeros",
                    action='store_true')
args = vars(parser.parse_args())

# Seed used for train/test splitting
seed = args["seed"]

impute_zeros = args['impute_zeros']
impute_addon = "_imputed" if impute_zeros else ""

# Load files as data frames
taxa_abundances = pd.read_csv(config.ibdmdb_root + "taxonomic_profiles.tsv",
                              sep="\t")
mgx_ecs = pd.read_csv(config.ibdmdb_root + "mgx_ecs.tsv",
                      sep="\t")
mgx_pa = pd.read_csv(config.ibdmdb_root + "mgx_pa.tsv",
                     sep="\t")
mtx_ecs = pd.read_csv(config.ibdmdb_root + "mtx_ecs.tsv",
                      sep="\t")
mpx_ecs = pd.read_csv(config.ibdmdb_root + "mpx_ecs.tsv",
                      sep="\t")
# We skip the first row because biom-convert conversion adds an extra row
mbx = pd.read_csv(config.ibdmdb_root + "mbx.tsv",
                  sep="\t",
                  skiprows=[0])

# Load metadata
metadata = pd.read_csv(config.ibdmdb_root + "hmp2_metadata_2018-08-20.csv",
                       sep=",",
                       low_memory=False)[["External ID", "Participant ID", "diagnosis"]]

# Create a dictionary to store samples per participant
# We want to split based on participants, not samples, to avoid overfitting
participant_samples = metadata.groupby(["Participant ID"])["External ID"].apply(lambda x: 
                                                                                x.values.tolist())
participant_samples = participant_samples.to_dict()
with open(config.ibdmdb_root + "participant_samples.pkl", "wb") as f:
    pkl.dump(participant_samples, f)

# Dictionary mapping each sample with a diagnosis
classes = pd.Series(metadata.diagnosis.values, index=metadata["External ID"]).to_dict()
classes = {k: data_processing_utils.map_to_ibd_diagnosis(v) for k, v in classes.items()}
with open(config.ibdmdb_root + "classes.pkl", "wb") as f:
    pkl.dump(classes, f)

# Pre-processing: drop empty samples, normalize, filter per scpecies, merge ECs/pathways, LC-MS technology, drop ungrouped proteins

# Zero-replacement
def replace_zeros(
        df,
        epsilon = 10 ** (-7)):
    for column in df.columns[1:]:
        df[column] += epsilon

    return df

# Group rows with same features (ECs or pathways)(currently they are stratified by bacteria)
def merge_features(
        df,
        id_column,
        data_type,
        split_on=":"):
    df[id_column] = df[id_column].map(lambda x: x.split(split_on)[0])
    df = df.groupby(id_column, as_index=False).sum()
    print(str(len(df)) + " " + data_type + " features from " + str(len(df.columns)) + " samples")
    
    return df

if impute_zeros: taxa_abundances = replace_zeros(taxa_abundances)
# Drop strains
taxa_abundances = taxa_abundances[~taxa_abundances["#SampleID"].str.contains("t_")]
# Drop all but species
taxa_abundances = taxa_abundances[taxa_abundances["#SampleID"].str.contains("s_")]
# Normalize and drop empty samples
taxa_abundances, empty_samples = data_processing_utils.normalize_per_sample(taxa_abundances, "#SampleID")
taxa_abundances.drop(empty_samples, axis=1, inplace=True)

if impute_zeros: mgx_ecs = replace_zeros(mgx_ecs)
# Merge ECs
mgx_ecs = merge_features(mgx_ecs, "# Gene Family", "MGX")
# Normalize and drop empty samples
mgx_ecs, empty_samples = data_processing_utils.normalize_per_sample(mgx_ecs, "# Gene Family")
mgx_ecs.drop(empty_samples, axis=1, inplace=True)

if impute_zeros: mtx_ecs = replace_zeros(mtx_ecs)
# Merge ECs
mtx_ecs = merge_features(mtx_ecs, "# Gene Family", "MTX")
# Normalize and drop empty samples
mtx_ecs, empty_samples = data_processing_utils.normalize_per_sample(mtx_ecs, "# Gene Family")
mtx_ecs.drop(empty_samples, axis=1, inplace=True)

if impute_zeros: mpx_ecs = replace_zeros(mpx_ecs, epsilon=1)
# Merge ECs
mpx_ecs = merge_features(mpx_ecs, "Gene", "MPX")
# Drop UNGROUPED proteins
mpx_ecs = mpx_ecs[~mpx_ecs["Gene"].str.contains("UNGROUPED")]
# Normalize and drop empty samples
mpx_ecs, empty_samples = data_processing_utils.normalize_per_sample(mpx_ecs, "Gene")
mpx_ecs.drop(empty_samples, axis=1, inplace=True)

if impute_zeros: mbx = replace_zeros(mbx, epsilon=1)
# Filter for one LC-MS technology
mbx = mbx[mbx["#OTU ID"].str.contains("C18n")]
# Normalize and drop empty samples
mbx, empty_samples = data_processing_utils.normalize_per_sample(mbx, "#OTU ID")
mbx.drop(empty_samples, axis=1, inplace=True)

if impute_zeros: mgx_pa = replace_zeros(mgx_pa)
# Merge pathways
mgx_pa = merge_features(mgx_pa, "# Pathway", "MGX pathways", split_on="|")
# Normalize and drop empty samples
mgx_pa, empty_samples = data_processing_utils.normalize_per_sample(mgx_pa, "# Pathway")
mgx_pa.drop(empty_samples, axis=1, inplace=True)

# Define models
data_id_columns = {"mGx_taxa": "#SampleID",
                   "mGx": "# Gene Family",
                   "mTx": "# Gene Family",
                   "mPx": "Gene",
                   "mBx": "#OTU ID",
                   "mGx_pa": "# Pathway"}
data_labels = {"mGx_taxa": taxa_abundances,
               "mGx": mgx_ecs,
               "mTx": mtx_ecs,
               "mPx": mpx_ecs,
               "mBx": mbx,
               "mGx_pa": mgx_pa}

# Rename index column so it's the same for all dataframes
for key, value in data_labels.items():
    value.rename(mapper={data_id_columns[key]: "ID"},
                 axis="columns",
                 inplace=True)

# For each output omics data type, define which omics you want to predict it from
# predict this: {from these}
models = {"mTx": {"mGx_taxa", "mGx", "mGx_pa"},
          "mPx": {"mGx_taxa", "mGx", "mTx", "mGx_pa"},
          "mBx": {"mGx_taxa", "mGx", "mTx", "mPx", "mGx_pa"}}

# Find common samples and filter
# Dictionary of dictionaries
# Output_data_type: {input_data_type: list of sample names}
samples = dict()
for model_output in models.keys():
    samples[model_output] = dict()
    
    for model_input in models[model_output]:
        in_columns = set(data_labels[model_input].columns)
        out_columns = set(data_labels[model_output].columns)
        
        common_samples = in_columns.intersection(out_columns)
        common_samples.remove("ID")
            
        samples[model_output][model_input] = list(common_samples)
            
        print("Predicting " + model_output + " from "
              + model_input + " -> " + str(len(common_samples)) + " samples")
    
    print("---")

# Dictionary of dictionaries
# Output data type: {input_data_type: (input_data, output_data)}
datasets = dict()

def filter_common_samples(
        common_samples,
        df,
        index="ID"):
    filtered_df = df[[index] + common_samples]
    return filtered_df.set_index(index, inplace=False)

for model_output in models.keys():
    datasets[model_output] = dict()
    
    for model_input in models[model_output]:
        common_samples = samples[model_output][model_input]
        input_dataset = filter_common_samples(common_samples, 
                                              data_labels[model_input])
        output_dataset = filter_common_samples(common_samples, 
                                               data_labels[model_output])
        
        assert len(input_dataset.columns) == len(output_dataset.columns)
        assert len(input_dataset.columns) == len(common_samples)

        # Transpose and reindex
        input_dataset = input_dataset.reindex(input_dataset.columns, axis=1).transpose()
        output_dataset = output_dataset.reindex(output_dataset.columns, axis=1).transpose()

        datasets[model_output][model_input] = [input_dataset, output_dataset]

for label, full_dataset in data_labels.items():
    print(data_labels[label])
    full_dataset.set_index("ID", inplace=True)
    data_labels[label] = full_dataset.transpose()
    print(data_labels[label])

# Add multi-omics input (double-omics)
for model_output, inputs in models.items():
    list_inputs = sorted(list(inputs))
    multi_omics_inputs = set([list_inputs[i] + "+" + list_inputs[j]
                              for i in range(len(list_inputs))
                              for j in range(i+1, len(list_inputs))])

    models[model_output] = models[model_output].union(multi_omics_inputs)

# Add triple-omics input (only for mBx)
models["mBx"].add("mGx+mPx+mTx")

# Save all possible models
f = open(config.ibdmdb_root + "all_models.pkl","wb")
pkl.dump(models,f)
f.close()

# Add multi-omics datasets, filtered based on common samples
def create_multi_omics_datasets(to_update, models, samples):
    for model_output in models.keys():
        inputs = sorted(models[model_output])
        
        for model_input in inputs:
            selected_datasets = to_update[model_output]

            if model_input not in selected_datasets:
                omics1, omics2 = model_input.split("+")[0], model_input.split("+")[1]
                # Select the input datasets
                df1 = selected_datasets[omics1][0].copy()
                df2 = selected_datasets[omics2][0].copy()
                
                # Change features based on dataframe they come from
                df1.columns = [c + " " + omics1 for c in df1.columns]
                df2.columns = [c + " " + omics2 for c in df2.columns]

                # Find common samples and concatenate
                # For input dataframe: common samples (dataframe indices)
                indices1, indices2 = df1.index.values.tolist(), df2.index.values.tolist()
                common_indices = list(set(indices1).intersection(set(indices2)))
                
                # For output dataframe: common features
                out_df1 = selected_datasets[omics1][1].copy()
                out_df2 = selected_datasets[omics2][1].copy()
                common_columns = list(set(out_df1.columns).intersection(set(out_df2.columns)))

                # Final input and output
                multi_omics_input = pd.concat([df1.loc[common_indices], 
                                               df2.loc[common_indices]], 
                                               axis=1)
                multi_omics_output = out_df1.loc[common_indices][common_columns]

                to_update[model_output][model_input] = [multi_omics_input, multi_omics_output]
                samples[model_output][model_input] = common_indices

        # To predict mBx: mGx+mPx+mTx
        selected_datasets = to_update["mBx"]
        df1, df2, df3 = selected_datasets["mGx"][0].copy(), selected_datasets["mPx"][0].copy(), selected_datasets["mTx"][0].copy()
        df1.columns = [c + " " + "mGx" for c in df1.columns]
        df2.columns = [c + " " + "mPx" for c in df2.columns]
        df3.columns = [c + " " + "mTx" for c in df3.columns]
        indices1, indices2, indices3 = df1.index.values.tolist(), df2.index.values.tolist(), df3.index.values.tolist()
        common_indices = list(set(indices1).intersection(set(indices2)).intersection(set(indices3)))
        out_df1 = selected_datasets["mGx"][1].copy()
        multi_omics_input = pd.concat([df1.loc[common_indices], 
                                       df2.loc[common_indices],
                                       df3.loc[common_indices]], 
                                      axis=1)
        multi_omics_output = out_df1.loc[common_indices]
        to_update["mBx"]["mGx+mPx+mTx"] = [multi_omics_input, multi_omics_output]
        samples["mBx"]["mGx+mPx+mTx"] = common_indices

create_multi_omics_datasets(datasets, models, samples)

# Filter low abundance features
high_abundance_threshold = 5 * (10 ** (-5))
low_abundance_threshold = -1 
samples_threshold = 0.1

for model_output in models.keys():
    for model_input in models[model_output]:
        print(model_output + " from " + model_input)

        for i in range(len(datasets[model_output][model_input])):
            before = len(datasets[model_output][model_input][i].columns)

            if i == 0:
                data_type = model_input
                if "+" in model_input:
                    data_type = "mixed"
            else:
                data_type = model_output

            if i == 0:
                print("Input")
            else:
                print("Output")

            to_drop = data_processing_utils.filter_out_low_abundance_features(datasets[model_output][model_input][i], 
                                                                              high_abundance_threshold, 
                                                                              low_abundance_threshold,
                                                                              data_type,
                                                                              samples_threshold)

            datasets[model_output][model_input][i].drop(to_drop, inplace=True, axis=1)

            print("Dropped ", len(to_drop), "out of", before, "features")
            print("Number of features:", before - len(to_drop))

# Apply filtering to full datasets as well
for label, full_dataset in data_labels.items():
    print("Full dataset", label)
    to_drop = data_processing_utils.filter_out_low_abundance_features(full_dataset, 
                                                                      high_abundance_threshold, 
                                                                      low_abundance_threshold,
                                                                      label,
                                                                      samples_threshold)

    full_dataset.drop(to_drop, inplace=True, axis=1)

    data_labels[label] = full_dataset

# Train/test split
# Same format as the datasets dictionary
train_datasets = dict()
test_datasets = dict()

def create_train_and_test(df, train_samples, test_samples):
    df_train = df.filter(train_samples, axis=0)
    df_train = df_train[sorted(df_train.columns)]

    df_test = df.filter(test_samples, axis=0)
    df_test = df_test[sorted(df_test.columns)]
    
    return df_train, df_test

for model_output in models.keys():
    train_datasets[model_output] = dict()
    test_datasets[model_output] = dict()
    
    for model_input in models[model_output]:        
        print(model_output + " from " + model_input)
        
        train_samples, test_samples = data_processing_utils.separate_samples(samples[model_output][model_input], 
                                                                             participant_samples, 
                                                                             seed, 
                                                                             classes)
        input_train_dataset, input_test_dataset = create_train_and_test(datasets[model_output][model_input][0], 
                                                                        train_samples, 
                                                                        test_samples)
        output_train_dataset, output_test_dataset = create_train_and_test(datasets[model_output][model_input][1],
                                                                          train_samples, 
                                                                          test_samples)
        
        train_datasets[model_output][model_input] = [input_train_dataset, output_train_dataset]
        test_datasets[model_output][model_input] = [input_test_dataset, output_test_dataset]
        
        # Print proportions of classes in training and testing
        print("Train/test split: " + str(len(train_samples)) + "/" + str(len(test_samples)))
        classes_train = [int(bool(classes[x])) for x in input_train_dataset.index.values.tolist()]
        classes_test = [int(bool(classes[x])) for x in input_test_dataset.index.values.tolist()]
        print("Majority class train/test", sum(classes_train) / len(classes_train), 
              "/", sum(classes_test) / len(classes_test))
        
    print("---")

full_datasets_train_test = dict()

# Also split full datasets into train/test
for label, full_dataset in data_labels.items():
    print("Full dataset", label)
    print(full_dataset)
    
    train_samples, test_samples = data_processing_utils.separate_samples(full_dataset.index.values.tolist(),
                                                                         participant_samples,
                                                                         seed,
                                                                         classes)

    train_data, test_data = create_train_and_test(full_dataset,
                                                  train_samples,
                                                  test_samples)

    full_datasets_train_test[label] = [train_data, test_data]
    
    # Print proportions of classes in training and testing
    print("Train/test split: " + str(len(train_samples)) + "/" + str(len(test_samples)))
    classes_train = [int(bool(classes[x])) for x in train_data.index.values.tolist()]
    classes_test = [int(bool(classes[x])) for x in test_data.index.values.tolist()]
    print("Majority class train/test", sum(classes_train) / len(classes_train), 
          "/", sum(classes_test) / len(classes_test))
        

# MelonnPan, MiMeNet, RF and deep NN
train_and_test = [train_datasets, test_datasets]
for model_output in models.keys():
    for model_input in models[model_output]:
        for i, t in enumerate(["train", "test"]):            
            for j, io in enumerate(["input", "output"]):
                save_to_dir = config.input_data_root + "normalized/seed_" + str(seed) + "/melonnpan/"
                save_to_dir += t + "/to_" + model_output + "/"

                if not os.path.exists(save_to_dir): os.makedirs(save_to_dir)

                save_file = "from_" + model_input + "_" + io
                save_file += impute_addon + ".txt"
                
                dataset = train_and_test[i][model_output][model_input][j]
                dataset.sort_index(inplace=True)
                dataset = dataset.reindex(sorted(dataset.columns), axis=1)
                
                assert (not dataset.isnull().values.any())

                # Normalize per sample: required after feature filtering
                dataset = dataset.div(dataset.sum(axis=1), axis=0)
                train_and_test[i][model_output][model_input][j] = dataset

                # MelonnPan
                dataset.to_csv(save_to_dir + save_file,
                              sep="\t", index_label="ID")
                if not os.path.exists(save_to_dir.replace("normalized", "default")):
                   os.makedirs(save_to_dir.replace("normalized", "default"))
                dataset.to_csv(save_to_dir.replace("normalized", "default") + save_file,
                              sep="\t", index_label="ID")

                # RF
                save_to_dir = save_to_dir.replace("melonnpan", "rf")
                if not os.path.exists(save_to_dir): os.makedirs(save_to_dir)
                dataset.to_csv(save_to_dir + save_file,
                              sep="\t", index_label="ID")

                # Deep NN
                save_to_dir = save_to_dir.replace("rf", "deep_nn")
                if not os.path.exists(save_to_dir): os.makedirs(save_to_dir)
                dataset.to_csv(save_to_dir + save_file,
                              sep="\t", index_label="ID")                

                # MiMeNet
                # Similar output format, but transposed and comma-separated
                save_to_dir = save_to_dir.replace("deep_nn", "mimenet")
                if not os.path.exists(save_to_dir): os.makedirs(save_to_dir)
                dataset.transpose().to_csv(save_to_dir + save_file,
                                           index_label="ID")
                if not os.path.exists(save_to_dir.replace("normalized", "default")):
                    os.makedirs(save_to_dir.replace("normalized", "default"))
                dataset.transpose().to_csv(save_to_dir.replace("normalized", "default") + save_file,
                                           index_label="ID")
                    

# BiomeNED
for model_output in models.keys():
    for model_input in models[model_output]:
        # No need to do this for the test set, we will reload the model and make predictions on the test sets used for MelonnPan
        for i, t in enumerate(["train"]): 
            save_to = config.input_data_root + "normalized/seed_" + str(seed) + "/biomened/"
            save_to += t + "/to_" + model_output + "/"

            if not os.path.exists(save_to):
                os.makedirs(save_to)

            save_to += "ibd_from_" + model_input + impute_addon

            samples = train_and_test[i][model_output][model_input][0].index.values.tolist()

            # We have to use a different seed, otherwise participants get split the same and the validation set is empty
            train_idx, val_idx = data_processing_utils.separate_samples(samples,
                                                        participant_samples,
                                                        seed,
                                                        classes)
            train_idx = [samples.index(x) for x in train_idx]
            val_idx = [samples.index(x) for x in val_idx]

            data_processing_utils.build_dataset_dict(train_and_test[i][model_output][model_input][0],
                                                     train_and_test[i][model_output][model_input][1],
                                                     save_to,
                                                     classes,
                                                     train_idx,
                                                     val_idx)

# Full datasets (only needed for IBD prediction, so store as .txt only)
for label, data in full_datasets_train_test.items():
    save_to = config.input_data_root + "normalized/seed_" + str(seed) + "/" + label
    train, test = data[0], data[1]
    
    train.sort_index(inplace=True)
    train = train.reindex(sorted(train.columns), axis=1)
    test.sort_index(inplace=True)
    test = test.reindex(sorted(test.columns), axis=1)
    
    train = train.div(train.sum(axis=1), axis=0)
    test = test.div(test.sum(axis=1), axis=0)

    train.to_csv(save_to + "_train" + impute_addon + ".txt", sep="\t", index_label="ID")
    test.to_csv(save_to + "_test" + impute_addon + ".txt", sep="\t", index_label="ID")

# # Define only main models
models = {"mTx": {"mGx"},
          "mPx": {"mGx", "mTx",
                  "mGx+mTx"},
          "mBx": {"mGx", "mTx", "mPx",
                  "mGx+mTx", "mGx+mPx", "mPx+mTx",
                  "mGx+mPx+mTx"}}

f = open(config.ibdmdb_root + "models.pkl","wb")
pkl.dump(models,f)
f.close()
