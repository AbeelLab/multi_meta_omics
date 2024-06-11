import os

import config
import numpy as np
import pandas as pd
import pickle as pkl

from feature_engine.transformation import ArcsinTransformer
from importlib.machinery import SourceFileLoader
from joblib import dump
from sklearn.preprocessing import QuantileTransformer
from utils import data_processing_utils

parser = argparse.ArgumentParser()
parser.add_argument("--transform",
                    choices=['clr', 'arcsin', 'quantile'])
args = vars(parser.parse_args())

def clr(x):
    return np.log(x) - np.log(gmean(x))

def apply_transform(transform):
    # Other models
    for model, sep in config.separators.items():
        if transform not in config.dps[model]: continue
        
        for seed in config.seeds:
            print("--Seed:", seed)
            for model_output, model_inputs in config.all_models.items():
                for model_input in model_inputs:
                    for io in ["input", "output"]:  
                        train_f = "%snormalized/seed_%s/%s/%s/to_%s/from_%s_%s.txt" % (config.input_data_root,
                                                                                       seed,
                                                                                       model,
                                                                                       "train",
                                                                                       model_output,
                                                                                       model_input,
                                                                                       io)

                        save_dir = "%s%s/seed_%s/%s/%s/to_%s/" % (config.input_data_root,
                                                                  transform,
                                                                  seed,
                                                                  model,
                                                                  "train",
                                                                  model_output)
                        os.makedirs(save_dir, exist_ok=True)

                        transformed_train_f = "%sfrom_%s_%s.txt" % (save_dir,
                                                                    model_input,
                                                                    io)

                        df = pd.read_csv(train_f, sep=sep, index_col="ID")
                        transformer = None
                        if transform == "arcsin":
                            transformer = ArcsinTransformer()
                        elif transform == "quantile":
                            transformer = QuantileTransformer(output_distribution="normal")

                        if transformer is not None:
                            if model == "mimenet":
                                df = transformer.fit_transform(df.transpose()).transpose()
                            else:
                                df = transformer.fit_transform(df)
                        else:
                            df = df.apply(clr, axis=1)
                        
                        df.to_csv(transformed_train_f, sep=sep, index_label="ID")
                        
                        save_dir_test = "%s%s/seed_%s/%s/%s/to_%s/" % (config.input_data_root,
                                                                       transform,
                                                                       seed,
                                                                       model,
                                                                       "test",
                                                                       model_output)
                        os.makedirs(save_dir_test, exist_ok=True)

                        test_df = pd.read_csv(train_f.replace("train", "test"), sep=sep, index_col="ID")
                        if transformer is not None:
                            if model == "mimenet":
                                test_df = transformer.transform(df.transpose()).transpose()
                            else:
                                test_df = transformer.transform(df)
                        else:
                            test_df = test_df.apply(clr, axis=1)
                            
                        test_df.to_csv(transformed_train_f.replace("train", "test"),
                                       sep=sep,
                                       index_label="ID")

                        # Save transformer
                        if transformer is not None:
                            save_to = "%sfrom_%s_%s_transformer.joblib" % (save_dir,
                                                                           model_input,
                                                                           io)
                            dump(transformer, save_to)


    # BiomeNED
    if transform == "arcsin": m = "mimenet"
    else: m = "melonnpan"
    for seed in config.seeds:
        for model_output, model_inputs in config.all_models.items():
            for model_input in model_inputs:
                for data_type in ["train"]:
                    transformed_fi = "%s%s/seed_%s/%s/%s/to_%s/from_%s_%s.txt" % (config.input_data_root,
                                                                                  transform,
                                                                                  seed,
                                                                                  m,
                                                                                  data_type,
                                                                                  model_output,
                                                                                  model_input,
                                                                                  "input")

                    transformed_fo = "%s%s/seed_%s/%s/%s/to_%s/from_%s_%s.txt" % (config.input_data_root,
                                                                                  transform,
                                                                                  seed,
                                                                                  m,
                                                                                  data_type,
                                                                                  model_output,
                                                                                  model_input,
                                                                                  "output")

                    save_dir = "%s%s/seed_%s/biomened/%s/to_%s/" % (config.input_data_root,
                                                                    transform,
                                                                    seed,
                                                                    data_type,
                                                                    model_output)
                    os.makedirs(save_dir, exist_ok=True)
                    save_to = "%sibd_from_%s" % (save_dir,
                                                 model_input)

                    # Load original: for consistency of train/validation split
                    og = "%snormalized/seed_%s/biomened/%s/to_%s/ibd_from_%s.pkl" % (config.input_data_root,
                                                                                     seed,
                                                                                     data_type,
                                                                                     model_output,
                                                                                     model_input)
                    with open(og, "rb") as f:
                        og = pkl.load(f)
                    f.close()

                    train_idx = og["train_idx"]
                    val_idx = og["val_idx"]

                    if m == "mimenet":
                        transformed_fi = pd.read_csv(transformed_fi, index_col="ID").transpose()
                        transformed_fo = pd.read_csv(transformed_fo, index_col="ID").transpose()
                    else:
                        transformed_fi = pd.read_csv(transformed_fi, sep="\t", index_col="ID")
                        transformed_fo = pd.read_csv(transformed_fo, sep="\t", index_col="ID")                       
                            
                        
                    data_processing_utils.build_dataset_dict(transformed_fi,
                                                             transformed_fo,
                                                             save_to,
                                                             config.classes,
                                                             train_idx,
                                                             val_idx)


    # Full datasets
    for seed in config.seeds:
        for data_type in config.data_types:
            train_f = "%snormalized/seed_%s/%s_train.txt" % (config.input_data_root,
                                                             seed,
                                                             data_type)
            transformed_train_f = train_f.replace("normalized", transform)

            df = pd.read_csv(train_f, sep="\t", index_col="ID")
            transformer = None
            if transform == "arcsin":
                transformer = ArcsinTransformer()
            elif transform == "quantile":
                transformer = QuantileTransformer(output_distribution="normal")

            if transformer is not None:
                df = transformer.fit_transform(df)
            else:
                df = df.apply(clr, axis=1)
                
            df = transformer.fit_transform(df)
            df.to_csv(transformed_train_f, sep="\t", index_label="ID")

            test_df = pd.read_csv(train_f.replace("train", "test"), sep="\t", index_col="ID")
            if transformer is not None:
                test_df = transformer.transform(test_df)
            else:
                test_df = test_df.apply(clr, axis=1)
            test_df.to_csv(transformed_train_f.replace("train", "test"), sep="\t", index_label="ID")

            # Save transformer
            save_to = "%s%s/seed_%s/%s_transformer.joblib" % (config.input_data_root,
                                                              transform,
                                                              seed,
                                                              data_type)
            dump(transformer, save_to)

            
apply_transform(args["transform"])
