import argparse
import os

import config
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Seed used for train/test splitting
parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--seed",
                    type=int)
args = vars(parser.parse_args())

seed = args["seed"]

for dp in config.dps["rf"]:
    for model_output in config.all_models.keys():
        for model_input in config.all_models[model_output]:
            # Load training data
            input_file = "%s%s/seed_%s/rf/train/to_%s/from_%s_input.txt" % (config.input_data_root,
                                                                            dp,
                                                                            seed,
                                                                            model_output,
                                                                            model_input)
            input_df = pd.read_csv(input_file, sep="\t", index_col="ID")
            
            # Output data
            output_file = "%s%s/seed_%s/rf/train/to_%s/from_%s_output.txt" % (config.input_data_root,
                                                                              dp,
                                                                              seed,
                                                                              model_output,
                                                                              model_input)
            output_df = pd.read_csv(output_file, sep="\t", index_col="ID")

            # Train random forest regressor with default parameters
            regressor = RandomForestRegressor(random_state=42)
            regressor.fit(input_df, output_df)

            # Predict on test data
            input_df_test = pd.read_csv(input_file.replace("train", "test"), sep="\t", index_col="ID")
            output_df_test = pd.read_csv(output_file.replace("train", "test"), sep="\t", index_col="ID")

            predictions = pd.DataFrame(regressor.predict(input_df_test),
                                       index=output_df_test.index,
                                       columns=output_df_test.columns)

            # Save test predictions
            output_folder = "%s%s/seed_%s/rf/output_%s_to_%s/" % (config.output_data_root,
                                                                  dp,
                                                                  seed,
                                                                  model_input,
                                                                  model_output)

            os.makedirs(output_folder, exist_ok=True)
            
            predictions.to_csv(output_folder + "test_predictions.txt",
                               index_label="ID",
                               sep="\t")

            # Also save train predictions
            train_predictions = pd.DataFrame(regressor.predict(input_df),
                                             index=output_df.index,
                                             columns=output_df.columns)
            # Normalize predictions
            train_predictions.to_csv(output_folder + "train_predictions.txt",
                                     index_label="ID",
                                     sep="\t")
