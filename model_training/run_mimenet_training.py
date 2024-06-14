import argparse
import os

import config
import pandas as pd

# Seed used for train/test splitting
parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--seed",
                    type=int)
parser.add_argument("--dp",
                    type=str)
args = vars(parser.parse_args())

seed = args["seed"]
dp = args["dp"]

for model_output in config.models:
    for model_input in config.models[model_output]:
        if "+" in model_input: continue

        input_data = "%s%s/seed_%s/mimenet/train/to_%s/from_%s_input.txt" % (config.input_data_root,
                                                                             dp,
                                                                             seed,
                                                                             model_output,
                                                                             model_input)
        output_folder = "%s%s/seed_%s/mimenet/output_%s_to_%s/" % (config.output_data_root,
                                                                   dp,
                                                                   seed,
                                                                   model_input,
                                                                   model_output)

        output_data = "%s%s/seed_%s/mimenet/train/to_%s/from_%s_output.txt" % (config.input_data_root,
                                                                               dp,
                                                                               seed,
                                                                               model_output,
                                                                               model_input)


        param_file = config.mimenet_root + "results/IBD/network_parameters.txt"
        training_command = "python " + config.mimenet_root + "MiMeNet_train.py"
        training_command += " -micro " + input_data
        training_command += " -metab " + output_data

        if dp != "default":
            training_command += " -micro_norm None -metab_norm None"

        training_command += " -external_micro " + input_data.replace("train", "test")
        training_command += " -external_metab " + output_data.replace("train", "test")
        training_command += " -num_background 25 -num_run 5 -num_cv 5"

        intermediate_folder = dp + "_" + str(seed) + "_" + model_output + "_from_" + model_input
        training_command += " -output " + intermediate_folder

        if os.path.exists(output_folder + "test_predictions.txt"):
            continue
        else:
            print(output_folder)
            print(training_command)
            os.makedirs(output_folder, exist_ok=True)

        # os.system(training_command)

        # # Move to desired directory (without results/)
        # os.system("cd " + config.scripts_root + "sbatch/results/" + intermediate_folder + " && mv external_predictions.csv " + output_folder)

        # # Rename and reformat predictions file
        # # Copy and go from csv to tsv
        # move_to = output_folder + "test_predictions.txt"
        # current = output_folder + "external_predictions.csv"

        # temp = pd.read_csv(current)
        # temp.set_index("Unnamed: 0", inplace=True)
        # temp.to_csv(move_to, sep="\t", index_label="ID")
