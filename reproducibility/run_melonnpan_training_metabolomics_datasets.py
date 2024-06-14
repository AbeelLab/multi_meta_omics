import argparse
import os

import config
import pickle as pkl

# Seed used for train/test splitting
parser = argparse.ArgumentParser()
parser.add_argument("-c",
                    "--cores",
                    type=int)
parser.add_argument("-ft",
                    "--filtering_type",
                    type=str)
parser.add_argument("--cut_features",
                    action='store_true')
args = vars(parser.parse_args())

cores = args["cores"]
filtering_type = args["filtering_type"]
cut_features = args["cut_features"]
add_on = ""
if cut_features: add_on = "_selected_features"

for x in [config.f2019_root,
          config.y2019_root,
          config.w2020_root]:
    input_data = x + "input/default/melonnpan/mgx_train_" \
        + filtering_type + ".tsv"
    output_data = x + "input/default/melonnpan/mbx_train_" \
        + filtering_type + ".tsv"
    out = x + "output/"
    output_folder = "%s%s/melonnpan/output_%s_to_%s_%s%s/" % (out,
                                                              "default",
                                                              "mGx",
                                                              "mBx",
                                                              filtering_type,
                                                              add_on)

    os.makedirs(output_folder, exist_ok=True)

    training_command = "Rscript " + config.melonnpan_root + "train_metabolites.R"
    training_command += " -p " + str(cores)
    # Cutoff should intuitively be 0 but I think there's a bug in the model
    if not cut_features:
        training_command += " --cutoff 1.1" 
    training_command += " --metag=" + input_data
    training_command += " --metab=" + output_data
    training_command += " -o " + output_folder

    os.system(training_command)

    # Make test predictions
    predictions_command = "Rscript " + config.melonnpan_root + "predict_metabolites.R"
    predictions_command += " -w " + output_folder + "MelonnPan_Trained_Weights.txt"
    predictions_command += " -r " + input_data
    predictions_command += " -i " + input_data.replace("train", "test")
    predictions_command += " -o " + output_folder

    os.system(predictions_command)

    # Move predictions to files with different names
    move_to = output_folder + "test_predictions.txt"
    current = output_folder + "MelonnPan_Predicted_Metabolites.txt"
    # Move test predictions
    os.system("mv " + current + " " + move_to)
    # Move train predictions
    move_to = move_to.replace("test", "train")
    current = current.replace("Predicted", "Trained")
    os.system("mv " + current + " " + move_to)
