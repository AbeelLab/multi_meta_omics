import argparse
import os

import config

# Seed used for train/test splitting
parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--seed",
                    type=int)
parser.add_argument("-c",
                    "--cores",
                    type=int)
parser.add_argument("--all_models",
                    action="store_true")
parser.add_argument("--single_omics",
                    action="store_true")
args = vars(parser.parse_args())

seed = args["seed"]
cores = args["cores"]

models = config.models
if args["all_models"]:
    models = config.all_models

for dp in ["default"]:
    for model_output in models.keys():
        for model_input in models[model_output]:
            # Skip multi-omics
            if args["single_omics"] and "+" in model_input: continue
            
            input_data = "%s%s/seed_%s/melonnpan/train/to_%s/from_%s_input.txt" % (config.input_data_root,
                                                                                   dp,
                                                                                   seed,
                                                                                   model_output,
                                                                                   model_input)
            # MelonnPan applies an arcsin transformation to the output by default
            output_folder = "%s%s/seed_%s/melonnpan/output_%s_to_%s/" % (config.output_data_root,
                                                                         dp,
                                                                         seed,
                                                                         model_input,
                                                                         model_output)

            # Skip computations that were already done
            if os.path.exists(output_folder + "test_predictions.txt"):
                continue
            else:
                os.makedirs(output_folder, exist_ok=True)

            output_data = "%s%s/seed_%s/melonnpan/train/to_%s/from_%s_output.txt" % (config.input_data_root,
                                                                                     dp,
                                                                                     seed,
                                                                                     model_output,
                                                                                     model_input)

            training_command = "Rscript " + config.melonnpan_root + "train_metabolites.R"
            training_command += " -p " + str(cores)
            # Cutoff should intuitively be 0 but I think there's a bug in the model
            training_command += " --cutoff 1.1" 
            training_command += " --metag=" + input_data
            training_command += " --metab=" + output_data
            training_command += " -o " + output_folder

            if dp != "default":
                training_command += " --no.transform.metag --no.transform.metab"

            os.system(training_command)

            # Make test predictions
            predictions_command = "Rscript " + config.melonnpan_root + "predict_metabolites.R"
            predictions_command += " -w " + output_folder + "MelonnPan_Trained_Weights.txt"
            predictions_command += " -r " + input_data
            predictions_command += " -i " + input_data.replace("train", "test")
            predictions_command += " -o " + output_folder

            if dp != "default":
                predictions_command += " --no.transform.metag --no.transform.metab"

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
