import warnings
warnings.filterwarnings("ignore")

import argparse
import config
import os
import shutil

import numpy as np
import pandas as pd
import pickle as pkl

from utils import eval_utils

# Seed used for train/test splitting
parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--seed",
                    type=int)
args = vars(parser.parse_args())

seed = args["seed"]

activations = ["tanh"]
activation_pairs = [x + "_" + y for x in activations for y in activations]

latent_sizes = [70]
sparsities = [0.06]
learning_rates = [0.01]

for dp in config.dps["biomened"]:
    print(dp)
    for model_output in config.models.keys():
        data_root = config.input_data_root
        data_root += dp + "/seed_" + str(seed) + "/biomened/train/to_" + model_output

        for model_input in config.models[model_output]:
            if "+" in model_input: continue
            for sparsity in sparsities:
                for latent_size in latent_sizes:
                    for activation_pair in activation_pairs:
                        for learning_rate in learning_rates:
                            final_output_folder = "%s%s/seed_%s/biomened/output_%s_to_%s/" % (config.output_data_root,
                                                                                              dp,
                                                                                              seed,
                                                                                              model_input,
                                                                                              model_output)

                            #if os.path.exists(final_output_folder + "test_predictions.txt"):
                            #    continue
                            
                            training_command = "python3 " + config.biomened_root + "main_cv_1dir.py"
                            training_command += " --model BiomeAESnip --nonneg_weight --normalize_input"
                            training_command += " --learning_rate " + str(learning_rate)
                            training_command += " --sparse " + str(sparsity)
                            training_command += " --batch_size 20 --topk 50"
                            training_command += " --latent_size " + str(latent_size)
                            training_command += " --activation " + activation_pair
                            training_command += " --data_root " + data_root

                            # Data type: used for the name
                            training_command += " --data_type from_" + model_input
                            batch_size = "20"

                            print(training_command)

                            # Sometimes the model can give an error for a batch size
                            # This happens when a batch contains just one sample
                            if os.system(training_command) != 0:
                                print("Wrong batch size...")
                                batch_size = "21"
                                training_command = training_command.replace("--batch_size 20", "--batch_size 21")
                                os.system(training_command)

                            model_alias = 'Translator+%s+%s-%s+cv_%d+%s+sparse_%s+ls_%d+%s' % ("ibd_from_" + model_input, 
                                                                                               "bac_group_fea",
                                                                                               "met_group_fea",
                                                                                               5,
                                                                                               "BiomeAESnip",
                                                                                               sparsity, 
                                                                                               latent_size,
                                                                                               "+nonneg_weight+normalize_input+bs_" + batch_size + "+ac_tanh_tanh+lr_0.01")
                            if os.path.exists(final_output_folder):
                                shutil.rmtree(final_output_folder)
                            os.makedirs(final_output_folder)

                            intermediary_output_folder = data_root + "/vis/" + model_alias

                            if os.system("cd " + intermediary_output_folder + " && mv * " + final_output_folder) != 0:
                                print(training_command)
                                print("No results... output_folder not created...")
                            else:
                                os.system("rm -r " + intermediary_output_folder)

                            # Create predictions file
                            if not os.path.exists(final_output_folder + 'x1_to_z_weight.txt'):
                                print("No model:", model_input, "to", model_output)
                                continue

                            # Load model weights
                            x1_to_z = np.genfromtxt(final_output_folder + 'x1_to_z_weight.txt', delimiter=' ')
                            z_to_x2 = np.genfromtxt(final_output_folder + 'z_to_x2_weight.txt', delimiter=' ')

                            # Scale input data
                            m = "melonnpan"
                            s = "\t"
                            if dp == "arcsin" or dp == "quantile":
                                m = "mimenet"
                                s = ","
                            true_dir = config.input_data_root + dp + "/seed_" + str(seed)
                            input_file = "%s/%s/test/to_%s/from_%s_input.txt" % (true_dir,
                                                                                 m,
                                                                                 model_output, 
                                                                                 model_input)
                            input_data = pd.read_csv(input_file,
                                                     index_col="ID",
                                                     sep=s)

                            if m == "mimenet": input_data = input_data.transpose()

                            input_data = eval_utils.scale_and_center(input_data.to_numpy())

                            temp = np.dot(np.dot(input_data, x1_to_z), z_to_x2)

                            output_file = "%s/%s/test/to_%s/from_%s_output.txt" % (true_dir,
                                                                                   m,
                                                                                   model_output, 
                                                                                   model_input)
                            real = pd.read_csv(output_file, index_col="ID", sep=s)

                            if m == "mimenet": real = real.transpose()
                            
                            temp = pd.DataFrame(temp, index=real.index, columns=real.columns)

                            # Save SCALED predictions
                            temp.to_csv(final_output_folder + "test_predictions.txt",
                                        sep="\t", index_label="ID")
                                
