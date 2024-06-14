import config
import os, argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_processing")
parser.add_argument("--regression_model")
args = vars(parser.parse_args())

dp = args['data_processing']
regression_model = args['regression_model']

def make_save_dir(classifier_name,
                  regression_model,
                  data_type,
                  dp):
    save_dir = config.ibd_classifier_root + dp
    save_dir += "/seed_" + str(seed)
    save_dir += "/classifier_" + classifier_name
    save_dir += "+regressor_" + regression_model
    save_dir += "+data_" + data_type
    save_dir += "_downsampled"

    return save_dir

def run_training_commands(training_data,
                          test_data,
                          out_dir,
                          seed,
                          classes=config.ibdmdb_root + "classes.pkl",
                          participant_samples=config.ibdmdb_root + "participant_samples.pkl"):
    # All three classes
    command = "python3 "
    command += config.python_scripts_root + "train_and_evaluate_ibd_classifiers.py"
    command += " --training_data " + training_data
    command += " --test_data " + test_data
    command += " --classes " + classes
    command += " --participant_samples " + participant_samples
    command += " --seed " + str(seed)
    command += " --downsample"
    command += " --out_dir " + out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    os.system(command + "/")

    # IBD vs. no IBD
    if not os.path.exists(out_dir + "_ibd_vs_noibd/"):
        os.makedirs(out_dir + "_ibd_vs_noibd/")
    os.system(command + "_ibd_vs_noibd/ --ibd_vs_noibd")

    # CD vs. no IBD
    if not os.path.exists(out_dir + "_cd_vs_noibd/"):
        os.makedirs(out_dir + "_cd_vs_noibd/")
    os.system(command + "_cd_vs_noibd/ --cd_vs_noibd")

for seed in config.seeds:
    for model_output in config.models.keys():
        # Path to full, real data
        training_data = config.input_data_root + dp
        training_data += "/seed_" + str(seed)
        training_data += "/" + model_output + "_train.txt"
        test_data = training_data.replace("train", "test")
        out_dir = make_save_dir("rf", regression_model, model_output, dp)
        run_training_commands(training_data, test_data, out_dir, seed)

        
        for model_input in config.models[model_output]:
            out_data_root = "%s%s/seed_%s/%s" % (config.output_data_root,
                                                 "normalized",
                                                 seed,
                                                 regression_model)
            true_data_root = "%s%s/seed_%s/%s" % (config.input_data_root,
                                                  dp,
                                                  seed,
                                                  "rf")
            # Input data
            input_data_file = "%s/%s/to_%s/from_%s_input.txt" % (true_data_root, 
                                                                 "train", 
                                                                 model_output, 
                                                                 model_input)
            input_data_file_test = input_data_file.replace("train", "test")
            data_type = "to-" + model_output + "-from-" + model_input + "-input"
            out_dir = make_save_dir("rf", regression_model, data_type, dp)
            run_training_commands(input_data_file, input_data_file_test, out_dir, seed)

            # Predicted data
            predicted_data_file =  "%s/output_%s_to_%s/train_predictions.txt" % (out_data_root,
                                                                                 model_input, 
                                                                                 model_output)
            predicted_data_file_test = predicted_data_file.replace("train", "test")
            data_type = "to-" + model_output + "-from-" + model_input + "-predicted"
            out_dir = make_save_dir("rf", regression_model, data_type, dp)
            run_training_commands(predicted_data_file, predicted_data_file_test, out_dir, seed)

            # Ground-truth data
            output_data_file = "%s/%s/to_%s/from_%s_output.txt" % (true_data_root, 
                                                                   "train", 
                                                                   model_output, 
                                                                   model_input)
            output_data_file_test = output_data_file.replace("train", "test")
            data_type = "to-" + model_output + "-from-" + model_input + "-output"
            out_dir = make_save_dir("rf", regression_model, data_type, dp)
            run_training_commands(output_data_file, output_data_file_test, out_dir, seed)
