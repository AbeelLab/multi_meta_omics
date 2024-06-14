import argparse
import os

import config
import numpy as np
import pickle as pkl

from utils import plotting_utils

# Parse command line arguments
# This is not ideal
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--downsample", action="store_true")
parser.add_argument("-p", "--pca", action="store_true") 
parser.add_argument("-dp", "--data_processing_technique", type=str, default="default")
parser.add_argument("-r", "--regressor", type=str, default="melonnpan")
parser.add_argument("-c", "--classifier", type=str, default="rf")
parser.add_argument("-m", "--metric", type=str, default="accuracy")
args = vars(parser.parse_args())

metric = args["metric"]

# Add-ons for file naming
downsampled_addon = "_downsampled" if args['downsample'] else ""
pca_addon = "_pca0.95" if args['pca'] else ""

benchmarking_labels = ["input", "predicted"]#, "concatenated"]
plotting_labels = benchmarking_labels

colors = {"input": '#00B0F0',
          "predicted": '#00B0F0',
          "concatenated": '#8AC926'}

fig_sizes = {"mTx": (4, 5),
             "mPx": (8, 5),
             "mBx": (12, 5)}

for model_output in config.models.keys():
    model_inputs = sorted([x for x in config.models[model_output]],
                          key = lambda x: (len(x), int("+" in x), x))#sorted(list(config.models[model_output]))

    colors["predicted"] = config.omics_colors[model_output]

    figsize = (12, 3)

    means = dict()
    errors = dict()

    for label in benchmarking_labels:
        means[label] = []
        errors[label] = []
        
        for model_input in model_inputs:
            results_for_this_input = []

            for seed in config.seeds:
                # Load model evaluation results
                data_type = "to-" + model_output + "-from-" + model_input + "-" + label
                model_dir = config.ibd_classifier_root + args["data_processing_technique"]
                model_dir += "/seed_" + str(seed)
                model_dir += "/classifier_" + args["classifier"]
                model_dir += "+regressor_" + args["regressor"]
                model_dir += "+data_" + data_type
                model_dir += downsampled_addon
                model_dir += pca_addon
                model_dir += "/"
                
                results_file = model_dir + "eval_results.pkl"

                if not os.path.exists(results_file):
                    print("No model:", model_dir)
                    results_for_this_input.append(0)
                    continue

                with open(results_file, "rb") as f:
                    eval_results = pkl.load(f)
                f.close()
                
                # Get accuracy
                results_for_this_input.append(eval_results[metric])

            means[label].append(np.mean(results_for_this_input))
            errors[label].append(np.std(results_for_this_input))

    # Compute performance and error using all data of model_output type
    results_from_all_seeds = []
    for seed in config.seeds:
        model_dir = config.ibd_classifier_root + args["data_processing_technique"]
        model_dir += "/seed_" + str(seed)
        model_dir += "/classifier_" + args["classifier"]
        model_dir += "+regressor_" + args["regressor"]
        model_dir += "+data_" + model_output
        model_dir += downsampled_addon
        model_dir += pca_addon
        model_dir += "/"
        results_file = model_dir + "eval_results.pkl"

        if not os.path.exists(results_file):
            print("No model:", model_dir)
            continue

        with open(results_file, "rb") as f:
            eval_results = pkl.load(f)
        f.close()

        results_from_all_seeds.append(eval_results[metric])

    dashed_line_mean = np.mean(results_from_all_seeds)
    dashed_line_error = np.std(results_from_all_seeds)
    print(dashed_line_error)

    save_to = config.figure_root + "explainability/to_" + model_output + "/"
    save_to += "predict_ibd_classifier_" + args["classifier"]
    save_to += "_regressor_" + args["regressor"]
    save_to += "_metric_" + args["metric"]
    save_to += downsampled_addon + pca_addon + ".svg"
    print(save_to)
    plotting_utils.make_multi_bar_chart(x_ticks = model_inputs,
                                        bar_width = 0.25,
                                        figsize = fig_sizes[model_output],
                                        benchmarking_labels = benchmarking_labels,
                                        plotting_labels = benchmarking_labels,
                                        values = means,
                                        stds = errors,
                                        colors = colors,
                                        y_label = "Accuracy",
                                        x_label = "Input data type",
                                        title = "Predicting IBD from various datasets",
                                        y_lim = (0, 1),
                                        save_to = save_to,
                                        dashed_line_mean = dashed_line_mean,
                                        dashed_line_error = dashed_line_error,
                                        dashed_line_color = colors["predicted"],
                                        last_bar_hatch = None,
                                        label_rotation=0)
