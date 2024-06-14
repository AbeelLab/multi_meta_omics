import argparse
import os

import config
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from ibd_classification import train_and_evaluate_ibd_classifiers
from sklearn import metrics 

# Parse arguments for feature filtering
parser = argparse.ArgumentParser()
parser.add_argument("-d",
                    "--data",
                    type=str)
parser.add_argument("-ft",
                    "--filtering_type",
                    type=str)
parser.add_argument("--cut_features",
                    action='store_true')
args = vars(parser.parse_args())

filtering_type = args["filtering_type"]
cut_features = args["cut_features"]
add_on = ""
if cut_features: add_on = "_selected_features"

# Set data root
data = args["data"]
data_root = None
if data == "f2019":
    data_root = config.f2019_root
if data == "y2019":
    data_root = config.y2019_root
if data == "w2020":
    data_root = config.w2020_root

filtering_type = args["filtering_type"]

training_sets = {"mbx": data_root + "input/default/melonnpan/mbx_train_" + filtering_type + ".tsv",
                 "mgx": data_root + "input/default/melonnpan/mgx_train_" + filtering_type + ".tsv",
                 "predicted": data_root + "output/default/melonnpan/output_mGx_to_mBx_" \
                 + filtering_type + add_on + "/train_predictions.txt"}

test_sets = {"mbx": data_root + "input/default/melonnpan/mbx_test_" \
             + filtering_type + ".tsv",
             "mgx": data_root + "input/default/melonnpan/mgx_test_" \
             + filtering_type + ".tsv",
             "predicted": data_root + "output/default/melonnpan/output_mGx_to_mBx_" \
             + filtering_type + add_on + "/test_predictions.txt"}

out_dirs = {"mbx": data_root + "classifiers/from_mbx/",
            "mgx": data_root + "classifiers/from_mgx/",
            "predicted": data_root + "classifiers/from_predicted/"}

results = {data: []
           for data in ["mgx", "mbx", "predicted"]}

for seed in config.seeds:
        for data in results:
            train_and_evaluate_ibd_classifiers.train_and_evaluate(training_data=training_sets[data],
                                                                  test_data=test_sets[data],
                                                                  out_dir=out_dirs[data],
                                                                  classes=data_root + "classes.pkl",
                                                                  ibd_vs_noibd=False,
                                                                  cd_vs_noibd=False,
                                                                  downsample=True,
                                                                  seed=seed,
                                                                  custom_splits=False,
                                                                  num_iter=1,
                                                                  participant_samples=None)

            with open(out_dirs[data] + "eval_results.pkl", "rb") as f:
                eval_results = pkl.load(f)
            f.close()

            results[data].append(eval_results["accuracy"])

            if data == "mbx" \
               and args["data"] == "f2019" \
               and filtering_type == "lenient" \
               and not cut_features:
                c_matrix = eval_results["c_matrix"]
                cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix)
                cm_display = cm_display.plot(cmap=plt.cm.Grays, values_format='g')
                plt.savefig("confusion_matrix.svg")
                plt.close()
                

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
for data in ["mgx", "mbx", "predicted"]:
    print(data)
    print(round(np.mean(results[data]), 2))
    print(round(np.std(results[data]), 2))
