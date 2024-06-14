import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

# Model trained on original data
original_scores_f = "/tudelft.net/staff-umbrella/abeellab/bcosma/" \
    + "omics-learning/data/melonnpan_output/MelonnPan_Training_Summary.txt"
original_scores = pd.read_csv(original_scores_f,
                              sep="\t")
original_scores = dict(zip(original_scores["ID"], original_scores["Spearman"]))

# Model trained on our processed data
lenient_scores_f = "/tudelft.net/staff-umbrella/abeellab/bcosma/omics-learning/data/f2019/" \
    + "output/default/melonnpan/output_mGx_to_mBx_lenient/MelonnPan_Training_Summary.txt"
lenient_scores = pd.read_csv(lenient_scores_f,
                             sep="\t")
lenient_scores = dict(zip(lenient_scores["ID"], lenient_scores["Spearman"]))

strict_scores_f = "/tudelft.net/staff-umbrella/abeellab/bcosma/omics-learning/data/f2019/" \
    + "output/default/melonnpan/output_mGx_to_mBx_strict/MelonnPan_Training_Summary.txt"
strict_scores = pd.read_csv(strict_scores_f,
                             sep="\t")
strict_scores = dict(zip(strict_scores["ID"], strict_scores["Spearman"]))

# Make plots
for name, scores in [("strict", strict_scores),
                     ("lenient", lenient_scores)]:
    common = set(scores.keys()).intersection(set(original_scores.keys()))

    x = []
    y = []
    for feature in common:
        x.append(original_scores[feature])
        y.append(scores[feature])

    plt.ylim(0.3, 1)
    plt.xlim(0.3, 1)
    
    plt.ylabel("Trained using the original MelonnPan data")
    plt.xlabel("Trained using our processed data")
    plt.title("Feature correlations measured during cross-validation")

    xs = np.linspace(0.3, 1, num=250)
    plt.plot(xs, xs, color="gray")

    plt.scatter(x, y, color="black")

    plt.savefig("reproducibility/comparison_" + name + ".png",
                dpi=300,
                bbox_inches = "tight")
    plt.savefig("reproducibility/comparison_" + name + ".svg",
                bbox_inches = "tight")
    plt.close()
