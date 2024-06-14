import argparse

import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import eval_utils

plt.style.use('seaborn-v0_8-whitegrid')

for model_output in config.models:
    all_inputs = config.models[model_output]
    single_omics_inputs = sorted([x for x in all_inputs
                                  if "+" not in x])
    multi_omics_inputs = sorted([x for x in all_inputs
                                 if "+" in x],
                                key = lambda x: (x, len(x)))


    plt.xlim(0.2, 0.9)
    axis = np.arange(len(multi_omics_inputs))
    
    # Bars
    scores = []
    stds = []
    for model_input in multi_omics_inputs:
        score, std, _ = eval_utils.evaluate_top_features("melonnpan",
                                                         model_input,
                                                         model_output,
                                                         config.seeds,
                                                         "default",
                                                         config,
                                                         k=50)
        scores.append(score)
        stds.append(std)    

    plt.barh(multi_omics_inputs,
             scores,
             color="#6694F6",
             xerr=stds,
             ecolor="black",
             ls='none')

    for model_input in single_omics_inputs:
        score, _, _ = eval_utils.evaluate_top_features("melonnpan",
                                                       model_input,
                                                       model_output,
                                                       config.seeds,
                                                       "default",
                                                       config,
                                                       k=50)
        plt.axvline(x=score,
                    color="black",
                    ls="dashed",
                    linewidth=3,
                    dashes=(3, 3))
        plt.text(score,
                 -len(axis),
                 model_input,
                 ha='left',
                 va='bottom',
                 color="black")

    plt.yticks()
    print(config.figure_root + model_output \
          + "_multi_omics_bars.svg")
    plt.savefig(config.figure_root + model_output \
                + "_multi_omics_bars.svg",
                dpi=300,
                bbox_inches="tight")
    plt.close()

# Table: normal vs. latent
table = []
for model_output in config.models:
    all_inputs = config.models[model_output]
    single_omics_inputs = sorted([x for x in all_inputs
                                  if "+" not in x])
    multi_omics_inputs = sorted([x for x in all_inputs
                                 if "+" in x],
                                key = lambda x: (x, len(x)))


    for model_input in multi_omics_inputs:
        row = [model_input + " -> " + model_output]
        score, std, _ = eval_utils.evaluate_top_features("melonnpan",
                                                         model_input,
                                                         model_output,
                                                         config.seeds,
                                                         "default",
                                                         config,
                                                         k=50,
                                                         latent=True)
        row.append('%.2f (+- %.2f)' % (score, std))

        score, std, _ = eval_utils.evaluate_top_features("melonnpan",
                                                         model_input,
                                                         model_output,
                                                         config.seeds,
                                                         "default",
                                                         config,
                                                         k=50,
                                                         latent=False)
        row.append('%.2f (+- %.2f)' % (score, std))

        table.append(row)
    
df = pd.DataFrame(np.array(table))
df.to_csv("latent_vs_normal.csv",
          index=False,
          header=False)      


# Make table with all scores
table = []
all_inputs = sorted([x for x in config.all_models["mBx"]],
                    key = lambda x: (x, int("+" in x), len(x)))
for model_input in all_inputs:
    row = [model_input]
    for model_output in ["mTx", "mPx", "mBx"]:
        if model_input not in config.all_models[model_output]: 
            row.append("-")
            continue
        corr, std, _ = eval_utils.evaluate_top_features("melonnpan",
                                                        model_input,
                                                        model_output,
                                                        config.seeds,
                                                        "default",
                                                        config,
                                                        k=50)
        row.append('%.2f (+- %.2f)' % (corr, std))

    table.append(row)

df = pd.DataFrame(np.array(table))
df.to_csv("multi_omics.csv",
          index=False,
          header=False)
