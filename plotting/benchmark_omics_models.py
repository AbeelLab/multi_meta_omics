import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import eval_utils
from utils import plotting_utils

# Define percentage/number of features used to compute average score
k = 50

omics_to_omics_models = ["melonnpan",
                         "mimenet",
                         "biomened",
                         "deep_nn",
                         "rf"]
dps = {"rf": "arcsin",
       "melonnpan": "default",
       "biomened": "quantile",
       "mimenet": "arcsin",
       "deep_nn": "quantile",
       "deep_nn_0.1": "quantile",
       "deep_nn_0.25": "quantile",
       "deep_nn_0.5": "quantile"}

# For plotting
labels = ["MelonnPan",
          "MiMeNet",
          "SparseNED",
          "Deep NN",
          "Random forest (baseline)"]

fig_sizes = {"mTx": (4, 5),
             "mPx": (8, 5),
             "mBx": (12, 5)}

# Initialize score dicts
scores = {omics_to_omics_model: {model_output: []
                                 for model_output in config.models}
          for omics_to_omics_model in omics_to_omics_models}
score_stds = {omics_to_omics_model: {model_output: []
                                     for model_output in config.models}
              for omics_to_omics_model in omics_to_omics_models}

# Table: output, from mgx model1, from mgx model2, ..., from mpx modeln
table = []
all_inputs = ["mGx", "mTx", "mPx"]
for model_output in ["mTx", "mPx", "mBx"]:
    row = [model_output]

    for model_input in all_inputs:
        for omics_to_omics_model in omics_to_omics_models:
            if model_input not in config.models[model_output]:
                row.append("-")
                continue

            score, std, _ = eval_utils.evaluate_top_features(omics_to_omics_model,
                                                             model_input,
                                                             model_output,
                                                             config.seeds,
                                                             dps[omics_to_omics_model],
                                                             config,
                                                             k)
            row.append('%.2f (+- %.2f)' % (score, std))
    table.append(row)

df = pd.DataFrame(np.array(table))
df.to_csv("table_benchmark.csv",
          index=False,
          header=False)

for omics_to_omics_model in omics_to_omics_models:
    print("---", omics_to_omics_model)
    for model_output in config.models:
        # Filter for single omics and sort
        model_inputs = sorted([x for x in config.models[model_output]
                               if "+" not in x])

        for model_input in model_inputs:
            # For now dummy data for mimenet because the model is not currently working
            #if omics_to_omics_model == "mimenet":
            #    scores[seed][omics_to_omics_model][model_output].append(np.random.uniform(0.32, 0.38))
            #    continue

            score, std, _ = eval_utils.evaluate_top_features(omics_to_omics_model,
                                                             model_input,
                                                             model_output,
                                                             config.seeds,
                                                             dps[omics_to_omics_model],
                                                             config,
                                                             k)

            scores[omics_to_omics_model][model_output].append(score)
            score_stds[omics_to_omics_model][model_output].append(std)

print(scores)

colors = {"melonnpan": "#1a3949",
          "mimenet": "#1f7296",
          "biomened": "#00b0eb",
          "deep_nn": "#69c1f0",
          "deep_nn_0.1": "#69c1f0",
          "deep_nn_0.25": "#69c1f0",
          "deep_nn_0.5": "#69c1f0",
          "rf": "#b0dcf7"}

for model_output in config.models:
    model_inputs = sorted([x for x in config.models[model_output]
                           if "+" not in x])
    
    #colors = {omics_to_omics_model: config.omics_colors[model_output]
    #          for omics_to_omics_model in omics_to_omics_models}

    alphas = {omics_to_omics_model: a
              for omics_to_omics_model, a in zip(omics_to_omics_models, [1, 1, 1, 1, 1])}

    means = {omics_to_omics_model: np.array(scores[omics_to_omics_model][model_output])
             for omics_to_omics_model in scores}
    errors = {omics_to_omics_model: np.array(score_stds[omics_to_omics_model][model_output])
              for omics_to_omics_model in score_stds}

    model_output = model_output.replace("M", "m").replace("X", "x")

    if not os.path.exists(config.figure_root):
        os.makedirs(config.figure_root)
        
    save_to = config.figure_root + "to_" + model_output + "_single_omics_benchmark"
    save_to += "_k=" + str(k) + ".png"

    print(save_to)

    y_label = "Mean Spearman's rank correlation coefficient \n (computed for the "
    y_label += str(int(k * 100)) + "% best-predicted features)"
    plotting_utils.make_multi_bar_chart(x_ticks=model_inputs,
                                        bar_width=0.15,
                                        figsize=fig_sizes[model_output],
                                        benchmarking_labels=omics_to_omics_models,
                                        plotting_labels=labels,
                                        values=means,
                                        stds=errors,
                                        colors=colors,
                                        y_label=y_label,
                                        x_label="Input data type",
                                        title="Predicting " + model_output,
                                        y_lim=(0, 1),
                                        alphas=alphas,
                                        save_to=save_to)

    overall_scores = {omics_to_omics_model: means[omics_to_omics_model].mean()
                      for omics_to_omics_model in omics_to_omics_models}
    print(overall_scores)
    print("Best model for " + model_output + ":", max(overall_scores, key=overall_scores.get))
