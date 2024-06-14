import argparse
import collections
import os

import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib_venn import venn3, venn2
from scipy.stats import ttest_1samp
from utils import eval_utils

plt.style.use('seaborn-v0_8-whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument("-o2o", "--omics_to_omics_model", type=str,
                    default="melonnpan",
                    choices=["rf", "melonnpan", "biomened", "mimenet"])
parser.add_argument("-dp", "--data_processing_technique", type=str,
                    default="default")
args = vars(parser.parse_args())

omics_to_omics_model = args["omics_to_omics_model"]
dp = args["data_processing_technique"]

save_dir = None

print(config.models)

def load_data(dp, model_output, model_input, seed):
        # Load ground truth
        true_dir = config.input_data_root + dp + "/seed_" + str(seed)
        ground_truth_file = "%s/%s/test/to_%s/from_%s_output.txt" % (true_dir,
                                                                     omics_to_omics_model,
                                                                     model_output, 
                                                                     model_input)
        real_df = eval_utils.prepare_data(ground_truth_file)
                        
        # Load input data
        input_file = "%s/%s/test/to_%s/from_%s_input.txt" % (true_dir,
                                                             omics_to_omics_model,
                                                             model_output, 
                                                             model_input)
        input_df = eval_utils.prepare_data(input_file)

        # Load predicted data from this input
        pred_dir = config.output_data_root + dp + "/seed_" + str(seed)
        pred_dir += "/" + omics_to_omics_model
        pred_dir += "/output_" + model_input + "_to_" + model_output + "_latent"
        predicted_file = pred_dir + "/test_predictions.txt"
        predicted_df = prepare_data(predicted_file)

        return input_df, real_df, predicted_df


# Hardcoded for 25% best features, according to https://stats.stackexchange.com/questions/62896/jaccard-similarity-from-data-mining-book-homework-problem
expected_js = 0.143

for model_output in ["mPx", "mBx"]:#config.models.keys():
        print("Model output:", model_output)
        model_inputs = sorted([x for x in config.models[model_output]],
                          key = lambda x: (len(x), int("+" in x), x))
                               #if "+" not in x])
        #sorted([x for x in config.models[model_output]],
                       #       key = lambda x: (int("+" in x), x))

        # First create a dictionary with correlations per seed, feature and input type
        all_corrs = {seed: {model_input: None
                            for model_input in model_inputs}
                     for seed in config.seeds}
        all_corrs_dict_format = {seed: {model_input: None
                                        for model_input in model_inputs}
                                 for seed in config.seeds}
        
        all_features = {model_input: set() for model_input in model_inputs}

        # Keep track of average feature variances (across seeds)
        # feature: [summed variance, normalization factor]
        variances = {}
        # Keep track of average correlations across input types
        correlations_across_inputs = {}

        for seed in config.seeds:
                # Path to save figures
                save_dir = config.figure_root + "explainability/to_" + model_output + "/"
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                
                for model_input in model_inputs:
                        _, real_df, predicted_df = load_data(dp, model_output, model_input, seed)
                        _, correlations = eval_utils.compute_spearman_top_k(real_df, predicted_df)
                        # Correlations should already be ordered: (feature, correlation)
                        all_corrs[seed][model_input] = correlations
                        all_corrs_dict_format[seed][model_input] = dict(correlations)

                        all_features[model_input] = all_features[model_input].union(set(dict(correlations).keys()))

                        for feature in dict(correlations):
                                var = np.var(real_df[feature])
                                
                                if feature not in variances:
                                        variances[feature] = [var, 1]
                                        correlations_across_inputs[feature] = [dict(correlations)[feature], 1]
                                else:
                                        variances[feature][0] += var
                                        variances[feature][1] += 1
                                        
                                        correlations_across_inputs[feature][0] += dict(correlations)[feature]
                                        correlations_across_inputs[feature][1] += 1

        # ---> Plot best predicted features across seeds, for each input
        for model_input in model_inputs:
                print("Input:", model_input)
                to_plot = np.zeros((len(config.seeds), len(config.seeds)))

                feature_intersection = set()
                feature_union = set()

                # Keep track of all Jaccard similarities to test for significance
                all_similarities = []
                # Compute Jaccard similarities between best features for each seed
                for i, seed1 in enumerate(config.seeds):
                        for j, seed2 in enumerate(config.seeds):
                                k = min(len(all_corrs[seed1][model_input]),
                                        len(all_corrs[seed2][model_input]))
                                k = int(0.25 * k)
                                
                                best1 = set([x[0]
                                             for x in all_corrs[seed1][model_input][:k]])
                                best2 = set([x[0]
                                             for x in all_corrs[seed2][model_input][:k]])

                                # Keep track of union and intersection
                                if len(feature_intersection) == 0:
                                        feature_intersection = best1.intersection(best2)
                                else:
                                        feature_intersection = feature_intersection.intersection(best1.intersection(best2))
                                feature_union = feature_union.union(best1.union(best2))

                                jaccard = len(best1.intersection(best2))
                                jaccard /= len(best1.union(best2))

                                to_plot[i][j] = jaccard

                                if j < i: all_similarities.append(jaccard)

                print("Intersection:", len(feature_intersection))
                print(feature_intersection)
                print("Union:", len(feature_union))

                sns.heatmap(to_plot, annot=True, fmt=".2f",
                            xticklabels=np.arange(1, len(config.seeds)+1),
                            yticklabels=np.arange(1, len(config.seeds)+1),
                            vmin=0, vmax=1,
                            cmap = 'gray_r')#cmap=sns.light_palette(config.omics_colors[model_output], as_cmap=True))

                # Compute t-test statistic: compare to expected Jaccard similarity
                pval = ttest_1samp(a=all_similarities,
                                   popmean=expected_js).pvalue
                print("--- Across seeds")
                print("p-value:", pval)
                print("Significant?", pval < 0.05)

                save_to = save_dir + "_latent_best_features_seeds_comparison_from_" + model_input + ".png"

                plt.savefig(save_to, dpi=300, bbox_inches = "tight")
                plt.savefig(save_to.replace("png", "svg"), bbox_inches = "tight")
                plt.close()

for model_output in config.models.keys():
        print("Model output:", model_output)
        model_inputs = sorted([x for x in config.models[model_output]],
                          key = lambda x: (len(x), int("+" in x), x))

        # First create a dictionary with correlations per seed, feature and input type
        all_corrs = {seed: {model_input: None
                            for model_input in model_inputs}
                     for seed in config.seeds}
        all_corrs_dict_format = {seed: {model_input: None
                                        for model_input in model_inputs}
                                 for seed in config.seeds}
        
        all_features = {model_input: set() for model_input in model_inputs}

        # Keep track of average feature variances (across seeds)
        # feature: [summed variance, normalization factor]
        variances = {}
        # Keep track of average correlations across input types
        correlations_across_inputs = {}

        for seed in config.seeds:
                # Path to save figures
                save_dir = config.figure_root + "explainability/to_" + model_output + "/"
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                
                for model_input in model_inputs:
                        _, real_df, predicted_df = load_data(dp, model_output, model_input, seed)
                        _, correlations = eval_utils.compute_spearman_top_k(real_df, predicted_df)
                        # Correlations should already be ordered: (feature, correlation)
                        all_corrs[seed][model_input] = correlations
                        all_corrs_dict_format[seed][model_input] = dict(correlations)

                        all_features[model_input] = all_features[model_input].union(set(dict(correlations).keys()))

                        for feature in dict(correlations):
                                var = np.var(real_df[feature])
                                
                                if feature not in variances:
                                        variances[feature] = [var, 1]
                                        correlations_across_inputs[feature] = [dict(correlations)[feature], 1]
                                else:
                                        variances[feature][0] += var
                                        variances[feature][1] += 1
                                        
                                        correlations_across_inputs[feature][0] += dict(correlations)[feature]
                                        correlations_across_inputs[feature][1] += 1

        # ---> Plot best predicted features across seeds, for each input
        for model_input in model_inputs:
                print("Input:", model_input)
                to_plot = np.zeros((len(config.seeds), len(config.seeds)))

                feature_intersection = set()
                feature_union = set()

                # Keep track of all Jaccard similarities to test for significance
                all_similarities = []
                # Compute Jaccard similarities between best features for each seed
                for i, seed1 in enumerate(config.seeds):
                        for j, seed2 in enumerate(config.seeds):
                                k = min(len(all_corrs[seed1][model_input]),
                                        len(all_corrs[seed2][model_input]))
                                k = int(0.25 * k)
                                
                                best1 = set([x[0]
                                             for x in all_corrs[seed1][model_input][:k]])
                                best2 = set([x[0]
                                             for x in all_corrs[seed2][model_input][:k]])

                                # Keep track of union and intersection
                                if len(feature_intersection) == 0:
                                        feature_intersection = best1.intersection(best2)
                                else:
                                        feature_intersection = feature_intersection.intersection(best1.intersection(best2))
                                feature_union = feature_union.union(best1.union(best2))

                                jaccard = len(best1.intersection(best2))
                                jaccard /= len(best1.union(best2))

                                to_plot[i][j] = jaccard

                                if j < i: all_similarities.append(jaccard)

                print("Intersection:", len(feature_intersection))
                print(feature_intersection)
                print("Union:", len(feature_union))

                sns.heatmap(to_plot, annot=True, fmt=".2f",
                            xticklabels=np.arange(1, len(config.seeds)+1),
                            yticklabels=np.arange(1, len(config.seeds)+1),
                            vmin=0, vmax=1,
                            cmap = 'gray_r')

                # Compute t-test statistic: compare to expected Jaccard similarity
                pval = ttest_1samp(a=all_similarities,
                                   popmean=expected_js).pvalue
                print("--- Across seeds")
                print("p-value:", pval)
                print("Significant?", pval < 0.05)

                save_to = save_dir + "best_features_seeds_comparison_from_" + model_input + ".png"

                plt.savefig(save_to, dpi=300, bbox_inches = "tight")
                plt.savefig(save_to.replace("png", "svg"), bbox_inches = "tight")
                plt.close()
        
        # ---> Get average correlation per feature, plot input types against each other
        to_plot = np.zeros((len(model_inputs), len(model_inputs)))
        # Determine mean correlations for each feature per seed
        avg_correlations = {model_input: {}
                            for model_input in model_inputs}

        for model_input in model_inputs:
                for feature in all_features[model_input]:
                        avg_correlations[model_input][feature] = 0
                        # Sometimes a feature can have NaN correlation
                        num_predictions = 0

                        for seed in config.seeds:
                                if feature in all_corrs_dict_format[seed][model_input]:
                                        num_predictions += 1
                                        corr = all_corrs_dict_format[seed][model_input][feature]
                                        avg_correlations[model_input][feature] += corr

                        avg_correlations[model_input][feature] /= num_predictions

        all_similarities = []
        for i, model_input1 in enumerate(model_inputs):
                for j, model_input2 in enumerate(model_inputs):
                        k = min(len(avg_correlations[model_input1]),
                                len(avg_correlations[model_input2]))
                        k = int(0.25 * k)
                        
                        best1 = collections.Counter(avg_correlations[model_input1]).most_common(k)
                        best2 = collections.Counter(avg_correlations[model_input2]).most_common(k)

                        best1 = set([x[0] for x in best1])
                        best2 = set([x[0] for x in best2])

                        jaccard = len(best1.intersection(best2))
                        jaccard /= len(best1.union(best2))

                        to_plot[i][j] = jaccard

                        if j < i: all_similarities.append(jaccard)

        pval = ttest_1samp(a=all_similarities,
                           popmean=expected_js).pvalue
        print("Across model inputs")
        print("p-value:", pval)
        print("Significant?", pval < 0.05)

        if model_output == "mPx":
                from_mgx = collections.Counter(avg_correlations["mGx"]).most_common(k)
                from_mtx = collections.Counter(avg_correlations["mTx"]).most_common(k)

                from_mgx = set([x[0] for x in from_mgx])
                from_mtx = set([x[0] for x in from_mtx])

                venn2([from_mgx, from_mtx], ('From mGx', 'From mTx'))
                plt.savefig(save_dir + "venn.svg")
                plt.close()

        if model_output == "mBx":
                from_mgx = collections.Counter(avg_correlations["mGx"]).most_common(k)
                from_mtx = collections.Counter(avg_correlations["mTx"]).most_common(k)
                from_mpx = collections.Counter(avg_correlations["mPx"]).most_common(k)

                from_mgx = set([x[0] for x in from_mgx])
                from_mtx = set([x[0] for x in from_mtx])
                from_mpx = set([x[0] for x in from_mpx])

                venn3([from_mgx, from_mtx, from_mpx], ('From mGx', 'From mTx', 'From mPx'))
                plt.savefig(save_dir + "venn.svg")
                plt.close()

        sns.heatmap(to_plot, annot=True, fmt=".2f",
                    xticklabels=model_inputs, yticklabels=model_inputs,
                    vmin=0, vmax=1,
                    cmap = 'gray_r',#cmap=sns.light_palette(config.omics_colors[model_output], as_cmap=True),
                    annot_kws={"size": 30 / np.sqrt(len(model_inputs))})

        save_to = save_dir + "best_features_input_comparison.png"
        print(save_dir)

        plt.savefig(save_to, dpi=300, bbox_inches = "tight")
        plt.savefig(save_to.replace("png", "svg"), bbox_inches = "tight")
        plt.close()

        # Plot feature variance vs. prediction rank
        avg_variances = {feature: variances[feature][0] /  variances[feature][1]
                         for feature in variances.keys()}
        avg_correlations_across_inputs = {feature: correlations_across_inputs[feature][0] /  correlations_across_inputs[feature][1]
                                          for feature in correlations_across_inputs.keys()}
        features = list(correlations_across_inputs.keys())
        x = [avg_variances[feature] for feature in features]
        y = [avg_correlations_across_inputs[feature] for feature in features]

        plt.figure(figsize=(18,6))
        plt.scatter(x, y, color=config.omics_colors[model_output])
        plt.xscale('log')
        plt.xlabel("Average feature variance")
        plt.ylabel("Average feature correlation")

        save_to = save_dir + model_output + "_variance_vs_corr.png"
        plt.savefig(save_to, dpi=300, bbox_inches = "tight")
        plt.savefig(save_to.replace("png", "svg"), bbox_inches = "tight")
        plt.close()

# Plot distributions of correlations
# Compare, for example: corr(MGX, MTX) and corr(MTX predicted, MTX)
related_data_types = {"mTx": ["mGx"], "mPx": ["mGx", "mTx"]}

for model_output in related_data_types.keys():
        common_features_across_seeds = set()

        print(model_output)
        for model_input in related_data_types[model_output]:
                avg_corrs_input_vs_real = dict()
                avg_corrs_real_vs_predicted = dict()

                print(model_input)
                
                for seed in config.seeds:
                        input_df, real_df, predicted_df = load_data(dp, model_output, model_input, seed)
                        
                        features = set(real_df.columns)
                        features = features.intersection(set(input_df.columns))
                        features = features.intersection(set(predicted_df.columns))
                        
                        corrs_real_vs_predicted =  dict(real_df[list(features)].corrwith(predicted_df[list(features)],
                                                                                         method='spearman'))
                        corrs_input_vs_real = dict(input_df[list(features)].corrwith(real_df[list(features)],
                                                                                     method='spearman'))

                        for feature in features:
                                if not np.isnan(corrs_real_vs_predicted[feature]) \
                                   and not np.isnan(corrs_input_vs_real[feature]):
                                        if feature not in avg_corrs_input_vs_real:
                                                avg_corrs_input_vs_real[feature] = [corrs_input_vs_real[feature]]
                                                avg_corrs_real_vs_predicted[feature] = [corrs_real_vs_predicted[feature]]
                                        else:
                                                avg_corrs_input_vs_real[feature].append(corrs_input_vs_real[feature])
                                                avg_corrs_real_vs_predicted[feature].append(corrs_real_vs_predicted[feature])
                                

                input_vs_real = np.array([np.mean(avg_corrs_input_vs_real[feature])
                                          for feature in avg_corrs_input_vs_real])
                assert not np.any(input_vs_real > 1)
                
                real_vs_predicted = np.array([np.mean(avg_corrs_real_vs_predicted[feature])
                                              for feature in avg_corrs_real_vs_predicted])
                print(np.mean(real_vs_predicted))
                assert not np.any(real_vs_predicted > 1)

                # Plot all
                ax = sns.kdeplot(data=input_vs_real, color="black", alpha=.5)
                ax = sns.kdeplot(data=real_vs_predicted, color="red", alpha=.5)

                plt.xlabel("Mean feature correlation across test splits")
                plt.ylabel("Density")
                plt.xlim(-1.5, 1.5)

                save_to = config.figure_root + "explainability/"
                save_to += model_output + "_from_" + model_input + "_distributions.png"

                plt.savefig(save_to, dpi=300, bbox_inches = "tight")
                plt.savefig(save_to.replace("png", "svg"), bbox_inches = "tight")
                plt.close()

                # Plot 50 best
                zipped = list(zip(input_vs_real, real_vs_predicted))
                zipped.sort(key=lambda x: x[1], reverse=True)
                zipped = zipped[:50]

                print(zipped)
                print(len(zipped))

                input_vs_real = [x[0] for x in zipped]
                real_vs_predicted = [x[1] for x in zipped]

                ax = sns.kdeplot(data=input_vs_real, color="black")
                ax = sns.kdeplot(data=real_vs_predicted, color="red")

                plt.xlabel("Mean feature correlation across test splits")
                plt.ylabel("Density")
                plt.xlim(-1.5, 1.5)

                save_to = config.figure_root + "explainability/"
                save_to += model_output + "_from_" + model_input + "_distributions_best.png"

                print("----- " + save_to.replace("png", "svg"))
                plt.savefig(save_to, dpi=300, bbox_inches = "tight")
                plt.savefig(save_to.replace("png", "svg"), bbox_inches = "tight")
                plt.close()              
