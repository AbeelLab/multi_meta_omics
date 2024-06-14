import pickle as pkl

# Global path to directory
root = "[...]"
# Other shared paths
data_root = root + "data/"
ibdmdb_root = data_root + "ibdmdb/"
input_data_root= ibdmdb_root +  "input/"
output_data_root = ibdmdb_root + "output/"
ibd_classifier_root = ibdmdb_root + "ibd_classifiers/"

# Other dataset(s)
# Franzosa 2019
f2019_root = data_root + "f2019/"
f2019_input_data_root= f2019_root +  "input/"
f2019_output_data_root = f2019_root + "output/"
f2019_ibd_classifier_root = f2019_root + "ibd_classifiers/"
# Wang 2020
w2020_root = data_root + "w2020/"
w2020_input_data_root= w2020_root +  "input/"
w2020_output_data_root = w2020_root + "output/"
w2020_ibd_classifier_root = w2020_root + "ibd_classifiers/"
# Yachida 2019
y2019_root = data_root + "y2019/"
y2019_input_data_root= y2019_root +  "input/"
y2019_output_data_root = y2019_root + "output/"
y2019_ibd_classifier_root = y2019_root + "ibd_classifiers/"

figure_root = root + "data/figures/"

scripts_root = root + "scripts/"
R_scripts_root = scripts_root + "R/"
python_scripts_root = scripts_root + "python/"

melonnpan_root = root + "melonnpan/"
biomened_root = root + "biomened/"
mimenet_root = root + "mimenet/"

# Pickle variables from pickle files
with open(ibdmdb_root + "classes.pkl", "rb") as f:
    classes = pkl.load(f)
f.close()

with open(ibdmdb_root + "participant_samples.pkl", "rb") as f:
    participant_samples = pkl.load(f)
f.close()

with open(ibdmdb_root + "models.pkl", "rb") as f:
    models = pkl.load(f)
f.close()

with open(ibdmdb_root + "all_models.pkl", "rb") as f:
    all_models = pkl.load(f)
f.close()

# Data types
data_types = {"mGx_taxa", "mGx", "mTx", "mPx", "mBx", "mGx_pa"}

# Benchmarked models
omics_to_omics_models = ["rf", "melonnpan", "biomened", "mimenet"]

# Data processing methods
dps = {"rf": ["arcsin"],
       "melonnpan": ["default", "normalized", "clr"],
       "mimenet": ["default", "normalized", "arcsin", "quantile"],
       "biomened": ["quantile", "normalized", "arcsin", "clr"],
       "deep_nn": ["quantile"]]}
separators = {"rf": "\t",
              "melonnpan": "\t",
              "mimenet": ",",
              "deep_nn": "\t"}
                     

# Seeds used for train/test splitting
seeds = [2, 3, 5, 7, 11, 13, 17, 23, 29, 31]

# For plotting
omics_colors = {"mPx": "#3DA79C",
                "mPx_kos": "#3DA79C",
                
                "mTx": "#ED107A",
                
                "mBx": "#F0A724",

                "mGx": "#00B0EB",
                "mGx_pa": "#00B0EB",
                "mGx_taxa": "#00B0EB"}

omics_to_omics_models_labels = {"rf": "Random Forest (baseline)",
                                "melonnpan": "MelonnPan",
                                "biomened": "SparseNED",
                                "mimenet": "MiMeNet"}
