from json import encoder
import pandas as pd
import numpy as np
import sys, os, pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
import logging
from rdkit.Chem import Descriptors
from multiprocessing import Pool, cpu_count
import h5py
from sklearn.random_projection import GaussianRandomProjection
from src.utils import encode_mols, load_compound_encoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pickle_file = sys.argv[1]
task_id = sys.argv[2]
input_dict = pickle.load(open(pickle_file, 'rb')) # dictionary with input information and array task ID, in this case, only input file path
input_file = str(input_dict[task_id]) 

# --- Get ECFP4 ---
def get_ECFP4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Invalid SMILES:', smiles)
        return None  # Return zero vector for invalid SMILES
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return fp

# To load pretrained dotP model:
class CompoundEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)/100  # [batch_size, d_proj]

filter_mols=True

compound_encoder = load_compound_encoder()
compound_encoder.eval()

logging.info("Loading input file: %s", input_file)
df = pd.read_csv(input_file) # ChEMBL dataset. Info on how to get them in scr/ChEMBL_data_preprocessing. Smiles must be standardized previously using scr/utils, and a new column named standardized_smiles must be added to the dataframe from ChEMBL

logging.info("Getting SMILES. Number of SMILES: %d", len(df))
if ('standardized_smiles' not in df.columns) or ('smiles' not in df.columns):
    raise ValueError("The input file must contain a 'standardized_smiles' column.")

if filter_mols: # default = True
    logging.info('Mol Weight filter')
    def compute_properties(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        mw = Descriptors.MolWt(mol)
        return mw

    def compute_mw_parallel(smiles_list, n_processes=None):
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)
        with Pool(n_processes) as pool:
            results = pool.map(compute_properties, smiles_list)
        # Optionally filter out None results
        return np.array([res for res in results if res is not None])
    
    smiles = df['standardized_smiles'].values #df['smiles'].values
    molecular_weights = compute_mw_parallel(smiles, n_processes=cpu_count())
    mask = (molecular_weights < 1000) & (molecular_weights > 50)

    df = df[mask]

logging.info(f"# Mols: {df.shape[0]}")
smiles = df['standardized_smiles'].values #df['smiles'].values

y = df[df.columns[-2]].values # df[df.columns[-1]].values
if len(np.unique(y)) != 2:
    raise ValueError("The target variable must be binary.")


descriptors = {}
C_new = []
keep_nmf = []
for s in smiles:
    x=get_ECFP4(s)
    if x is not None:
        C_new.append(x)
        keep_nmf.append(True)
    else:
        keep_nmf.append(False)

C_new = np.array(C_new)

descriptors['ECFP4'] = C_new

# Get dotP descriptors
logging.info("Getting dotP descriptors")
embeddings = np.array([encode_mols(sm, model=compound_encoder) for sm in tqdm(C_new)])
descriptors['dotP'] = embeddings

logging.info("Combining descriptors")
# descriptors['dotP_ECFP4'] = np.concatenate((descriptors['dotP'], descriptors['ECFP4']), axis=1)
# descriptors['dotP400_ECFP4'] = np.concatenate((descriptors['dotP400'], descriptors['ECFP4']), axis=1)
# descriptors['M3_ECFP4'] = np.concatenate((descriptors['M3'], descriptors['ECFP4']), axis=1)
# descriptors['dotPA_ECFP4'] = np.concatenate((descriptors['dotPA'], descriptors['ECFP4']), axis=1)
# descriptors['dotP512_ECFP4'] = np.concatenate((descriptors['dotP512'], descriptors['ECFP4']), axis=1)

# Load Chemprop descriptors
logging.info("Loading Chemprop descriptors")
path = input_file.replace('.csv', '_standardized_chemprop_fingerprints.pkl')
with open(path, 'rb') as f:
    chemprop_descriptors = pickle.load(f)

descriptors['chemprop'] = chemprop_descriptors[mask]

# Model
logging.info("Model training")

from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
auroc_table = pd.DataFrame()
avg_prec_table = pd.DataFrame()

bin_count = np.bincount(y)
logging.info('pos:neg ratio = %d:%d', bin_count[1], bin_count[0])

for d in ['dotP', 'ECFP4', 'chemprop']: #, 'dotP400_ECFP4', 'M3_ECFP4', 'dotPA_ECFP4']:
    logging.info("Processing descriptor: %s", d)
    X = descriptors[d]

    #############
    # CV groups #
    #############
    skf.get_n_splits(X, y)

    ###################################
    # Model training and evalutation  #
    ###################################
    ys, ypreds, yprobs,ids = [],[], [],[]
    aurocs, auprs = [],[]

    ## CV training and evalution
    for train_index, test_index in skf.split(X, y):
        clf = ExtraTreesClassifier(n_estimators=100, random_state=25,criterion='gini',
                            class_weight='balanced',n_jobs=-1)

        clf.fit(X[train_index], y[train_index])

        y_pred = clf.predict(X[test_index])
        y_prob = clf.predict_proba(X[test_index])[:,1]
        aurocs.append(roc_auc_score(y[test_index],y_prob))
        auprs.append(average_precision_score(y[test_index],y_prob))


    auroc_table[d] = pd.DataFrame({'auroc':aurocs+list([np.mean(aurocs)])})
    avg_prec_table[d] = pd.DataFrame({'aupr':auprs+list([np.mean(auprs)])})


logging.info("Save results")
path = input_file.split('/')[:-3]
species = input_file.split('/')[-3].split('_')[0]
output_path = '../../../data/figures_source_data/evaluation_prediction_models_chembl_datasets/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
name = input_file.split('/')[-1].replace('.csv', '')
results_file = os.path.join(output_path, f"{name}_results_standardized_b1_a0.6.pkl")
with open(results_file, 'wb') as f:
    pickle.dump({'n mols': len(smiles), 'pos:neg ratio': bin_count[1] / bin_count[0], 'auroc': auroc_table, 'avg_prec': avg_prec_table}, f)
