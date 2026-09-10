import sys
import os
import h5py
import joblib
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from src.utils import encode_mols, load_compound_encoder

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Check Arguments ---
if len(sys.argv) < 2:
    logging.error("Usage: python script.py <species_name>")
    sys.exit(1)

target_species = sys.argv[1]
device = 'cpu'

# --- 1. Load Data & Pretrained Encoder ---
# --- Get ECFP4 ---
def get_ECFP4(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Return zero vector for invalid SMILES
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return fp

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

onehot_species = False
with h5py.File(<rx_preprocessed_file_path>, 'r') as f:
    species = f['species'][:]
    species = [s.decode('utf-8').replace(' ','').lower() for s in species]
    if onehot_species:
        logging.info("Using one-hot encoding for species")
        E_bug = np.eye(Y.shape[1], dtype=float)
    else:
        E_bug = f['S'][:]

filter_mols=True
compound_encoder = load_compound_encoder()
compound_encoder.eval()

chembl_data = pd.read_csv('../../../data/figures_source_data/chembl_compound_species_activity_datapoints.csv')

logging.info(f"Starting pipeline for species: {target_species}")

# Filter species data
ds_species = chembl_data[chembl_data['species'] == target_species]
if ds_species.empty:
    logging.error(f"No data found for species: {target_species}")
    sys.exit(1)

smiles = ds_species['smiles'].values
y = ds_species['active'].values

# Compute ECFP4 and filter by MW
# [MW Filtering logic here as in your code]
ecfp4_bits = np.vstack([get_ECFP4(sm) for sm in tqdm(smiles)])
X_ecfp4 = torch.tensor(ecfp4_bits, dtype=torch.float32).to(device)
if True:
    molecular_weights = compute_mw_parallel(smiles, n_processes=cpu_count())
    mask = (molecular_weights < 1000) & (molecular_weights > 50)
    smiles = smiles[mask]
    ecfp4_bits = ecfp4_bits[mask]
    X_ecfp4 = X_ecfp4[mask]
    y = y[mask]

# Get Latent Embeddings (Z)
X_latent = encode_mols(X_ecfp4, model=compound_encoder)

# --- 2. Define Model Comparison Dictionary ---
models_to_test = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True, random_state=42)
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 3. Manual CV Loop for Comparison ---
for model_name, model_obj in models_to_test.items():
    logging.info(f"Testing model: {model_name}")
    
    # We test on both Latent Embeddings (Z) and raw ECFP4
    for feature_type, X_data in [("Latent", X_latent), ("ECFP4", ecfp4_bits)]:
        
        fold_aupr, fold_auroc = [], []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_data, y)):
            X_tr, X_te = X_data[train_idx], X_data[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            # Train
            model_obj.fit(X_tr, y_tr)
            
            # Save the fold model
            model_dir = f"your_dir/{target_species}/{model_name}_{feature_type}" # models are saved at /models/activity_predictors
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(model_obj, f"{model_dir}/fold_{fold}.joblib")
            
            # Evaluate
            probs = model_obj.predict_proba(X_te)[:, 1]
            fold_aupr.append(average_precision_score(y_te, probs))
            fold_auroc.append(roc_auc_score(y_te, probs))
            
        # Store Aggregated Results
        results.append({
            'species': target_species,
            'model': model_name,
            'features': feature_type,
            'mean_aupr': np.mean(fold_aupr),
            'mean_auroc': np.mean(fold_auroc)
        })

# --- 4. Final Export ---
results_df = pd.DataFrame(results)
results_df.to_csv(f"{target_species}/comparison_results.csv", index=False)
logging.info(f"Process complete. Results saved in ./results/{target_species}/")