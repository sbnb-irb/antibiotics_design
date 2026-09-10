from xml.parsers.expat import model
import h5py, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import pandas as pd
import multiprocessing
import random
from tqdm import tqdm
import logging, sys, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from rdkit.Chem import Descriptors
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from rdkit.Chem.MolStandardize import rdMolStandardize
import argparse


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

filter_mols = True

parser = argparse.ArgumentParser(description="Example using argparse")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

# Add a flag-style argument: --name
parser.add_argument(
    "--alpha",
    type=float,
    required=True,
    help="reconstruction loss weight"
)
parser.add_argument(
    "--beta",
    type=float,
    required=True,
    help="bioactivity loss weight"
)
parser.add_argument(
    "--epochs",
    type=int,
    required=True,
    help="number of training epochs"
)

parser.add_argument(
    "--species-one-hot",
    type=str2bool,
    default=False,
    required=False,
    help="whether to use one-hot encoding for species",
)

args = parser.parse_args()
onehot_species = args.species_one_hot

logging.info("Loading data...")
with h5py.File(<rx_preprocessed_file_path>, 'r') as f:
    Y = f['A'][:]
    X_chem = f['C'][:]
    if onehot_species:
        logging.info("Using one-hot encoding for species")
        E_bug = np.eye(Y.shape[1], dtype=float)
    else:
        E_bug = f['S'][:]
    species = f['species'][:]
    species = [s.decode('utf-8').replace(' ','').lower() for s in species]
    species = np.array(species)
    smiles = f['smiles'][:]
    smiles = np.array([s.decode('utf-8') for s in smiles])

if filter_mols:
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

    molecular_weights = compute_mw_parallel(smiles, n_processes=cpu_count())
    mask = (molecular_weights < 1000) & (molecular_weights > 50)

    X_chem = X_chem[mask]
    Y = Y[mask]

logging.info(f"# Mols: {X_chem.shape[0]}")
logging.info(f"X shape: {X_chem.shape[1]}")

# Model training and evaluation
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import pandas as pd
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
   
# --- Config ---
n_folds = 5
batch_size = 64
epochs = args.epochs
lr = 1e-3
alpha = args.alpha
beta = args.beta
dropout = 0.2 #float(sys.argv[1]) 
logging.info(f"Alpha: {alpha}")
logging.info(f"Beta: {beta}")
logging.info(f"Epochs: {epochs}")
logging.info(f"Dropout: {dropout}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Data ---
# X_chem = torch.tensor(X_chem, dtype=torch.float32).to(device)  # [n_compounds, d_chem]
Y = torch.tensor(Y, dtype=torch.float32).to(device)            # [n_compounds, n_bugs]
E_bug = torch.tensor(E_bug, dtype=torch.float32).to(device)    # [n_bugs, d_proj]

# Optional: normalize bug embeddings
# E_bug = nn.functional.normalize(E_bug, dim=1)

n_compounds = X_chem.shape[0]

# --- Model ---
class CompoundEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        z = self.net(x) / 100  # [batch_size, d_proj]
        return z
    
class CompoundDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)  # Output should match original X_chem dim
        )

    def forward(self, z):
        return self.net(z)  # Reconstructed X_chem
    
# --- Loss ---
def compute_loss_decoder(X_true, X_recon):
    # Reconstruction loss
    loss = nn.BCEWithLogitsLoss()(X_recon, X_true)
    return loss

def compute_loss_bioactivity(Z, E_bug, Y_true):
    # Bioactivity prediction loss
    Y_pred = torch.matmul(Z, E_bug.T)  # [batch_size, n_bugs]
    loss = nn.BCEWithLogitsLoss()(Y_pred, Y_true)
    return loss

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.min_delta = min_delta

    def step(self, loss):
        if loss < (self.best_loss - self.min_delta):
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False  # Continue training

# --- Get ECFP4 ---
def get_ECFP4(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Return zero vector for invalid SMILES
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return fp

def compute_jaccard_similarity(X_recon, X_true):
    # Jaccard similarity between reconstructed and true X_chem
    X_recon = (X_recon > 0.5).float()  # Binarize reconstruction
    X_true = (X_true > 0.5).float()    # Binarize true values
    intersection = (X_recon * X_true).sum(dim=1)
    union = (X_recon + X_true).clamp(0, 1).sum(dim=1)
    jaccard_similarity = intersection / union
    return jaccard_similarity

with open("../../data/species_chembl_list.txt", "r") as f:
    species_list_chembl = [line.strip() for line in f if line.strip()]
    
chembl_data = pd.read_csv('../../data/figures_source_data/chembl_compound_species_activity_datapoints.csv')
def evaluate_chembl_models():
    results_dict = {}
    for s in species_list_chembl:
        logging.info(f"Evaluating species: {s}")
        ds_species = chembl_data[chembl_data['species'] == s]
        ds_species = ds_species[['species', 'smiles_standardized', 'activity']].drop_duplicates().dropna()
        idx = ds_species.groupby(['species', 'smiles_standardized'])['activity'].idxmax()
        ds_species = ds_species.loc[idx].reset_index(drop=True)
        smiles = ds_species['smiles_standardized'].values
        y_true = ds_species['activity'].values
        # Compute ECFP4
        ecfp4 = np.vstack([get_ECFP4(sm) for sm in tqdm(smiles)])
        ecfp4 = torch.tensor(ecfp4, dtype=torch.float32).to(device)
        if filter_mols:
            molecular_weights = compute_mw_parallel(smiles, n_processes=cpu_count())
            mask = (molecular_weights < 1000) & (molecular_weights > 50)
            smiles = smiles[mask]
            if len(smiles) < 10:
                logging.info(f"Skipping species {s} due to insufficient valid molecules after filtering.")
                continue
            ecfp4 = ecfp4[mask]
            y_true = y_true[mask]
        # Get latent representations
        encoder.eval()
        with torch.no_grad():
            z = encoder(ecfp4)
            y_pred = torch.sigmoid(torch.tensor(z, dtype=torch.float32).to(device) @ E_bug.T).cpu().numpy()[:,np.where(np.array(species) == s)[0]]
            aupr_mm = average_precision_score(y_true, y_pred)
            auroc_mm = roc_auc_score(y_true, y_pred)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier()
        scores = cross_val_score(rf, z.cpu().numpy(), y_true, cv=cv, scoring='average_precision', n_jobs=-1)
        results_dict[s] = {'aupr_mm': aupr_mm, 'auroc_mm': auroc_mm, 'rf_aupr': scores.mean()}
    return results_dict

# --- CV Loop ---
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store results across folds
auroc_scores = []
auprc_scores = []
auprcs_by_cpds = []
aurocs_by_cpds = []
count = 0

cv_results = {}

for fold, (train_idx, test_idx) in enumerate(kf.split(X_chem)):
    logging.info(f"\nFold {fold+1}/{n_folds}")

    # Split
    X_train, Y_train = X_chem[train_idx], Y[train_idx]
    X_test, Y_test = X_chem[test_idx], Y[test_idx]

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Create mask for valid entries
    train_mask = ~torch.isnan(Y_train)
    test_mask = ~torch.isnan(Y_test)

    # Replace NaN with zero (loss uses mask)
    Y_train = torch.nan_to_num(Y_train, nan=0.0)
    Y_test = torch.nan_to_num(Y_test, nan=0.0)

    # Dataloaders
    train_dataset = TensorDataset(X_train, Y_train, train_mask)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Init model
    input_dim = X_train.shape[1]
    latent_dim = E_bug.shape[1]
    encoder = CompoundEncoder(input_dim, latent_dim).to(device)
    decoder = CompoundDecoder(input_dim, latent_dim).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # --- Training ---
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        bio_loss_total = 0
        recon_loss_total = 0
        for x_batch, y_batch, mask_batch in train_loader:
            z_batch = encoder(x_batch)
            x_recon = decoder(z_batch)

            recon_loss = compute_loss_decoder(x_batch, x_recon)
            bio_loss = compute_loss_bioactivity(z_batch, E_bug, y_batch)
            loss = recon_loss * alpha + bio_loss * beta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            bio_loss_total += bio_loss.item()
        logging.info(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Recon Loss: {recon_loss_total/len(train_loader):.4f} | Bio Loss: {bio_loss_total/len(train_loader):.4f}")

        if early_stopping.step(total_loss / len(train_loader)):
            logging.info("Early stopping triggered.")
            break

    # --- Evaluation ---
    encoder.eval()
    decoder.eval()
    logging.info("Evaluating model...")
    with torch.no_grad():
        z_test = encoder(X_test)
        y_pred = torch.sigmoid(z_test @ E_bug.T)

        # Reconstruction
        x_recon = decoder(z_test)
        
    # Evaluate performance
    y_pred = y_pred.cpu().numpy()
    y_test = Y_test.cpu().numpy()
    y_test = np.where(test_mask.cpu().numpy(), y_test, 0)  # Apply mask
    y_pred = np.where(test_mask.cpu().numpy(), y_pred, 0)  # Apply mask
    # Flatten arrays for evaluation
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    # Compute AUROC and AUPR bioactivity
    auroc_bio = roc_auc_score(y_test_flat, y_pred_flat)
    aupr_bio = average_precision_score(y_test_flat, y_pred_flat)

    logging.info(f"AUROC (bioactivity): {auroc_bio:.4f}")
    logging.info(f"AUPR (bioactivity): {aupr_bio:.4f}")

    # logging.info("Calculating AUROC and AUPR by compounds...")
    # for i in range(y_test.shape[0]):
    #     y_true = y_test[i, :]
    #     y_pred_ = y_pred[i, :]
    #     aurocs_by_cpds.append(roc_auc_score(y_true, y_pred_))
    #     auprcs_by_cpds.append(average_precision_score(y_true, y_pred_))

    # logging.info(f"Evaluate reconstruction")
    # jaccard similarity to truth by rows
    jaccard_sim = compute_jaccard_similarity(x_recon, X_test)
    # auroc and auprc
    auroc_recon = roc_auc_score(X_test.flatten(), x_recon.flatten())
    auprc_recon = average_precision_score(X_test.flatten(), x_recon.flatten())
    logging.info(f"Jaccard Similarity: {jaccard_sim.mean():.4f}")
    logging.info(f"AUROC (Reconstruction): {auroc_recon:.4f}")
    logging.info(f"AUPR (Reconstruction): {auprc_recon:.4f}")

    # Store test predictions and y trues for later analysis
    if count == 0:
        y_trues_test = y_test
        test_predictions = y_pred
    else:
        y_trues_test = np.vstack((y_trues_test, y_test))
        test_predictions = np.vstack((test_predictions, y_pred))
    count += 1

    # Evaluate in ChEMBL datasets
    chembl_results = evaluate_chembl_models()
    mean_aupr_mm = np.mean([chembl_results[s]['aupr_mm'] for s in chembl_results])
    mean_auroc_mm = np.mean([chembl_results[s]['auroc_mm'] for s in chembl_results])
    mean_rf_aupr = np.mean([chembl_results[s]['rf_aupr'] for s in chembl_results])
    logging.info(f"ChEMBL Results: Mean AUPR MM: {mean_aupr_mm:.4f}, Mean AUROC MM: {mean_auroc_mm:.4f}, Mean RF AUPR: {mean_rf_aupr:.4f}")

    cv_results[f'fold_{fold+1}'] = {
        'auroc_bio': auroc_bio,
        'aupr_bio': aupr_bio,
        'auroc_recon': auroc_recon,
        'auprc_recon': auprc_recon,
        'jaccard_sim_mean': jaccard_sim.mean(),
        'mean_auroc_direct_chembl': mean_auroc_mm,
        'mean_aupr_direct_chembl': mean_aupr_mm,
        'mean_aupr_rf_chembl': mean_rf_aupr
    }
    
# Save CV results
of = f'/cv_results_alpha{alpha}_beta{beta}_epochs{epochs}.pkl'
if onehot_species:
    of = of.replace('.pkl', '_onehot_species.pkl')
with open(of, 'wb') as f:
    pickle.dump(cv_results, f)
logging.info(f"Cross-validation results saved at {of}")

# logging.info("Calculating bioactivity AUROC and AUPR in the test set...")
# y_true = y_trues_test.flatten()
# y_pred = test_predictions.flatten()
# auroc = roc_auc_score(y_true, y_pred)
# auprc = average_precision_score(y_true, y_pred)
# logging.info(f"AUROC: {auroc:.4f}")
# logging.info(f"AUPR: {auprc:.4f}")

# logging.info("Calculating AUROC and AUPR by species...")
# auroc_by_species = []
# auprc_by_species = []
# for i in range(test_predictions.shape[1]):
#     y_true = y_trues_test[:, i]
#     y_pred = test_predictions[:, i]
#     auroc_by_species.append(roc_auc_score(y_true, y_pred))
#     auprc_by_species.append(average_precision_score(y_true, y_pred))
# Save results by species and compounds
# with open(f'auroc_by_species_alpha{alpha}_beta{beta}_epochs{epochs}.pkl', 'wb') as f:
#     pickle.dump(auroc_by_species, f)
# with open(f'auprc_by_species_alpha{alpha}_beta{beta}_epochs{epochs}.pkl', 'wb') as f:
#     pickle.dump(auprc_by_species, f)
# with open(f'aurocs_by_cpds_alpha{alpha}_beta{beta}_epochs{epochs}.pkl', 'wb') as f: 
#     pickle.dump(aurocs_by_cpds, f)
# with open(f'auprcs_by_cpds_alpha{alpha}_beta{beta}_epochs{epochs}.pkl', 'wb') as f:
#     pickle.dump(auprcs_by_cpds, f)



