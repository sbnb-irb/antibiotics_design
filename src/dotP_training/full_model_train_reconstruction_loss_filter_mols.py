from xml.parsers.expat import model
import h5py, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm
import logging, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from rdkit.Chem import Descriptors
from multiprocessing import Pool, cpu_count
import multiprocessing

from joblib import Parallel, delayed
from rdkit.Chem.MolStandardize import rdMolStandardize
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

onehot_species = True

def standardize_smiles(input_smiles):
    try:
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            return None

        mol = rdMolStandardize.Cleanup(mol)

        # These must be instantiated INSIDE the function
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)

        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    except Exception as e:
        return None  # Optionally log or collect errors

filter_mols = True

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
    species = np.array([s.decode('utf-8') for s in species])
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
batch_size = 64
epochs = 100
lr = 1e-3
alpha = 0.6 #sys.argv[1] 
beta = 1
dropout = 0.2 #float(sys.argv[1]) 
alpha = float(alpha)  # Convert to float
logging.info(f"Alpha: {alpha}")
logging.info(f"Beta: {beta}")
logging.info(f"Epochs: {epochs}")
logging.info(f"Dropout: {dropout}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_name = f"final_model_{alpha}_beta{beta}_epochs{epochs}_dropout{dropout}"

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

def save_activation(name):
    def hook(module, input, output):
        intermediates[name] = output.detach()
    return hook

logging.info(f"Training model with {n_compounds} compounds...")

X_train, Y_train = X_chem, Y
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)  # [n_compounds, d_chem]
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)  # [n_compounds, n_bugs]

# Dataloaders
train_dataset = TensorDataset(X_train, Y_train)
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
    for x_batch, y_batch in train_loader:
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

# --- Save model ---
torch.save(encoder, <your_path>)

# --- Evaluation in training set ---
encoder.eval()
decoder.eval()
logging.info("Evaluating model...")
with torch.no_grad():
    z_train = encoder(X_train)
    y_pred = torch.sigmoid(z_train @ E_bug.T)

    # Reconstruction
    x_recon = decoder(z_train)
        
    # Evaluate performance
    y_pred = y_pred.cpu().numpy()
    y_test = Y_train.cpu().numpy()
    # Flatten arrays for evaluation
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    # Compute AUROC and AUPR
    auroc = roc_auc_score(y_test_flat, y_pred_flat)
    aupr = average_precision_score(y_test_flat, y_pred_flat)

    logging.info(f"AUROC: {auroc:.4f}")
    logging.info(f"AUPR: {aupr:.4f}")

    logging.info(f"Evaluate reconstruction")
    # jaccard similarity to truth by rows
    jaccard_sim = compute_jaccard_similarity(x_recon, X_train)
    # auroc and auprc
    auroc_recon = roc_auc_score(X_train.flatten(), x_recon.flatten())
    auprc_recon = average_precision_score(X_train.flatten(), x_recon.flatten())
    logging.info(f"Jaccard Similarity: {jaccard_sim.mean():.4f}")
    logging.info(f"AUROC (Reconstruction): {auroc_recon:.4f}")
    logging.info(f"AUPR (Reconstruction): {auprc_recon:.4f}")


