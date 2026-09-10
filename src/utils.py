from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import logging
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


## SMILES standardization ##
def standardize_smiles(input_smiles):
    try:
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            return None

        mol = rdMolStandardize.Cleanup(mol)

        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)

        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)

    except Exception as e:
        return None  

## DotP encoder ##
# To load it and predict:
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
        return self.net(x) / 100  # [batch_size, d_proj]

def get_project_root():
    """
    Adjust this depending on where utils.py is located.
    If this file is src/dotp/utils.py, parents[2] should be the repo root.
    """
    return Path(__file__).resolve().parents[2]


DEFAULT_DOTP_ENCODER_PATH = (
    get_project_root()
    / "models"
    / "dotp"
    / "dotP_encoder_alpha0.6_beta1.pt"
)


@lru_cache(maxsize=1)
def load_compound_encoder(
    model_path=str(DEFAULT_DOTP_ENCODER_PATH),
    device="cpu"
):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model.eval()
    return model


def encode_mols(ecfp4, model=None, model_path=None, device="cpu", reduced=False):
    """
    Encode ECFP4 fingerprints using the DotP compound encoder.

    Parameters
    ----------
    ecfp4 : array-like
        Array of ECFP4 fingerprints, shape (n_molecules, n_bits) or (n_bits,).
    model : torch.nn.Module, optional
        Preloaded compound encoder. Recommended when encoding repeatedly.
    model_path : str or Path, optional
        Path to model file. Used only if model is not provided.
    device : str
        "cpu" or "cuda".
    reduced : bool
        If True, return the hidden representation after the first ReLU layer.
        If False, return the final latent representation.
    """

    if model is None:
        if model_path is None:
            model = load_compound_encoder(device=device)
        else:
            model = load_compound_encoder(str(model_path), device=device)

    model = model.to(device)
    model.eval()

    x = torch.tensor(ecfp4, dtype=torch.float32, device=device)

    if x.ndim == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        if reduced:
            # Output after Linear + ReLU
            z = model.net[:2](x)
        else:
            z = model(x)

    return z.cpu().numpy()