import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm.auto import tqdm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CHEMBL_DATA_PATH = (
    "../../antibiotics_design/data/figures_source_data/"
    "chembl_compound_species_activity_datapoints.csv"
)

ENCODER_PATH = (
    "../../models/dotP/dotP_encoder_alpha0.6_beta1.pt"
)

MODELS_ROOT = (
    "../../models/activity_predictors/"
)


# ---------------------------------------------------------------------
# Select the species
# ---------------------------------------------------------------------

# Option 1: provide the species as a command-line argument:
# python evaluate_saved_models.py escherichiacoli

if len(sys.argv) < 2:
    raise ValueError(
        "Provide the species name as a command-line argument, for example:\n"
        "python evaluate_saved_models.py escherichiacoli"
    )

target_species = sys.argv[1]

# Option 2: alternatively, comment out the block above and define it manually:
# target_species = "escherichiacoli"


# ---------------------------------------------------------------------
# ECFP4 calculation
# ---------------------------------------------------------------------

def get_ecfp4(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    """
    Calculate a 2,048-bit ECFP4 fingerprint.

    ECFP4 corresponds to a Morgan fingerprint with radius 2.
    """
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=n_bits,
    )

    return np.asarray(fingerprint, dtype=np.float32)


def compute_molecular_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return np.nan

    return Descriptors.MolWt(mol)


def compute_molecular_weights(
    smiles_list,
    n_processes: int | None = None,
) -> np.ndarray:
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    with Pool(n_processes) as pool:
        molecular_weights = pool.map(
            compute_molecular_weight,
            smiles_list,
        )

    return np.asarray(molecular_weights, dtype=float)


# ---------------------------------------------------------------------
# dotP compound encoder
# ---------------------------------------------------------------------

class CompoundEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.net(x) / 100


compound_encoder = torch.load(
    ENCODER_PATH,
    map_location="cpu",
    weights_only=False,
)

compound_encoder.eval()


# ---------------------------------------------------------------------
# Load and filter the species-specific ChEMBL data
# ---------------------------------------------------------------------

chembl_data = pd.read_csv(CHEMBL_DATA_PATH)

species_df = chembl_data.loc[
    chembl_data["species"] == target_species
].copy()

if species_df.empty:
    raise ValueError(
        f"No ChEMBL data found for species: {target_species}"
    )

# Preserve the original row order used by the training script.
smiles = species_df["smiles"].astype(str).to_numpy()
y = species_df["active"].astype(int).to_numpy()

print(f"Species: {target_species}")
print(f"Initial molecules: {len(smiles)}")
print(f"Initial actives: {y.sum()}")
print(f"Initial inactives: {len(y) - y.sum()}")


# ---------------------------------------------------------------------
# Generate ECFP4 fingerprints
# ---------------------------------------------------------------------

fingerprints = [
    get_ecfp4(smiles_value)
    for smiles_value in tqdm(
        smiles,
        desc=f"Calculating ECFP4 for {target_species}",
    )
]

valid_smiles_mask = np.asarray([
    fingerprint is not None
    for fingerprint in fingerprints
])

if not valid_smiles_mask.all():
    print(
        f"Removing {(~valid_smiles_mask).sum()} molecules "
        "with invalid SMILES."
    )

smiles = smiles[valid_smiles_mask]
y = y[valid_smiles_mask]

ecfp4_bits = np.vstack([
    fingerprint
    for fingerprint in fingerprints
    if fingerprint is not None
]).astype(np.float32)


# ---------------------------------------------------------------------
# Apply the same molecular-weight filter used during training
# ---------------------------------------------------------------------

molecular_weights = compute_molecular_weights(
    smiles,
    n_processes=cpu_count(),
)

molecular_weight_mask = (
    np.isfinite(molecular_weights)
    & (molecular_weights > 50)
    & (molecular_weights < 1000)
)

smiles = smiles[molecular_weight_mask]
y = y[molecular_weight_mask]
ecfp4_bits = ecfp4_bits[molecular_weight_mask]
molecular_weights = molecular_weights[molecular_weight_mask]

print(f"Molecules after filtering: {len(smiles)}")
print(f"Actives after filtering: {y.sum()}")
print(f"Inactives after filtering: {len(y) - y.sum()}")


# ---------------------------------------------------------------------
# Generate dotP latent representations
# ---------------------------------------------------------------------

X_ecfp4_tensor = torch.from_numpy(ecfp4_bits).float()

with torch.inference_mode():
    X_latent = (
        compound_encoder(X_ecfp4_tensor)
        .cpu()
        .numpy()
    )


# ---------------------------------------------------------------------
# Variables required for evaluating the saved CV models
# ---------------------------------------------------------------------

print("ecfp4_bits shape:", ecfp4_bits.shape)
print("X_latent shape:", X_latent.shape)
print("y shape:", y.shape)

assert len(ecfp4_bits) == len(X_latent) == len(y)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL_TYPES = [
    "RandomForest",
    "XGBoost",
    "LogisticRegression",
    "MLP",
]

FEATURE_SETS = {
    "Latent": X_latent,
    "ECFP4": ecfp4_bits,
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

fold_results = []

n_molecules = len(y)

# Stores one out-of-fold prediction vector per model/feature combination.
oof_predictions = {}

for model_name in MODEL_TYPES:
    for feature_type, X_data in FEATURE_SETS.items():
        combination_name = f"{model_name}_{feature_type}"

        combination_oof = np.full(
            n_molecules,
            np.nan,
            dtype=float,
        )

        model_dir = os.path.join(
            MODELS_ROOT,
            target_species,
            f"{model_name}_{feature_type}",
        )

        for fold, (_, test_idx) in enumerate(cv.split(X_data, y)):

            model_path = os.path.join(
                model_dir,
                f"fold_{fold}.joblib",
            )

            if not os.path.isfile(model_path):
                print(f"Missing model: {model_path}")
                continue

            classifier = joblib.load(model_path)

            X_test = X_data[test_idx]
            y_test = y[test_idx]

            probabilities = classifier.predict_proba(X_test)

            class_labels = np.asarray(classifier.classes_)
            active_positions = np.where(class_labels == 1)[0]

            if len(active_positions) != 1:
                raise ValueError(
                    f"Could not identify active class in {model_path}. "
                    f"Classes: {class_labels}"
                )

            y_score = probabilities[:, active_positions[0]]

            fold_results.append({
                "species": target_species,
                "model": model_name,
                "features": feature_type,
                "fold": fold,
                "n_test": len(test_idx),
                "n_active_test": int(y_test.sum()),
                "active_fraction_test": float(y_test.mean()),
                "average_precision": average_precision_score(
                    y_test,
                    y_score,
                ),
                "auroc": roc_auc_score(
                    y_test,
                    y_score,
                ),
            })

            # Store predictions only for the held-out molecules in this fold.
            combination_oof[test_idx] = y_score

        if np.isnan(combination_oof).any():
            missing_count = np.isnan(combination_oof).sum()
            raise RuntimeError(
                f"{combination_name} is missing "
                f"{missing_count} out-of-fold predictions."
            )

        oof_predictions[combination_name] = combination_oof

fold_results_df = pd.DataFrame(fold_results)



output_path = os.path.join(
    MODELS_ROOT,
    target_species,
    "fold_metrics.csv",
)

fold_results_df.to_csv(output_path, index=False)

print(fold_results_df)
print(f"\nSaved fold-level metrics to: {output_path}")

# ---------------------------------------------------------------------
# Calculate consensus out-of-fold scores
# ---------------------------------------------------------------------

expected_combinations = len(MODEL_TYPES) * len(FEATURE_SETS)

if len(oof_predictions) != expected_combinations:
    raise RuntimeError(
        f"Expected {expected_combinations} model/feature combinations, "
        f"but obtained {len(oof_predictions)}."
    )

oof_matrix = np.column_stack([
    oof_predictions[
        f"{model_name}_{feature_type}"
    ]
    for model_name in MODEL_TYPES
    for feature_type in FEATURE_SETS
])

consensus_oof_score = oof_matrix.mean(axis=1)

oof_df = pd.DataFrame({
    "species": target_species,
    "smiles": smiles,
    "active": y.astype(int),
    "consensus_score": consensus_oof_score,
})

# Keep the individual model scores as well.
for combination_name, scores in oof_predictions.items():
    oof_df[combination_name] = scores

oof_output_path = os.path.join(
    MODELS_ROOT,
    target_species,
    "out_of_fold_consensus_scores.csv",
)

oof_df.to_csv(oof_output_path, index=False)

print(f"Saved out-of-fold predictions to: {oof_output_path}")