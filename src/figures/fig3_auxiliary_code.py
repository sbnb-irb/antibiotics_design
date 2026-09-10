#!/usr/bin/env python

import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


# ============================================================
# Suppress RDKit / warning noise
# ============================================================
RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


# ============================================================
# Parameters
# ============================================================
N_CPUS = 56

BASE_DIRS = [
    "../../data/generated_molecules/extra900",
    "../../data/generated_molecules/initial100",
]

DATASET_PATH = (
    "../../data/figures_source_data/chembl_compound_species_activity_datapoints.csv"
)

MABSCESSUS_ACTIVE_PATH = (
    "../../data/figures_source_data/mycobacteroidesabscessus_chembl_data.csv"
)

OUTDIR = Path("../../data/figures_source_data/")

CFG = 35
SAMPLING_STEPS = 100
CFG_STR = str(CFG).zfill(3)
STEPS_STR = str(SAMPLING_STEPS).zfill(4)

FP_RADIUS = 2
FP_NBITS = 2048


# ============================================================
# Helper functions
# ============================================================
def standardize_smiles(input_smiles):
    try:
        if pd.isna(input_smiles):
            return None

        input_smiles = str(input_smiles).strip()

        if input_smiles == "" or input_smiles.lower() in {"nan", "none"}:
            return None

        mol = Chem.MolFromSmiles(input_smiles)

        if mol is None:
            return None

        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)

        Chem.SanitizeMol(mol)

        return Chem.MolToSmiles(mol, canonical=True)

    except Exception:
        return None


def smiles_to_fp(smi, radius=FP_RADIUS, n_bits=FP_NBITS):
    try:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            return None

        return AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=n_bits,
        )

    except Exception:
        return None


def get_species_from_dirs(base_dirs):
    species = set()

    for base_dir in base_dirs:
        base_dir = Path(base_dir)

        if not base_dir.exists():
            print(f"Base dir not found: {base_dir}")
            continue

        for p in base_dir.iterdir():
            if p.is_dir():
                species.add(p.name)

    return sorted(species)


def find_generation_files(species):
    files = []

    for base_dir in BASE_DIRS:
        base_dir = Path(base_dir)

        gen_dir = (
            base_dir
            / species
            / f"CfgScale_{CFG_STR}_SamplingSteps_{STEPS_STR}"
        )

        latent_path = gen_dir / "generated_latents.npy"
        decoded_path = (
            gen_dir
            / "decoded_molecules"
            / "generated_latents.npy_decoded_latents.csv"
        )

        if latent_path.exists() and decoded_path.exists():
            files.append({
                "species": species,
                "base_dir": str(base_dir),
                "latent_path": str(latent_path),
                "decoded_path": str(decoded_path),
            })

    return files


def closest_intraset_tanimoto(fps):
    n = len(fps)

    if n <= 1:
        return [np.nan] * n, [None] * n

    closest_sims = []
    closest_idx = []

    for i, fp in enumerate(fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps), dtype=float)
        sims[i] = -1.0

        j = int(np.argmax(sims))

        closest_sims.append(float(sims[j]))
        closest_idx.append(j)

    return closest_sims, closest_idx


def closest_to_reference_tanimoto(query_fps, ref_fps):
    if len(ref_fps) == 0:
        return [np.nan] * len(query_fps), [None] * len(query_fps)

    closest_sims = []
    closest_idx = []

    for fp in query_fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, ref_fps), dtype=float)
        j = int(np.argmax(sims))

        closest_sims.append(float(sims[j]))
        closest_idx.append(j)

    return closest_sims, closest_idx


# ============================================================
# Parallel active standardization
# ============================================================
def load_active_smiles():
    print("Loading active molecules...")

    data = pd.read_csv(DATASET_PATH)

    active_df_main = (
        data.loc[data["active"] == 1, ["species", "smiles"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    mabs_df = pd.read_csv(MABSCESSUS_ACTIVE_PATH)

    mabs_active_df = (
        mabs_df.loc[mabs_df["active"] == 1, ["smiles"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    mabs_active_df["species"] = "mycobacteroidesabscessus"

    active_df = pd.concat(
        [
            active_df_main,
            mabs_active_df[["species", "smiles"]],
        ],
        ignore_index=True,
    )

    active_df = active_df.drop_duplicates(["species", "smiles"]).reset_index(drop=True)

    print(f"Raw active species-molecule pairs: {active_df.shape[0]:,}")

    return active_df


def standardize_unique_smiles_parallel(smiles_list):
    unique_smiles = pd.Series(smiles_list).dropna().drop_duplicates().tolist()

    print(f"Standardizing unique SMILES: {len(unique_smiles):,}")

    std_lookup = {}

    with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        futures = {
            executor.submit(standardize_smiles, smi): smi
            for smi in unique_smiles
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Standardizing SMILES"):
            smi = futures[future]

            try:
                std_lookup[smi] = future.result()
            except Exception:
                std_lookup[smi] = None

    return std_lookup


def build_active_fps_by_species(active_df):
    active_std_lookup = standardize_unique_smiles_parallel(active_df["smiles"].tolist())

    active_df["smiles_std"] = active_df["smiles"].map(active_std_lookup)

    active_df = (
        active_df
        .dropna(subset=["smiles_std"])
        .drop_duplicates(["species", "smiles_std"])
        .reset_index(drop=True)
    )

    print(f"Standardized active species-molecule pairs: {active_df.shape[0]:,}")

    active_df.to_csv(OUTDIR / "fig3_active_molecules_standardized.csv", index=False)

    unique_active_smiles = active_df["smiles_std"].drop_duplicates().tolist()

    print(f"Computing fingerprints for unique active molecules: {len(unique_active_smiles):,}")

    active_fp_lookup = {}

    for smi in tqdm(unique_active_smiles, desc="Computing active FPs"):
        fp = smiles_to_fp(smi)
        if fp is not None:
            active_fp_lookup[smi] = fp

    active_fps_by_species = {}

    for species, df_sp in active_df.groupby("species"):
        smiles = []
        fps = []

        for smi in df_sp["smiles_std"].drop_duplicates():
            fp = active_fp_lookup.get(smi)

            if fp is not None:
                smiles.append(smi)
                fps.append(fp)

        active_fps_by_species[species] = {
            "smiles": smiles,
            "fps": fps,
        }

    print(f"Species with active FPs: {len(active_fps_by_species):,}")

    return active_fps_by_species


# ============================================================
# Globals for worker processes
# ============================================================
ACTIVE_FPS_BY_SPECIES = None


def init_worker(active_fps_by_species):
    global ACTIVE_FPS_BY_SPECIES

    RDLogger.DisableLog("rdApp.*")
    rdBase.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore")

    ACTIVE_FPS_BY_SPECIES = active_fps_by_species


# ============================================================
# Per-species worker
# ============================================================
def process_species(species):
    generation_files = find_generation_files(species)

    if len(generation_files) == 0:
        return None, None

    decoded_rows = []
    species_generated = 0

    for file_info in generation_files:
        latent_path = Path(file_info["latent_path"])
        decoded_path = Path(file_info["decoded_path"])

        try:
            latents = np.load(latent_path, mmap_mode="r")
            n_generated = latents.shape[0]
            species_generated += n_generated

            decoded_df = pd.read_csv(decoded_path)

            if "decoded_smiles" not in decoded_df.columns:
                continue

            tmp = decoded_df[["decoded_smiles"]].copy()
            tmp["species"] = species
            #tmp["source_dir"] = file_info["base_dir"]
            tmp["decoded_index"] = np.arange(len(tmp))

            decoded_rows.append(tmp)

        except Exception:
            continue

    if len(decoded_rows) == 0:
        return None, None

    gen_df = pd.concat(decoded_rows, ignore_index=True)

    gen_df["smiles_std"] = [
        standardize_smiles(s)
        for s in gen_df["decoded_smiles"]
    ]

    n_decoded_raw = len(gen_df)

    gen_df = (
        gen_df
        .dropna(subset=["smiles_std"])
        .reset_index(drop=True)
    )

    n_decoded_valid = len(gen_df)
    n_unique_valid = gen_df["smiles_std"].nunique()

    pct_decoded = (
        100 * n_decoded_valid / species_generated
        if species_generated > 0
        else np.nan
    )

    pct_unique_among_decoded = (
        100 * n_unique_valid / n_decoded_valid
        if n_decoded_valid > 0
        else np.nan
    )

    pct_unique_among_generated = (
        100 * n_unique_valid / species_generated
        if species_generated > 0
        else np.nan
    )

    summary_row = {
        "species": species,
        "n_files": len(generation_files),
        "n_generated": species_generated,
        "n_decoded_raw": n_decoded_raw,
        "n_decoded_valid": n_decoded_valid,
        "n_standardization_failed": n_decoded_raw - n_decoded_valid,
        "n_unique_valid": n_unique_valid,
        "pct_decoded": pct_decoded,
        "pct_unique_among_decoded": pct_unique_among_decoded,
        "pct_unique_among_generated": pct_unique_among_generated,
    }

    # Fingerprints for generated molecules
    gen_fps = []
    valid_fp_mask = []

    for smi in gen_df["smiles_std"]:
        fp = smiles_to_fp(smi)

        if fp is None:
            gen_fps.append(None)
            valid_fp_mask.append(False)
        else:
            gen_fps.append(fp)
            valid_fp_mask.append(True)

    gen_df = gen_df.loc[valid_fp_mask].reset_index(drop=True)
    gen_fps = [fp for fp in gen_fps if fp is not None]

    if len(gen_df) == 0:
        return summary_row, None

    # Closest generated molecule within the same species
    intra_sims, intra_idx = closest_intraset_tanimoto(gen_fps)

    gen_df["closest_intraset_tanimoto"] = intra_sims
    gen_df["closest_intraset_index"] = intra_idx
    gen_df["closest_intraset_smiles"] = [
        gen_df.loc[j, "smiles_std"] if j is not None else None
        for j in intra_idx
    ]

    # Closest active molecule for that species
    active_info = ACTIVE_FPS_BY_SPECIES.get(species, {"smiles": [], "fps": []})
    active_smiles = active_info["smiles"]
    active_fps = active_info["fps"]

    active_sims, active_idx = closest_to_reference_tanimoto(gen_fps, active_fps)

    gen_df["closest_active_tanimoto"] = active_sims
    gen_df["closest_active_index"] = active_idx
    gen_df["closest_active_smiles"] = [
        active_smiles[j] if j is not None else None
        for j in active_idx
    ]

    gen_df["n_generated_valid_for_species"] = len(gen_df)
    gen_df["n_active_reference_for_species"] = len(active_fps)

    return summary_row, gen_df


# ============================================================
# Main
# ============================================================
def main():
    print("Starting Fig. 3 generation metrics script")
    print(f"Using {N_CPUS} CPUs")
    print(f"Output directory: {OUTDIR}")

    all_species = get_species_from_dirs(BASE_DIRS)
    print(f"Species found in generation folders: {len(all_species):,}")

    # Actives
    active_df = load_active_smiles()
    active_fps_by_species = build_active_fps_by_species(active_df)

    # Generated molecules and Tanimoto metrics
    summary_rows = []
    tanimoto_results = []
    errors = []

    with ProcessPoolExecutor(
        max_workers=N_CPUS,
        initializer=init_worker,
        initargs=(active_fps_by_species,),
    ) as executor:

        futures = {
            executor.submit(process_species, species): species
            for species in all_species
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing species"):
            species = futures[future]

            try:
                summary_row, gen_df = future.result()

                if summary_row is not None:
                    summary_rows.append(summary_row)

                if gen_df is not None and len(gen_df) > 0:
                    tanimoto_results.append(gen_df)

            except Exception as e:
                errors.append(f"{species}: {repr(e)}")

    generation_summary = pd.DataFrame(summary_rows)

    if len(generation_summary) > 0:
        generation_summary = generation_summary.sort_values("species").reset_index(drop=True)

    tanimoto_results_df = (
        pd.concat(tanimoto_results, ignore_index=True)
        if len(tanimoto_results) > 0
        else pd.DataFrame()
    )

    # Species-level Tanimoto summary
    if len(tanimoto_results_df) > 0:
        species_tanimoto_summary = (
            tanimoto_results_df
            .groupby("species")
            .agg(
                n_generated_valid=("smiles_std", "count"),
                n_active_reference=("n_active_reference_for_species", "first"),
                mean_closest_intraset_tanimoto=("closest_intraset_tanimoto", "mean"),
                median_closest_intraset_tanimoto=("closest_intraset_tanimoto", "median"),
                std_closest_intraset_tanimoto=("closest_intraset_tanimoto", "std"),
                mean_closest_active_tanimoto=("closest_active_tanimoto", "mean"),
                median_closest_active_tanimoto=("closest_active_tanimoto", "median"),
                std_closest_active_tanimoto=("closest_active_tanimoto", "std"),
            )
            .reset_index()
        )
    else:
        species_tanimoto_summary = pd.DataFrame()

    # Save outputs
    generation_summary.to_csv(
        OUTDIR / "fig3_generation_decoding_uniqueness_summary.csv",
        index=False,
    )

    tanimoto_results_df.to_csv(
        OUTDIR / "fig3_generated_closest_intraset_and_active_tanimoto_per_molecule.csv",
        index=False,
    )

    species_tanimoto_summary.to_csv(
        OUTDIR / "fig3_generated_closest_tanimoto_species_summary.csv",
        index=False,
    )

    if len(errors) > 0:
        pd.Series(errors).to_csv(
            OUTDIR / "fig3_generation_metrics_errors.txt",
            index=False,
            header=False,
        )

    print("\nDone.")
    print(f"Saved: {OUTDIR / 'fig3_generation_decoding_uniqueness_summary.csv'}")
    print(f"Saved: {OUTDIR / 'fig3_generated_closest_intraset_and_active_tanimoto_per_molecule.csv'}")
    print(f"Saved: {OUTDIR / 'fig3_generated_closest_tanimoto_species_summary.csv'}")

    if len(errors) > 0:
        print(f"Errors/warnings saved: {OUTDIR / 'fig3_generation_metrics_errors.txt'}")
        print(f"Number of errors/warnings: {len(errors)}")


if __name__ == "__main__":
    main()