from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import sys, pickle
import pandas as pd
import logging
import os
import collections
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root))
from default_parameters import *

parser = argparse.ArgumentParser(description='Binarize fetched pathogen data')
parser.add_argument('--pathogen_code', type=str, help='Pathogen code')
parser.add_argument('--output_dir', type=str, help='Data directory')

##############################
#### ORGANISM TARGET TYPES ###
##############################

args = parser.parse_args()
data_dir = args.output_dir
pathogen_code = args.pathogen_code

# Loading the data
df = pd.read_csv(os.path.join(data_dir, "012_{0}_cleaned.csv".format(pathogen_code)))
print("Considering only organism target types")
print("Before: {0}".format(df.shape))
df = df[df["target_type"] == "ORGANISM"]
print("After: {0}".format(df.shape))
df.drop(columns=["target_type"], inplace=True)
print("Considering only functional assay types")
print("Before: {0}".format(df.shape))
print(df.value_counts("assay_type"))
df = df[df["assay_type"] == "F"]
print("After: {0}".format(df.shape))
df.drop(columns=["assay_type"], inplace=True)

tasks_dir = os.path.join(data_dir, "joined_data_binarization")
if not os.path.exists(tasks_dir):
    os.makedirs(tasks_dir)

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
        return Chem.MolToSmiles(mol, canonical=True)

    except Exception as e:
        return None  # Optionally log or collect errors

def pchembl_binarizer(df, prefix):

    """
    Description:
    Binarizes compound activity data based on pChEMBL thresholds.
    Generates multiple datasets labeling compounds as active (1) or inactive (0)
    for machine learning tasks.

    Input:
    df (pd.DataFrame): Input DataFrame containing 'pchembl_value', 'pchembl_relation',
                    'inchikey', and 'smiles' columns.
    prefix (str): Prefix to label the resulting datasets.

    Output:
    dict: A dictionary of DataFrames with keys indicating the binarization method
        (e.g., 'prefix_pchembl_value_6.0').
        Each DataFrame contains 'inchikey', 'smiles', and binary activity labels.
    """
    
    df = df[df["pchembl_value"].notnull()]
    df = df[df["pchembl_relation"].notnull()]
    data = {}
    for pchembl_cutoff in PCHEMBL_CUTOFFS:
        da = df[df["pchembl_value"] >= pchembl_cutoff]
        da = da[da["pchembl_relation"] != "<"]
        if da.shape[0] < MIN_POSITIVES:
            print("Not enough positives for pchembl cutoff {0}, {1}".format(pchembl_cutoff, da.shape[0]))
            continue
        di = df[df["pchembl_value"] < pchembl_cutoff]
        di = di[di["pchembl_relation"] != ">"]
        actives = [(ik, smi) for ik, smi in da[["inchikey", "smiles"]].values]
        inactives = [(ik, smi) for ik, smi in di[["inchikey", "smiles"]].values]
        data["{0}_pchembl_value_{1}".format(prefix, pchembl_cutoff)] = pd.DataFrame({"inchikey": [x[0] for x in actives] + [x[0] for x in inactives],
                                             "smiles": [x[1] for x in actives] + [x[1] for x in inactives],
                                             "pchembl_value_{0}".format(pchembl_cutoff): [1] * len(actives) + [0] * len(inactives)})
    print("Collected {0} datasets".format(len(data)))
    return data

def percentage_activity_binarizer(df, prefix):

    """
    Description:
    Binarizes compound bioactivity data based on percentage activity thresholds
    and percentiles. Filters and labels compounds as active (1) or inactive (0)
    for classification tasks.

    Input:
    df (pd.DataFrame): DataFrame containing 'standard_value', 'standard_relation',
                    'standard_units', 'direction_flag', 'inchikey', and 'smiles'.
    prefix (str): Prefix to label the resulting binarized datasets.

    Output:
    dict: A dictionary of DataFrames keyed by binarization method
        (e.g., 'prefix_percentage_activity_50', 'prefix_percentage_activity_percentile_10'),
        each with 'inchikey', 'smiles', and binary activity labels.
    """

    df = df[df["standard_value"].notnull()]
    df = df[df["standard_relation"].notnull()]
    df = df[df["standard_units"] == "%"]
    df = df[df["direction_flag"] == 1]
    data = {}
    for percentage_activity_cutoff in PERCENTAGE_ACTIVITY_CUTOFFS:
        da = df[df["standard_value"] >= percentage_activity_cutoff]
        da = da[da["standard_relation"] != "<"]
        if da.shape[0] < MIN_POSITIVES:
            continue
        di = df[df["standard_value"] < percentage_activity_cutoff]
        di = di[di["standard_relation"] != ">"]
        actives = [(ik, smi) for ik, smi in da[["inchikey", "smiles"]].values]
        inactives = [(ik, smi) for ik, smi in di[["inchikey", "smiles"]].values]
        data["{0}_percentage_activity_{1}".format(prefix, percentage_activity_cutoff)] = pd.DataFrame({"inchikey": [x[0] for x in actives] + [x[0] for x in inactives],
                                             "smiles": [x[1] for x in actives] + [x[1] for x in inactives],
                                             "percentage_activity_{0}".format(percentage_activity_cutoff): [1] * len(actives) + [0] * len(inactives)})
    print("Collected {0} datasets".format(len(data)))
    return data

def active_inactive_binarizer(df, prefix):

    """
    Description:
    Creates a binary classification dataset based on labeled activity flags.
    Compounds marked as active (1) or inactive (-1) are retained and labeled
    accordingly for downstream use.

    Input:
    df (pd.DataFrame): DataFrame containing 'activity_flag', 'inchikey', and 'smiles'.
    prefix (str): Prefix used to label the resulting dataset key.

    Output:
    dict: A dictionary with a single DataFrame containing 'inchikey', 'smiles',
        and a binary 'labeled_active' column (1 = active, 0 = inactive).
    """

    df = df[df["activity_flag"] != 0]
    data = {}
    da = df[df["activity_flag"] == 1]
    di = df[df["activity_flag"] == -1]
    actives = [(ik, smi) for ik, smi in da[["inchikey", "smiles"]].values]
    inactives = [(ik, smi) for ik, smi in di[["inchikey", "smiles"]].values]
    data["{0}_labeled_active".format(prefix)] = pd.DataFrame({"inchikey": [x[0] for x in actives] + [x[0] for x in inactives],
                                            "smiles": [x[1] for x in actives] + [x[1] for x in inactives],
                                            "labeled_active": [1] * len(actives) + [0] * len(inactives)})
    print("Collected {0} datasets".format(len(data)))
    return data

def append_data(all_datasets, data):
    for k,v in data.items():
        if k not in all_datasets:
            all_datasets[k] = v
        else:
            raise Exception("Dataset {0} already exists".format(k))
    return all_datasets

# Dataset creation functions

def create_combined_dataset(df, all_datasets, priority):
    assay_ids = [x for x in df.value_counts("assay_id").index]
    counts = [x for x in df.value_counts("assay_id")]
    sel_assay_ids = []
    for aid, count in zip(assay_ids, counts):
        if count >= MIN_SIZE_ASSAY_TASK:
            sel_assay_ids.append(aid)
        else:
            break
    sel_assay_ids = sel_assay_ids[:MAX_NUM_INDEPENDENT_ASSAYS]
    for aid in sel_assay_ids:
        print("Assay ID: {0}".format(aid))
        dt = df[df["assay_id"] == aid]
        activity_types = [x for x in dt.value_counts("standard_type").index]
        counts = [x for x in dt.value_counts("standard_type")]
        sel_activity_types = []
        for at, count in zip(activity_types, counts):
            if count >= MIN_SIZE_ASSAY_SUBTASK:
                sel_activity_types.append(at)
        sel_activity_types = sel_activity_types[:MAX_NUM_ASSAY_SUBTASKS]
        for activity_type in activity_types:
            prefix = "{0}_assay_{1}_{2}".format(priority, aid, activity_type)
            dtt = dt[dt["standard_type"] == activity_type]
            for has_pchembl in [True, False]:
                if has_pchembl:
                    dttp = dtt[dtt["pchembl_value"].notnull()]
                    if dttp.shape[0] < MIN_SIZE_ASSAY_SUBTASK:
                        continue
                    data = pchembl_binarizer(dttp, prefix=prefix)
                    all_datasets = append_data(all_datasets, data)
                else:
                    dttp = dtt[dtt["pchembl_value"].isnull()]
                    if dttp.shape[0] < MIN_SIZE_ASSAY_SUBTASK:
                        continue
                    dttpp = dttp[dttp["standard_units"] == "%"]
                    if dttpp.shape[0] < MIN_SIZE_ASSAY_SUBTASK:
                        continue
                    data = percentage_activity_binarizer(dttpp, prefix=prefix)
                    all_datasets = append_data(all_datasets, data)

    return all_datasets

# Global active / inactive binarizer

def create_datasets_by_active_inactive(df, all_datasets, priority):
    prefix = "{0}_all".format(priority)
    data = active_inactive_binarizer(df, prefix=prefix)
    all_datasets = append_data(all_datasets, data)
    return all_datasets

def create_datasets_by_major_types(df, all_datasets, priority):
    counter = collections.defaultdict(int)
    for v in df[["target_id", "standard_type", "standard_units"]].values:
        counter[(v[0], v[1].lower(), v[2])] += 1
    selected_units = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:MAX_NUM_INDEPENDENT_ASSAYS]
    # print(selected_units)
    for r in selected_units:
        r = r[0]
        target_id = r[0]
        standard_type = r[1]
        standard_units = r[2]
        dt = df[(df["target_id"] == target_id) & (df["standard_type"].str.lower() == standard_type) & (df["standard_units"] == standard_units)]
        prefix = "{0}_target_{1}_{2}_{3}".format(priority, target_id, standard_type.lower(), standard_units.lower())
        for has_pchembl in [True, False]:
            if has_pchembl:
                dtp = dt[dt["pchembl_value"].notnull()]
                if dtp.shape[0] < MIN_SIZE_ASSAY_SUBTASK:
                    continue
                data = pchembl_binarizer(dtp, prefix=prefix)
                all_datasets = append_data(all_datasets, data)
            else:
                dtp = dt[dt["pchembl_value"].isnull()]
                if dtp.shape[0] < MIN_SIZE_ASSAY_SUBTASK:
                    continue
                for has_percentage_activity in [True, False]:
                    if has_percentage_activity:
                        dtpp = dtp[dtp["standard_units"] == "%"]
                        if dtpp.shape[0] < MIN_SIZE_ASSAY_SUBTASK:
                            continue
                        data = percentage_activity_binarizer(dtpp, prefix=prefix)
                        all_datasets = append_data(all_datasets, data)
                    else:
                        dtpp = dtp[dtp["standard_value"].isnull()]
                        if dtpp.shape[0] < MIN_SIZE_ASSAY_SUBTASK:
                            continue
                        data = others_binarizer(dtpp, prefix=prefix)
                        all_datasets = append_data(all_datasets, data)
    return all_datasets 

def create_datasets_by_all_pchembl(df, all_datasets, priority):
    print("Considering only pchembl values")
    dtp = df[df["pchembl_value"].notnull()]
    if dtp.shape[0] < MIN_SIZE_ANY_TASK:
        return all_datasets
    prefix = "{0}_all".format(priority)
    data = pchembl_binarizer(dtp, prefix=prefix)
    all_datasets = append_data(all_datasets, data)
    return all_datasets

def create_datasets_by_all_percentage(df, all_datasets, priority):
    print("Considering only percentage activity")
    dtp = df[df["standard_units"] == "%"]
    dtp = dtp[dtp["standard_value"].notnull()]
    dtp = dtp[dtp["standard_relation"].notnull()]
    dtp = dtp[dtp["direction_flag"] == 1]
    if dtp.shape[0] < MIN_SIZE_ANY_TASK:
        return all_datasets
    prefix = "{0}_all".format(priority)
    data = percentage_activity_binarizer(dtp, prefix=prefix)
    all_datasets = append_data(all_datasets, data)
    return all_datasets

all_datasets = {}
all_datasets = create_datasets_by_all_pchembl(df, all_datasets, priority=10)
all_datasets = create_datasets_by_all_percentage(df, all_datasets, priority=10)
all_datasets = create_datasets_by_active_inactive(df, all_datasets, priority=10)

def disambiguate_data(df):
    ik2smi = {}
    ik2act = collections.defaultdict(list)
    columns = list(df.columns)
    assert len(columns) == 3, "Expected 3 columns"
    for k,v in df[[columns[0], columns[1]]].values:
        ik2smi[k] = v
    for k,v in df[[columns[0], columns[2]]].values:
        ik2act[k] += [v]
    ik2act = {k: int(np.max(v)) for k,v in ik2act.items()}
    R = []
    for k,v in ik2act.items():
        R += [[k, ik2smi[k], v]]
    return pd.DataFrame(R, columns=columns)

all_datasets = {k: disambiguate_data(v) for k,v in all_datasets.items()}
summary_raw_tasks = []

# print("Printing created datasets:")
# print([[i, len(all_datasets[i])] for i in sorted(all_datasets)])

print("Printing tasks before last filtering...")
for dt in sorted(all_datasets):
    l = len(all_datasets[dt])
    columns = list(all_datasets[dt].columns)
    if l != 0:
        ratio = round(sum(all_datasets[dt][columns[2]].tolist()) / l, 3)
    else:
        ratio = 0
    print("{0} -- {1} -- {2}".format(dt, str(l), str(ratio)))

for k,v in all_datasets.items():
    if v.shape[0] < MIN_SIZE_ANY_TASK:
        continue
    columns = list(v.columns)
    assert len(columns) == 3, "Expected 3 columns"
    n = v[columns[2]].sum()
    if n < MIN_POSITIVES:
        continue
    # if n / len(v) < 0.5:
    file_name = os.path.join(tasks_dir, "{0}_ORGANISM.csv".format(k))
    print("Saving data in {0}".format(file_name))
    v.to_csv(file_name, index=False)
    summary_raw_tasks.append([k, 'ORGANISM', len(v), n])

# Joined all
joined_df = pd.DataFrame()
for k,v in all_datasets.items():
    activity_column = list(v.columns)[-1]
    v = v.rename(columns={activity_column: 'active'})
    joined_df = pd.concat([joined_df, v], axis=0)

joined_df.drop_duplicates(inplace=True)

n_jobs = multiprocessing.cpu_count()

s_standardized = Parallel(n_jobs=n_jobs)(
    delayed(standardize_smiles)(smi) for smi in tqdm(joined_df.smiles)
)

joined_df['smiles'] = s_standardized

joined_df = joined_df[joined_df['smiles'].notnull()]

joined_df = joined_df[['smiles', 'active']].drop_duplicates()

# group by smiles and take max active
joined_df = joined_df.groupby('smiles', as_index=False).agg({'active': 'max'})

joined_df.to_csv(os.path.join(tasks_dir, "joined_all_max_activity.csv"), index=False)

# Store summary file
summary_raw_tasks = pd.DataFrame(summary_raw_tasks, columns=["task_id", "target_type", "num_molecules", "num_positives"])
summary_raw_tasks.to_csv(os.path.join(data_dir, "013a_raw_tasks_MOD_summary_Gema.csv"), index=False)