import re
from fuzzywuzzy import fuzz, process
from tqdm import tqdm
import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map  # For easy tqdm integration with Pool
import multiprocessing
import logging
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Log job start time
logging.info("Job started")


# Necessary functions
def get_species(name):
    """Cleaning bacterial names in the reference lists from Ensembl"""
    # Eliminar corchetes alrededor del nombre
    cleaned_name = name.replace("[", "").replace("]", "")
    # Eliminar contenido entre paréntesis
    start_index = name.find("(")
    end_index = name.find(")")
    if start_index != -1 and end_index != -1:
        cleaned_name = cleaned_name.replace(name[start_index : end_index + 1], "")
    cleaned_name = cleaned_name.strip()
    try:
        genus_species = cleaned_name.split(" ")[0] + " " + cleaned_name.split(" ")[1]
        return genus_species  # Eliminar espacios al principio y al final
    except:
        print(name)
        return name


def get_short_species(name):
    """Get the abbreviated species name (like E. coli)"""
    return name[0] + ". " + name.split(" ")[1]


def map_bacterial_species(input_species, scorer=fuzz.partial_ratio):
    """Map species to reference names"""
    # Find the closest match in the reference list using fuzzywuzzy's process.extractOne
    matches = process.extract(
        input_species, reference_species_list, scorer=scorer, limit=10
    )
    # You can adjust the similarity threshold as needed (e.g., 80 is a reasonable value)
    if matches[0][1] >= 80:
        # Extract the max similarity value
        max_score = matches[0][1]
        # Keep all the matches that share the highest score
        top_matches = [tup for tup in matches if tup[1] == max_score]
        return top_matches
    else:
        return None


#################
# Load the data #
#################
# Query species
data = pd.read_csv(
    <private_file>, # reaxys raw data. “Antiinfective agent” records from release 221041 of reaxys flat files
    low_memory=False,
)
data = data.dropna(subset=["SPECIE"])
species_list = np.unique(data.SPECIE.values)

# Reference bacterial names
bacterial_species_ref = pd.read_csv(
    # species_EnsemblBacteria.txt from https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/release-56/,
    sep="\t",
)

columns = bacterial_species_ref.columns  # Fix displacement of column names
bacterial_species_ref.reset_index(inplace=True)
bacterial_species_ref = bacterial_species_ref.dropna(axis=1, how="all")
bacterial_species_ref.columns = columns

logging.info("Reference bacterial names")
ref_names_list = [get_species(s) for s in tqdm(bacterial_species_ref["#name"].values)]
ref_names_list += [get_short_species(s) for s in ref_names_list]

# Refernece fungi names
fungi_species_ref = pd.read_csv(
    # species_EnsemblFungi.txt from https://ftp.ensemblgenomes.ebi.ac.uk/pub/fungi/release-58/,
    sep="\t",
)

logging.info("Reference fungi names")
ref_names_list_fungi = [get_species(s) for s in tqdm(fungi_species_ref["Name"].values)]
ref_names_list_fungi += [get_short_species(s) for s in ref_names_list_fungi]
ref_names_list_fungi = np.unique(ref_names_list_fungi)

reference_species_list = list(
    np.unique(list(ref_names_list) + list(ref_names_list_fungi))
)

# Map names
logging.info("# total names to standardize: " + str(len(species_list)))

std_names = process_map(
    map_bacterial_species,
    species_list,
    max_workers=20,
    chunksize=100,
)

std_names_dict = dict(zip(species_list, std_names))

logging.info("names standardized and saved")
