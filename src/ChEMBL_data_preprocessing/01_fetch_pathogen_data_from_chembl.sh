#!/bin/bash
# Initialize conda
eval "$(conda shell.bash hook)"
conda activate /aloy/home/grojas/anaconda3/envs/ersilia

set -e  # Exit immediately if any command fails

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Function to display usage
usage() {
    echo "Usage: $0 --pathogen_code <code> --output_dir <directory> [--organism | --protein]"
    echo "  --pathogen_code   Specify the pathogen code (e.g., abaumannii)"
    echo "  --output_dir      Specify the output directory"
    echo "  --organism        Specify organism-level task"
    echo "  --protein         Specify protein-level task"
    exit 1
}

# Check for arguments
if [ $# -lt 5 ]; then
    echo "Error: Insufficient arguments."
    usage
fi

# Initialize variables
PATHOGEN_CODE=""
OUTPUT_DIR=""
TASK_TYPE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pathogen_code)
            PATHOGEN_CODE=$2
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --organism)
            if [ -n "$TASK_TYPE" ]; then
                echo "Error: Cannot specify both --organism and --protein."
                usage
            fi
            TASK_TYPE="organism"
            shift
            ;;
        --protein)
            if [ -n "$TASK_TYPE" ]; then
                echo "Error: Cannot specify both --organism and --protein."
                usage
            fi
            TASK_TYPE="protein"
            shift
            ;;
        *)
            echo "Error: Unknown argument $1"
            usage
            ;;
    esac
done

# Validate arguments
if [ -z "$PATHOGEN_CODE" ]; then
    echo "Error: --pathogen_code is required."
    usage
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output_dir is required."
    usage
fi

if [ -z "$TASK_TYPE" ]; then
    echo "Error: You must specify either --organism or --protein."
    usage
fi

echo "Fetching data for pathogen $PATHOGEN_CODE and storing in $OUTPUT_DIR"
[ -d "folder_name" ] || mkdir -p $OUTPUT_DIR

if ls "$OUTPUT_DIR"/012_* &>/dev/null; then
    echo "Data is already in $OUTPUT_DIR. Jumping to binarisation."
else
    echo "No cleaned data found. Downloading and cleaning..."
    python $SCRIPT_DIR/../src/011_pathogen_getter.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR 
    python $SCRIPT_DIR/../src/012_clean_fetched_pathogen_data.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR
fi

# If after running the script the file STILL doesn't exist, remove the directory
if ! ls "$OUTPUT_DIR"/012_* &>/dev/null; then
    echo "Error: No data was found. Removing $OUTPUT_DIR..."
    rm -rf "$OUTPUT_DIR"
    exit 1
fi

# # This if loop needs to be removed for normal use
# # I'm using it because of some problems in which I had to run the script in 2 parts
# if ls "$OUTPUT_DIR"/joined_data_binarization &>/dev/null; then
#     echo "Already binarised $OUTPUT_DIR..."
#     exit 1
# fi

# Conditionally run 013a or 013b based on task type
if [ "$TASK_TYPE" == "organism" ]; then
    # python $SCRIPT_DIR/../src/013a_binarize_fetched_pathogen_data_ORG.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR
    python $SCRIPT_DIR/../src/013a_binarize_fetched_pathogen_data_ORG_modified.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR
    # if you want to downlaod  all the individual datasets for dotP evaluation, use the original 013a_binarize_fetched_pathogen_data_ORG.py
elif [ "$TASK_TYPE" == "protein" ]; then
    python $SCRIPT_DIR/../src/013b_binarize_fetched_pathogen_data_SP.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR
fi


# python $SCRIPT_DIR/../src/014_datasets_modelability.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR --$TASK_TYPE  # Task type is organism or protein
# python $SCRIPT_DIR/../src/015_datasets_distinguishability.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR --$TASK_TYPE  # Task type is organism or protein
# python $SCRIPT_DIR/../src/016_select_tasks_MOD_DIS.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR
# python $SCRIPT_DIR/../src/017_select_tasks_RED.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR --$TASK_TYPE  # Task type is organism or protein
# python $SCRIPT_DIR/../src/018_calculate_correlations.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR --$TASK_TYPE  # Task type is organism or protein


# python $SCRIPT_DIR/../src/017_wrapup_tasks_and_clean_output_folder.py --pathogen_code $PATHOGEN_CODE --output_dir $OUTPUT_DIR