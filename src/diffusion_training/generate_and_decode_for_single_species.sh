#!/bin/bash
 
echo "$(hostname)"
echo "Runner: $(realpath "$0")"
echo "Args: ${*:-<none>}"

# ---------- Inputs ----------
SPECIES="${1:?Usage: $0 <species> <sampling_steps> <species>}"
CFGSCALE="${2:?Usage: $0 <cfg_scale> <sampling_steps>}"
SAMPLING_STEPS="${3:?Usage: $0 <cfg_scale> <sampling_steps>}"
CKPT="${4:?Usage: $0 <cfg_scale> <sampling_steps> <species> <ckpt_path>}"
OUT_BASE="${5:?Usage: $0 <cfg_scale> <sampling_steps> <species> <ckpt_path> <output_base_dir>}"

echo "CFGSCALE: ${CFGSCALE}"
echo "SAMPLING_STEPS: ${SAMPLING_STEPS}"
echo "SPECIES: ${SPECIES}"

GUIDANCE=""

# Base working dir and script
WDIR=""
GEN_LATENT_FILE="${WDIR}/ldmol_generate_latents.py"
DECODE_LATENT_FILE="${WDIR}decode_latents.py"

# ---------- Unique output per combo ----------
SCFGSCALE=$(printf "%03d" "${CFGSCALE}")
SSAMPLING_STEPS=$(printf "%04d" "${SAMPLING_STEPS}")
DEC_DIR="CfgScale_${SCFGSCALE}_SamplingSteps_${SSAMPLING_STEPS}"

# Put outputs into a per-combo folder 
mkdir -p "${OUT_BASE}"
OUT_DIR_GEN="${OUT_BASE}/${DEC_DIR}"
mkdir -p "${OUT_DIR_GEN}"

OUT_DIR_DEC="${OUT_DIR_GEN}/decoded_molecules"
mkdir -p "${OUT_DIR_DEC}"

OUTPUT_LATENTS="${OUT_DIR_GEN}/generated_latents.npy"
RUN_LOG="${OUT_DIR_GEN}/run.log"

echo "Output dir: ${OUT_DIR_GEN}"
echo "OUTPUT_LATENTS: ${OUTPUT_LATENTS}"
echo "Logging to: ${RUN_LOG}"

# ---------- Launch ----------
{
  date
  echo "CFGSCALE=${CFGSCALE}"
  echo "SAMPLING_STEPS=${SAMPLING_STEPS}"
  echo "CKPT=${CKPT}"
  echo "GUIDANCE=${GUIDANCE}"
  echo "Running: python ${PFILE} ..."
} | tee -a "${RUN_LOG}"

# --- Conda environment ---
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate your_env

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python "${GEN_LATENT_FILE}" \
  --ckpt "${CKPT}" \
  --output_latents_file "${OUTPUT_LATENTS}" \
  --guidance "${GUIDANCE}" --embedding_size 1024 \
  --num_sampling_steps "${SAMPLING_STEPS}" --cfg_scale "${CFGSCALE}" \
  --batch_size 256 --nsample -1 \
  --artificial_latent_size 300 --working_directory "${WDIR}" \
  --samples_per_guidance 450 

MODEL_CKPT=""
VOCAB=""
MODEL_CONFIG=""

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python "${DECODE_LATENT_FILE}" \
  --latents_file "${OUTPUT_LATENTS}" \
  --vocab "${VOCAB}" \
  --save_dir "${OUT_DIR_DEC}" \
  --trained_model "${MODEL_CKPT}" \
  --model_config "${MODEL_CONFIG}" \
  --log_name "${DEC_DIR}" \

echo "Done."
