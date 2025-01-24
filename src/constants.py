from typing import Final  # , Literal

#############################
######   DIRECTORIES   ######
#############################
# project root
BASE_DIR = ""

# directory to read the input data (sentences to translate, 
# e.g. HK-news dataset or opus dataset) from 
BASE_DATA_DIR = f"{BASE_DIR}data/"

# store the fine-tuned models 
FINETUNED_MODELS_DIR = f"{BASE_DIR}finetuned_models/"

# save the FID and intermediate files
FID_DIR = f"{BASE_DIR}fid_dir/"

# store the files with the evaluation metrics of each fine-tuned model
TMP_METRICS_DIR = f"{FID_DIR}tmp_metrics/"

# store intermediate FID feature files  
FID_TMP_FILES_DIR = f"{FID_DIR}fid_tmp_files/"

# contains the PRISM models and property files
PRISM_DIR = f"{BASE_DIR}PRISM/"


###########################################
######   EXPERIMENTAL PARAMS FILES   ######
###########################################
GEN_FID_PARAMS_FILE = f"{BASE_DIR}src/fid/gen_fid_exp_params.yaml"

# file that contains the fine-tuning process details
FINETUNE_EXP_PARAMS_FILE = f"{BASE_DIR}src/finetune/finetune_exp_params.yaml"


################################
######   EXTERNAL TOOLS   ######
################################
PYTHON = "python3.8"
PRISM = ""

WEEK_LENGTH: Final = 7
