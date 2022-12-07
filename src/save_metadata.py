from yaml import safe_load 
import os

# Get The Credentials 
credentials = safe_load(open("src/config.yaml"))

# DVC Credentials
#dvc_config = credentials["dvc_config"]
DVC_REMOTE_URL = credentials["DAGSHUB_REMOTE_URL"]
USERNAME = credentials["MLFLOW_TRACKING_USERNAME"]
PASSWORD = credentials["MLFLOW_TRACKING_PASSWORD"]

# Metadata values
#meta_data = credentials["metadata_path"]

MODEL = meta_data["model_path"]

# Configure DVC
os.system("dvc remote add origin {DVC_REMOTE_URL}")
os.system("dvc remote modify origin --local auth basic")
os.system("dvc remote modify origin --local user {USERNAME}}")
os.system("dvc remote modify origin --local password {PASSWORD}")

# Add Metadata to DVC
os.system("dvc add {MODEL}")
os.system("dvc push")