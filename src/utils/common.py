import logging
import os
from zipfile import ZipFile
import yaml
from pathlib import Path

logging.basicConfig(
    filename=os.path.join("Logs", "running.log"),
    format="[%(asctime)s: %(module)s: %(levelname)s]: %(message)s",
    level=logging.INFO,
    filemode="a"
)

def read_yaml(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"config file {config_path} loaded")
    return config

def create_directory(dir_path: list):
    try:
        full_dir_path = ""
        for path in dir_path:
            full_dir_path = os.path.join(full_dir_path, path)
        os.makedirs(full_dir_path, exist_ok=True)
        logging.info(f"directory {dir_path} created")
    except Exception as e:
        logging.exception(e)
        raise e

def unzip_File(source_dir:str, destination_dir:str):
    try:
        with ZipFile(source_dir, 'r') as zipfile:
            zipfile.extractall(destination_dir)
        logging.info(f"file {source_dir} unzipped to {destination_dir}")
    except Exception as e:
        logging.exception(e)
        raise e