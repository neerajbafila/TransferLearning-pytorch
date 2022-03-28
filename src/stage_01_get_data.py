import logging
import os

from src.utils.common import read_yaml, create_directory
from src.utils.data_mgmt import download_data
import argparse

STAGE = "stage_01_get_data"
logging.basicConfig(
    filename=os.path.join("Logs",'running.log'),
    format="[%(asctime)s:%(module)s:%(levelname)s]:%(message)s",
    level=logging.INFO,
    filemode='a'
    )

def main(config_path):
    logging.info("Starting the application...")
    content = read_yaml(config_path)
    download_data(config_path)
    

if __name__=='__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", help="path to config file", default="config/config.yaml")
        args = parser.parse_args()
        main(args.config)
    except Exception as e:
        print(e)
        logging.exception(e)
        raise e