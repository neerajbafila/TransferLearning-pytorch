import logging
import os
from pathlib import Path
import urllib.request as req
from tqdm import tqdm
from src.utils.common import read_yaml, create_directory, unzip_File

def download_data(config_path):
    content = read_yaml(config_path)
    local_dir = Path(content['data_dir']['local_dir'])
    create_directory([local_dir])
    
    zip_data = Path(local_dir, content['data_dir']['zip_data'])
    unzip_data_file = Path(local_dir, content['data_dir']['unzip_data_dir'])
    
    try:
        if not os.path.isdir(unzip_data_file):
            if not os.path.isfile(zip_data):
                url = content['data_url']
                logging.info(f"downloading data from {url}")
                print((f"downloading data from {url}"))
                with tqdm(req.urlretrieve(url, zip_data)) as t:
                    t.set_description("Downloading")
                    for _ in t:
                        pass
            else:
                unzip_File(zip_data, local_dir)
                logging.info(f"file {zip_data} unzipped to {local_dir}")
                print(f"file {zip_data} unzipped to {local_dir} folder")
        else:
            logging.info(f"{unzip_data_file} already exists")
            print(f"{unzip_data_file} already exists")
            
    except Exception as e:
        logging.exception(e)
        raise e
