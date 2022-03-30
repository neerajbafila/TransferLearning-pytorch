import logging
import os
import torch
import argparse
from torchvision import models
from src.utils.common import read_yaml, create_directory
from src.utils.model_operation import get_param_details
from pathlib import Path

STAGE = "stage_02_getting_preTrained_model_alexnet"
logging.basicConfig(
    filename=os.path.join("Logs",'running.log'),
    format="[%(asctime)s:%(module)s:%(levelname)s]:%(message)s",
    level=logging.INFO,
    filemode='a'
    )

def main(config_path):
    content = read_yaml(config_path)
    logging.info(f"Getting pretrained Model")
    preTrained_model =  models.alexnet(pretrained=True) ## Imagenet dataset
    df, total = get_param_details(preTrained_model)
    logging.info(f"The details of the parameters in the preTrained model: {df}, total params are {total}")
    print(f"Total parameters are {total}")
    preTrained_model_path = Path(content['artifacts']['artifacts_dir'], content['artifacts']['preTrained_model_dir'])
    preTrained_model_name = content['artifacts']['preTrained_model_name']
    create_directory([preTrained_model_path])
    preTrained_model_full_path = Path(preTrained_model_path, preTrained_model_name)
    torch.save(preTrained_model, preTrained_model_full_path)
    # torch.save(preTrained_model.state_dict(), preTrained_model_full_path)
    logging.info(f"The preTrained model is saved at {preTrained_model_full_path}")


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", help="path to config file", default="config/config.yaml")
        args = parser.parse_args()
        main(args.config)
    except Exception as e:
        logging.exception(e)
        raise e


