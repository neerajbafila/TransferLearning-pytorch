import logging
import os
from pathlib import Path
from matplotlib.style import available
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from src.utils.common import read_yaml, create_directory
from src.utils.model_operation import get_param_details
from src import stage_01_get_data

STAGE = ""

logging.basicConfig(
    filename=os.path.join("Logs",'running.log'),
    format="[%(asctime)s:%(module)s:%(levelname)s]:%(message)s",
    level=logging.INFO,
    filemode='a'
    )

def main(config_path):
    content = read_yaml(config_path)
    preTrained_model_path = Path(content['artifacts']['artifacts_dir'], content['artifacts']['preTrained_model_dir'])
    preTrained_model_name = content['artifacts']['preTrained_model_name']
    preTrained_model_full_path = Path(preTrained_model_path, preTrained_model_name)
    preTrained_model = torch.load(preTrained_model_full_path)
    logging.info(f"The preTrained model is loaded from {preTrained_model_full_path}")

    # freeze the preTrained model
    for parameters in preTrained_model.parameters():
        parameters.requires_grad = False
    
    df, total = get_param_details(preTrained_model)
    print(f"param details after freeze operatation {total}")

    # changes the classifier layer as per need:
    preTrained_model.classifier = nn.Sequential(

        nn.Linear(in_features=9216, out_features=4000, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=1000, out_features=100, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=100, out_features=2)
    )

    df, total = get_param_details(preTrained_model)
    logging.info(f"The details of the parameters in the preTrained model: {df}, total params are {total}")
    print(f"param details after change in classifier layer {total}")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"available Device is {DEVICE}")    
    preTrained_model.to(DEVICE)

    EPOCHS = content['parameters']['EPOCH']
    LR = content['parameters']['LR']
    BATCH_SIZE = content['parameters']['BATCH_SIZE']
    train_data_loader, test_data_loader, label_map = stage_01_get_data.main(config_path)
    for epoch in range(EPOCHS):
        with tqdm(train_data_loader) as tqdm_epoch:
            tqdm_epoch.set_description(f"EPOCH {epoch}/{EPOCHS}")
            print('continue')

        


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", help="path to config file", default="config/config.yaml")
        args = parser.parse_args()
        main(args.config)
    except Exception as e:
        logging.exception(e)
        raise e
