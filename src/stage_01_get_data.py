import logging
import os
from pathlib import Path
from torchvision import transforms, datasets, models
import torch
from torch.utils.data import DataLoader

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
    content = read_yaml(config_path)
    download_data(config_path)
    train_data_path = Path(content['data_dir']['local_dir'], content['data_dir']['unzip_data_dir'], content['data_dir']['train_data_path'])
    test_data_path = Path(content['data_dir']['local_dir'], content['data_dir']['unzip_data_dir'], content['data_dir']['test_data_path'])

    # let take mean=0.5 and std=0.5 for normalization
    mean = torch.Tensor([0.5, 0.5, 0.5])
    std = torch.Tensor([0.5, 0.5, 0.5])
    # make transform for train data
    
    logging.info(content['parameters']['IMAGE_SIZE'])
    image_size = content['parameters']['IMAGE_SIZE']

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(degrees=content['parameters']['rotation_degrees']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    logging.info(f"train transform created with {train_transform}")

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    logging.info(f"test transform created with {test_transform}")
    # get train data

    train_data = datasets.ImageFolder(root=train_data_path, transform=train_transform)
    test_data = datasets.ImageFolder(root=test_data_path, transform=test_transform)
    logging.info(f"train and test data created with defined transforms")
    # make dataloader
    train_data_loader = DataLoader(dataset=train_data, batch_size=content['parameters']['BATCH_SIZE'], shuffle=True)
    test_data_loader = DataLoader(dataset=test_data, batch_size=content['parameters']['BATCH_SIZE'], shuffle=False)
    logging.info(f"dataloader created with BATCH_SIZE {content['parameters']['BATCH_SIZE']}")
    # get class_to_idx
    label_map = train_data.class_to_idx
    label_map = {val: key for key, val in label_map.items()}

    logging.info(f"label_map created with {label_map}")
    return train_data_loader, test_data_loader, label_map


if __name__=='__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", help="path to config file", default="config/config.yaml")
        args = parser.parse_args()
        logging.info("\n************************************************")
        logging.info(f">>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<")
        main(args.config)
        logging.info(f'>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<')
    except Exception as e:
        print(e)
        logging.exception(e)
        raise e