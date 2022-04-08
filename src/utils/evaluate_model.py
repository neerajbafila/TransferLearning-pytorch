import torch
import logging
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src import stage_01_get_data
from src.utils.common import create_directory, read_yaml 

logging.basicConfig(
    filename=os.path.join("Logs",'running.log'),
    format="[%(asctime)s:%(module)s:%(levelname)s]:%(message)s",
    level=logging.INFO,
    filemode='a'
    )

def main(config_path):
    content = read_yaml(config_path)
    model_path = Path(content['artifacts']['artifacts_dir'], content['artifacts']['final_trained_model_dir'], content['artifacts']['final_trained_model'])
    model = torch.load(model_path)
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    logging.info(f"model {model_path} loaded")
    pred = np.array([])
    target = np.array([])
    # get test data
    train_data_loader, test_data_loader, label_map = stage_01_get_data.main(config_path)

    with torch.no_grad():
        for images, lables in test_data_loader:
            # put data in cuda

            images = images.to(device)
            lables = lables.to(device)

            outputs = model(images)
            actual = lables.cpu().numpy()

            probability = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probability,dim=1)
            
            pred = np.concatenate((pred, prediction.cpu().numpy()))
            target = np.concatenate((target, actual))
        
    confusion_matrix_dir = Path(content['artifacts']['artifacts_dir'], content['artifacts']['final_trained_model_dir'], content['artifacts']['confusion_matrix_dir'])
    create_directory([confusion_matrix_dir]) 
    confusion_matrix_name = content['artifacts']['confusion_matrix_fig_name']
    confusion_matrix_full_path = os.path.join(confusion_matrix_dir, confusion_matrix_name)
    cm = confusion_matrix(target, pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, cbar=False, annot=True, fmt='d', xticklabels=label_map.values(), yticklabels=label_map.values())
    plt.savefig(confusion_matrix_full_path)
    plt.show()
    logging.info(f"confusion matrix figure is saved at {confusion_matrix_full_path}")


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", help="path to config file", default="config/config.yaml")
        args = parser.parse_args()
        main(args.config)
    except Exception as e:
        logging.exception(e)
        raise e
        
