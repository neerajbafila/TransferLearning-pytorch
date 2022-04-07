import logging
import os
from pathlib import Path
from typing import final
from matplotlib.style import available
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from src.utils.common import read_yaml, create_directory
from src.utils.model_operation import get_param_details
from src import stage_01_get_data

STAGE = "stage_03_train_newModel_by_preTrained_model"

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
    logging.info("freezing the  all the layers of preTrained model")
    for parameters in preTrained_model.parameters():
        parameters.requires_grad = False
    logging.info("freezing the  all the layers of preTrained model completed")
    
    df, total = get_param_details(preTrained_model)
    print(f"param details after freeze operatation {total}")

    # changes the classifier layer as per need:
    logging.info("changing the classifier layer of preTrained model")
    preTrained_model.classifier = nn.Sequential(
        nn.Linear(in_features=9216, out_features=4000, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4000, out_features=100, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=100, out_features=2)
    )

    df, total = get_param_details(preTrained_model)
    logging.info(f"The details of the parameters in the preTrained model after changes in classifier layer: \n{df}, total params are {total}")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"available Device is {DEVICE}")    
    preTrained_model.to(DEVICE)

    EPOCHS = content['parameters']['EPOCH']
    LR = content['parameters']['LR']
    BATCH_SIZE = content['parameters']['BATCH_SIZE']
    train_data_loader, test_data_loader, label_map = stage_01_get_data.main(config_path)
    # make loss function
    criterion = nn.CrossEntropyLoss()
    # make optimizer
    optimizer = torch.optim.Adam(params=preTrained_model.parameters(), lr=LR)
    logging.info("model training started")
    for epoch in range(EPOCHS):
        with tqdm(train_data_loader) as tqdm_epoch:
            tqdm_epoch.set_description(f"EPOCH {epoch}/{EPOCHS}")
            for images, lables in tqdm_epoch:

                # put images in cuda
                images = images.to(DEVICE)
                lables = lables.to(DEVICE)

                # forward pass
                outputs = preTrained_model(images)
                loss = criterion(outputs, lables)

                # backward pass
                ## delete old gradients
                optimizer.zero_grad()
                loss.backward()
                ## update gradients
                optimizer.step()
                tqdm_epoch.set_postfix(loss=loss.item())
    logging.info("model training completed")
     #save the final model
    final_model_path = Path(content['artifacts']['artifacts_dir'], content['artifacts']['final_trained_model_dir'])
    final_model_name = content['artifacts']['final_trained_model']
    create_directory([final_model_path])
    full_path = Path(final_model_path, final_model_name)
    torch.save(preTrained_model, full_path)
    logging.info(f"Final Model saved at {full_path}")

if __name__ == '__main__':
    try:
        logging.info("\n************************************************")
        logging.info(f">>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<")
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", help="path to config file", default="config/config.yaml")
        args = parser.parse_args()
        main(args.config)
        logging.info(f">>>>>>>>>>>>{STAGE} Completed<<<<<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
