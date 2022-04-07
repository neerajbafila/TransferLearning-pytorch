import logging
import os
import torch
import pandas as pd
from src.utils.common import read_yaml, create_directory

logging.basicConfig(
    filename=os.path.join("Logs",'running.log'),
    format="[%(asctime)s:%(module)s:%(levelname)s]:%(message)s",
    level=logging.INFO,
    filemode='a'
    )

def get_param_details(model):
    """
    Get the details of the parameters in the model.
    """
    # logging.info("The details of the parameters in the model:")
    parameters = {'Module': list(), 'Parameter': list()}
    total = {'Trainable': 0, 'Non-Trainable': 0}
    for name, param in model.named_parameters():
        parameters['Module'].append(name)
        parameters['Parameter'].append(param.numel())
        if param.requires_grad:
            total['Trainable'] += param.numel()
        else:
            total['Non-Trainable'] += param.numel()

    df = pd.DataFrame(parameters)
    df.style.set_caption(f"{total}")
    return df, total