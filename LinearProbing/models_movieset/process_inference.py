from dataset.moviebyset_dataset import MovieBySet_KFold
from models.lightning_model import DynMLP2HiddenLit

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def run_embed2level_1fold(visual_dir, gt_csv,  dico_level,  exp_name,version_name, valid_num, test_num, model_path):
    """Run the inference for one experiment

    Args:
        visual_dir (str or pathlike): path to the directory containing the extracted features. 
        gt_csv (str or pathlike): path to the csv containing the objectification annotation
        dico_level (dict): dictionnary containing the labels to use for the classification
        exp_name (str): name of the experiment
        version_name (str): name of the version of the experiment
        valid_num (int): number of the fold to use as the validation set
        test_num (int): number of the fold to use as the test set
        model_path (str or pathlike): path to checkpoints
    """

    test_dataset = MovieBySet_KFold( visual_dir, gt_csv, valid_num, test_num,split="test")
    test_loader= DataLoader(test_dataset, batch_size=len(test_dataset),num_workers=19)
    layers_config = [test_dataset[0][0].shape,128,len(dico_level)]
    model = DynMLP2HiddenLit.load_from_checkpoint(checkpoint_path = model_path,
         layers_config=layers_config, activation = nn.ReLU(),w = None)


    es = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    logger = TensorBoardLogger(save_dir = "lightning_logs",name= exp_name, version=version_name)
    trainer = pl.Trainer(min_epochs = 5, max_epochs=100,callbacks=[es], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=40)  

    test_results = trainer.test(model,test_loader)



def run_embed2level_nfold(visual_dir, gt_csv,dico_level,  exp_name,valid_num, test_num, model_path):
    """Run cross-validation

    Args:
        visual_dir (str or pathlike): path to the directory containing the extracted features. 
        gt_csv (str or pathlike): path to the csv containing the objectification annotation
        dico_level (dict): dictionnary containing the labels to use for the classification
        exp_name (str): global name of the experiment
        valid_num (int): number of the fold to use as the validation set
        test_num (int): number of the fold to use as the test set
        model_path (str or pathlike): path to checkpoints
    """
    version =  f"ValidOn{valid_num}"
    run_embed2level_1fold(visual_dir, gt_csv, dico_level, exp_name,version,valid_num, test_num, model_path)


def all_proces_inference(visual_dir, checkpoints, annotation_dir, saving_dir):
    """Run the inference for all the experiments

    Args:
        visual_dir (str or pathlike): path to the directory containing the extracted features. 
        checkpoints (str or pathlike): path to the directory containing the checkpoints. 
        annotation_dir (str or pathlike): path to the directory containing the objectification annotation. 
        saving_dir (str or pathlike): path where the logs should be stored.
    """

    file_names = {"ENHN_S.csv":{"ENHN":0, "S":1}}

    model_path = checkpoints
    for file_name,dico_level in file_names.items():
        output_size = len(dico_level)
        layers_config =  [512,128,output_size]
        gt_file = os.path.join(annotation_dir, file_name)

        for i in range(12): # test 
            for j in range(12): # validation

                if i != j and i != 3 and i!= 7 and j!=3 and j!=7 :
                    total_model_dir = os.path.join(model_path,"HN_S" ,f"TestedOn_{i}", f"ValidOn{j}","checkpoints")
                    total_model_path = os.path.join(total_model_dir, os.listdir(total_model_dir)[0])
                    exp_name = os.path.join(saving_dir, file_name.split(".")[0] , f"TestedOn_{i}")
                    run_embed2level_nfold(visual_dir, gt_file, dico_level, layers_config, exp_name,j, i, total_model_path)