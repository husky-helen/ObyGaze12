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


def run_embed2level_1fold(visual_dir, gt_csv, layers_config, exp_name,version_name, valid_num, test_num):
    """Run the training for one experiment

    Args:
        visual_dir (str or pathlike): path to the directory containing the extracted features. 
        gt_csv (str or pathlike): path to the csv containing the objectification annotation
        exp_name (str): name of the experiment
        version_name (str): name of the version of the experiment
        valid_num (int): number of the fold to use as the validation set
        test_num (int): number of the fold to use as the test set
        model_path (str or pathlike): path to checkpoints
    """
    
    train_dataset = MovieBySet_KFold( visual_dir, gt_csv,valid_num, test_num ,split="train" )
    val_dataset = MovieBySet_KFold( visual_dir, gt_csv,valid_num, test_num ,split="val")
    test_dataset = MovieBySet_KFold( visual_dir, gt_csv, valid_num, test_num,split="test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=19)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),num_workers=19)
    test_loader= DataLoader(test_dataset, batch_size=len(test_dataset),num_workers=19)

    model = DynMLP2HiddenLit(layers_config, activation = nn.ReLU(),w = None)


    es = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    logger = TensorBoardLogger(save_dir = "lightning_logs",name= exp_name, version=version_name)
    trainer = pl.Trainer(min_epochs = 5, max_epochs=100,callbacks=[es], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=40)  


    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(model,test_loader)

def run_embed2level_nfold(visual_dir, gt_csv, exp_name,valid_num, test_num):
    """Run cross-val

    Args:
        visual_dir (str or pathlike): path to the directory containing the extracted features. 
        gt_csv (str or pathlike): path to the csv containing the objectification annotation. 
        exp_name (str): name of the experiments
        valid_num (int): number of the fold to use as the validation set
        test_num (int): number of the fold to use as the test set
    """
        
    version =  f"ValidOn{valid_num}"
    run_embed2level_1fold(visual_dir, gt_csv, exp_name,version,valid_num, test_num)


def all_process_train(visual_dir , annotation_dir , saving_dir ):
    """Run the training for all the experiments

    Args:
        visual_dir (str or pathlike): path to the directory containing the extracted features. 
        annotation_dir (str or pathlike): path to the directory containing the objectification annotation. 
        saving_dir (str or pathlike): path to the directory where the logs should be saved. 
    """
    
    file_names = {"HN_S.csv":{"Hard Neg":0, "Sure":1}}
    for file_name,dico_level in file_names.items():
        output_size = len(dico_level)
        layers_config = [512,128,output_size]
        gt_file = os.path.join(annotation_dir, file_name)

            
        for i in range(12): # test (12 movies annotated with objectification tags)
            
            for j in range(12): # validation

                if i != j and i != 3 and i!= 7 and j!=3 and j!=7 : # movies corresponding to folds 3 and 7 have distributions that would not be relevant to study in a test

                    exp_name =os.path.join(saving_dir, file_name.split(".")[0] , f"TestedOn_{i}")
                    run_embed2level_nfold(visual_dir, gt_file, layers_config, exp_name,j, i)