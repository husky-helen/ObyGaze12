import pandas as pd
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.balanced_dataset import XCLIP_KFold
from models.lightning_model import DynMLP2HiddenLit

def run_embed2level_1fold(feature_dir, gt_csv, nb_fold, dico_level, layers_config, exp_name,version_name, saving_dir):
    """Run one experiment

    Args:
        feature_dir (str or pathlike): path to the directory where the features are stored
        gt_csv (str or pathlike): path to the csv containing the objectifcation annotation
        dico_level (dict): classes to use for the objectification classification
        layers_config (list): list containing the number of neurons for each layer
        exp_name (str): name used to stored the results
        nb_fold (int): number of the fold to use as the validation set
        version_name (str): version_name of the experiment (each validation fold has a different number)
        saving_dir (str or pathlike) : path to the dir where the results should be stored
    """

    train_dataset = XCLIP_KFold( feature_dir, gt_csv, classification="level",split="train", nb_fold = nb_fold, dico_level =  dico_level)
    val_dataset = XCLIP_KFold( feature_dir, gt_csv, classification="level",split="val", nb_fold = nb_fold, dico_level =  dico_level)
    test_dataset = XCLIP_KFold( feature_dir, gt_csv, classification="level",split="test", nb_fold = nb_fold, dico_level = dico_level )


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=19)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),num_workers=19)
    test_loader= DataLoader(test_dataset, batch_size=len(test_dataset),num_workers=19)

   
    model = DynMLP2HiddenLit(layers_config, activation = nn.ReLU(),w = None)


    es = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    logger = TensorBoardLogger(save_dir = saving_dir,name= exp_name, version=version_name)
    trainer = pl.Trainer(min_epochs = 5, max_epochs=100,callbacks=[es], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=40,devices=1)  

   
    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(model,test_loader)

def run_embed2level_nfold(feature_dir, gt_csv,dico_level, layers_config, exp_name,saving_dir):
    """Run the process for different validation fold (cross-validation)

    Args:
        feature_dir (str or pathlike): path to the directory where the features are stored
        gt_csv (str or pathlike): path to the csv containing the objectifcation annotation
        dico_level (dict): classes to use for the objectification classification
        layers_config (list): list containing the number of neurons for each layer
        exp_name (str): name used to stored the results
        saving_dir (str or pathlike) : path to the dir where the results should be stored
    """

    tot_fold = pd.read_csv(gt_csv, index_col= 0, sep=";").fold.max()-1
    for i in range(tot_fold):
        version_name = f'fold_{i}'
        run_embed2level_1fold(feature_dir, gt_csv, i, dico_level, layers_config, exp_name,version_name,saving_dir)


def process_XCLIP(feature_dir, annotation_dir = "/media/LaCie/balanced_level",saving_dir="/home/julie/Results/LinearProbing/Balanced_XCLIP"):
    """Run the classification for several configuration with features extracted from XCLIP

    Args:
        feature_dir (str or pathlike): path to the directory where the features are stored
        annotation_dir (str or pathlike): path to the annotation files
        saving_dir (str or pathlike) : path to the dir where the results should be stored
    """

    file_names = {"HN_S.csv" : {"Hard Neg":0, "Sure":1},"EN_S.csv":{"Easy Neg":0, "Sure":1}}
 
  

    for file_name,dico_level in file_names.items():
        output_size = len(dico_level)
        model_configs = {"small": [512,128,output_size]}
        for _, layers_config in model_configs.items():
            
        
            gt_file = os.path.join(annotation_dir, "02"+"_"+file_name)
            exp_name = "XCLIP_" + file_name.split(".")[0]
        
            run_embed2level_nfold(feature_dir, gt_file, dico_level, layers_config, exp_name, saving_dir)