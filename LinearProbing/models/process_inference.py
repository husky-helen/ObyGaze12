import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models.lightning_model import DynMLP2HiddenLit
from dataset.balanced_dataset import LinearProbing_KFold

def infer_one_model_one_infertype(model_path, features_dir, gt_csv, dico_level, logger_dir= "/home/xxx/Results/LinearProbing/Inference", logger_name = "test", logger_version = "version"):
    """Does the inference for one experiment

    Args:
        model_path (str or pathlike): path to the file containing the checkpoint to load
        features_dir (str or pathlike): path to the directory containing features extracted 
        gt_csv (str or pathlike): path to the objectification annotation file
        dico_level (dict): dictionnary containing the classes to use
        logger_dir (str, optional): Path to the directory where the logs should be saved. Defaults to "/home/xxx/Results/LinearProbing/Inference".
        logger_name (str, optional): name of the exeperiment. Defaults to "test".
        logger_version (str, optional): name of the version of the experiment. Defaults to "version".

    Returns:
        float: f1 score
    """
    testdataset = LinearProbing_KFold(features_dir, gt_csv, split="test", dico_level=dico_level)
    layers = [testdataset[0][0].shape[-1],128, len(dico_level)]
    print("Model path", model_path)
    print("Layers : ", layers)
    model = DynMLP2HiddenLit.load_from_checkpoint(checkpoint_path=model_path,
                                    layers_config=layers,
                                    activation=nn.ReLU(),
                                    x=None)
    
        
    testloader = DataLoader(testdataset, batch_size = len(testdataset), shuffle = False, num_workers=19)

    logger = TensorBoardLogger(save_dir = logger_dir ,name= logger_name, version=logger_version)
    trainer = pl.Trainer(min_epochs = 5, max_epochs=100, logger=logger, check_val_every_n_epoch=1, log_every_n_steps=40, devices=1)  

    test_results = trainer.test(model,testloader)
    
    return test_results[0]["test_f1"]


def run_inference(root_checkpoints, root_annotations, root_logger, features_dir, print_ = True):
    """Run inference for all the experiments

    Args:
        root_checkpoints (str or pathlike): path to the directory containing the checkpoints
        root_annotations (str or pathlike): path to the objectification annotation directory
        root_logger (str or pathlike): Path to the directory where the logs should be saved.
        features_dir (str or pathlike): path to the directory containing features extracted 
        print_ (bool, optional): If the mean f1 score should be printed. Defaults to True.
    """

    file_names = {"ENHN_S.csv":{"EN_HN":0, "S":1},  "EN_S.csv":{"Easy Neg":0, "Sure":1}}
    experiments = os.listdir(root_checkpoints)
    f1_dico = {}
    for experiment in experiments: # TODO erase
    
        f1_dico[experiment] = {}
        th, neg, pos = experiment.split("_")

        folds = os.listdir(os.path.join(root_checkpoints, experiment))

        for fold in folds:

            model_path_tmp = os.path.join(root_checkpoints, experiment, fold, "checkpoints")

            model_name = os.listdir(model_path_tmp)[0] # check if several models have been saved
            model_path = os.path.join(model_path_tmp, model_name)
           

            model_train_type = "TrainedOn_" + neg + "_" + pos
            for infer_type, dico_level in file_names.items():
                
                gt_csv = os.path.join(root_annotations, "02_"+infer_type ) 
                logger_dir = os.path.join(root_logger, model_train_type)
                logger_name = 'InferOn_' + infer_type
                logger_version = fold

                score_f1 = infer_one_model_one_infertype(model_path, 
                                            features_dir, 
                                            gt_csv, 
                                            dico_level, 
                                            logger_dir, 
                                            logger_name, 
                                            logger_version)
                
                if infer_type in f1_dico[experiment].keys():
                    f1_dico[experiment][infer_type] += [score_f1]
                else: f1_dico[experiment][infer_type] = [score_f1]
                
    if print_:

        for exp, tmp_dico in f1_dico.items():

            for infer, list_f1 in tmp_dico.items():

                print(f"Trained On {exp}, Tested On {infer}, f1 score : {np.mean(list_f1)}" )
