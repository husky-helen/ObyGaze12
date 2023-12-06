import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os

from sklearn.metrics import f1_score, accuracy_score
from lightning_model import DynMLP2HiddenLit
from balanced_dataset import XCLIPBalancedKFoldVisualFeaturesLitDataset, BalancedViVitKFoldVisualFeaturesLitDataset, BalancedLSMDCKFoldVisualFeaturesLitDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def infer_one_model_one_infertype(model_path, model_params, infer_type, visual_dir, gt_csv, dico_level, logger_dir= "/home/julie/Results/LinearProbing/Inference", logger_name = "test", logger_version = "version"):

    # Load model
    model = DynMLP2HiddenLit.load_from_checkpoint(checkpoint_path=model_path,
                                    layers_config=model_params[0],
                                    activation=model_params[1],
                                    x=model_params[2])
    
    
    testdataset = BalancedLSMDCKFoldVisualFeaturesLitDataset(visual_dir, gt_csv, classification="level", split="test", dico_level=dico_level )    
    testloader = DataLoader(testdataset, batch_size = len(testdataset), shuffle = False, num_workers=19)

    # test
    logger = TensorBoardLogger(save_dir = logger_dir ,name= logger_name, version=logger_version)
    trainer = pl.Trainer(min_epochs = 5, max_epochs=100, logger=logger, check_val_every_n_epoch=1, log_every_n_steps=40, devices=1)  

    test_results = trainer.test(model,testloader)
    model.eval()
    elem = next(iter(testloader))
    y_pred = np.argmax(model(elem[0]).detach(), axis = 1)
    y_true = np.argmax(elem[1].detach(),axis=1)

    
def run_lsmdc():

    ###### args
    inference_type = ["02_EN_S.csv", "02_HN_S.csv", "02_ENHN_S.csv"]


    model_checkpoint="~/Results/LinearProbing/BalancedViVit/13Nov2113/ViVit_Weighted_Embed_big__Lucile_02_ENHN_S/fold_0/checkpoints/epoch=5-step=108.ckpt"
    model_params = [[768,1024,2],nn.ReLU(), None]
    infer_type = inference_type[0]
    visual_dir = "/media/LaCie/Features/ViVit/ViVit/"
    gt_csv = "/media/LaCie/balanced_level/AnnotationsLucile/02_EN_S.csv"
    dico_level = {"Easy Neg":0, "Sure":1}



        
    #infer_one_model_one_infertype(model_checkpoint, model_params, infer_type, visual_dir, gt_csv, dico_level)

    visual_dir_vivit = "/media/LaCie/Features/ViVit/ViVit" #"/home/julie/ssd/data1/Features/ViVit"
    visual_dir_xclip="/media/LaCie/Features/XCLIP/Visual"
    visual_dir="/media/LaCie/Features/LSMDC_XCLIP"

    ths = ["02"]
    model_configs = {"small": [768,128,2]} #"big":[768,1024,2],
    file_names = {"ENHN_S.csv":{"EN_HN":0, "S":1}, "HN_S.csv" : {"Hard Neg":0, "Sure":1}, "EN_S.csv":{"Easy Neg":0, "Sure":1}} #

    annotators = {"Merged":"AnnotationsMerged"} #"Lucile":"AnnotationsLucile","Magali": "AnnotationsMagali", "Helen":"AnnotationsHelen", "Clement":"AnnotationsClement",
    factorstype = "VisualFactors"

    root_checkpoints_vivit = "/home/julie/Results/LinearProbing/ModelViVit/BalancedViVit_14Nov1628_Merged" #"/home/julie/Results/LinearProbing/BalancedViVit/13Nov2113/"
    root_checkpoints_xclip = "/home/julie/Results/LinearProbing/Balanced_XCLIP_14Nov1628/Merged/Small/"
    root_checkpoints = "/home/julie/Results/LinearProbing/BalancedLSMDC_26Nov/"

    root_annotations = "/media/LaCie/balanced_level" #"/home/julie/ssd/data1/MovieGraphs/Meta/new_metas1"


    root_logger_vivit = "/home/julie/Results/LinearProbing/Inference/ModelViVit/BalancedViVit_14Nov1628_Merged"
    root_logger_xclip = "/home/julie/Results/LinearProbing/Inference/ModelXCLIP/BalancedXCLIP_16Nov1138_Merged_Small"
    root_logger = "/home/julie/Results/LinearProbing/Inference/ModelLSMDC/"

    experiments = os.listdir(root_checkpoints)

    # LSMDC
    print("ROOT CHECKPOINT : ", root_checkpoints)
    for experiment in experiments:
        print("Experiment :", experiment)
    
        _,_,_,model_size,_, annotator, th, neg, pos = experiment.split("_")

        folds = os.listdir(os.path.join(root_checkpoints, experiment))

        for fold in folds:

            model_path_tmp = os.path.join(root_checkpoints, experiment, fold, "checkpoints")

            model_name = os.listdir(model_path_tmp)[0] # checker s'il y a plusieurs modèles
            model_path = os.path.join(model_path_tmp, model_name)
            model_params = [model_configs[model_size],nn.ReLU(), None]

            model_train_type = "ModelTrainedOn_" + neg + "_" + pos
            for infer_type, dico_level in file_names.items():

                gt_csv = os.path.join(root_annotations,"Annotations"+annotator, th+"_"+infer_type ) 
                logger_dir = os.path.join(root_logger, model_train_type,annotator)
                logger_name = 'InferOn_' + infer_type
                logger_version = fold
                print(logger_dir, logger_name, logger_version)
                print(model_path)
                print('----------------')
                infer_one_model_one_infertype(model_path, 
                                            model_params, 
                                            infer_type, 
                                            visual_dir, 
                                            gt_csv, 
                                            dico_level, 
                                            logger_dir, 
                                            logger_name, 
                                            logger_version)
                




"""# XCLIP
print("ROOT CHECKPOINT : ", root_checkpoints)
for experiment in experiments:
    print("Experiment :", experiment)
    #ViVit_Weighted_Embed_big__Lucile_02_ENHN_S
    _,_,_,model_size,_, annotator, th, neg, pos = experiment.split("_")

    folds = os.listdir(os.path.join(root_checkpoints, experiment))

    for fold in folds:

        model_path_tmp = os.path.join(root_checkpoints, experiment, fold, "checkpoints")

        model_name = os.listdir(model_path_tmp)[0] # checker s'il y a plusieurs modèles
        model_path = os.path.join(model_path_tmp, model_name)
        model_params = [model_configs[model_size],nn.ReLU(), None]

        model_train_type = "ModelTrainedOn_" + neg + "_" + pos
        for infer_type, dico_level in file_names.items():

            gt_csv = os.path.join(root_annotations,"Annotations"+annotator, th+"_"+infer_type ) 
            logger_dir = os.path.join(root_logger, model_train_type,annotator)
            logger_name = 'InferOn_' + infer_type
            logger_version = fold
            print(logger_dir, logger_name, logger_version)
            print(model_path)
            print('----------------')
            infer_one_model_one_infertype(model_path, 
                                        model_params, 
                                        infer_type, 
                                        visual_dir, 
                                        gt_csv, 
                                        dico_level, 
                                        logger_dir, 
                                        logger_name, 
                                        logger_version)
"""
# VIVIT
"""for experiment in experiments:
    print("Experiment :", experiment)
    #ViVit_Weighted_Embed_big__Lucile_02_ENHN_S
    _,_,_,model_size,_, annotator, th, neg, pos = experiment.split("_")

    folds = os.listdir(os.path.join(root_checkpoints, experiment))

    for fold in folds:

        model_path_tmp = os.path.join(root_checkpoints, experiment, fold, "checkpoints")

        model_name = os.listdir(model_path_tmp)[0] # checker s'il y a plusieurs modèles
        model_path = os.path.join(model_path_tmp, model_name)
        model_params = [model_configs[model_size],nn.ReLU(), None]

        model_train_type = "ModelTrainedOn_" + neg + "_" + pos
        for infer_type, dico_level in file_names.items():

            gt_csv = os.path.join(root_annotations,"Annotations"+annotator, th+"_"+infer_type ) 
            logger_dir = os.path.join(root_logger, model_train_type,annotator)
            logger_name = 'InferOn_' + infer_type
            logger_version = fold
            print(logger_dir, logger_name, logger_version)
            print(model_path)
            print('----------------')
            infer_one_model_one_infertype(model_path, 
                                        model_params, 
                                        infer_type, 
                                        visual_dir, 
                                        gt_csv, 
                                        dico_level, 
                                        logger_dir, 
                                        logger_name, 
                                        logger_version)"""

"""
for ann, dir_ann in annotators.items():
    for th in ths:
        for file_name,dico_level in file_names.items():
            output_size = len(dico_level)
            print("output_size",output_size)
            model_configs = {"big":[768,1024,2],"small": [768,128,2]}

            for k_model, layers_config in model_configs.items():
                
                gt_file = os.path.join(root_dir, dir_ann , th+"_"+file_name)
                exp_name = "ViVit_Weighted_Embed_"+k_model + "_"  + "_" + ann + "_" +th+"_"+ file_name.split(".")[0]
                print("----------------", exp_name, "------------------------")
                #run_embed2level_nfold(visual_dir, gt_file, dico_level, layers_config, exp_name)
                infer_one_model_one_infertype(model_path, model_params, infer_type, visual_dir, gt_csv, dico_level, logger_dir= "/home/julie/Results/LinearProbing/Inference", logger_name = "test", logger_version = "version"):


"""