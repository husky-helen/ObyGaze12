import torch
from collections import defaultdict
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

import pickle
import os
import pandas as pd


def create_svm(C = 0.1, kernel = "linear", probability = True):
    return SVC(C=C, kernel=kernel, probability = probability, class_weight='balanced')

def train_svm(svm, X_train, Y_train ):
    svm.fit(X_train, Y_train)
    return svm

def log_acc(svm,log_path, X_train, Y_train, X_val=None, Y_val=None, X_test=None, Y_test=None):
    acc_train = evaluate_svm(svm, X_train, Y_train)
    acc_val = evaluate_svm(svm, X_val, Y_val)
    acc_test = evaluate_svm(svm, X_test, Y_test)

    log_txt = 'Acc on train : '+ str(acc_train) + "\nAcc on val : "+ str(acc_val) + "\nAcc on test : "+ str(acc_test)+ "\n"
    print("LOG :", log_path)
    with open(log_path, 'w') as fichier:
        fichier.write(log_txt)


def evaluate_svm(svm, X_test, Y_test):
    accuracy = svm.score(X_test, Y_test)
    return accuracy

def compute_distance2sephyperplan(svm, X_test):
    return svm.decision_function(X_test) / np.linalg.norm(svm.coef_)

def distance2clips(distances, clips_name, nb_clips):
    idx_tmp = np.argsort(distances)
    
    clip_idx = idx_tmp[::-1][:nb_clips]
    res = [clips_name[clid] for clid in clip_idx]
    return res
    
def log_clip_names(clip_names, log_file,prefix = "\n"):

    log_txt = prefix
    for clip_name in clip_names:
        log_txt += clip_name+"\n"
    
    with open(log_file, 'a') as fichier:
        fichier.write(log_txt)

def save_svm(svm, pickle_file):
    with open(pickle_file, 'wb') as model_file:
        pickle.dump(svm, model_file)

def load_svm(pickle_file):
    with open(pickle_file, 'rb') as model_file:
        loaded_clf = pickle.load(model_file)

    return loaded_clf

def create_mini_df(distances, clip_names, labels):

    split = ["train"] * len(distances[0]) + ["val"] * len(distances[1]) + ["test"] * len(distances[2])
    distance = np.concatenate(distances)
    clip_name = np.concatenate(clip_names)
    label = np.concatenate(labels)

    concept_df = pd.DataFrame(clip_name, columns = ["video_name"])
    concept_df["distance"] = distance
    #concept_df["split"] = split
    #concept_df["gt"] = label
    return concept_df

def create_mini_df2(distances, clip_names, labels):

    split = ["train"] * len(distances[0]) + ["test"] * len(distances[0])
    distance = np.concatenate(distances)
    clip_name = np.concatenate(clip_names)


    concept_df = pd.DataFrame(clip_name, columns = ["video_name"])
    concept_df["distance"] = distance

    return concept_df


def create_big_df(list_paths, visual_concepts):
    dataframes = [pd.read_csv(chemin,sep=";", index_col=0) for chemin in list_paths if os.path.exists(chemin)]

    #print(dataframes[0].columns)
    
    resultat_fusion = pd.merge(dataframes[0], dataframes[1], on='video_name',suffixes=(''.join(visual_concepts[0].split()), ''.join(visual_concepts[1].split())), how='outer')
    for i in range(2, len(dataframes)):
        resultat_fusion = pd.merge(resultat_fusion, dataframes[i], on='video_name', suffixes=(None, ''.join(visual_concepts[i].split())), how='outer')

    return resultat_fusion


