import numpy as np
import pandas as pd
import os
import ast
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score


def compute_clips_ratio(df_proba, K):
    """Compute the ratio :
    Number of truly positive films among the K first films / predicted positive among the K first films
    Select 5 clips with a 

    Args:
        df_proba (pd.DataFrame): DataFrame containing the probability/similarity obtained for each concept
        K (int): number of films to use to compute the ratio

    Returns:
        film_list (list): List containg the 5 clips with the highest probability for the studied concept
        mean_ratio (float): Mean ratio (over the cross_validation)
    """

    nb_l, nb_c = df_proba.shape
    film_list = []
    ratios = []
    
    # Loop in the cross-validation
    for col in df_proba.columns:
        if "distance" in col:
            gt_name = col[8:].split('_')[0]+"_gt_label"
            sub_df = df_proba.loc[:,["video_name", col, "label", "reason",gt_name]]
            df_trie = sub_df.sort_values(by=col, ascending=False)
            
            df_trie_k = df_trie.iloc[:K,:] # Keep K films with highest probability
            denum = sum(df_trie_k[col]>0) # Number of positive films among the K films
            if denum == 0:
                ratio = 0
            else:   
                ratio = sum(df_trie_k[gt_name]) / denum # Number of truly positive films among the K first films / predicted positive among the K first films
            ratios.append(ratio)
            nb_clips = 5
            if df_trie.shape[0]< 5: nb_clips = df_trie.shape[0]
            for i in range(nb_clips):
                elem = df_trie.iloc[i,:]

                dico_infos = {"video_name": elem["video_name"], "level": elem["label"], "annotated_concepts": elem["reason"], "positive_exemple":elem[gt_name], "rank":i}
                if dico_infos not in film_list: film_list.append(dico_infos)

    return film_list,np.nanmean(ratios)

def load_annotations(annot_path):
    """Load the csv containing the objectification annotation

    Args:
        annot_path (str or pathlike): path to the csv containing the objectification annotation (should also contain a 'fold' column)

    Returns:
        pandas.DataFrame: Objectification annotation df
    """
    annot = pd.read_csv(annot_path, sep=";", index_col=0)
    annot = annot.loc[:,["video_name", "label", "reason", "fold"]] # Keep interesting columns
    annot["reason"] = annot["reason"].apply(ast.literal_eval) 
    dummy_df =  annot["reason"].apply(lambda x: pd.Series([1] * len(x), index=x, dtype=object)).fillna(0, downcast='int')
    annot = pd.concat([annot, dummy_df], axis = 1)
    acs = annot.columns
    annot.columns = ["".join(ac.split(" ")) for ac in acs]
    return annot

def compute_merged_df(annot_path, path_svm_experiment):
    """Merge the result df of each cross validation into one dataframe

    Args:
        annot_path (str or pathlike): path to the csv containing the objectification annotation (should also contain a 'fold' column)
        path_svm_experiment (str or pathlike): path to the dir where the trained svms are stored

    Returns:
        pandas.DataFrame: Merged dataframe of the cross-validation results
    """
    annot = load_annotations(annot_path)
    splits = os.listdir(path_svm_experiment)
    splits.sort()

    svm_df = pd.read_csv(os.path.join(path_svm_experiment, splits[0], "big_df.csv"), sep=";", index_col=0)
    svm_df.rename(columns={'distance': 'distanceClothes'}, inplace=True)

    for i in range(1, len(splits)):
        split = splits[i]
        path_1 = os.path.join(path_svm_experiment, split, "big_df.csv")
        if os.path.exists(path_1):
            tmp_df = pd.read_csv(path_1, sep=";", index_col=0)
            tmp_df.rename(columns={'distance': 'distanceClothes'}, inplace=True)
            svm_df = svm_df.merge(tmp_df, on = "video_name", suffixes=( None ,"_" + split ))

    merged_df = svm_df.merge(annot, on='video_name', how='outer')
    merged_df.to_csv("merged_.csv", sep=";")
    return merged_df


def gt_from_exptype(concept_merged_df, exp_type, concept):
    """Add a colummn to the merged df that stores the true label for the concept

    Args:
        concept_merged_df (pandas.DataFrame): Merged dataframe of the cross-validation results
        exp_type (str): Experiment type (Sc_HNc_vs_conceptbar)
        concept (str): concept of interest

    Returns:
        pandas.DataFrame: updated merged df
    """
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

    concept = "".join(concept.split(" "))
    concept_merged_df.reset_index(drop=True, inplace=True)
    
    concept_merged_df.loc[:,concept + '_gt_label'] = 0
    
    #sub_df = concept_merged_df.query("label=='Sure'")  before
    sub_df = concept_merged_df.query("label=='Sure' or label=='Hard Neg'") 
    positive_concept = sub_df[sub_df[concept]==1].index
    concept_merged_df.iloc[positive_concept, -1] = 1
    
    return concept_merged_df


def find_concept_in_columns(columns, concept):
    """find the columns corresponding to the concept

    Args:
        columns (str): set of columns
        concept (str): concept to find

    Returns:
        list: list of the columns corresponding to the concept
    """
    concept = "".join(concept.split(" "))
    new_cols = []
    for i, col_name in enumerate(columns):
        if col_name == 'video_name' or col_name == "label" or col_name =="reason" or col_name == "fold":
            new_cols.append(i)
        elif concept in col_name:
            new_cols.append(i)
    return new_cols

def compute_new_metrics(concept_merged_df):
    """compute metrics
    Args:
        concept_merged_df (pandas.DataFrame): df containing results for the experiment

    Returns:
        pandas.DataFrame: contains the metrics for each experiment
    """
    video_names = concept_merged_df.video_name
    concept_merged_df_cols = concept_merged_df.columns
    max_fold = concept_merged_df.fold.max()

    acc_train,acc_val,acc_test = [],[],[]
    recall_train, recall_val, recall_test = [],[],[]
    precision_train, precision_val, precision_test = [],[],[]
    f1_train, f1_val, f1_test = [],[],[]

    for col in concept_merged_df_cols:
   
        if "distance" in col:
            cs = col.split('_')
            if len(cs)==2:
                _, split_num = cs
                split_num=int(split_num)
            else:
                split_num = 0

      
            val_tmp = concept_merged_df.query("fold == @split_num").loc[:,["video_name", col, concept_merged_df_cols[-1]]]
            val_tmp.dropna(axis="index", inplace=True)
            
            train_tmp = concept_merged_df.query("fold != @split_num and fold != @max_fold ").loc[:,["video_name", col, concept_merged_df_cols[-1]]]
            train_tmp.dropna(axis="index", inplace=True)
           

            test_tmp = concept_merged_df.query('fold == @max_fold').loc[:, ["video_name", col, concept_merged_df_cols[-1]]]
            test_tmp.dropna(axis="index", inplace=True)
         
            
            ### calcul des metriques
            gt_labels_ = train_tmp.loc[:,col]>0
            gt_labels_ = gt_labels_.astype(int)
            pred_label= train_tmp.loc[:,concept_merged_df_cols[-1]]
            acc_train.append(accuracy_score(y_true = gt_labels_, y_pred = pred_label ))
            precision_train.append(precision_score(gt_labels_, pred_label, zero_division=0))
            recall_train.append(recall_score(gt_labels_, pred_label, zero_division=0))
            f1_train.append(f1_score(gt_labels_, pred_label, zero_division=0))

            gt_labels_ = val_tmp.loc[:,col]>0
            gt_labels_ = gt_labels_.astype(int)
            pred_label= val_tmp.loc[:,concept_merged_df_cols[-1]]
            acc_val.append(accuracy_score(y_true = gt_labels_, y_pred = pred_label ))
            precision_val.append(precision_score(gt_labels_, pred_label, zero_division=0))
            recall_val.append(recall_score(gt_labels_, pred_label, zero_division=0))
            f1_val.append(f1_score(gt_labels_, pred_label, zero_division=0))

            gt_labels_ = test_tmp.loc[:,col]>0
            gt_labels_ = gt_labels_.astype(int)
            pred_label= test_tmp.loc[:,concept_merged_df_cols[-1]]
            acc_test.append(accuracy_score(y_true = gt_labels_, y_pred = pred_label ))
            precision_test.append(precision_score(gt_labels_, pred_label, zero_division=0))
            recall_test.append(recall_score(gt_labels_, pred_label, zero_division=0))
            f1_test.append(f1_score(gt_labels_, pred_label, zero_division=0))
                
    train_mean_acc,val_mean_acc,test_mean_acc = np.mean(acc_train),np.mean(acc_val),np.mean(acc_test)
    train_mean_precision,val_mean_precision,test_mean_precision = np.mean(precision_train),np.mean(precision_val),np.mean(precision_test)
    train_mean_recall,val_mean_recall,test_mean_recall = np.mean(recall_train),np.mean(recall_val),np.mean(recall_test)
    train_mean_f1,val_mean_f1,test_mean_f1 = np.mean(f1_train),np.mean(f1_val),np.mean(f1_test)

    return [train_mean_acc,train_mean_precision, train_mean_recall, train_mean_f1],[val_mean_acc,val_mean_precision,val_mean_recall,val_mean_f1],[test_mean_acc, test_mean_precision, test_mean_recall, test_mean_f1]
