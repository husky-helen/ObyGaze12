from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import pickle
import os
import torch 
from model.utils_classiflevel import load_annotation, load_elected_svms, project_XCLIP_on_CAV



def train_final_tree(projected_df_,classif_exps, hyperparameter_dico):
    """Train a decision tree based on the hyperparameters found thanks to a cross-validation previously done. 
    The final tree is trained on the training and validation set since the hyperparameters are already tuned.

    Args:
        projected_df_ (pandas.DataFrame): contains objectification annotations and the projected values of the embedding on each CAV
        classif_exps (Dict): Experiments on which the tree should be trained (ex: {"EN_vs_S":["Easy Neg":0, "Sure":1]})
        hyperparameter_dico (Dict): Tree hyperparameters 

    Returns:
        pandas.DataFrame: results dataframe (metrics)
        Dict: {classif_exp: {exp_type: tree}}
    """

    final_arbre_res = {}

    final_liste_res = []


    max_fold = projected_df_.fold.max()

    for classif_exp, dico_level in classif_exps.items():
        
        final_arbre_res[classif_exp] = {}

        for exp_type in ["Sc_HNc_vs_conceptbar"]:
            
            projected_df_["new_label"] = projected_df_["label"].map(dico_level)
            projected_df_exp_type = projected_df_.query("exp_type==@exp_type")
            exp_df = projected_df_exp_type.dropna(subset="new_label")

            df_train = exp_df.query("fold != @max_fold")
            X_train= df_train.iloc[:, 2:10]
            nans = X_train[X_train.isna().any(axis=1)].index

            y_train = df_train.loc[:,"new_label"]

            df_test = exp_df.query("fold == @max_fold")
            X_test= df_test.iloc[:, 2:10]
            y_test = df_test.loc[:,"new_label"]


            max_depth, min_samples_leaf = hyperparameter_dico[classif_exp][exp_type]

            my_tree = DecisionTreeClassifier(random_state=42, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
            
            my_tree.fit(X_train, y_train)
            y_test_pred = my_tree.predict(X_test)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            f1_test = f1_score(y_test, y_test_pred)

            final_arbre_res[classif_exp][exp_type] = my_tree

            elem_list = [classif_exp, exp_type, max_depth, min_samples_leaf,accuracy_test,f1_test]
            final_liste_res.append(elem_list)
    res_final = pd.DataFrame(final_liste_res, columns = ["classif_exp","svm_exp", "max_depth","min_samples_leaf", "acc_test","f1_test"])

    return res_final, final_arbre_res


def infer_different_task(final_arbre_res, projected_df_, classif_exps, hyperparameter_dico):
    """Test the decision tree on tasks for which it has not necessarily been trained

    Args:
        final_arbre_res (Dict): {classif_exp:{"Sc_HNc_vs_conceptbar":decision_tree}}
        projected_df_ (pandas.DataFrame): _description_
        classif_exps (Dict): Experiments on which the tree should be trained (ex: {"EN_vs_S":["Easy Neg":0, "Sure":1]})
        hyperparameter_dico (Dict): Tree hyperparameters 

    Returns:
        pandas.DataFrame: contains metrics obtained on each task
    """
    lines_inferences = []
    max_fold = projected_df_.fold.max()
    for training_exptype, dico1 in final_arbre_res.items():

        for svm_exptype, my_tree in dico1.items():

            for predicting_exptype in ["EN_S", "ENHN_S"]:
                dico_level = classif_exps[predicting_exptype]
                projected_df_["new_label"] = projected_df_["label"].map(dico_level)
                projected_df_exp_type = projected_df_.query("exp_type==@svm_exptype")
                exp_df = projected_df_exp_type.dropna(subset="new_label")
                
                df_train = exp_df.query("fold != @max_fold")
                X_train= df_train.iloc[:, 2:10]
                nans = X_train[X_train.isna().any(axis=1)].index

                y_train = df_train.loc[:,"new_label"]

                df_test = exp_df.query("fold == @max_fold")
                X_test= df_test.iloc[:, 2:10]
                y_test = df_test.loc[:,"new_label"]


                max_depth, min_samples_leaf = hyperparameter_dico[training_exptype][svm_exptype]
            
                y_test_pred = my_tree.predict(X_test)
                accuracy_test = accuracy_score(y_test, y_test_pred)
                f1_test = f1_score(y_test, y_test_pred)

            

                elem_list = [training_exptype, svm_exptype, predicting_exptype,  max_depth, min_samples_leaf,accuracy_test,f1_test]
                lines_inferences.append(elem_list)

    res_inference = pd.DataFrame(lines_inferences, columns = ["training_exptype","svm_exp","predicting", "max_depth","min_samples_leaf", "acc_test","f1_test"])
    return res_inference

def tree_process(path_annotation,path_svm_elected, path_embedding, classif_exps, hyperparameter_dico):
    """Whole process

    Args:
        path_annotation (str or pathlike): Path to the csv file containing objectification annotation
        path_svm_elected (str or pathlike): Path to the dir storing svms elected from the cross-validation 
        path_embedding (str or pathlike): Path to the dir storing XCLIP embeddings
        classif_exps (Dict): Experiments on which the tree should be trained (ex: {"EN_vs_S":["Easy Neg":0, "Sure":1]})
        hyperparameter_dico (Dict): Tree hyperparameters 

    Returns:
        pandas.DataFrame: Final results
    """
    df_annotation = load_annotation(path_annotation)
    dico_svms = load_elected_svms(path_svm_elected)
    projected_df_ = project_XCLIP_on_CAV(df_annotation, path_embedding, dico_svms)
    _, final_arbre_res = train_final_tree(projected_df_,classif_exps, hyperparameter_dico)
    res_inference = infer_different_task(final_arbre_res, projected_df_,classif_exps, hyperparameter_dico)
    return res_inference