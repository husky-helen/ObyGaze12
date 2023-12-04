import pandas as pd
import os 

from model.utils_evaluate import compute_clips_ratio, compute_merged_df,gt_from_exptype, find_concept_in_columns, compute_new_metrics


def process_all_exp_all_concept2(root_svm, root_annotations, saving_path):
    """Evaluate the classification of each concept, by using the svm previously trained (works for several experiments)

    Args:
        root_svm (str or pathlike): path to the directory where the trained svm are stored
        root_annotations (str or pathlike): path to the csv containing objectification annotations
        saving_path (str or pathlike): path to the file where the evaluation results should be stored (.csv)
    """

   
    total_results = pd.DataFrame([])

    exp_types = os.listdir(root_svm)

    for exp_type in exp_types:

        
        dico_infos = { "kernel" : "linear", "threshold":"02", "levels": "EN_HN_S"}
        dico_infos["annotator"] = "merged"
        dico_infos["exp_type"] = exp_type
        dico_infos["exp_name"] = "linear_02_EN_HN_S"
        
        th = dico_infos["threshold"]
        levels = dico_infos["levels"]
        total_exp_name = dico_infos["annotator"] + "_" + dico_infos["exp_type"] + "_" + dico_infos["exp_name"]
        path_svm_experiment = os.path.join(root_svm, exp_type)


        annot_path = root_annotations
        result_exp= process_one_exp_all_concept(annot_path, path_svm_experiment, dico_infos)
        total_results = pd.concat([total_results, result_exp], axis=0)
        
    total_results.to_csv(saving_path, sep=";")

    
def process_one_exp_all_concept(annot_path, path_svm_experiment, exp_infos):
    """Evaluate one experiment (compute metrics such as accuracy or f1 for each concept)

    Args:
        annot_path (str or pathlike): path to the csv containing objectification annotations
        path_svm_experiment (str or pathlike): path to the directory containing the trained svm for a specific experiment
        exp_infos (dict): dict containing infos about the experiment

    Returns:
        pandas.DataFrame: DataFrame containing the averaged metrics for each concept
    """
    visual_concepts = ['Body', 'Type of plan', 'Clothes', 'Posture', 'Look', 'Activities', 'Exp of  emotion', 'Appearance']
    merged_df = compute_merged_df(annot_path, path_svm_experiment)
    
    result_exp = pd.DataFrame([])
    for visual_concept in visual_concepts:
        result_df= process_one_exp_one_concept(merged_df.copy(deep=True), exp_infos, visual_concept)
        result_exp = pd.concat([result_exp, result_df], axis=0)
   
    return result_exp


def process_one_exp_one_concept(merged_df, exp_infos, concept, K = 15):
    """Evaluate the classification of one specific concept

    Args:
        merged_df (pd.DataFrame): Dataframe containing infos for each cross validation
        exp_infos (dict): dict containing infos about the experiment
        concept (str): concept to evaluate
        K (int, optional): Not really used here. Defaults to 15.

    Returns:
        pandas.DataFrame: Dataframe containing averaged metrics for one concept
    """
    
    exp_type = exp_infos["exp_type"]
    annotator = exp_infos["annotator"]
    exp_name = exp_infos["exp_name"]
    threshold = exp_infos["threshold"]
    levels = exp_infos["levels"]


    merged_df_col = merged_df.columns
    cols_concept = find_concept_in_columns(merged_df_col,concept)

    concept_merged_df = merged_df.iloc[:,cols_concept]
    concept_merged_df.reset_index(drop=True, inplace=True)


    concept_merged_df=gt_from_exptype(concept_merged_df, exp_type, concept)

    

    nb_split= concept_merged_df.fold.max()

    train_metrics, val_metrics, test_metrics = compute_new_metrics(concept_merged_df)

 
    train_mean_acc,train_mean_precision, train_mean_recall, train_mean_f1 = train_metrics
    val_mean_acc,val_mean_precision,val_mean_recall,val_mean_f1 = val_metrics
    test_mean_acc, test_mean_precision, test_mean_recall, test_mean_f1 = test_metrics

    train_val_df = concept_merged_df.query("fold != @nb_split and fold != @nb_split-1")
    test_df = concept_merged_df.query("fold == @nb_split or fold == @nb_split-1")


    _, mean_ratio_train = compute_clips_ratio(train_val_df, K)

    _, mean_ratio_test = compute_clips_ratio(test_df, K)


    results_ligne = [[exp_name,exp_type, annotator, threshold, levels, concept,
                      train_mean_acc,train_mean_precision, train_mean_recall, train_mean_f1, mean_ratio_train,
                      val_mean_acc, val_mean_precision, val_mean_recall, val_mean_f1 ,
                      test_mean_acc, test_mean_precision, test_mean_recall, test_mean_f1, mean_ratio_test]]

    columns = ["exp_name","exp_type", "annotator", "threshold", "levels", "concept",
               "mean_acc", "mean_precision", "mean_recall", "mean_f1", "mean_ratio",
               "val_mean_acc", "val_mean_precision", "val_mean_recall", "val_mean_f1",
               "test_mean_acc", "test_mean_precision", "test_mean_recall", "test_mean_f1", "test_mean_ratio"]
    results_df = pd.DataFrame(results_ligne, columns = columns)

    return results_df

