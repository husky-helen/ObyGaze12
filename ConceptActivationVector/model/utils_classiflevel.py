import pandas as pd
import torch 
import os
import pickle

def load_annotation(path_annotation):
    """Load objectification annotation

    Args:
        path_annotation (str or pathlike): Path to the csv file containing objectification annotation

    Returns:
        pandas.DataFrame: DataFrame containing the objectification annotation
    """
    df_annotation = pd.read_csv(path_annotation, index_col=0, sep=";")
    df_annotation.dropna(subset = ["video_name"], inplace=True)
    return df_annotation

def load_elected_svms(path_svm_elected):
    """Load all the necessary svms

    Args:
        path_svm_elected (str): path to the dir storing svms elected from the cross-validation 

    Returns:
        Dict: {"Sc_HNc_vs_conceptbar": {concept: svm}}
    """
    dico_svms = {}
    for exp_type in ["Sc_HNc_vs_conceptbar"]:
        dico_svms[exp_type] = {}
        for concept in ["Body", "Type of plan", "Clothes", "Posture", "Look", "Activities", "Exp of  emotion", "Appearance"]:
            svm_dir = os.path.join(path_svm_elected, exp_type, concept)
            svm_name = os.listdir(svm_dir)[0]
            svm_path = os.path.join(svm_dir, svm_name)
            with open(svm_path, "rb") as file:
                svm = pickle.load(file)
            dico_svms[exp_type][concept] = svm
    return dico_svms

def project_XCLIP_on_CAV(df_annotation, path_embedding, dico_svms):
    """Project the XCLIP embedding on each CAV

    Args:
        df_annotation (pandas.DataFrame): contains objectification annotation
        path_embedding (str or pathlike): path to the dir storing XCLIP embeddings
        dico_svms (dict): dict containing the elected svms (see load_elected_svms function)

    Returns:
        pandas.DataFrame: contains objectification annotations and the projected values of the embedding on each CAV
    """
    bad_video_name = []
    lines = []
    for index, row in df_annotation.iterrows():

        movie = row.video_name.split("_")[0]
        embed_path = os.path.join(path_embedding, movie, row.video_name[:-3]+"pth")

        if os.path.exists(embed_path):
            embed = torch.load(embed_path)
            embed = embed.detach().numpy()

           
            for exp_type in ["Sc_HNc_vs_conceptbar"]:
                concept_vals = []
      
                for concept in ["Body", "Type of plan", "Clothes", "Posture", "Look", "Activities", "Exp of  emotion", "Appearance"]:
                    
                    my_svm = dico_svms[exp_type][concept]
                    svm_val = my_svm.decision_function(embed)[0]

                    concept_vals.append(svm_val)
                lines.append([row.video_name, exp_type]+concept_vals)
   
        else:
            bad_video_name.append(row.video_name)
    projected_df = pd.DataFrame(lines, columns = ["video_name", "exp_type","Body", "Type of plan", "Clothes", "Posture", "Look", "Activities", "Exp of  emotion", "Appearance" ])
    projected_df_ = projected_df.merge(df_annotation.loc[:,["video_name", "label", "reason", "fold"]], on="video_name", how="left")

    return projected_df_