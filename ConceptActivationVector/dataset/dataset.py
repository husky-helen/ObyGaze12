import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class XCLIP_Multi_SHN(Dataset):
    def __init__(self, annotation_path, split_number, concept, embedding_path, split_type = "train", exp_type = "Sc_HNc_vs_conceptbar"):
        """Initiate XCLIP_Multi_SHN class

        Args:
            annotation_path (str or pathlike): path to the csv containing the objectification annotation and the splits
            split_number (int): split number equivalent to the number of the fold to use for the validation
            concept (str): concept to clasdify
            embedding_path (str or pathlike): path to the folder containing the features extracted from XCLIP
            split_type (str, optional): train, val or test. Defaults to "train".
            exp_type (str, optional):experiment type. Defaults to "Sc_HNc_vs_conceptbar".
        """
        
        assert split_type in ["train", "val", "test"]
        assert concept in ['Body', 'Type of plan', 'Clothes', 'Posture', 'Look', 'Activities', 'Exp of  emotion', 'Appearance']
        assert exp_type in ["Sc_HNc_vs_EN", "Sc_HNc_vs_conceptbar","Sc_HNc_vs_conceptbar_inter_ENbar"]
        
        self.concept = concept
        self.split_type = split_type
        self.exp_type = exp_type
        self.embedding_path = embedding_path
        self.annotation_df = pd.read_csv(annotation_path, index_col=0, sep=";")
        self.annotation_df = self.annotation_df.dropna(subset=['video_name'])
        self.annotation_df = self.annotation_df.query("video_name!='False'")
        self.annotation_df.reset_index(drop=True, inplace=True)
        self.create_concept_col()
        self.annotation_df.reset_index(drop=True, inplace=True)
        self.select_sub_df()
        self.annotation_df.reset_index(drop=True, inplace=True)
        
        self.max_fold =self.annotation_df.fold.max()
        self.split_number = split_number
        self.get_split()
        self.embeddings = self.load_embeddings()
        
        assert self.split_number < self.max_fold 

    def create_concept_col(self):
        concept_bool = []
        for i in range(self.annotation_df.shape[0]):

            elem = self.annotation_df.iloc[i,:]
            if self.concept in elem["reason"]: concept_bool.append(1)
            else: concept_bool.append(0)

        self.annotation_df["concept_bool"] = concept_bool

    
    def select_sub_df(self):

        if self.exp_type == "Sc_HNc_vs_EN": self.annotation_df = self.annotation_df.query("( (label == 'Sure' and concept_bool==1) or (label == 'Hard Neg' and concept_bool==1)) or label == 'Easy Neg'")
        elif self.exp_type == "Sc_HNc_vs_conceptbar": self.annotation_df = self.annotation_df.query("label == 'Sure' or label == 'Easy Neg' or label=='Hard Neg'")
        elif self.exp_type == "Sc_HNc_vs_conceptbar_inter_ENbar": self.annotation_df = self.annotation_df.query("label == 'Sure' or label=='Hard Neg'")
        else: assert False

    def get_label(self, level_obj, concept_bool):

        if level_obj == "Sure" and concept_bool == 1: return 1
        elif level_obj == "Hard Neg" and concept_bool ==1: return 1
        return 0


    def load_embeddings(self):
        tensor_list = []
        bad_index = []
        for index, elem in self.annotation_df.iterrows():
            movie = elem.movie
            clip_name = elem.video_name
            
   
        
            embed_path = os.path.join(self.embedding_path, movie, clip_name[:-3]+'pth')
            if not os.path.exists(embed_path):
                bad_index.append(index)
            else:
                embed = torch.load(embed_path)
                embed.requires_grad=False
                tensor_list.append(embed)
        
        self.annotation_df.drop(bad_index, inplace = True)
        if len(tensor_list)==0: print("error")
        embeddings = torch.cat(tensor_list, dim = 0)
        
        return embeddings

    def get_split(self):

        if self.split_type == 'test':
            self.annotation_df = self.annotation_df.query("fold==@self.max_fold")
        
        elif self.split_type == 'val':
            self.annotation_df = self.annotation_df.query("fold==@self.split_number")
        else:
            self.annotation_df = self.annotation_df.query("fold!=@self.max_fold and fold!=@self.split_number")
        self.annotation_df.reset_index(drop=True,inplace=True)
    
    def __len__(self):
        return self.annotation_df.shape[0]
    

    def __getitem__(self, idx):

        elem = self.annotation_df.iloc[idx,:]
        
        clip_name = elem["video_name"]
        
        label = self.get_label(level_obj=elem["label"], concept_bool=elem["concept_bool"])
        embed = self.embeddings[idx,:]
        return clip_name, embed,label
