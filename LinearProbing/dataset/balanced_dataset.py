import torch
import os
import ast
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

class LinearProbing_KFold(Dataset):
    def __init__(self, visual_dir, gt_csv, split="train", nb_fold = 0, dico_level =  {'Easy Neg':0,'Hard Neg':1, "Not Sure":2, "Sure":3}):
        """Init the LinearProbing_KFold class

        Args:
            visual_dir (str or pathlike): path to the directory containing the extracted features
            gt_csv (str or pathlike): path to the csv containing the objectification annotation
            split (str, optional): "train" or "val" or "test". Defaults to "train".
            nb_fold (int, optional): _description_. Defaults to 0.
            dico_level (dict, optional): _description_. Defaults to {'Easy Neg':0,'Hard Neg':1, "Not Sure":2, "Sure":3}.
        """
        assert split in ["train", "val", "test"]
        
        self.visual_dir = visual_dir
        self.gt_csv = gt_csv
        self.split = split
        self.nb_fold = nb_fold
        self.dico_level = dico_level

        self.gt_df = pd.read_csv(self.gt_csv, index_col=0,sep=";")
        self.gt_df = self.gt_df.dropna(subset=["video_name"])
        self.get_label()
        self.weight = self.compute_weight()
        assert nb_fold < self.gt_df.fold.max()
        self.getsplit()

    def get_label(self):
        new_label = []
        for index, row in self.gt_df.iterrows():
            new_label.append(self.dico_level[row.label])
        self.gt_df["label"] = new_label


    def compute_weight(self):
        max_fold = self.gt_df.fold.max()
        sub_df = self.gt_df.query("fold!=@max_fold").label
        return torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(sub_df), y=sub_df))
        
        
    def getsplit(self):
        max_fold = self.gt_df.fold.max()
        if self.split == "test":

            self.gt_df = self.gt_df.query("fold==@max_fold")

        elif self.split == "val":
            self.gt_df = self.gt_df.query("fold==@self.nb_fold")
        
        else:
            self.gt_df = self.gt_df.query("fold != @max_fold and fold != @self.nb_fold")
            count_values = Counter(self.gt_df.label)
            nb_classes = list(count_values.values())
            names_classes = list(count_values.keys())
            folds_size  = []
            for name_classe in names_classes:

                df_train_val_class = self.gt_df.query("label==@name_classe")
                mini_fold_size = int(df_train_val_class.shape[0] * 0.1)
                folds_size.append(mini_fold_size)

            plus_grand_fold = np.max(folds_size)
            plus_petit_fold = np.min(folds_size)
            nb_folds_to_use = [round(plus_petit_fold *8/fold_size) for fold_size in folds_size]

            nb_splits = np.max(nb_folds_to_use) // np.min(nb_folds_to_use)
            val_fold = 1 + self.nb_fold % max_fold
            possible_train_splits = [j for j in range(1,10) if j != val_fold] * (nb_splits*15)
            split_folds = []
            for nb_fold_to_use in nb_folds_to_use:
                split_folds.append(possible_train_splits[self.nb_fold*nb_fold_to_use: (self.nb_fold+1)*nb_fold_to_use])
            
            list_idx_to_select = []
            for name_class, split_fold in zip(names_classes, split_folds):

                list_idx_to_select+=list(self.gt_df.query("label==@name_class and fold in @split_fold").index)

            self.gt_df = self.gt_df.loc[list_idx_to_select,:]

    def __len__(self):
        return self.gt_df.shape[0]
    
    def __getitem__(self, idx):
        gt_data = self.gt_df.iloc[idx,:]
        video_name = gt_data.video_name
        movie = gt_data.movie          
        label = gt_data.label
        one_hot_encoded = F.one_hot(torch.tensor(label), len(self.dico_level)).float()
 
        if not os.path.exists(os.path.join(self.visual_dir, movie, video_name[:-3]+"pth")):
            idx_new=torch.randint(low = 0, high = self.__len__(), size = (1,)).item()
            return self.__getitem__(idx_new)
        else:
            video_features = torch.load(os.path.join(self.visual_dir, movie, video_name[:-3]+"pth"), map_location="cpu")
            video_features.requires_grad = False
            shape_vf =  video_features.shape
            if len(shape_vf)>2:
                dim = shape_vf[-1]
                m = nn.AdaptiveMaxPool2d((1,dim))
                video_features = m(video_features)[0][0]
            else: video_features = video_features[0]
       
        return video_features, one_hot_encoded

