import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd

class MovieBySet_KFold(Dataset):
    """MovieBySet_KFold dataset is a dataset which is not balanced, 
    clips from the same film are always in the same set. Designed for XCLIP features
    """

    def __init__(self, visual_dir, gt_csv,num_fold_val, num_fold_test,split="train"):
        """Init the MovieBySet_KFold dataset

        Args:
            visual_dir (str or pathlike): path to the directory containing the extracted features
            gt_csv (str or pathlike): path to the csv containing the objectification annotation
            num_fold_val (int): fold number to consider as the validation set
            num_fold_test (int): fold number to consider as the testing set
            split (str, optional): "train" or "val" or "test". Defaults to "train".
        """
        
 
        assert split in ["train", "val", "test"]

        
        self.visual_dir = visual_dir
        self.gt_csv = gt_csv
        self.split = split

     
        self.num_fold_val = num_fold_val
        self.num_fold_test = num_fold_test
        
        
        self.gt_df = pd.read_csv(self.gt_csv, index_col=0,sep=";")
        self.gt_df = self.gt_df.dropna(subset=["video_name"])

        
        assert self.num_fold_val <= self.gt_df.fold.max()
        assert self.num_fold_test <= self.gt_df.fold.max()
        self.getsplit()

        
        
    def getsplit(self):
        
        if self.split == "train":
            self.gt_df = self.gt_df.query("fold!=@self.num_fold_val and fold!=@self.num_fold_test")

        elif self.split == "val":
            self.gt_df = self.gt_df.query("fold==@self.num_fold_val")
        
        else:
            self.gt_df = self.gt_df.query("fold==@self.num_fold_test")

    def __len__(self):
        return self.gt_df.shape[0]
    
    def __getitem__(self, idx):
        gt_data = self.gt_df.iloc[idx,:]
        video_name = gt_data.video_name
        movie = gt_data.movie
        label = gt_data.label
        one_hot_encoded = F.one_hot(torch.tensor(label), 2).float()

        
        if not os.path.exists(os.path.join(self.visual_dir, movie, video_name[:-3]+"pth")):
            idx_new=torch.randint(low = 0, high = self.__len__(), size = (1,)).item()
            return self.__getitem__(idx_new)
        else:
            video_features = torch.load(os.path.join(self.visual_dir, movie, video_name[:-3]+"pth"))
            video_features.requires_grad = False
       
        return video_features[0], one_hot_encoded