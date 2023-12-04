
import torch
import numpy as np
import json
import os


from utils_models import load_model
from utils_video import load_video

def xclip_extract_video_features(video_dir, model_name, saving_path):

    processor, model = load_model(model_name=model_name)
    movies = os.listdir(video_dir)
    
    for movie in movies:
        print("Movie : ", movie)
        movies_path = os.path.join(video_dir, movie)

        clips = os.listdir(movies_path)
        clips.sort()

        for clip in clips:
            print(clip)
            video_name = os.path.join(movies_path, clip)
    
            # load video 
            video = load_video(video_name=video_name)
            inputs = processor(videos=list(video), return_tensors="pt")

            # get features with get_video_features
            video_features = model.get_video_features(**inputs)

            # save features
            total_saving_path = os.path.join(saving_path, movie, clip[:-3]+ "pth")

            if not os.path.exists(os.path.join(saving_path, movie)):
                os.makedirs(os.path.join(saving_path, movie))
            torch.save(video_features, total_saving_path)


