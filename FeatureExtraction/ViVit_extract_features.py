from utils_video import load_video
from utils_models import load_model
from transformers import VivitImageProcessor, VivitModel
from huggingface_hub import hf_hub_download
import numpy as np
import torch
import os


def vivit_extract_features(clip_dir, features_dir):

    movies = os.listdir(clip_dir)

    image_processor, model = load_model("google/vivit-b-16x2-kinetics400")


    for movie in movies:

        clips = os.listdir(os.path.join(clip_dir, movie))
        clips.sort()
        if not os.path.exists(os.path.join(features_dir, movie)):
            os.makedirs(os.path.join(features_dir, movie))

        for clip in clips:
            print(clip)
            video_name = os.path.join(clip_dir, movie, clip)
        
            video = load_video(video_name,max_frame = 32, target_size = (224,224) )

            inputs = image_processor(list(video), return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                
                embedding = outputs["last_hidden_state"]

                torch.save(embedding, os.path.join(features_dir, movie, clip[:-3]+"pth"))

