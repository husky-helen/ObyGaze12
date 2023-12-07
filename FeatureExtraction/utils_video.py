from decord import VideoReader, cpu
import numpy as np
import cv2

def calculate_frame_indices(video_length, num_frames_to_extract):
    """
    Args:
        video_length (int): number of frame of the video
        num_frames_to_extract (int): number of frame to extract from the video

    Returns:
        list of int: list of the frames' index to keep 
    """

   
    interval = video_length / num_frames_to_extract
    frame_indices = []

    for i in range(num_frames_to_extract):
        frame_indices.append(int(i * interval))
        
    return frame_indices

def load_video(video_name,max_frame = 32, target_size = (224,224) ):
    """_
    Args:
        video_name (str or pathlike): path to the location of the video
        max_frame (int, optional): number of frame to keep from the video. Defaults to 32.

    Returns:
        numpy array : batch numpy array of the frame extracted (max_frame,H,W,C )
    """
    videoreader = VideoReader(video_name, num_threads=1, ctx=cpu(0))
    videoreader.seek(0)
    video_length = len(videoreader)
    indices_ = calculate_frame_indices(video_length, max_frame)
    video_frames = videoreader.get_batch(indices_).asnumpy()
    resized_frames = np.array([cv2.resize(frame, target_size) for frame in video_frames])
    return resized_frames #Old : video_frames
