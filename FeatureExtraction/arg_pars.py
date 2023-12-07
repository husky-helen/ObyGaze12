import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--video_dir', default="/home/xxx/ssd/data1/Clips_mp4_2/")
parser.add_argument('--feature_dir', default="/home/xxx/ssd/data1/Features/XCLIP/Visual")
parser.add_argument('--model_type', default="XCLIP")
parser.add_argument('--model_name', default = "microsoft/xclip-base-patch16-zero-shot")


opt = parser.parse_args()
