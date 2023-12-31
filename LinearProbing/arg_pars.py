import argparse

parser = argparse.ArgumentParser()

# XCLIP
"""parser.add_argument('--features_dir', default="/home/xxx/ssd/data1/Features/XCLIP/Visual")
parser.add_argument('--annotation_dir', default= "/home/xxx/Code/CleanCVPR/LinearProbing/balanced_level")
parser.add_argument('--training_saving_dir', default="/home/xxx/Results/LinearProbing/XCLIP")
parser.add_argument('--model_type', default="XCLIP", help="XCLIP or ViVit or XCLIP_LSMDC")
parser.add_argument('--checkpoints', default =  "/home/xxx/Results/LinearProbing/XCLIP")
parser.add_argument('--infering_saving_dir', default="/home/xxx/Results/LinearProbing/Inference/XCLIP/")"""

# XCLIP_LSMDC
"""parser.add_argument('--features_dir', default="/home/xxx/ssd/data1/Features/LSMDC_XCLIP")
parser.add_argument('--annotation_dir', default= "/home/xxx/Code/CleanCVPR/LinearProbing/balanced_level")
parser.add_argument('--training_saving_dir', default="/home/xxx/Results/LinearProbing/LSMDC_XCLIP")
parser.add_argument('--model_type', default="XCLIP_LSMDC", help="XCLIP or ViVit or XCLIP_LSMDC")
parser.add_argument('--checkpoints', default =  "/home/xxx/Results/LinearProbing/LSMDC_XCLIP")
parser.add_argument('--infering_saving_dir', default="/home/xxx/Results/LinearProbing/Inference/LSMDC_XCLIP/")
"""
# ViVit
"""parser.add_argument('--features_dir', default="/home/xxx/ssd/data1/Features/ViVit")
parser.add_argument('--annotation_dir', default= "/home/xxx/Code/CleanCVPR/LinearProbing/balanced_level")
parser.add_argument('--training_saving_dir', default="/home/xxx/Results/LinearProbing/ViVit")
parser.add_argument('--model_type', default="ViVit", help="XCLIP or ViVit or XCLIP_LSMDC")
parser.add_argument('--checkpoints', default =  "/home/xxx/Results/LinearProbing/ViVit")
parser.add_argument('--infering_saving_dir', default="/home/xxx/Results/LinearProbing/Inference/ViVit/")"""

# MovieSet
parser.add_argument('--features_dir', default="/home/xxx/ssd/data1/Features/XCLIP/Visual")
parser.add_argument('--annotation_dir', default= "/home/xxx/Code/CleanCVPR/LinearProbing/models_movieset/annotation_files")
parser.add_argument('--training_saving_dir', default="/home/xxx/Results/TrainingMoviesSeparated/")
parser.add_argument('--model_type', default="XCLIP", help="XCLIP or ViVit or XCLIP_LSMDC")
parser.add_argument('--checkpoints', default =  "/home/xxx/Results/TrainingMoviesSeparated/")
parser.add_argument('--infering_saving_dir', default="/home/xxx/Results/InferenceMoviesSeparated/")
opt = parser.parse_args()
