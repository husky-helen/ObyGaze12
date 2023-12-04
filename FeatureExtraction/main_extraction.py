from XCLIP_extract_features import xclip_extract_video_features
from ViVit_extract_features import vivit_extract_features
from arg_pars import opt


if __name__ == '__main__':

    if opt.model_type == "XCLIP":
        xclip_extract_video_features(opt.video_dir, opt.model_name, opt.feature_dir)

    elif opt.model_type == "ViVit":
        vivit_extract_features(opt.video_dir, opt.feature_dir)

    else: 
        assert False


