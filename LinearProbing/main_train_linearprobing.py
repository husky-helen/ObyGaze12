from arg_pars import opt
from models.process_train import process_all_exps

if __name__ == '__main__':
    
    feature_dir = opt.features_dir
    annotation_dir = opt.annotation_dir
    saving_dir = opt.training_saving_dir
    
    process_all_exps(feature_dir , annotation_dir, saving_dir)

