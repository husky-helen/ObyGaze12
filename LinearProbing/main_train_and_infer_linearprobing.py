from arg_pars import opt
from models.process_train import process_all_exps
from models.process_inference_tmp import run_inference
from arg_pars import opt




if __name__ == '__main__':
    
    features_dir = opt.features_dir
    annotation_dir = opt.annotation_dir
    training_saving_dir = opt.training_saving_dir
    infering_saving_dir = opt.infering_saving_dir
    root_checkpoints = opt.training_saving_dir

    process_all_exps(features_dir , annotation_dir, training_saving_dir)
    run_inference(root_checkpoints, annotation_dir, infering_saving_dir, features_dir)
