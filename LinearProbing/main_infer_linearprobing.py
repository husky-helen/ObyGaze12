from models.process_inference_tmp import run_inference
from arg_pars import opt


if __name__ == '__main__':
    
    root_checkpoints = opt.checkpoints
    root_annotations = opt.annotation_dir
    root_logger = opt.infering_saving_dir
    features_dir = opt.features_dir

    infering_saving_dir = opt.infering_saving_dir
    root_checkpoints = opt.training_saving_dir

    
    run_inference(root_checkpoints, root_annotations, infering_saving_dir, features_dir)
    

