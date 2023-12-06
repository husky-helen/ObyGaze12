from arg_pars import opt
from models_movieset.process_inference import all_proces_inference
from models_movieset.process_train import all_process_train

if __name__ == '__main__':
    
    features_dir = opt.features_dir
    annotation_dir = opt.annotation_dir
    training_saving_dir = opt.training_saving_dir
    infering_saving_dir = opt.infering_saving_dir

    all_process_train(visual_dir = features_dir, 
                      annotation_dir =annotation_dir , 
                      saving_dir = training_saving_dir)
    

    all_proces_inference(visual_dir = features_dir, 
               checkpoints = training_saving_dir, 
               annotation_dir=annotation_dir,
               saving_dir = infering_saving_dir)
