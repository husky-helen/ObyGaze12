from arg_pars import opt
from ConceptActivationVector.model.svm_train import run_multiparams

if __name__ == '__main__':
    
    run_multiparams(annotation_path = opt.annotation_df, 
                    embedding_path = opt.embedding_dir,
                    log_dir = opt.result_dir,
                    C_opt_json = opt.C_opt_json)


