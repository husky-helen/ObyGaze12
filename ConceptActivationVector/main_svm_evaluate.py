from arg_pars import opt
import os
from model.svm_evaluate import process_all_exp_all_concept2

if __name__ == '__main__':

    saving_path = os.path.join(opt.result_dir, "evaluate_svm_df.csv")
    process_all_exp_all_concept2(root_svm = opt.result_dir, 
                                 root_annotations = opt.annotation_df, 
                                 saving_path = saving_path
                                 )