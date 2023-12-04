from model.utils_tree import tree_process
from arg_pars import opt
import pandas as pd


hyperparameter_dico = {'ENHN_S': {'Sc_HNc_vs_conceptbar': (4, 6)},
 'EN_S': {'Sc_HNc_vs_conceptbar': (8, 6)},
 'HN_S': {'Sc_HNc_vs_conceptbar': (10, 10)}}

classif_exps = {"ENHN_S":{"Hard Neg":0, "Easy Neg":0, "Sure":1}, "EN_S":{"Easy Neg":0, "Sure":1}, "HN_S":{"Hard Neg":0, "Sure":1}}


if __name__ == '__main__':

    res_df = tree_process(path_annotation = opt.annotation_df,
                 path_svm_elected = opt.elected_svms_dir, 
                 path_embedding = opt.embedding_dir, 
                 classif_exps = classif_exps,
                 hyperparameter_dico=hyperparameter_dico)

    res_df.to_csv(opt.result_tree)