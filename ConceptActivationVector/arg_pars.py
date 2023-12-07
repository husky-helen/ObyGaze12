import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--annotation_df", default = "/home/xxx/ssd/data1/MovieGraphs/Meta/new_metas1/AnnotationsMerged/VisualFactors/02_EN_HN_S.csv")
parser.add_argument('--embedding_dir', default="/home/xxx/ssd/data1/Features/XCLIP/Visual")
parser.add_argument("--result_dir", default="/home/xxx/ssd/data1/CVPR_TEST/ResultsCAV")
parser.add_argument("--elected_svms_dir", default="/home/xxx/ssd/data1/RESULTS/SVMs_pickle_final17Nov/Elected")
parser.add_argument('--C_opt_json', default="/home/xxx/Code/CleanCVPR/ConceptActivationVector/model/C_opt.json")
parser.add_argument("--result_tree", default="/home/xxx/ssd/data1/CVPR_TEST/ResultsCAV/tree_res.csv")
parser.add_argument("--result_logreg", default="/home/xxx/ssd/data1/CVPR_TEST/ResultsCAV/logreg_res.csv")
opt = parser.parse_args()


