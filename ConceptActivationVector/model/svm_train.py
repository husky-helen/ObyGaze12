from model.util_svm import *
from dataset.dataset import XCLIP_Multi_SHN
from torch.utils.data import DataLoader
import json

def run_one_svm(annotation_path, split_number, concept, embedding_path,C, svm_path, df_path,exp_type):
    """Train the classification of the concept

    Args:
        annotation_path (str or pathlike): path to the csv containing the objectification annotation
        split_number (int): number of the split to use for validation 
        concept (str): concept to classify 
        embedding_path (str or pathlike): path to the dir containing the XCLIP embeddings
        C (float): parameters of the svm (see sklearn for more infos)
        svm_path (str or pathlike): path to the directory where svms will be saved 
        df_path (str or pathlike): path to the csv where the results for this experiments will be saved
        exp_type (str): name of the experiment
    """

    train_dataset = XCLIP_Multi_SHN(annotation_path, split_number, concept, embedding_path ,split_type = "train", exp_type=exp_type)
    val_dataset = XCLIP_Multi_SHN(annotation_path, split_number, concept, embedding_path ,split_type = "val", exp_type=exp_type)
    test_dataset = XCLIP_Multi_SHN(annotation_path, split_number, concept, embedding_path ,split_type = "test", exp_type=exp_type)
    
    
    train_loader = DataLoader(train_dataset, batch_size = len(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size = len(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size = len(test_dataset))

    train_elem = next(iter(train_loader))
    val_elem = next(iter(val_loader))
    test_elem = next(iter(test_loader))
    
    my_svm = create_svm(C, "linear", True)
    try:
        my_svm = train_svm(my_svm, train_elem[1], train_elem[2])


        train_dist = compute_distance2sephyperplan(my_svm, train_elem[1])
        val_dist = compute_distance2sephyperplan(my_svm, val_elem[1])
        test_dist = compute_distance2sephyperplan(my_svm, test_elem[1])


        concept_df = create_mini_df([train_dist, val_dist, test_dist], [train_elem[0], val_elem[0], test_elem[0]], [train_elem[2], val_elem[2], test_elem[2]])
        concept_df.to_csv(df_path, sep = ';')
        save_svm(my_svm, svm_path)
    except ValueError:
        print("Value error")

    except RuntimeError:
        print("Not enough data")
    
def run_svms(annotation_path, split_number, embedding_path, log_dir,exp_type, C_exp):
    """Train for each concept the classification

    Args:
        annotation_path (str or pathlike): path to the csv containing the objectification annotation
        split_number (int): number of the split to use for validation 
        embedding_path (str or pathlike): path to the dir containing the XCLIP embeddings
        exp_type (str): name of the experiment
        log_dir (str or pathlike): path to the directory where the results are going to be saved
        C_exp (dict): Dict that contains the C parameter for each concept
    """
    visual_concepts = ['Body', 'Type of plan', 'Clothes', 'Posture', 'Look', 'Activities', 'Exp of  emotion', 'Appearance']
    concept_df_paths = []
    
    for visual_concept in visual_concepts:
        C = C_exp[visual_concept]
        concept_dir = os.path.join(log_dir, visual_concept)
        if not os.path.exists(concept_dir):
            os.makedirs(concept_dir)

        svm_path = os.path.join(concept_dir, 'svm.pkl')
        df_path = os.path.join(concept_dir, 'mini_df.csv')
        concept_df_paths.append(df_path)
        run_one_svm(annotation_path, split_number, visual_concept, embedding_path,C, svm_path, df_path,exp_type)
    big_df = create_big_df(concept_df_paths, visual_concepts)
    big_df_path = os.path.join(log_dir,"big_df.csv" )
    big_df.to_csv(big_df_path, sep=";")



def run_crossval(annotation_path,  embedding_path,  log_dir,exp_type, C_exp):
    """Train with the cross-validation

    Args:
        annotation_path (str or pathlike): path to the csv containing the objectification annotation
        embedding_path (str or pathlike): path to the dir containing the XCLIP embeddings
        log_dir (str or pathlike): path to the directory where the results are going to be saved
        exp_type (str): Experiment type, defaults = "Sc_HNc_vs_conceptbar"
        C_exp (dict): Dict that contains the C parameter for each concept
    """

    annotation_df = pd.read_csv(annotation_path, sep=";", index_col=0)
    nb_crossval = annotation_df.fold.max() 

    for i in range(nb_crossval):
        crossval_log_dir = os.path.join(log_dir, str(i))
        run_svms(annotation_path, i, embedding_path, crossval_log_dir,exp_type, C_exp)


def run_multiparams(annotation_path, embedding_path,log_dir, C_opt_json):
    """Train the concept classification for several experiments

    Args:
        annotation_path (str or pathlike): path to the csv containing the objectification annotation
        embedding_path (str or pathlike): path to the dir containing the XCLIP embeddings
        log_dir (str or pathlike): path to the directory where the results are going to be saved
        C_opt_json (dict): path to the json that contains the C parameter for each concept
    """

    with open(C_opt_json,'r') as f:
        C_exp = json.load(f)
    
    precise_log_dir = os.path.join(log_dir, "Sc_HNc_vs_conceptbar")
    run_crossval(annotation_path,  embedding_path, precise_log_dir,"Sc_HNc_vs_conceptbar", C_exp)



