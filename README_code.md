## 1. Movie Downloading

We are unable to provide you with the original clips and films, so we advise you to request access to the MovieGraphs dataset, which provides purchase links for the films as well as the breakdown of the films into clips.

## 2. Feature Extraction

We extracted features by using 2 models. Some parameters are set in FeatureExtraction/arg_pars.py. Please replace ours with yours or call the main files by passing your own parameters. In the **FeatureExtraction folder** there is an **environment.yml** file which can be used to create a conda environment with the libraries needed to run python files.

### 2.1. XCLIP
```
python main_extraction.py --model_type "XCLIP"
```

### 2.2 ViVit

```
python main_extraction.py --model_type "ViVit"
```

## 3. Concept Activation Vector

Some parameters are set in FeatureExtraction/arg_pars.py. Please replace ours path with yours or call the main files by passing your own parameters.

To **train** the svms : 
```
python main_svm_train.py
```

To **evaluate** the svms : 
```
python main_svm_evaluate.py
```

To **train and evaluate** the objectification level with a decision tree : 
```
python main_decision_tree.py
```

To **train and evaluate** the objectification level with a logistic regression
```
python main_logreg.py
```

## 4. Classification of the level of objectification using embedding extracted with XCLIP and ViVit

Default parameters can be changed directly in the **arg_parse.py** file in the **LinearProbing** folder or can be given when calling functions. Detailed metrics can be studied by launching tensorboard in the folder containing the logs of the experiment of interest. In the **LinearProbing folder** there is an **environment.yml** file which can be used to create a conda environment with the libraries needed to run python files.

To **train and infer** the model on the level objectification task
```
python main_train_and_infer_linearprobing.py
```

To **train** the model on the level objectification task
```
python main_train_linearprobing.py
```

To **infer** with the model on the level objectification task
```
python main_infer_linearprobing.py
```

To **train and infer** the model on the level objectification task with a dataset for which film clips are part of the same set 

```
python main_movieset.py
```

## 5. XCLIP retraining 

We followed the steps of the https://github.com/xuguohai/X-CLIP/tree/main repository
