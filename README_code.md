## 1. Movie Downloading

We are unable to provide you with the original clips and films, so we advise you to request access to the MovieGraphs dataset, which provides purchase links for the films as well as the breakdown of the films into clips.

## 2. Feature Extraction

This section is dedicated to feature extraction with 2 foundation models (X-CLIP and ViVit).
We extracted features from the movie clips mentioned above by using these 2 models. 

In the **FeatureExtraction folder** there is :
* an **environment.yml** file which can be used to create a conda environment with the libraries needed to run python files.
* **utils_video.py** and **utils_models.py** contain auxiliary functions.
* **ViVit_extract_features.py** and **XCLIP_extract_features.py** contain the process of feature extraction for each model.
* **main_extraction.py** contains the process of feature extraction for both models.
To run the following 2 commands, you need to be in the FeatureExtraction folder.
Some parameters are set in FeatureExtraction/arg_pars.py. Please replace ours with yours or call the main files by passing your own parameters. 

### 2.1. Feature extraction with X-CLIP

To extract features with the X-CLIP model, execute this command in the FeatureExtraction folder.

```
python main_extraction.py --model_type "XCLIP"
```

### 2.2 Feature extraction with ViVit

To extract features with the ViVit model, execute this command in the FeatureExtraction folder.

```
python main_extraction.py --model_type "ViVit"
```

## 3. Concept Activation Vector

Some parameters are set in FeatureExtraction/arg_pars.py. Please replace our path with yours or call the main files by passing your own parameters.

To **train** the SVMs : 
```
python main_svm_train.py
```

To **evaluate** the SVMs : 
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

## 4. Classification of the level of objectification using embedding extracted with X-CLIP and ViVit

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

## 5. X-CLIP retraining 

We followed the steps of the https://github.com/xuguohai/X-CLIP/tree/main repository
