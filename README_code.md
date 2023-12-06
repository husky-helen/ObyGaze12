## 1. Movie Downloading

We are unable to provide you with the original clips and films, so we advise you to request access to the MovieGraphs dataset, which provides purchase links for the films as well as the breakdown of the films into clips.

## 2. Feature Extraction

We extracted features by using 2 models. Path to the clips, embeddingd are stored in FeatureExtraction/arg_pars.py. Please replace our path with yours or call the main files by passing your own paths.

### 2.1. XCLIP
```
python main_extraction.py --model_type "XCLIP"
```

### 2.2 ViVit

```
python main_extraction.py --model_type "ViVit"
```

## 3. Concept Activation Vector

Path to the clips, embeddingd are stored in FeatureExtraction/arg_pars.py. Please replace our path with yours or call the main files by passing your own paths.

To train the svms : 
```
python main_svm_train.py
```

To evaluate the svms : 
```
python main_svm_evaluate.py
```

To train and evaluate the objectification level with a decision tree : 
```
python main_decision_tree.py
```

To train and evaluate the objectification level with a logistic regression
```
python main_logreg.py
```

## 4. Classification of the level of objectification using embedding extracted with XCLIP and ViVit

Default parameters can be changed directly in the **arg_parse.py** file in the **LinearProbing** folder or can be given when calling functions. If you want details of the metrics you can launch tensorboard in the folder containing the logs of the experiment of interest.

To train and infer the model on the level objectification task
```
python main_train_and_infer_linearprobing.py
```

To train the model on the level objectification task
```
python main_train_linearprobing.py
```

To infer with the model on the level objectification task
```
python main_infer_linearprobing.py
```

To train and infer the model on the level objectification task with a dataset for which film clips are part of the same set 

```
python main_movieset.py
```

## 5. XCLIP retraining 

We followed the steps of the https://github.com/xuguohai/X-CLIP/tree/main repository
