# Visual Objectification in Films: Towards a New AI Task for Video Interpretation



## Citing this work

```
@inproceedings{,
author = {Tores, Julie and Sassatelli, Lucile and Wu, Hui-Yin and Bergman, Clement and Andolfi, Léa and Ecrement, Victor and Precioso, Frédéric and Devars, Thierry and Guaresi, Magali and Julliard, Virginie and Lecossais, Sarah},
title = {Visual Objectification in Films: Towards a New AI Task for Video Interpretation},
year = {2024},
isbn = {},
publisher = {},
address = {},
url = {
},
doi = {},
booktitle = {},
pages = {},
numpages = {},
keywords = {visual objectification, films, video interpretation},
location = {},
series = {}
}
```

## Authors

- [Julie Tores](mailto:julie.tores@univ-cotedazur.fr) - Université Côte d'Azur, CNRS, Inria, I3S, France
- Lucile Sassatelli - Université Côte d'Azur, CNRS, I3S, France, Institut Universitaire de France
- Hui-Yin Wu - Université Côte d’Azur, Inria, France
- Clement Bergman - Université Côte d’Azur, Inria, France
- Léa Andolfi - Sorbonne Université, GRIPIC
- Victor Ecrement - Université Côte d’Azur, Inria, France
- Frédéric Precioso - Université Côte d'Azur, CNRS, Inria, I3S, France
- Thierry Devars - Sorbonne Université, GRIPIC
- Magali Guaresi - Université Côte d’Azur, CNRS, BCL, France
- Virginie Julliard - Sorbonne Université, GRIPIC
- Sarah Lecossais - Université Sorbonne Paris Nord, LabSIC

## Repository structure

### Datasets

The dataset that we used to train and evaluate our model are directly included in this repository. 
The dataset is stored in the file named **ObyGaze12_thresh_02.csv**.

In the LinearProbing/balanced_level folder there are the files (with the split) used in the experiment.
In the LinearProbing/models_movieset/annotation_files there are the files (with the split) used when the experiment requires a dataset where clips in the training, validation and testing set cannot originate from the same movie.

Further details can be found in the README_data.md.


### Model

The model was implemented using PyTorch.

All the code is in the 3 following folders: LinearProbing, FeatureExtraction, ConceptActivationVector. 
Subfolders named **dataset** contain classes and functions which allow to manipulate the dataset.
Subfolders named **models** contain classes and functions which allow to manipulate the model.

Further details can be found in the README_code.md.


## Requirements
Two files discribing the environment used for the feature extraction and the linear probing can be found at  `FeatureExtraction/environment.yml` and  `LinearProbing/environment.yml`
The versions specified in this file are the ones that have been tested to work with our code, but other versions may work.
Regarding the version of PyTorch that is specified, you may need to change the pytorch version according to your CUDA version. 


