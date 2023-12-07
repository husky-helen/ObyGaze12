# 1. Repository architecture 
This is the global architecture of the repository 

* Feature Extraction
* Linear Probing
  * balanced_level
  * dataset
  * models
  * models_movieset   
* Concept Activation Vector
  * dataset
  * model

There are also 2 READMEs : 
* README_data.md 
* README_code.md
  
# 2. Data

The dataset is stored in the file named **ObyGaze12_thresh_02.csv**.

In the LinearProbing/balanced_level folder there are the files (with the split) used in the experiment.

In the LinearProbing/models_movieset/annotation_files there are the files (with the split) used when the experiment requires a dataset where clips in the training, validation and testing set cannot originate from the same movie.

Further details can be found in the README_data.md.

# 3. Code 

All the code is in the 3 following folders (LinearProbing, FeatureExtraction, ConceptActivationVector). 

Subfolders named **dataset** contain classes and functions which allow to manipulate the dataset.

Subfolders named **models** contain classes and functions which allow to manipulate the model.

Further details can be found in the README_code.md.

