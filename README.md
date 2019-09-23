# Deep learning for liver NAS and fibrosis scoring

Repository for the upcoming publication "Deep learning enables pathologist-like scoring of NASH models".

Required input:
- Tiles of microscopy images of of mouse or rat liver tissue, stained with Masson's trichrome (see below for details on tiles).

Output / result: 
- Discrete pathologist-like liver scores on ballooning, inflammation, steatosis (NAS-score) and fibrosis with scores optimized to follow the scores.
- Continous liver scores on ballooning, inflammation, steatosis (NAS-score) and fibrosis.
- Spatial feature distribution of ballooning, inflammation, steatosis (NAS-score) and fibrosis.

![alt text](https://github.com/FabianHeinemann/Deep_learning_for_liver_NAS_and_fibrosis_scoring/blob/master/image/Fig1_for_GIT.png)

## Requirements:

- Python 3.x
- Libraries: keras, tensorflow, matplotlib, pandas, sklearn, yaml (plus a number of common libraries, you will find them as imports)

## Set-up:

1. Clone this repository
2. Download images, model weights and other files from: https://osf.io/p48rd/ and extract them under ./model/

*Important note:* The OSF repository with images and CNN weights will be made public <b>after</b> acceptance of the publication. Contact: fabian.heinemann@boehringer-ingelheim.com if you like to get access before publication (e.g as a reviewer).

## Training a new model:

1. Complete train.yaml (or a copy). A least you need to set:
* model_path (Path where the .h5 file with the CNN weights will be stored)
* model_file_name (Filename of model)
* ground_truth_path (Path where the tiles with the ground truth are located)

2. Run:
``` 
$python train.py -c train.yaml
```
This will generate a new model file.

Please note, that trained models for ballooning, inflammation, steatosis and fibrosis have been uploaded to: https://osf.io/p48rd/ and there is no need to train a new model unless you want to add own data.
 
## Classification of a scanned liver section:

1. Prerequisite: 
Scanned liver slide stained with Masson's trichrome and cut into tiles:
* Higher resolution tiles (0.44 µm/px, 299x299 px², for ballooning, inflammation and steatosis)
(Placed under: ./classification_data/<exp_no>/tiles/tiles/)

* Lower resolution tiles (1.32 µm/px, 299x299 px², for fibrosis)
(Placed under: ./classification_data/<exp_no>/big_tiles/tiles/)

<exp_no> is your experiment id.

2. Complete classify.yaml (or a copy). A least you need to set:
* model_path (Path where the .h5 file with the CNN weights is located)
* model_file_name (Filename of model)
* list_of_classes (add correct names of individual classes for the respective model, see example in yaml)
* thresholds_json (Thresholds json file (expected under model path))
* ground_truth_path (Path where the tiles with the ground truth are located)
* tile_path (Path where the tiles are located)
* score_name (Name of the score in output column)
* results_path (Path where results are written to)
* experiment_name (Name of experiment and readout)

3. Run:
``` 
$python classify.py -c classify.yaml
```
Two files will be created:
* <experiment_score_name>_summary.csv
* <experiment_score_name>.csv

## Determination of new thresholds, and / or computation of evaluation parameters:

1. Complete fit_threshold_settings.yaml

2. Run:
``` 
$python fit_thresholds.py -c fit_threshold_settings.yaml
```

Output: 
* Thresholds to map continuous liver scores to discrete pathologist scores with minimized error (if fit_new_thresholds = True).
* Output of various evaluation parameters (mean absolute error, weighted precision, weighted F1, Cohens Kappa) comparing ground truth of NAS and fibrosis score with computeted result of NAS and fibrosis score

## Class activation maps:

1. Run within jupyter notebook and complete paths inside:
```
CNN_class_activation_map.ipynb
```
![alt text](https://github.com/FabianHeinemann/Deep_learning_for_liver_NAS_and_fibrosis_scoring/blob/master/class_activation_map_images/test/16_224_606_47_24_cam.png)
