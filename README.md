# Deep learning for liver NAS and fibrosis scoring

Repository for the upcoming publication "Deep learning enables pathologist-like scoring of NASH models". Nature scientific reports, 2019 (accepted).

**Input:**
- Tiles of microscopy images of of mouse or rat liver tissue, stained with Masson's trichrome (see below for details on tiles).

**Output / result:**
- Discrete pathologist-like liver scores on ballooning, inflammation, steatosis (NAS-score) and fibrosis with scores optimized to follow the scores of a given ground truth with minimal error (e.g. a pathologist).
- Continuous liver scores on ballooning, inflammation, steatosis (NAS-score) and fibrosis.
- Spatial feature distribution of ballooning, inflammation, steatosis (NAS-score) and fibrosis.

![alt text](https://github.com/FabianHeinemann/Deep_learning_for_liver_NAS_and_fibrosis_scoring/blob/master/image/Fig1_for_GIT.png)

## Requirements:

- Python 3.x
- Libraries: keras, tensorflow, matplotlib, pandas, sklearn, yaml (plus a number of common libraries, you will find them as imports)

## Set-up:

1. Clone this repository
2. Download images, model weights and other files from: https://osf.io/p48rd/ and extract all data under ./model/. Some subfolders contain .zip files, which need to be extracted by hand at their respective locations.

## Training a new model:

**Complete train.yaml (or a copy):** 

A least you need to set:
* model_path (Path where the .h5 file with the CNN weights will be stored)
* model_file_name (Filename of model)
* ground_truth_path (Path where the tiles with the ground truth are located)

**Start training:**
``` 
$python train.py -c train.yaml
```
This will generate a new model file.

Please note, that trained models for ballooning, inflammation, steatosis and fibrosis have been uploaded to: https://osf.io/p48rd/ and there is no need to train a new model unless you want to add own data.
 
## Classification of a scanned liver section:

**Prerequisite**

* Higher resolution tiles of Masson trichrome stained liver (0.44 µm/px, 299x299 px², for ballooning, inflammation and steatosis)
(Placed under: ./classification_data/<exp_no>/tiles/tiles/)

* Lower resolution tiles of Masson trichrome stained liver (1.32 µm/px, 299x299 px², for fibrosis)
(Placed under: ./classification_data/<exp_no>/big_tiles/tiles/)

* Completed settings file: classify_Kleiner_score.yaml (or a copy).

**Run:**
``` 
$python classify_Kleiner_score.py -c classify_Kleiner_score.yaml
```
This will create the following files

* <exp_no>_summary.csv
* <exp_no>_Ballooning_sub_score.csv
* <exp_no>_Inflammation_sub_score.csv
* <exp_no>_Steatosis_sub_score.csv
* <exp_no>_Fibrosis_score.csv

The first file contains results summarized per liver including discrete pathologist-like scores and continuous scores. The last four contain spatial data for each result (full results).

## Alternative (optional): 
(Classify each score individually:)

**Prerequisite**

* Higher resolution tiles of Masson trichrome stained liver (0.44 µm/px, 299x299 px², for ballooning, inflammation and steatosis)
(Placed under: ./classification_data/<exp_no>/tiles/tiles/)

* Lower resolution tiles of Masson trichrome stained liver (1.32 µm/px, 299x299 px², for fibrosis)
(Placed under: ./classification_data/<exp_no>/big_tiles/tiles/)

* Completed settings file: classify.yaml (or a copy).

**Run:**
``` 
$python classify.py -c classify.yaml
```
Two files will be created:
* <exp_no_score_name>_summary.csv
* <exp_no_score_name>.csv

The first file contains results summarized per liver including discrete pathologist-like scores and continuous scores. The second file contains spatial data for the score specified in classify.yaml (full results).

## Determination of new thresholds, and / or computation of evaluation parameters:

**Complete fit_threshold_settings.yaml**

**Run:**
``` 
$python fit_thresholds.py -c fit_threshold_settings.yaml
```
Output: 

* Thresholds to map continuous liver scores to discrete pathologist scores with minimized error (if fit_new_thresholds = True).
* Output of various evaluation parameters (mean absolute error, weighted precision, weighted F1, Cohens Kappa) comparing ground truth of NAS and fibrosis score with computeted result of NAS and fibrosis score

## Class activation maps:

**Run within jupyter notebook and complete paths inside:**
```
CNN_class_activation_map.ipynb
```
![alt text](https://github.com/FabianHeinemann/Deep_learning_for_liver_NAS_and_fibrosis_scoring/blob/master/class_activation_map_images/test/16_224_606_47_24_cam.png)
