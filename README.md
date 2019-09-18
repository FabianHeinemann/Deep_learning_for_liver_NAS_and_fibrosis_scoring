# Deep learning for liver NAS and fibrosis scoring

Repository for the upcoming publication "Deep learning enables pathologist-like scoring of NASH models"

## Requirements:

- CUDA compatible GPU
- Python 3.x
- Libraries: yaml, keras, tensorflow, matplotlib, pandas, sklearn

## Set-up:

1. Clone this repository to your local machine.
2. Download images and model weights from: https://osf.io/p48rd/ and extract them under ./model/ 
*Imporant note:* The OSF repository with images and CNN weights will be made public <b>after</b> acceptance of the publication. Contact: fabian.heinemann@boehringer-ingelheim.com if you like to get access before publication.

## Training a new model:

1. Complete train.yaml (or a copy). A least you need to set:
-- model_path (Path where the .h5 file with the CNN weights will be stored)
-- model_file_name (Filename of model)
-- ground_truth_path (Path where the tiles with the ground truth are located)
2. Run
> $python classify.py -c train.yaml

This will generate a new model file.

Please note, that trained models for ballooning, inflammation, steatosis and fibrosis have been uploaded to: https://osf.io/p48rd/ and there is no need to train a new model unless you want to add own data.
 
## Classification:

If you want to classify new samples, extract image patches under ./classification_data/exp_no/ in two subfolders:
- tiles/tiles/ (tiles for analysis of ballooning, inflammation, steatosis)
- tiles_big/tiles/ (tiles for analysis of fibrosis)
