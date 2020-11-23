# Digital-Peter solution
Solution for Digital Peter competition.

![alt text](https://github.com/t0efL/Digital-Peter/blob/main/img.jpg)

## Overview

This is our team’s solution for Artificial Intelligence Journey 2020 Competition (Digital Peter track). This contest was about line-by-line recognition of Peter the Great’s manuscripts. The task is related to several AI technologies (Computer Vision, NLP, and knowledge graphs). Competition data was prepared by Sberbank of Russia, Saint Petersburg Institute of History (N.P.Lihachov mansion) of Russian Academy of Sciences, Federal Archival Agency of Russia and Russian State Archive of Ancient Acts.

https://ods.ai/competitions/aij-petr

## Files
1. [custom_functions.py](https://github.com/t0efL/Digital-Peter/blob/main/custom_functions.py) - custom functions, classes and augmentations we used for training and prediction.
2. [dict.json](https://github.com/t0efL/Digital-Peter/blob/main/dict.json) - dictionary of 9k words for post processing.
3. [large_dict.json](https://github.com/t0efL/Digital-Peter/blob/main/large_dict.json) - dictionary of 160k words for post processing, we were forced to give it up due to the submission runtime limit.
4. [hparams.py](https://github.com/t0efL/Digital-Peter/blob/main/hparams.py) - module containing hyperparameters and some functions.
5. [metadata.json](https://github.com/t0efL/Digital-Peter/blob/main/metadata.json) - file for docker-format submission.
6. [model1train.py](https://github.com/t0efL/Digital-Peter/blob/main/model1train.py) - code for training the first model (DenseNet161 with clean dataset).
7. [model2train.py](https://github.com/t0efL/Digital-Peter/blob/main/model2train.py) - code for training the second model (ResNext101 with clean dataset).
8. [model3train.py](https://github.com/t0efL/Digital-Peter/blob/main/model3train.py) - code for training the third model (ResNext101 with default dataset).

## Set up

## Training
We've used 3 different models for the final ensemble. So we have three different trainings. To run each of them use the following commands:

`python model1train.py`  // DenseNet161 with clean data

`python model2train.py`  // ResNext101 with clean data

`python model3train.py`  // ResNext101 with default data

Clean dataset can be found [here](https://drive.google.com/file/d/1Qki21iEcg_iwMo3kWuaHi5AlxxpLKpof/view).  
Default dataset can be found [here](https://drive.google.com/file/d/1GyeiNYTh3a1S-CukmLJmbLkAWjnSpmja/view?usp=sharing).

## Inference

`python ocr.py`

## Results

**Final Ensemble**:
1. DenseNet161 (with latest clean samples, CER 5.025, Val CER 4.553)
2. ResNeXt101 (with latest clean samples, CER 5.047, Val CER 4.750)
3. ResNeXt101 (with standard samples, CER 5.011, Val CER 4.711)

**Public LB scores**:

**Private LB scores**:


## More information

## Team
[Vadim Timakin](https://github.com/t0efL)  
[Maksim Zhdanov](https://github.com/xzcodes)
