# Digital-Peter solution

![alt text](https://github.com/t0efL/Digital-Peter/blob/main/img.jpg)

## Overview

This is our team’s 2nd place solution for Artificial Intelligence Journey Junior 2020 Competition (Digital Peter track). This contest was about line-by-line recognition of Peter the Great’s manuscripts. The task is related to several AI technologies (Computer Vision, NLP, and knowledge graphs). Competition data was prepared by Sberbank of Russia, Saint Petersburg Institute of History (N.P.Lihachov mansion) of Russian Academy of Sciences, Federal Archival Agency of Russia and Russian State Archive of Ancient Acts.

## Files
1. [custom_functions.py](https://github.com/t0efL/Digital-Peter/blob/main/custom_functions.py) - custom functions, classes and augmentations we used for training and prediction.
2. [dict.json](https://github.com/t0efL/Digital-Peter/blob/main/dict.json) - dictionary of 9k words for post processing.
3. [large_dict.json](https://github.com/t0efL/Digital-Peter/blob/main/large_dict.json) - dictionary of 160k words for post processing, we were forced to give it up due to the submission runtime limit.
4. [hparams.py](https://github.com/t0efL/Digital-Peter/blob/main/hparams.py) - module containing hyperparameters and some functions.
5. [metadata.json](https://github.com/t0efL/Digital-Peter/blob/main/metadata.json) - file for docker-format submission.
6. [model1train.py](https://github.com/t0efL/Digital-Peter/blob/main/model1train.py) - code for training the first model (DenseNet161 with Smart Resize).
7. [model2train.py](https://github.com/t0efL/Digital-Peter/blob/main/model2train.py) - code for training the second model (ResNext101 with Smart Resize).
8. [model3train.py](https://github.com/t0efL/Digital-Peter/blob/main/model3train.py) - code for training the third model (ResNext101 with Default Resize).
9. [ocr.py](https://github.com/t0efL/Digital-Peter/blob/main/ocr.py) - ensemble inference.

## Set up

You can start using it by installing:

`$ git clone https://github.com/t0efL/Digital-Peter.git`

All the files containing code and hyperparameters from the original training. One thing you might want to change is working directory - the folder where the logs and the weights saved will be saved. You can do it in hparams.py. 

## Training
We've used 3 different models for the final ensemble. So we have three different trainings. To run each of them use the following commands:

`$ python model1train.py`  // DenseNet161 with Smart Resize

`$ python model2train.py`  // ResNext101 with Smart Resize

`$ python model3train.py`  // ResNext101 with Default Resize

By default, all the logs and weights will be saved in the "log/" folder. If you want to change this working directory, you can do it in hparams.py. We recommend you to train each model appoximatly 100 epochs; our training loop contains early stopping, so if the loss stops decreasing, the training will stop. Besides, by default the logs from each training are saved in the same folder, so we recommend you to clean log folder after each training or change working directory in hparams.py. Finally, we recommend you to take the weights from each training according the CER (this will be indicated in the name of the weights), not number of epochs.

*Approximate time for each training session - 10 hours (Google Colab Pro).*

## Inference

Put your weights in the folder as "weights1.pt", "weights2.pt" and "weights3.pt" or just download ours (link below) and run the following command:

`$ python ocr.py`

You'll find predictions in stdout and "/output" folder as well.

## Quick start

Check out [quickstart.ipynb](https://github.com/t0efL/Digital-Peter/upload) notebook for quick start.

## Results

Finally, we implemented ensemble technique for 3 backbones. We didn’t use our best model (CER: 5.011) with smart resize for this ensemble as its submission failed due to the time limit (moreover we kept our 9k dictionary instead of 160k dictionary due to the same problem).

**Final Ensemble**:
1. DenseNet161 (with Smart Resize, CER 5.025, Val CER 4.553)
2. ResNeXt101 (with Smart Resize, CER 5.047, Val CER 4.750)
3. ResNeXt101 (with Default Resize, CER 5.286, Val CER 4.711)

**Public scores**:
CER - 4.861, WER - 23.954, String Accuracy - 46.27

**Private scores**:
CER - 4.814, WER - 24.72, String Accuracy -	45.942

## More information
* Our article on the Medium (5 min read with solution explained):  [Link](https://xzcodes.medium.com/historical-manuscripts-recongition-using-attentionocr-ai-journey-2020-5c5db333799a)  
* Our full competition report (information about all of submissions and approaches):  [Link](https://docs.google.com/document/d/1yce8MAW8OxfuLEPgxq8yhqttRzegP3_mTp8rbKB0SFA/edit?usp=sharing)  
* We've used [OCR-transformer](https://github.com/vlomme/OCR-transformer) pipeline by Vladislav Kramarenko as the baseline.  
* The dataset: [Link](https://drive.google.com/file/d/1Qki21iEcg_iwMo3kWuaHi5AlxxpLKpof/view).   
* Our weights for each model: [Link](https://drive.google.com/drive/folders/1UYH9q7BvZnBEUL8VxbjpOr-5uZGljFCj?usp=sharing)

## Team
[Vadim Timakin](https://github.com/t0efL)  
[Maksim Zhdanov](https://github.com/maxnygma)
