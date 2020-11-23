# Digital-Peter solution
Solution for Digital Peter competition.

![alt text](https://github.com/t0efL/Digital-Peter/blob/main/img.jpg)

## Overview

This is our team’s solution for Artificial Intelligence Journey 2020 Competition (Digital Peter track). This contest was about line-by-line recognition of Peter the Great’s. The task is related to several AI technologies (Computer Vision, NLP, and knowledge graphs). Competition data was prepared by Sberbank of Russia, Saint Petersburg Institute of History (N.P.Lihachov mansion) of Russian Academy of Sciences, Federal Archival Agency of Russia and Russian State Archive of Ancient Acts.

https://ods.ai/competitions/aij-petr

## Set up

## Training
We've used 3 different models for the final ensemble. So we have three different trainings. To run each of them use the following commands:

`python model1train.py`  // DenseNet161 with clear data

`python model2train.py`  // ResNext101 with clear data

`python model3train.py`  // ResNext101 with default data

Clear dataset can be found [here](https://drive.google.com/file/d/1Qki21iEcg_iwMo3kWuaHi5AlxxpLKpof/view).  
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
