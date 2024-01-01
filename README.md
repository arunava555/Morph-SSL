# Morph-SSL: Self-Supervision with Longitudinal Morphing to Predict AMD Progression from OCT
This is the official Pytorch Implementation of  [Morph-SSL: Self-Supervision with Longitudinal Morphing to Predict AMD Progression from OCT](https://arxiv.org/abs/2304.08439) 

Morph-SSL is a self-supervised learning method developed to leverage a wider availability of unlabeled longitudinal OCT scans for training. It uses pairs of unlabeled scans acquired at irregular time-intervals from each subject to solve the *pretext* task of morphing the scan from the prior visit to the next (see step1_Train_MorphSSL_pretrain.ipynb for details). 

The efficacy of the learned feature representations is demonstrated on the challenging task of predicting the future risk of conversion of eyes (currently in the iAMD stage) to the late nAMD stage (see step3_Train_Downstream_Classifier.ipynb). 

Considering practical issues such as GPU memory and the computation time required in a 3D convolutional network, we also developed a lightweight CNN architecture (See /model/Encoder_model.py)
