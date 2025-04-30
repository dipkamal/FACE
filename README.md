# FACE: Faithful Automatic Concept Extraction

This repository contains code for the paper "FACE: Faithful Automatic Concept Extraction" submitted to the The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025).

## Introduction 
 Existing automatic concept discovery methods naively rely on clustering or factorization of activation vectors, raising concerns about the faithfulness of extracted concepts to the model's true decision-making process. To address this, we propose FACE, Faithful Automatic Concept Extraction (FACE), a principled approach combining non-negative matrix factorization (NMF) with a Kullback-Leibler (KL) divergence constraint to align concept-based reconstructions with model predictions, ensuring faithful explanations.


## Requirements
The current implementation is on PyTorch and requires some existing libraries that you can install using the following file:
```
pip install -r requirements.txt
```
## Models under evaluation 

We pretrained ResNet34 and MobileNetV2 for ImageNet and COCO. For CelebA dataset, we train ResNet and MobileNet models on an exclusive set of four facial attributes. See CelebA folder for details on model training. 

## Demo of FACE
We have provided a jupyter notebook demonstrating FACE implementation on an imagenet class 'Church'. You will need to prepare correctly classified images as .npz file to execute the notebook. You can download the sample here https://www.dropbox.com/scl/fi/xtp2fz5jky9fwlkgfqad6/final_filtered_church_images.npz?rlkey=qxsdzt55kgv9qixgxm9ec6c97&st=gdwxwbcs&dl=0. 

