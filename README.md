# FACE: Faithful Automatic Concept Extraction

This repository contains code for the paper "FACE: Faithful Automatic Concept Extraction" submitted to the The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025).

## Introduction 
Interpreting deep neural networks through concept-based explanations offers a bridge between low-level features with high-level human-understandable semantics. However, existing concept-based methods face significant limitations. Early methods like TCAV require a manually-labeled dataset for extracting concepts, greatly limiting scalability and practical applicability. Recent unsupervised methods such as ACE, ICE and CRAFT automate the concept discovery using unsupervised clustering or matrix factorization of encoder activations. Non-negative matrix factorization (NMF) based approaches like CRAFT, in particular, have been shown to produce spatially coherent and interpretable concepts. However, these methods focus solely on reconstructing the latent representation and do not account for the behavior of the model’s downstream classifier. As a result, the resulting explanations may appear interpretable and yet do misinterpret the original model's true reasoning.

In this work, we propose FACE: Faithful Automatic Concept Extraction, a novel framework that combines Non-negative Matrix Factorization (NMF) with a Kullback-Leibler (KL) divergence regularization term to ensure alignment between the model’s original and concept-based predictions. Unlike prior methods that operate solely on encoder activations, FACE incorporates classifier supervision during concept learning, enforcing predictive consistency and enabling explanations that are faithful.

We propose a faithfulness-aware variant of NMF that explicitly aligns the reconstructed activations with the model’s predictive behavior. We achieve this by incorporating a Kullback-Leibler (KL) divergence between the classifier head prediction on $\mathbf{A}$ and $\mathbf{UW}^\top$, leading to the following objective:

```math
\min_{\substack{\mathbf{U} \geq 0, \, \mathbf{W} \geq 0}} \frac{1}{2} \|\mathbf{A} - \mathbf{U}\mathbf{W}^\top\|_F^2 + \lambda \cdot \text{KL}(h(\mathbf{A}) \| h(\mathbf{UW}^\top))
```


## Requirements
The current implementation is on PyTorch and requires some existing libraries that you can install using the following file:
```
pip install -r requirements.txt
```
## Models under evaluation 

We use pretrained ResNet34 and MobileNetV2 for ImageNet and COCO. For CelebA dataset, we train ResNet and MobileNet models on an exclusive set of four facial attributes. See CelebA folder for details on celebA models training.

## Demo of FACE
We have provided a jupyter notebook demonstrating FACE implementation on an imagenet class 'Church'. You will need to prepare correctly classified images as .npz file to execute the notebook. You can download the sample here https://drive.google.com/file/d/1OvASMUfeHbQTdx2DGx1FhiOBdbEXIkF1/view?usp=drive_link
