'''
Code for different evaluation metrics of NMF-based explanations
Author: Dipkamal Bhusal

'''


from utility import *
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
from torchvision import transforms
from math import ceil
from torch.linalg import svd
from utility import *
from sampler import HaltonSequence
from estimators import JansenEstimator
import argparse 
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment


def _batch_inference(model, dataset, batch_size=128, resize=None, device='cuda'):
    nb_batchs = ceil(len(dataset) / batch_size)
    start_ids = [i*batch_size for i in range(nb_batchs)]
    results = []
    with torch.no_grad():
        for i in start_ids:
            x = torch.tensor(dataset[i:i+batch_size])
            x = x.to(device)
            if resize:
                x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)
            results.append(model(x).cpu())
    results = torch.cat(results)
    return results

def transform_to_nmf_basis2(images_preprocessed, W, g):
    with torch.no_grad():
        activations = _batch_inference(g, images_preprocessed, batch_size=4)
        activations = activations.to('cuda')
        activations_avg = torch.mean(activations, dim=(2, 3)) 

        Wt = W  
        WWt_inv = torch.linalg.pinv(Wt @ Wt.T)  
        U_new = activations_avg @ Wt.T @ WWt_inv 

    return U_new

def transform_to_nmf_basis(images_preprocessed, W, g, device):
    activations = _batch_inference(g, images_preprocessed, batch_size=64, device=device)  
    activations = activations.to(device)
    activations_avg = torch.mean(activations, dim=(2, 3))  
    U_new = torch.matmul(activations_avg, W.T)  
    return U_new


def compute_accuracy_after_nmf(images_preprocessed, W, g, h_2d, imagenet_class, device):
    U_new = transform_to_nmf_basis(images_preprocessed, W, g, device)
    activations_now = U_new @ W 
    prediction_now = h_2d(activations_now)
    prediction_now_ = torch.argmax(prediction_now, dim=-1)
    accuracy_now = torch.sum(prediction_now_ == imagenet_class)/len(prediction_now_)
    return accuracy_now


def estimate_importance(U, W, h, h_2d, class_id, batch_size,number_of_concepts,device, nb_design=32): 
    masks = HaltonSequence()(number_of_concepts, nb_design=nb_design).astype(np.float32)
    estimator = JansenEstimator()
    latent_to_logit = h
    batch_size = batch_size
    importances = []
    
    latent_to_logit = h_2d
    
    if len(U.shape) == 2:
        for u in U:
            u_perturbated = u[None, :] * masks
            a_perturbated = u_perturbated @ W

            y_pred = _batch_inference(latent_to_logit, a_perturbated, batch_size,
                                          device=device)
            y_pred = y_pred[:, class_id]
            stis = estimator(torch_to_numpy(masks),
                                 torch_to_numpy(y_pred),
                                 nb_design)

            importances.append(stis)
    
    elif len(U.shape) == 4:
            for u in U:
                u_perturbated = u[None, :] * masks[:, None, None, :]
                a_perturbated = np.reshape(u_perturbated,(-1, u.shape[-1])) @ W
                a_perturbated = np.reshape(a_perturbated, (len(masks), U.shape[1], U.shape[2], -1))
                a_perturbated = np.moveaxis(a_perturbated, -1, 1)

                y_pred = _batch_inference(h, a_perturbated, batch_size,
                                          device=device)
                y_pred = y_pred[:, class_id]

                stis = estimator(torch_to_numpy(masks),
                                 torch_to_numpy(y_pred),
                                 nb_design)

                importances.append(stis)

    
    return np.mean(importances, 0)



def compute_gini_index(concept_importance):
   
    
    if len(concept_importance.shape) != 1:
        raise ValueError("Input concept_importance must be a 1D tensor.")

    sorted_importance = torch.sort(concept_importance)[0]
    n = sorted_importance.size(0)
    mean_importance = torch.mean(sorted_importance)
    
    if mean_importance == 0:
        return 0.0  # Gini index is 0 if all values are zero

    cumulative_indices = torch.arange(1, n + 1, dtype=torch.float32, device=concept_importance.device)
    numerator = torch.sum((2 * cumulative_indices - n - 1) * sorted_importance)
    denominator = n * torch.sum(sorted_importance)
    gini_index = numerator / denominator
    return gini_index.item()


def concept_deletion(images_u, W, h_2d, imagenet_class, concept_importance):
    activations_org = images_u @ W 
    prediction_org = h_2d(activations_org)
    prediction_org_ = torch.argmax(prediction_org, dim=-1)
    accuracy_org = torch.sum(prediction_org_ == imagenet_class)/len(prediction_org_)

    images_u = torch.tensor(images_u, device = 'cuda')
    W = torch.tensor(W, device = 'cuda')
    images_u = images_u.clone()
    concept_importance = torch.tensor(concept_importance)
    concept_importance = concept_importance.clone().to('cuda')
    
    sorted_indices = torch.argsort(concept_importance, descending=True)
    rank = images_u.shape[-1]
    results = {}
    
    results[0] = accuracy_org.item()
    for i in range(1, rank + 1):
        u_mod = images_u.clone()
        u_mod[:, sorted_indices[:i]] = 0
        UW = u_mod @ W  
        pred = h_2d(UW) 
        c = torch.argmax(pred, dim=-1)
        acc = torch.sum(c == imagenet_class).item() / len(c)
        results[i] = acc

    return results


def concept_insertion(images_u, w, h_2d, imagenet_class, concept_importance):
    images_u = torch.tensor(images_u, device = 'cuda')
    W = torch.tensor(w, device = 'cuda')
    images_u = images_u.clone().to('cuda')
    concept_importance = torch.tensor(concept_importance)
    concept_importance = concept_importance.clone().to('cuda')
    sorted_indices = torch.argsort(concept_importance, descending=True)
    rank = images_u.shape[-1]
    results = {}
    
    u_insert = torch.zeros_like(images_u).to('cuda')
    
    UW_org=u_insert@w
    UW_org = torch.tensor(UW_org, device='cuda')
    pred_org  = h_2d(UW_org)
    c= torch.argmax(pred_org, dim=-1)
    org_accuracy = torch.sum(c==imagenet_class)/len(c)
    results[0] = org_accuracy.item()

    for i in range(1, rank + 1):
        u_insert[:, sorted_indices[:i]] = images_u[:, sorted_indices[:i]]
        UW = u_insert @ W 
        pred = h_2d(UW)
        c = torch.argmax(pred, dim=-1)
        acc = torch.sum(c == imagenet_class).item() / len(c)
        results[i] = acc

    return results


def compute_deletion_score(insertion_deletion_score):
    concept_count = np.array(list(insertion_deletion_score.keys()))
    accuracy = np.array(list(insertion_deletion_score.values()))
    initial_acc = accuracy[0]
    aopc_values = initial_acc - accuracy
    aopc = np.sum(aopc_values) / (len(concept_count) + 1)
    
    return aopc


def compute_insertion_auc(insertion_score):
    concept_count = np.array(list(insertion_score.keys()))
    accuracy = np.array(list(insertion_score.values()))
    auc = np.trapz(accuracy, concept_count) / (concept_count[-1] - concept_count[0])
    return auc

def compute_sparsity(U):
    if not isinstance(U, torch.Tensor):
        U = torch.tensor(U)

    U = U.to('cuda')
    U_flat = U.view(-1, U.shape[-1])  # [N*H*W, K]
    non_zero_counts = (U_flat != 0).float().sum(dim=1)
    k = U.shape[-1]
    sparsity_scores = non_zero_counts / k
    return sparsity_scores.mean().item()

def extract_w_list(analyzer, images_np, k_folds=5):
    from sklearn.model_selection import KFold
    to_pil = transforms.ToPILImage()
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    w_list = []

    for train_idx, _ in kf.split(images_np):
        subset = images_np[train_idx]
        analyzer = analyzer
        images_preprocessed = analyzer.preprocess_images(subset)
        patches, activations_avg = analyzer.compute_activations(images_preprocessed)
        U, W, mse_losses, kl_losses, total_losses = analyzer.nmf_kl_pgd(patches, activations_avg,
                                                                                images_preprocessed)
   

        w_list.append(W)

    return w_list

def compute_stability(w_list):
    #w_list = torch.tensor(w_list)
    similarities = []
    num_models = len(w_list)
    for i in range(num_models):
        W1 = F.normalize(w_list[i], p=2, dim=1).to('cuda')
        for j in range(i + 1, num_models):
            W2 = F.normalize(w_list[j], p=2, dim=1).to('cuda')
            sim_matrix = W1 @ W2.T
            cost = 1 - sim_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_sims = sim_matrix[row_ind, col_ind]
            similarities.append(matched_sims.mean().item())
    return np.mean(similarities) if similarities else 0.0

def kl_divergence(logits_A, logits_UW):
        return F.kl_div(F.log_softmax(logits_UW, dim=-1), F.softmax(logits_A, dim=-1), reduction='batchmean')

def evaluate_nmf_projection(images_preprocessed, W, g, h_2d, device, A_original, logits_original):
    U_new = transform_to_nmf_basis2(images_preprocessed, W, g)  # [n, r]
    A_reconstructed = torch.matmul(U_new, W)  # [n, p]
    logits_reconstructed = h_2d(A_reconstructed)
    mse_loss = F.mse_loss(A_reconstructed, A_original)
    kl_loss = kl_divergence(logits_original, logits_reconstructed)
    return mse_loss.item(), kl_loss.item()
