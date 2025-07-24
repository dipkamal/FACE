'''
Code for FACE: NMF optimization with KL-regularization

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


class KL_NMF:
    def __init__(self, model, imagenet_class, images_np, rank, patch_size, batch_size, epsilon, steps, lambda_val, learning_rate, device):
        self.model = model.eval()
        self.imagenet_class = imagenet_class
        self.images_np = images_np
        self.rank = rank
        self.patch_size = patch_size
        self.batch_size = batch_size 
        self.epsilon = epsilon
        self.steps = steps
        self.lambda_val = lambda_val
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(self.device)
        self.g = torch.nn.Sequential(*(list(self.model.children())[:-2]))
        self.h = lambda x: self.model.fc(torch.mean(x, (2, 3)))
        self.h_2d = lambda x: self.model.fc(x)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_images(self, images_np):
        to_pil = transforms.ToPILImage()
        return torch.stack([self.transform(to_pil(img)) for img in images_np], 0).to(self.device)

    def _batch_inference(self, model_fn, dataset, batch_size=128, resize=None):
        from math import ceil
        nb_batchs = ceil(len(dataset) / batch_size)
        start_ids = [i * batch_size for i in range(nb_batchs)]
        results = []
        with torch.no_grad():
            for i in start_ids:
                x = dataset[i:i+batch_size]
                x = x.to(self.device)
                if resize:
                    x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)
                results.append(model_fn(x).cpu())
        return torch.cat(results)


    def evaluate_accuracy_before_nmf(self):
        images_preprocessed = self.preprocess_images(self.images_np)
        predictions = self._batch_inference(self.model, images_preprocessed, batch_size=4)
        predicted_classes = torch.argmax(predictions, dim=-1)
        accuracy = torch.sum(predicted_classes == self.imagenet_class) / len(predicted_classes)
        return accuracy.item()

    def compute_activations(self, images_preprocessed):
        strides = int(self.patch_size * 0.80)
        patches = torch.nn.functional.unfold(images_preprocessed, kernel_size=self.patch_size, stride=strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size).to(self.device)
        activations_org = self._batch_inference(self.g, patches, batch_size=self.batch_size)
        activations_org = activations_org.to(self.device)
        if len(activations_org.shape) == 4:
            activations_avg = torch.mean(activations_org, dim=(2, 3)) 
        return patches, activations_avg
                                                                                                        
    # def transform_to_nmf_basis(self, images_preprocessed, W):
    #     with torch.no_grad():
    #         activations = self._batch_inference(self.g, images_preprocessed, batch_size=4)  # [n, 512, 7, 7]
    #         activations = activations.to(self.device)
    #         activations_avg = torch.mean(activations, dim=(2, 3))  # [n, p]

    #         Wt = W  # W is [r, p]
    #         WWt_inv = torch.linalg.pinv(Wt @ Wt.T)  # [r, r]
    #         U_new = activations_avg @ Wt.T @ WWt_inv  # [n, p] @ [p, r] @ [r, r] = [n, r]

    #     return U_new

    def nndsvd_initialization(self, A, r):
        U_svd, S_svd, V_svd = svd(A)
        U = torch.zeros_like(U_svd[:, :r], dtype=torch.float32)
        W = torch.zeros_like(V_svd[:r, :], dtype=torch.float32)
        U[:, 0] = torch.sqrt(S_svd[0]) * torch.abs(U_svd[:, 0])
        W[0, :] = torch.sqrt(S_svd[0]) * torch.abs(V_svd[0, :])
        for i in range(1, r):
            u, v = U_svd[:, i], V_svd[i, :]
            u_pos, u_neg = torch.clamp(u, min=0), torch.clamp(-u, min=0)
            v_pos, v_neg = torch.clamp(v, min=0), torch.clamp(-v, min=0)
            u_norm, v_norm = torch.norm(u_pos), torch.norm(v_pos)
            if u_norm * v_norm > torch.norm(u_neg) * torch.norm(v_neg):
                U[:, i] = torch.sqrt(S_svd[i]) * u_pos / u_norm
                W[i, :] = torch.sqrt(S_svd[i]) * v_pos / v_norm
            else:
                U[:, i] = torch.sqrt(S_svd[i]) * u_neg / torch.norm(u_neg)
                W[i, :] = torch.sqrt(S_svd[i]) * v_neg / torch.norm(v_neg)
        return U, W
    
    def kl_divergence(self, logits_A, logits_UW):
        return F.kl_div(F.log_softmax(logits_UW, dim=-1), F.softmax(logits_A, dim=-1), reduction='batchmean')

    def nmf_kl_pgd(self, patches, activations_avg, input_images):
        
        A = activations_avg
        U, W = self.nndsvd_initialization(A, self.rank)
        U = U.to(self.device).requires_grad_(True)
        W = W.to(self.device).requires_grad_(True)

        optimizer = torch.optim.Adam([U, W], lr=self.learning_rate)
        prev_loss = float('inf')

        
        mse_losses, kl_losses, sparsity_losses, total_losses = [], [], [], []

        original_logits = self._batch_inference(self.model, patches, batch_size=4).to(self.device)

        for step in range(self.steps):
            optimizer.zero_grad()
            UW_reconstructed = torch.matmul(U, W)
            predicted_logits = self.h_2d(UW_reconstructed)
            kl_loss = self.kl_divergence(original_logits, predicted_logits)
            mse_loss = F.mse_loss(UW_reconstructed, A)
            total_loss = mse_loss + self.lambda_val * kl_loss
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                U.clamp_(min=0)
                W.clamp_(min=0)
            mse_losses.append(mse_loss.item())
            kl_losses.append(kl_loss.item())
            total_losses.append(total_loss.item())
            if abs(prev_loss - total_loss.item()) < self.epsilon:
                break
            prev_loss = total_loss.item()

        return U.detach(), W.detach(), mse_losses, kl_losses, total_losses
