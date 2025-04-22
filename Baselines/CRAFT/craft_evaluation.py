from craft.craft_torch import Craft, torch_to_numpy
from scipy.stats import spearmanr
import torch
import numpy as np
from torchvision import transforms
from math import ceil
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torchvision
import torch.nn as nn


class craft_implementation:
    def __init__(self, rank, imagenet_class, images_np, model, patch_size, device):
        self.r = rank
        self.device = device
        self.imagenet_class = imagenet_class
        self.model = model.eval().to(self.device)
        self.patch_size = patch_size
        
        self.transform = transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
        
        self.images = images_np 
        self.g = nn.Sequential(*(list(self.model.children())[:-2]))  
        self.h = lambda x: self.model.fc(torch.mean(x, (2, 3))) 
        self.h_2d = lambda x: self.model.fc(x)
        
        
    def preprocess_images(self, images):
        to_pil = transforms.ToPILImage()
        images_preprocessed = torch.stack([self.transform(to_pil(img)) for img in images], 0)
        return images_preprocessed
    
    
    def _batch_inference(self, model_fn, dataset, batch_size=128, resize=None, device='cuda'):
        nb_batchs = ceil(len(dataset) / batch_size)
        start_ids = [i*batch_size for i in range(nb_batchs)]
        results = []
        with torch.no_grad():
            for i in start_ids:
                x = torch.tensor(dataset[i:i+batch_size])
                x = x.to(device)
                if resize:
                    x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)
                results.append(model_fn(x).cpu())
        results = torch.cat(results)
        return results    

        
    def compute_craft(self):
        images_preprocessed =  self.preprocess_images(self.images)
        images_preprocessed =  images_preprocessed.to(self.device)
        
        
        predictions = self._batch_inference(self.model, images_preprocessed, batch_size=4, resize=None, device='cuda')
        predicted_classes = torch.argmax(predictions, dim=-1)
        accuracy = torch.sum(predicted_classes == self.imagenet_class)/len(predicted_classes)
        print('accuracy before NMF is:', accuracy)
        craft = Craft(input_to_latent = self.g,
              latent_to_logit = self.h,
              number_of_concepts = self.r,
              patch_size = self.patch_size,
              batch_size = 64,
              device = self.device)
        
        
        crops, crops_u, w = craft.fit(images_preprocessed)
        crops = np.moveaxis(torch_to_numpy(crops), 1, -1)
        
        importances = craft.estimate_importance(images_preprocessed, class_id=self.imagenet_class)
        images_u = craft.transform(images_preprocessed)
        
        #accuracy after NMF 
        UW=images_u@w
        UW = torch.tensor(UW, device='cuda')
        UW_reconstructed = UW.permute(0, 3, 1, 2)  # Shape: (233, 512, 7, 7)
        pred  = self.h(UW_reconstructed)
        c= torch.argmax(pred, dim=-1)
        accuracy = torch.sum(c==self.imagenet_class)/len(c)
        print('accuracy after NMF is:', accuracy)

        return crops, crops_u, w, importances, images_u, craft
    
    
    def transform_to_nmf_basis(self, images_preprocessed, W):
        activations = self._batch_inference(self.g, images_preprocessed, batch_size=64, device=self.device)  
        activations = activations.to(self.device)
        activations_avg = torch.mean(activations, dim=(2, 3)) 
        U_new = torch.matmul(activations_avg, W.T) 
        return U_new
    
    def transform_to_nmf_basis2(self, images_preprocessed, W):
        """
        Projects activations onto learned NMF basis W using least squares.

        Args:
            images_preprocessed (Tensor): [n, 3, 224, 224]
            W (Tensor): [r, p]

        Returns:
            U_new (Tensor): [n, r] projection coefficients
        """
        with torch.no_grad():
            activations = self._batch_inference(self.g, images_preprocessed, batch_size=4)  
            activations = activations.to('cuda')
            activations_avg = torch.mean(activations, dim=(2, 3))  

            Wt = W  
            WWt_inv = torch.linalg.pinv(Wt @ Wt.T) 
            U_new = activations_avg @ Wt.T @ WWt_inv 

        return U_new
    
    
    def compute_gini_index(self, concept_importance):
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
    
    
    def concept_deletion(self, images_u, w, W, concept_importance):
       
        UW=images_u@w
        UW = torch.tensor(UW, device='cuda')
        UW_reconstructed = UW.permute(0, 3, 1, 2)  # this is of shape torch.Size([233, 512, 7, 7])
        pred  = self.h(UW_reconstructed)
        c= torch.argmax(pred, dim=-1)
        org_accuracy = torch.sum(c==self.imagenet_class)/len(c)

        images_u = torch.tensor(images_u, device = 'cuda')
        W = torch.tensor(w, device = 'cuda')

        images_u = images_u.clone()
        concept_importance = torch.tensor(concept_importance)
        concept_importance = concept_importance.clone().to('cuda')

        sorted_indices = torch.argsort(concept_importance, descending=True)
        rank = images_u.shape[-1]
        results = {}

        results[0] = org_accuracy.item()

        for i in range(1, rank + 1):
            u_mod = images_u.clone()
            u_mod[:, :, :, sorted_indices[:i]] = 0
            UW = u_mod @ W  
            UW = UW.permute(0, 3, 1, 2)  
            pred = self.h(UW)
            c = torch.argmax(pred, dim=-1)
            acc = torch.sum(c == self.imagenet_class).item() / len(c)
            results[i] = acc

        return results
    
    def concept_insertion(self, images_u, w, concept_importance):
        images_u = torch.tensor(images_u, device = 'cuda')
        W = torch.tensor(w, device = 'cuda')

        images_u = images_u.clone().to('cuda')
        W = W.to('cuda')
        concept_importance = torch.tensor(concept_importance)
        concept_importance = concept_importance.clone().to('cuda')

        sorted_indices = torch.argsort(concept_importance, descending=True)
        rank = images_u.shape[-1]
        results = {}

        u_insert = torch.zeros_like(images_u).to('cuda')

        UW=u_insert.cpu().numpy()@w
        UW = torch.tensor(UW, device='cuda')
        UW_reconstructed = UW.permute(0, 3, 1, 2)  
        pred  = self.h(UW_reconstructed)
        c= torch.argmax(pred, dim=-1)
        org_accuracy = torch.sum(c==self.imagenet_class)/len(c)


        results[0] = org_accuracy.item()

        for i in range(1, rank + 1):

            u_insert[:, :, :, sorted_indices[:i]] = images_u[:, :, :, sorted_indices[:i]]
            UW = u_insert @ W  
            UW = UW.permute(0, 3, 1, 2) 
            pred = self.h(UW)
            c = torch.argmax(pred, dim=-1)
            acc = torch.sum(c == self.imagenet_class).item() / len(c)
            results[i] = acc

        return results
    
    def compute_deletion_score(self, insertion_deletion_score):
        concept_count = np.array(list(insertion_deletion_score.keys()))
        accuracy = np.array(list(insertion_deletion_score.values()))
        initial_acc = accuracy[0]
        aopc_values = initial_acc - accuracy
        aopc = np.sum(aopc_values) / (len(concept_count) + 1)

        return aopc

    def compute_insertion_auc(self, insertion_score):
        concept_count = np.array(list(insertion_score.keys()))
        accuracy = np.array(list(insertion_score.values()))
        auc = np.trapz(accuracy, concept_count) / (concept_count[-1] - concept_count[0])
        return auc
    
    def kl_divergence(self, logits_A, logits_UW):
        return F.kl_div(F.log_softmax(logits_UW, dim=-1), F.softmax(logits_A, dim=-1), reduction='batchmean')

    def evaluate_nmf_projection(self, images_preprocessed, w, craft):
        original_activation = self._batch_inference(self.g, images_preprocessed, batch_size=4, resize=None, device='cuda')
        original_activation = original_activation.to('cuda')
        original_activation = torch.mean(original_activation, dim=(2, 3))
        logits_original = self.h_2d(original_activation)
        
        U_new = craft.transform(images_preprocessed)
        A_reconstructed = torch.matmul(torch.tensor(U_new), torch.tensor(w))  # [n, p]
        A_reconstructed = A_reconstructed.permute(0,3,1,2)
        A_reconstructed = torch.mean(A_reconstructed, dim=(2, 3))
        A_reconstructed = A_reconstructed.to('cuda')
        logits_reconstructed = self.h_2d(A_reconstructed.to('cuda'))
        mse_loss = F.mse_loss(A_reconstructed, original_activation)
        kl_loss = self.kl_divergence(logits_original, logits_reconstructed)
        return mse_loss.item(), kl_loss.item()

    
    def compute_sparsity(self, U):
        if not isinstance(U, torch.Tensor):
            U = torch.tensor(U)

        U = U.to('cuda')
        U_flat = U.view(-1, U.shape[-1])  # [N*H*W, K]
        non_zero_counts = (U_flat != 0).float().sum(dim=1)
        k = U.shape[-1]
        sparsity_scores = non_zero_counts / k
        return sparsity_scores.mean().item()
    
    
    def extract_w_list(self, k_folds=5):
        from sklearn.model_selection import KFold

        to_pil = transforms.ToPILImage()
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        w_list = []

        for train_idx, _ in kf.split(self.images):
            subset = self.images[train_idx]
            images_preprocessed = torch.stack([self.transform(to_pil(img)) for img in subset], 0)

            craft = Craft(
                input_to_latent=self.g,
                latent_to_logit=self.h,
                number_of_concepts=self.r,
                patch_size=self.patch_size,
                batch_size=64,
                device='cuda')

            _, _, w = craft.fit(images_preprocessed)
            w_list.append(w)

        return w_list
    
    def compute_stability(self, w_list):
        w_list = torch.tensor(w_list)
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
    
    
