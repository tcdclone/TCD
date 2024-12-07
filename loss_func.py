import torch
import torch.nn.functional as F
import torch.nn as nn

def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """
    Compute the InfoNCE loss for the given embeddings and labels.

    Args:
        embeddings (torch.Tensor): Tensor of shape (N, d), where N is the number of samples and d is the embedding dimension.
        labels (torch.Tensor): Tensor of shape (N,) containing the labels for each sample.
        temperature (float): Temperature parameter for scaling the logits.

    Returns:
        torch.Tensor: The computed InfoNCE loss.
    """
    # Normalize embeddings to have unit length
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity matrix (N, N) using dot product
    similarity_matrix = torch.matmul(embeddings, embeddings.t())

    # Scale by temperature
    similarity_matrix /= temperature

    # Mask to remove self-similarity from positive pairs
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.t()).float()  # (N, N) mask with 1 for positive pairs, 0 otherwise
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask *= logits_mask

    # Compute log-softmax of the similarity matrix
    log_probs = F.log_softmax(similarity_matrix, dim=1)

    # Select the log-probabilities for the positive pairs
    positives = mask * log_probs
    positive_sum = positives.sum(dim=1)
    num_positives = mask.sum(dim=1)

    # Avoid division by zero by adding a small epsilon
    loss = -positive_sum / (num_positives + 1e-8)
    return loss.mean()

def contrastive_loss(embeddings, labels, temperature=0.07):
    similarity_matrix = F.cosine_similarity(
        embeddings.unsqueeze(1), 
        embeddings.unsqueeze(0), 
        dim=-1
    )
    
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    
    mask.fill_diagonal_(0)
    
    positive_mask = mask
    negative_mask = 1 - mask - torch.eye(len(labels), device=labels.device)
    
    exp_sim = torch.exp(similarity_matrix / temperature)
    
    pos_numerator = exp_sim * positive_mask
    
    neg_denominator = exp_sim * negative_mask
    
    loss = -torch.log(
        pos_numerator.sum(1) / (pos_numerator.sum(1) + neg_denominator.sum(1))
    ).mean()
    
    return loss


def cal_total_loss(lm_loss, embedding_loss, lamb):
    return lm_loss*lamb + embedding_loss*(1-lamb)
