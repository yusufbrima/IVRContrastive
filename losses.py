import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy as sp

from math import log
eps = 1e-8 # a small number to prevent division by zero

def get_triplet_mask(labels):
  """compute a mask for valid triplets
  Args:
    labels: Batch of integer labels. shape: (batch_size,)
  Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
  """
  # step 1 - get a mask for distinct indices

  # shape: (batch_size, batch_size)
  indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
  indices_not_equal = torch.logical_not(indices_equal)
  # shape: (batch_size, batch_size, 1)
  i_not_equal_j = indices_not_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_not_equal_k = indices_not_equal.unsqueeze(1)
  # shape: (1, batch_size, batch_size)
  j_not_equal_k = indices_not_equal.unsqueeze(0)
  # Shape: (batch_size, batch_size, batch_size)
  distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

  # step 2 - get a mask for valid anchor-positive-negative triplets

  # shape: (batch_size, batch_size)
  labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
  # shape: (batch_size, batch_size, 1)
  i_equal_j = labels_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_equal_k = labels_equal.unsqueeze(1)
  # shape: (batch_size, batch_size, batch_size)
  valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

  # step 3 - combine two masks
  mask = torch.logical_and(distinct_indices, valid_indices)

  return mask


class BatchAllTtripletLoss(nn.Module):
  """Uses all valid triplets to compute Triplet loss
  Args:
    margin: Margin value in the Triplet Loss equation
  """
  def __init__(self, margin=1.):
    super().__init__()
    self.margin = margin
    
  def forward(self, embeddings, labels):
    """computes loss value.
    Args:
      embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
      labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
    Returns:
      Scalar loss value.
    """

    # print("before distance matrix")
    # step 1 - get distance matrix
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    # print("after distance matrix")

    # print('distance_matrix', distance_matrix)

    # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

    # shape: (batch_size, batch_size, 1)
    anchor_positive_dists = distance_matrix.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    anchor_negative_dists = distance_matrix.unsqueeze(1)
    # get loss values for all possible n^3 triplets
    # shape: (batch_size, batch_size, batch_size)
    triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

    # step 3 - filter out invalid or easy triplets by setting their loss values to 0

    # shape: (batch_size, batch_size, batch_size)
    mask = get_triplet_mask(labels)
    triplet_loss *= mask
    # easy triplets have negative loss values
    triplet_loss = F.relu(triplet_loss)

    # step 4 - compute scalar loss value by averaging positive losses
    num_positive_losses = (triplet_loss > eps).float().sum()
    triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

    return triplet_loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, epsilon=1e-7):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.epsilon = epsilon

    def forward(self, features, labels=None):
        """
        Compute the Supervised Contrastive Loss.
        
        Args:
            features: tensor of shape [n_samples, feature_dim].
            labels: ground truth of shape [n_samples].
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) != 2:
            raise ValueError('`features` needs to be [n_samples, feature_dim]')

        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)

        # Normalize features
        features = F.normalize(features, dim=1)

        anchor_dot_contrast = torch.matmul(features, features.T)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits / self.temperature) * logits_mask
        log_prob = logits / self.temperature - torch.log(exp_logits.sum(1, keepdim=True) + self.epsilon)

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.epsilon)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        # Debugging information
        if torch.isnan(loss):
            print("NaN detected in loss calculation")
            print("anchor_dot_contrast stats:", torch.min(anchor_dot_contrast).item(), torch.max(anchor_dot_contrast).item(), torch.mean(anchor_dot_contrast).item())
            print("logits stats:", torch.min(logits).item(), torch.max(logits).item(), torch.mean(logits).item())
            print("exp_logits stats:", torch.min(exp_logits).item(), torch.max(exp_logits).item(), torch.mean(exp_logits).item())
            print("log_prob stats:", torch.min(log_prob).item(), torch.max(log_prob).item(), torch.mean(log_prob).item())
            print("mean_log_prob_pos stats:", torch.min(mean_log_prob_pos).item(), torch.max(mean_log_prob_pos).item(), torch.mean(mean_log_prob_pos).item())

        return loss