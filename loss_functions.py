import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedLoss(nn.Module):
    def __init__(self, loss_fn, targets, reduction='mean', beta=0.9999, **kwargs):
        """
            Initialize the Class Balanced Loss.

            :param loss_fn: The loss function to be used (e.g., nn.BCELoss()).
            :param targets: The ground truth labels or targets.
            :param reduction: Specifies the reduction to apply to the output ('mean', 'sum', etc.), defaults to 'mean'.
            :param beta: A balancing factor for class weights, defaults to 0.9999.
            :param kwargs: Additional keyword arguments for custom behavior.
            """
        super(ClassBalancedLoss, self).__init__()
        self.loss_fn = loss_fn
        self.reduction = reduction
        self.beta = beta
        self.target = targets
        self.target_binary = self.target.flatten().to(torch.int)
        self.class_weights = self.compute_class_weights()
        self.kwargs = kwargs

    def compute_class_weights(self):
        """
        Compute class weights based on the effective number of samples.

        :return: Computed class weights.
        :rtype: torch.Tensor
        """
        class_counts = torch.bincount(self.target_binary)  # Count occurrences of each class
        effective_num = 1.0 - torch.pow(self.beta, class_counts)  # Compute effective number of samples per class
        weights = (1.0 - self.beta) / effective_num  # Compute weights inversely proportional to sample counts
        weights /= torch.sum(weights)  # Normalize to sum to 1
        return weights

    def forward(self, inputs, target):
        """
            Compute the Class Balanced loss using the specified loss function.

            :param inputs: Model predictions.
            :param target: Ground truth labels.
            :return: Computed loss value.
            :rtype: torch.Tensor
        """
        batch_weights = self.class_weights[self.target_binary].unsqueeze(dim=1)
        # Calculate the base loss
        loss = self.loss_fn(inputs, target)

        # Apply class weights
        balanced_loss = batch_weights * loss
        # Reduction: mean or sum
        if self.reduction == 'mean':
            return torch.mean(balanced_loss)
        elif self.reduction == 'sum':
            return torch.sum(balanced_loss)
        else:
            return balanced_loss


## I made this with chat but never checked it
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize the FocalLoss.

        Parameters:
        - alpha (Tensor or float, optional): Weighting factor for class imbalance.
          If a float is given, it applies the same weight to all classes. If a tensor,
          it applies per-class weights.
        - gamma (float, optional): Focusing parameter to adjust the rate at which easy
          examples are down-weighted. Default is 2.0.
        - reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
          'mean': the mean of the loss over all elements, 'sum': sum of the loss, 'none': no reduction.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = self._gamma_checker(gamma)
        self.reduction = reduction

    def _gamma_checker(self, gamma):
        if 1 <= gamma <= 5:
            return gamma
        else:
            new_gamma = int(input('Pleas input a new values for gamma between 1 and 5'))
            self._gamma_checker(new_gamma)

    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Parameters:
        - inputs (Tensor): Predicted logits (not probabilities), shape (N, C) where C is the number of classes.
        - targets (Tensor): Ground truth labels, shape (N,) for classification.

        Returns:
        - loss (Tensor): Computed focal loss value.
        """
        # Convert logits to probabilities

        probs = F.softmax(inputs, dim=1)
        targets = targets.long()
        num_classes = len(torch.unique(targets))

        # Gather the probabilities of the correct class for each example
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # Shape: (N,)

        # Compute the focal loss component (1 - p_t)^gamma
        focal_weight = (1 - pt).pow(self.gamma)

        # Calculate log of predicted probabilities
        log_pt = torch.log(pt)

        # Apply alpha weighting factor if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)) and num_classes == 2:
                # If binary classification and alpha is a scalar, use [alpha, 1 - alpha]
                alpha_t = torch.tensor([self.alpha, 1 - self.alpha])[targets]
            elif isinstance(self.alpha, (float, int)):
                # Apply the same alpha to all classes
                alpha_t = self.alpha
            else:
                # Use per-class alpha
                alpha_t = self.alpha[targets]  # Per-class alpha for multi-class
            focal_loss = -alpha_t * focal_weight * log_pt
        else:
            focal_loss = -focal_weight * log_pt

        # Reduction: mean or sum-
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
