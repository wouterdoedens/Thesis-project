import torch
import numpy as np
from captum.attr import IntegratedGradients
import pandas as pd
import matplotlib.pyplot as plt


class GeneralExplainer:
    def __init__(self, trainer):
        """
        Initialize the Explainer with a trained model from a Trainer object.

        :param trainer: Trainer object that contains a trained model and data
        """
        self.trainer = trainer
        self.model = trainer.model
        self.data = trainer.data
        self.criterion = trainer.criterion
        self.target = trainer.target

    def permutation_importance(self):
        """
        Calculate feature importance based on the change in loss when shuffling each feature.

        :return: A numpy array containing the importance score for each feature.
        """
        self.model.eval()  # Set the model to evaluation mode to avoid any training-specific behavior (e.g., dropout).

        # Get the original model loss on the unmodified data (baseline loss).
        base_loss = self.evaluate_loss(self.data.x, self.data.edge_index)

        importances = []  # List to store feature importance values.

        # Loop over each feature in the node feature matrix.
        for feature_idx in range(self.data.x.shape[1]):
            X_permuted = self.data.x.clone()  # Create a copy of the original node features.

            # Shuffle the values in the feature column at index 'feature_idx'.
            permuted_feature = X_permuted[:, feature_idx]
            permuted_feature = permuted_feature[
                torch.randperm(permuted_feature.size(0))]  # Randomly shuffle the feature values.

            # Replace the original feature column with the shuffled one.
            X_permuted[:, feature_idx] = permuted_feature

            # Calculate the model loss using the permuted features.
            permuted_loss = self.evaluate_loss(X_permuted, self.data.edge_index)

            # Calculate the feature importance as the increase in loss due to shuffling.
            importance = permuted_loss - base_loss
            importances.append(importance.item())  # Add the importance score to the list.

        return np.array(importances)  # Return the importance scores as a NumPy array.

    def evaluate_loss(self, X, edge_index):
        """
        Helper function to evaluate loss for given input X and edge_index.

        :param X: Input features
        :param edge_index: Graph edges
        :return: Loss value
        """
        with torch.no_grad():
            output = self.model(X, edge_index)
            loss = self.criterion(output, self.target)
        return loss

    def gradient_based_importance(self):
        """
        Calculate feature importance using the gradients of the loss w.r.t the inputs.

        :return: A numpy array containing the gradient-based importance score for each feature.
        """
        self.model.eval()
        self.model.zero_grad()  # Clear any previous gradients

        # Forward pass
        output = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(output, self.target)

        # Backward pass to calculate gradients
        loss.backward()

        # Get the gradient of the loss w.r.t each input feature and take the mean over the batch
        grads = self.data.x.grad.abs().sum(dim=0).detach().numpy()

        return grads

    def integrated_gradients_importance(self):
        """
        Use the Integrated Gradients method to compute feature importance.

        :return: A numpy array containing the Integrated Gradients attributions for each feature.
        """
        input_test = self.data.x  # Input features
        baseline = torch.zeros_like(input_test)  # Baseline: all zeros

        # Instantiate Integrated Gradients with the model
        ig = IntegratedGradients(self.model)

        # Compute the attributions using Integrated Gradients
        attributions, delta = ig.attribute(input_test, baseline,
                                           additional_forward_args=(self.data.edge_index),  # Pass edge_index for GCN
                                           target=0,  # Target output class
                                           return_convergence_delta=True)

        # Sum the attributions across all features for each input
        feature_importance = attributions.sum(dim=0).detach().numpy()

        return feature_importance


    def combine_importances(self):
        """
        Combine the feature importances from all methods into a single DataFrame and print it.

        :return: A pandas DataFrame containing the feature importances from all methods.
        """
        # Calculate feature importances using different methods
        perm_importance = self.permutation_importance()
        grad_importance = self.gradient_based_importance()
        ig_importance = self.integrated_gradients_importance()

        # Create a pandas DataFrame
        df = pd.DataFrame({
            'Feature': self.data.feature_list,
            'Permutation Importance': perm_importance,
            'Gradient Importance': grad_importance,
            'Integrated Gradients Importance': ig_importance
        })

        # Print the DataFrame
        print(df)

        return df
