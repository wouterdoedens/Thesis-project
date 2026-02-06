from visualisations import Visualiser
from GNN_Explainers_class import GraphVisualisations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch_geometric.data import Data
import torch
import numpy as np


class Trainer(Visualiser, GraphVisualisations):
    """
    A class for training and managing a machine learning model on city network data,
    with functionality for visualizing predictions and network structure.
    """

    def __init__(self,
                 model=None,
                 city_data=None,
                 criterion=None,
                 optimizer=None,
                 schedular=None,
                 n_epochs=500,
                 set_Trainer=None):
        """
        Initializes the Trainer object with the model, data, and training parameters.

        :param model: The neural network model to be trained. (e.g. MLP, GCN)
        :param city_data: The Data object returned by GeometricDataTransformer
        :param criterion: Loss function to evaluate model performance.
        :param optimizer: Optimizer for updating model parameters.
        :param schedular: Learning rate scheduler for dynamic adjustments.
        :param n_epochs: Total number of training epochs. Default is 500.
        :param set_Trainer: If provided, initializes the instance with parameters from another `Trainer` object.
        """

        super().__init__()
        self.x = None  # Store model predictions
        self.losses = []

        if set_Trainer is None:
            self.model = model
            self.city_data = city_data
            self.data = self.city_data.data_object_PyG if city_data else None
            self.num_epochs = n_epochs
            self.optimizer = optimizer
            self.schedular = schedular
            self.criterion = criterion
            self.target = self.data.y.view(-1, 1) if self.data else None

            self.train()
            self.set_city_name(city_data.city_name if city_data else "Unknown")
            self.pdp, _ = self.partial_dependence_plots()

        else:
            self.model = set_Trainer.model
            self.city_data = set_Trainer.city_data
            self.data = set_Trainer.data
            self.num_epochs = set_Trainer.num_epochs
            self.optimizer = set_Trainer.optimizer
            self.schedular = set_Trainer.schedular
            self.criterion = set_Trainer.criterion
            self.target = set_Trainer.target
            self.x = set_Trainer.x
            self.set_city_name(set_Trainer.city_data.city_name)
            self.pdp, _ = self.partial_dependence_plots()

        self.y = self.target.detach().numpy() if self.target is not None else None
        self.counts0, self.counts1, self.bin_edges = self.histogram()

    def handle_training_prompt(self):
        """Handle user input for running the training process."""
        run_training = input("Do you want to run the training now? (yes/no): ").strip().lower()

        if run_training == 'yes':
            with_printing = input("Do you want to print the loss during training? (yes/no): ").strip().lower()
            self.train(pr=(with_printing == 'yes'))
        else:
            print("Training skipped. You can run it later using the `train` method.")

    def train(self, pr=None):
        """
        Train the model for the specified number of epochs.
        :param pr: If provided, prints the loss at each epoch, defaults to None.
        """
        for epoch in range(self.num_epochs):
            self.model.train() # Set model to training mode
            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(self.data.x, self.data.edge_index) # Zero the gradients

            # Calculate loss using the criterion (ClassBalancedLoss or other)
            loss = self.criterion(out, self.target)

            if torch.isnan(out).any(): # Check for NaNs in the output and loss
                print("Output contains NaNs")
            if torch.isnan(loss).any():
                print("Loss contains NaNs")
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.schedular is not None:
                self.schedular.step() # Adjust learning rate if a scheduler is provided
            # Record loss for analysis
            self.losses.append(loss.item())

            if pr:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
        self.x = out.detach().numpy()

        # After training, store the output and target for further use or visualization

    def get_losses(self):
        """Return the list of losses recorded during training."""
        return self.losses

    def reset(self, model=None, criterion=None, optimizer=None, num_epochs=None):
        """
        Reset the model, criterion, optimizer, and num_epochs.
        Parameters can be provided to update corresponding attributes.

        :param model: The new model to set. If None, the existing model remains unchanged.
        :param criterion: The loss function to set. If None, the existing criterion remains unchanged.
        :param optimizer: The optimizer to set. If None, the existing optimizer remains unchanged.
        :param num_epochs: The number of epochs for training. If None, the existing value remains unchanged.
        :return: The updated instance of the class.
        """
        if model is not None:
            self.model = model
        if criterion is not None:
            self.criterion = criterion
            print(self.model, self.criterion)
        if optimizer is not None:
            self.optimizer = optimizer
        if num_epochs is not None:
            self.num_epochs = num_epochs
        self.handle_training_prompt()
        return self

    def visualize_network(self, based_on='prediction', colormapping=None) -> None:
        """
            Visualizes a city network using a specified data source ('prediction' or 'ground_truth')
            for coloring nodes or edges.

            :param based_on: Specifies whether to visualize the network based on 'prediction' or 'ground_truth'.
                             Defaults to 'prediction'.
            :param colormapping: The colormap used to represent values on the network. Defaults to 'coolwarm' if None.

            :raises ValueError: If 'based_on' is set to anything other than 'prediction' or 'ground_truth'.
            :return: None. This function visualizes the network but does not return any values.
            """

        df = self.city_data.c_df[['NWB_ID']].copy()
        if based_on == 'prediction':
            prediction = self.x.flatten()
            df['prediction'] = prediction
        elif based_on == 'ground_truth':
            ground_truth = self.y.flatten()
            df['ground_truth'] = ground_truth
        else:
            raise ValueError(
                'Invalid input selection. Please use "prediciton" or "ground_truth"')
        if colormapping is None:
            colormapping = 'coolwarm'
        self._network_visualizer(name=self._city_name, gdf=self.city_data.gdf, dataframe=df, colormap=colormapping)

    def partial_dependence_plots(self, percentile=5, steps=100, operation='mean', save=False):
        """
        Create Partial Dependence Plots (PDPs) for all features in the dataset.

        :param save:
        :param percentile: Percentile to define feature range. Default is 5
        :param steps: Number of steps to vary each feature. Default is 100
        :param operation: Aggregation operation (mean or median). Default is mean
        :return: Dictionary containing PDP values for each feature.
        """
        feature_names = self.city_data.feature_names  # Get feature names.
        data_clone = self.data.x.clone()  # Clone the data.
        self.model.eval()  # Set model to evaluation mode.

        # Initialize column information (1 if non-binary, 2 if binary 'Speed', 3 for other binary)
        column_information = np.zeros([2, len(feature_names)])
        for feature_idx in range(self.data.x.shape[1]):  # Go through the columns
            unique_vals = torch.unique(self.data.x[:, feature_idx])  # Show the number of unique values
            if len(unique_vals) > 2:  # If continuous
                column_information[0, feature_idx] = 1  # Non-binary
            elif 'Speed' in feature_names[feature_idx]:
                column_information[1, feature_idx] = 1  # Binary 'Speed' feature
            else:
                column_information[1, feature_idx] = 2  # Other binary features
        binary_columns_speed = np.where(column_information[1] == 1)[0]  # Get the indices of binary columns
        pdp_results = {}  # To store results
        figures = []

        def generate_pdp(data_input, binary_name):
            """
            Generates Partial Dependence Plots (PDPs) for all non-binary features.

            :param data_input: torch.Tensor
                The input data used for PDP calculations.
            :param binary_name: str
                The identifier for the binary configuration of the dataset.

            :return: None
                Updates the `pdp_results` dictionary with computed PDP values.
            """

            local_pdp_results = {}  # Dictionary to store PDP results for individual features

            # Iterate over non-binary features based on column information
            for feature_idx in np.where(column_information[0] > 0)[0]:
                X_pdp = data_input.clone()  # Clone the input data to avoid modifications to the original

                # Set non-binary feature values to their mean or median based on the specified operation
                if operation == 'mean':
                    column_means = torch.mean(data_clone, dim=0)
                    X_pdp[:, column_information[0] > 0] = column_means[column_information[0] > 0]
                elif operation == 'median':
                    column_medians = torch.median(data_clone, dim=0)[0]
                    X_pdp[:, column_information[0] > 0] = column_medians[column_information[0] > 0]

                # Define the range of feature values based on percentiles
                feature_values = data_clone[:, feature_idx]
                feature_10th = torch.quantile(feature_values, percentile / 100).item()
                feature_90th = torch.quantile(feature_values, (100 - percentile) / 100).item()
                feature_values = torch.linspace(feature_10th, feature_90th, steps)  # Create a range of feature values

                predictions = []  # Store predicted values for each feature variation

                # Iterate over different values of the selected feature
                for val in feature_values:
                    X_pdp[:, feature_idx] = val  # Modify the feature value in the input data
                    with torch.no_grad():  # Disable gradient tracking to save memory
                        prediction = self.model(X_pdp, self.data.edge_index)  # Get model predictions
                    predictions.append(prediction.mean().item())  # Store the average prediction

                # Store results for this feature in a dictionary
                local_pdp_results[feature_names[feature_idx]] = {
                    'feature_values': feature_values.numpy(),
                    'predictions': np.array(predictions)
                }

            # Store the PDP results under the given binary configuration name
            pdp_results[binary_name] = local_pdp_results

            # **Step 1: Generate PDPs for original data**

        generate_pdp(data_clone, "Original")

        linestyles = ['solid', (0, (5, 5)), 'dashdot', (0, (1, 10)), (0, (1, 5)), (0, (1, 1))]

        # **Step 2: Generate PDPs for each binary feature set to 1**
        for binary_idx in np.where(column_information[1] > 0)[0]:  # Loop over binary 'Speed' features
            modified_data = data_clone.clone()
            if column_information[1][binary_idx] == 1:
                modified_data[:, binary_columns_speed] = 0
            modified_data[:, binary_idx] = 1  # Set binary feature to 1
            binary_name = f"{feature_names[binary_idx]} = 1"
            generate_pdp(modified_data, binary_name)

        # **Plot Results**
        for binary_name, pdp_data in pdp_results.items():
            fig = plt.figure(figsize=(10, 6))
            plt.suptitle(f'Partial Dependence Plots ({binary_name})', fontsize=20, fontweight='bold')

            for i, (feature_name, feature_data) in enumerate(pdp_data.items()):
                linestyle = linestyles[i % len(linestyles)]
                x_vals = np.linspace(percentile, (100 - percentile), len(feature_data['feature_values']))
                plt.plot(
                    x_vals,
                    feature_data['predictions'],
                    label=f'{feature_name}',
                    linestyle=linestyle,  # Set the linestyle
                    linewidth=4
                )
                plt.xlabel('Percentile of features', fontsize=20, fontweight='bold')
                plt.ylabel('Model Prediction average', fontsize=20, fontweight='bold')
                plt.ylim(0, 1)
                plt.tick_params(axis='both', labelsize=16, labelcolor='black')  # Increase tick label sizes
                for ax in fig.get_axes():  # Loop through all axes in the figure
                    for label in ax.get_xticklabels():  # Modify x-axis tick labels
                        label.set_fontweight('bold')
                    for label in ax.get_yticklabels():  # Modify y-axis tick labels
                        label.set_fontweight('bold')

                if save:
                    base_directory = r"C:\Users\woute\PycharmProjects\ThesisTesting\Results\PDP"
                    model_name = self.model.__class__.__name__ + self.criterion.__class__.__name__
                    parent_directory = os.path.join(base_directory, model_name)  # Parent directory for the model
                    city_directory = self.city_data.city_name
                    child_directory = os.path.join(parent_directory, city_directory)  # Subdirectory for figures

                    # Ensure both directories are created
                    os.makedirs(child_directory, exist_ok=True)

                    filenaam = binary_name + ".png"

                    # Full path to save the plot
                    full_path = os.path.join(child_directory, filenaam)
                    fig.savefig(full_path)

            figures.append(fig)
            # Return both PDP results and figures
        return pdp_results, figures


class MLPModel(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model for processing graph-based data.

    :param data_obj: object
        An object containing the PyTorch Geometric data structure (data_object_PyG).
    :param hidden_channels_list: list of int, optional
        A list specifying the number of neurons in each hidden layer. Defaults to [1] if not provided.

    :return: torch.nn.Module
        A fully connected neural network with ReLU activations and a final sigmoid activation for binary classification.
    """

    def __init__(self, data_obj, hidden_channels_list=None):
        super(MLPModel, self).__init__()
        data = data_obj.data_object_PyG  # Extract the PyG data object

        # If no hidden_channels_list is provided, default to one hidden layer with 1 output neuron
        if hidden_channels_list is None:
            hidden_channels_list = [1]

        # Define a list to hold the layers
        self.layers = nn.ModuleList()

        # Create the first layer (input to first hidden layer)
        self.layers.append(nn.Linear(data.num_features, hidden_channels_list[0]))

        # Create hidden layers dynamically from the hidden_channels_list
        for i in range(1, len(hidden_channels_list)):
            self.layers.append(nn.Linear(hidden_channels_list[i - 1], hidden_channels_list[i]))

        # Create the final output layer (last hidden layer to 1 output)
        self.layers.append(nn.Linear(hidden_channels_list[-1], 1))

    def forward(self, x, edge_index):
        """
        Forward pass of the MLP model.

        :param x: torch.Tensor
            Input node features.
        :param edge_index: torch.Tensor
            Edge indices representing the graph structure (not directly used in this MLP).

        :return: torch.Tensor
            Output tensor after passing through the network, with sigmoid activation applied.
        """
        test = edge_index  # Placeholder to ensure edge_index is passed but not used

        # Pass the input through each layer, followed by ReLU activation (except the last layer)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)

        # Apply the output layer without ReLU, followed by a sigmoid for binary classification
        x = self.layers[-1](x)
        x = torch.sigmoid(x)

        return x


class GCNModel(nn.Module):
    """
    A Graph Convolutional Network (GCN) model for node classification.

    :param data_obj: An object containing the PyTorch Geometric data structure (`data_object_PyG`).
    :param hidden_channels_list: A list specifying the number of neurons in each hidden layer.
                                 Defaults to a single-layer GCN if not provided.
    :param args: Additional arguments for the parent `nn.Module` class.
    :param kwargs: Additional keyword arguments for the parent `nn.Module` class.
    """

    def __init__(self, data_obj, hidden_channels_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = data_obj.data_object_PyG

        self.num_classes = len(torch.unique(data.y))  # Determine the number of unique classes
        self.convs = nn.ModuleList()  # Store convolutional layers

        # Define the number of input features
        in_channels = data.num_features

        if hidden_channels_list is None:
            # If no hidden_channels_list is provided, create a single convolutional layer
            self.convs.append(GCNConv(in_channels, 1))
        else:
            # Create convolutional layers based on the hidden_channels_list
            for hidden_channels in hidden_channels_list:
                self.convs.append(GCNConv(in_channels, hidden_channels))
                in_channels = hidden_channels  # Update in_channels for the next layer

            # Final layer that outputs a single value per node (binary classification)
            self.convs.append(GCNConv(in_channels, 1))

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        :param x: Node feature matrix of shape `(num_nodes, num_features)`.
        :param edge_index: Graph connectivity in COO format with shape `(2, num_edges)`.
        :return: Output tensor with shape `(num_nodes, 1)`, where each value is a probability (after sigmoid activation).
        """
        if len(self.convs) == 1:
            x = self.convs[0](x, edge_index)
        else:
            # Pass the input through each GCN layer
            for conv in self.convs[:-1]:  # Apply all but the last layer
                x = conv(x, edge_index)
                x = F.relu(x)  # ReLU activation

            # Apply the last layer (output should be a single value per node)
            x = self.convs[-1](x, edge_index)

        # Sigmoid activation for binary classification
        return torch.sigmoid(x)
