from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
import os
import matplotlib.colors as mcolors
from geopandas import GeoDataFrame
from pandas import DataFrame
from typing import Tuple, Optional
from torch_geometric.utils import k_hop_subgraph
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import time


class GraphVisualisations:
    @classmethod
    def _network_visualizer(cls,
                            name: str,
                            gdf: GeoDataFrame,
                            dataframe: DataFrame,
                            colormap: str,
                            value_column: Optional[str] = None,
                            legend: bool = True,
                            figsize: Tuple[int, int] = (10, 8),
                            title: Optional[str] = None,
                            save=False,
                            path=None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualizes a network by merging a GeoDataFrame and a DataFrame, then plotting it.
        :param name: The name of the city
        :param gdf: GeoDataFrame of the city with linestrings
        :param dataframe: DataFrame containing data to merge with the GeoDataFrame.
        :param colormap: Color map for plotting.
        :param value_column: Column name in `dataframe` to visualize. Default is the second column.
        :param legend: Whether to display the legend. Default is True.
        :param figsize: Size of the plot figure. Default is (10, 8).
        :param title: Custom title for the plot. Default is None, using a generated title.

        :returns: A tuple containing the Matplotlib figure and axes with the plotted visualization.
        """

        # Set the default value column to the second column if not specified
        if value_column is None:
            value_column = dataframe.columns[1]

        if value_column not in dataframe.columns:
            raise ValueError(f"Value column '{value_column}' not found in DataFrame.")

        # Merge the dataframes
        merged_gdf = gdf.merge(dataframe, on='NWB_ID')

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        merged_gdf.plot(column=value_column, cmap=colormap, legend=legend, ax=ax)

        # Set title and labels
        plot_title = title if title else f"Visualization of {name} based on {value_column}"
        ax.set_title(plot_title, fontsize=16, fontweight='bold')
        ax.set_xticks([])  # remove the numbers on the x and y labels
        ax.set_yticks([])
        ax.tick_params(axis='both', labelsize=16, labelcolor='black')  # Increase tick label sizes

        for label in ax.get_xticklabels():  # Modify x-axis tick labels
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():  # Modify y-axis tick labels
            label.set_fontweight('bold')
        # Access the colorbar
        cbar = ax.get_figure().axes[-1]  # The colorbar is added as the last axis in GeoPandas

        # Customize the colorbar tick labels
        cbar.tick_params(labelsize=14, labelcolor='black')  # Set size and color of ticks
        for tick_label in cbar.get_yticklabels():  # Make colorbar tick labels bold
            tick_label.set_fontweight('bold')
        if save:
            fig.savefig(path)
        return fig, ax


class GNNExplainers(GraphVisualisations):
    """
    A class for generating explanations of Graph Neural Networks (GNNs) using GNNExplainer.

    :param trainer: The training object containing the trained model and dataset.
    :param path_name: Path to store or load precomputed explanations.
    :param device: The computing device ('cpu' or 'cuda'). Defaults to GPU if available.
    :param start: The starting index for explanation generation.
    :param end: The ending index for explanation generation.
    :param set_Dictionary: Precomputed dictionary of explanations (if provided).
    """

    def __init__(self, trainer, path_name=None, device=None, start=None, end=None, set_Dictionary=None):
        # Set device to GPU if available
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Move trainer model and data to the device
        self.trainer = trainer
        self.trainer.model = self.trainer.model.to(self.device)
        self.trainer.data = self.trainer.data.to(self.device)
        self.start = start
        self.end = end

        # Initialize explainer
        self.gnnexplainer = self.initiate_gnnexplainer()
        if set_Dictionary is None:
            self.explainer_dictionary = self.get_explainers_dictionary(path_name=path_name)
        else:
            self.explainer_dictionary = set_Dictionary
        self.counts, self.feature_importance = self.get_counts_and_values()

    def initiate_gnnexplainer(
            self,
            num_epochs=200,
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            mode='binary_classification',
            task_level='node',
            return_type='raw'
    ):
        """
        Initializes an instance of the GNNExplainer.

        :param num_epochs: Number of training epochs for the explainer.
        :param explanation_type: Type of explanation, e.g., 'model' or 'phenomenon'.
        :param node_mask_type: Type of node mask (e.g., 'attributes' for feature-based masking).
        :param edge_mask_type: Type of edge mask (e.g., 'object' for edge-based masking).
        :param mode: The model's mode, e.g., 'binary_classification' or 'regression'.
        :param task_level: The level of explanation, e.g., 'node' or 'edge'.
        :param return_type: Type of output returned by the model (e.g., 'raw' or 'probabilities').
        :return: A configured instance of the `Explainer` class.
        """
        # Ensure the model is on the correct device
        self.trainer.model = self.trainer.model.to(self.device)

        gnnexplainer = Explainer(
            model=self.trainer.model,  # The GNN model being explained
            algorithm=GNNExplainer(epochs=num_epochs),  # Configured GNNExplainer
            explanation_type=explanation_type,  # Explanation scope, e.g., 'model'
            node_mask_type=node_mask_type,  # Explanation basis: node features
            edge_mask_type=edge_mask_type,  # Explanation basis: edge attributes
            model_config=dict(
                mode=mode,  # Operational mode of the model, e.g., binary classification
                task_level=task_level,  # Level of explanation, e.g., node-level
                return_type=return_type,  # Type of prediction returned, e.g., raw logits
            ),
        )
        self.gnnexplainer = gnnexplainer
        return gnnexplainer

    def get_counts_and_values(self):
        """
        Computes feature importance and node activation counts based on the explainer dictionary.

        :return: A tuple containing:
            - counts: A NumPy array representing the number of times each node was active in an explanation.
            - feature_importance: A NumPy array indicating the summed importance of each feature for each node.
        """
        data = self.explainer_dictionary  # Retrieve the stored explainer dictionary
        shape = self.trainer.data.x.shape  # Get the shape of the input data (nodes × features)
        # Initialize arrays to store node activation counts and feature importance values
        node_counts = np.zeros(shape[0])
        zero_counts = np.zeros(shape[0])
        feature_individual_values = np.zeros([shape[0], shape[1]])
        has_explainer_list = []

        for i in range(self.trainer.data.x.shape[0]):
            node_mask = data[str(i)][0]  # Retrieve the node mask from the explainer dictionary
            active_nodes = torch.where(node_mask > 0)[0].unique()  # Find unique active nodes
            if active_nodes.numel() > 0:  # Check if there are any active nodes
                has_explainer_list.append(i)  # Add the node to the explainer list
                values = node_mask.sum(dim=0)  # Sum feature importance values for the node
                feature_individual_values[i, :] = values.detach().cpu().numpy()   # Store feature importance
                node_counts[active_nodes.detach().cpu().numpy()] += 1  # Store feature importance

        zero_counts[has_explainer_list] += 1 # Mark nodes that have explanations
        self.counts = node_counts
        self.feature_importance = feature_individual_values
        return self.counts, self.feature_importance

    def get_explainers_dictionary(self, overwrite=False, path_name=None):
        """
        Generates and saves explanations for the GNN model using GNNExplainer. Has a large computation time

        :param overwrite: If True, overwrite the existing explanation file. Defaults to False.
        :param path_name: Path to save or load the explainer dictionary. If None, a default path is generated.
        :return: A dictionary where keys are node indices (or 'graph' for global explanations), and values are tuples
                 containing (node_mask, edge_mask).
        """
        # Ensure the data is on the correct device
        self.trainer.data = self.trainer.data.to(self.device)
        number_nodes = self.trainer.data.x.shape[0]

        # Define the path for saving/loading explanations
        if path_name is None:
            path_name = (self.trainer.city_data.city_name +
                         self.trainer.model.__class__.__name__ +
                         self.gnnexplainer.algorithm.__class__.__name__ + '.pth')

        # Load existing explanations if available and not overwriting
        if os.path.exists(path_name) and not overwrite:
            self.explainer_dictionary = torch.load(path_name)
            return self.explainer_dictionary

        start_time = time.time()
        explainer_dictionary = {}
        explainer = self.gnnexplainer

        # Generate graph-level explanation (if applicable)
        if self.start is None or self.start == 0:
            explanation = explainer(self.trainer.data.x, self.trainer.data.edge_index)
            explainer_dictionary['graph'] = (explanation['node_mask'], explanation['edge_mask'])

        # Initialize count tracking for active nodes in explanations
        counts = np.zeros([number_nodes])

        # Set end index to the total number of nodes if not specified
        if self.end is None:
            self.end = number_nodes

        # Generate node-level explanations
        for i in range(self.start, self.end):
            new_time = time.time()
            explanation = explainer(self.trainer.data.x, self.trainer.data.edge_index, index=i)

            # Identify and count active nodes
            used_node_indices = torch.where(explanation['node_mask'] > 0)[0].unique()
            counts[used_node_indices.cpu().numpy()] += 1

            # Store explanation for the current node
            explainer_dictionary[str(i)] = (explanation['node_mask'], explanation['edge_mask'])

            # Save intermediate results every 1000 nodes
            if (i + 1) % 1000 == 0:
                print(f"Saving intermediate file after {i + 1} runs...")
                torch.save(explainer_dictionary, path_name)
                print(f"Intermediate file saved: {path_name}")

            print(f"It took {time.time() - new_time:.4f} seconds for explainer {i} to run")

        print(f"The total time is {time.time() - start_time:.4f} seconds")

        # Save final explainer dictionary
        self.explainer_dictionary = explainer_dictionary
        torch.save(self.explainer_dictionary, path_name)
        return self.explainer_dictionary

    def _visualize_network(self, node_index='graph', based_on='counts', save=False, path=None):
        """
        Generates network visualization data based on node importance, neighbors, or specific features.

        :param node_index: Node index to visualize. Use 'graph' for global visualization or a specific node index for local visualization.
        :param based_on: Determines the visualization metric. Options:
                         - 'counts': Highlights active nodes based on their importance counts.
                         - 'neighbours': Colors nodes based on their k-hop neighborhood relationships.
                         - Any feature name: Uses the specified feature importance values.
        :param save: If True, the visualization is saved to the specified path.
        :param path: File path to save the visualization. If None, a default path is generated.
        :return: Tuple (DataFrame containing node visualization data, colormap, save flag, path).
        """
        df = self.trainer.city_data.c_df[['NWB_ID']].copy()
        model = self.trainer.model
        max_khop = 0
        edge_index = self.trainer.data.edge_index
        subset = 0

        if hasattr(model, 'convs'):
            max_khop = len(model.convs)
        elif hasattr(model, 'layers'):
            max_khop = len(model.layers)

        new_counts = np.zeros(len(df))
        gdf = self.trainer.city_data.gdf

        # Determine active nodes based on explanations
        if node_index == 'graph':
            node_mask = self.explainer_dictionary['graph'][0]
            active_nodes = torch.where(node_mask > 0)[0].unique().cpu().numpy()
            new_counts[active_nodes] = self.counts[active_nodes]
        elif node_index in range(len(self.counts)):
            subset, edge_index_khop, _, _ = k_hop_subgraph(node_index, max_khop, edge_index, relabel_nodes=True)
            node_mask = self.explainer_dictionary[str(node_index)][0]
            active_nodes = torch.where(node_mask > 0)[0].unique().cpu().numpy()
            df = df.iloc[subset]
            new_counts[active_nodes] = self.counts[active_nodes]
            new_counts = new_counts[subset]
        else:
            raise ValueError("Please input a valid node index")

        # Define colormap and assign visualization values based on the metric
        if based_on == 'counts':
            df['node_counts'] = new_counts
            cmap = 'viridis'
        elif based_on == 'neighbours':
            if node_index == 'graph':
                new_strings_list = ["Graph Nodes"] * len(self.counts)
                new_strings_list = ["Active Nodes" if i in active_nodes else new_strings_list[i] for i in
                                    range(len(new_strings_list))]
            else:
                new_strings_list = [str(max_khop) + "-hop neighbour"] * len(self.counts)
                new_strings_list = [
                    "Active " + str(max_khop) + "-hop neighbour" if i in active_nodes else new_strings_list[i] for i in
                    range(len(new_strings_list))
                ]
                for j in range(len(self.trainer.model.convs) - 1, 0, -1):
                    subset2, edge_index_khop, _, _ = k_hop_subgraph(node_index, j, edge_index, relabel_nodes=True)
                    new_strings_list = [str(j) + "-hop neighbour" if i in subset2 else new_strings_list[i] for i in
                                        range(len(new_strings_list))]
                    common_elements = subset2[torch.isin(subset2, active_nodes)]
                    new_strings_list = [
                        "Active " + str(j) + "-hop neighbour" if i in common_elements else new_strings_list[i] for i in
                        range(len(new_strings_list))
                    ]
                new_strings_list[node_index] = "Center Node"
                new_strings_list = [new_strings_list[i] for i in subset]
            df['neighbours'] = new_strings_list
            colors = ['#FFFF00', '#008000', '#00FFFF', '#0000FF', '#FF0000']  # Yellow -> Green -> Cyan -> Blue -> Red
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_red_blue", colors)
        elif based_on in self.trainer.city_data.feature_names:
            new_counts = np.zeros(len(self.counts))
            idx = self.trainer.city_data.feature_names.get_loc(based_on)
            feature_importance_values = self.feature_importance[:, idx]
            new_counts[active_nodes] = feature_importance_values[active_nodes]
            if node_index in range(len(self.counts)):
                new_counts = new_counts[subset]
            df[str(based_on)] = new_counts
            cmap = 'viridis'
        else:
            raise ValueError('Please input a valid "based_on" value')

        # Save visualization if required
        if save:
            base_directory = r"C:\Users\woute\PycharmProjects\ThesisTesting\Results\GNNExplainer"
            network_directory = 'Networks'
            child_directory = os.path.join(base_directory, network_directory)
            model_name = self.trainer.model.__class__.__name__ + self.trainer.criterion.__class__.__name__
            child_directory2 = os.path.join(child_directory, model_name)
            child_directory3 = os.path.join(child_directory2, based_on)
            os.makedirs(child_directory3, exist_ok=True)
            filename = self.trainer.city_data.city_name + ".png"
            path = os.path.join(child_directory3, filename)

        return df, cmap, save, path

    def visualize_network(self, node_index='graph', based_on='counts', save=False, path=None, set_title=None):
        """
        Generates and displays a network visualization based on a selected metric.

        :param node_index: Node index to visualize. Use 'graph' for global visualization or a specific node index for local visualization.
        :param based_on: Determines the visualization metric. Options:
                         - 'counts': Highlights active nodes based on their importance counts.
                         - 'neighbours': Colors nodes based on their k-hop neighborhood relationships.
                         - Any feature name: Uses the specified feature importance values.
        :param save: If True, the visualization is saved to the specified path.
        :param path: File path to save the visualization. If None, a default path is generated.
        :param set_title: Custom title for the visualization. If 'based_on', it uses the `based_on` value as the title.
        """
        if set_title == 'based_on':
            set_title = based_on

        df, cmap, save2, path2 = self._visualize_network(node_index, based_on, save, path)
        gdf = self.trainer.city_data.gdf

        self._network_visualizer(self.trainer.city_data.city_name, gdf, df, colormap=cmap, save=save2, path=path2,
                                 title=set_title)

    def visualize_features(self, top_k=None, return_type='histogram', node_index=None, save=False):
        """
        Visualizes feature importance using histogram or boxplot.

        :param top_k: (Optional) Number of top features to display. Defaults to all features.
        :param return_type: Type of visualization ('histogram' or 'boxplot').
        :param node_index: (Optional) Node index for node-specific feature visualization.
        :param save: If True, saves the visualization to a specified directory.
        :return: A pandas DataFrame containing feature names and their importance values (only for histogram).
        :raises ValueError: If an invalid return_type is provided.
        """

        num_features = len(self.trainer.city_data.feature_names)
        if (top_k is None) or (top_k >= num_features):
            top_k = num_features
        if return_type == 'histogram':
            if node_index is None:
                values = self.feature_importance.sum(axis=0)
                top_k_values, top_k_indices = torch.topk(values, top_k)
                top_k_indices = top_k_indices.cpu().numpy()
                top_k_values = top_k_values.cpu().numpy()
                title = "Histogram"
            elif node_index in range(self.trainer.data.x.shape[0]):
                top_k_indices = np.argsort(self.feature_importance[node_index, :])[-top_k:][::-1]
                top_k_indices = top_k_indices.cpu().numpy()
                top_k_values = self.feature_importance[node_index, :][top_k_indices]
                title = "Histogram for node " + str(node_index)
            else:
                raise ValueError("Please input a valid node index")
            top_k_labels = self.trainer.city_data.feature_names[top_k_indices]
            fig = plt.figure(figsize=(10, 6))
            plt.bar(top_k_labels, top_k_values, color='skyblue')
            plt.xlabel("Feature", fontsize=20, fontweight='bold')
            plt.ylabel("Feature Importance", fontsize=20, fontweight='bold')
            plt.title(f"Top {top_k} Highest Values " + title, fontsize=24, fontweight='bold')
            plt.xticks(rotation=45, ha="right", fontsize=16, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.tight_layout()
            loss_name = self.trainer.criterion.__class__.__name__
            city_name = self.trainer.city_data.city_name
            if save:
                base_directory = r"C:\Users\woute\PycharmProjects\ThesisTesting\Results\GNNExplainer"
                parent_directory = os.path.join(base_directory, "Features")
                histogram_directory = return_type + 's'
                child_directory = os.path.join(parent_directory, histogram_directory)
                model_name = self.trainer.model.__class__.__name__ + loss_name
                child_directory2 = os.path.join(child_directory, model_name)
                os.makedirs(child_directory2, exist_ok=True)
                filename = city_name + ".png"
                # Full path to save the plot
                full_path = os.path.join(child_directory2, filename)
                fig.savefig(full_path)
            plt.show()
            column2 = loss_name + city_name
            importance_df = pd.DataFrame({
                "Feature": top_k_labels,
                column2: top_k_values
            })
            return importance_df
        elif return_type == 'boxplot':
            df = pd.DataFrame(self.feature_importance, columns=self.trainer.city_data.feature_names)
            fig = plt.figure(figsize=(10, 6))  # Optional: adjust the figure size
            sns.boxplot(data=df)  # This will create box plots for each column in the DataFrame
            plt.xlabel('Features', fontsize=20, fontweight='bold')
            plt.ylabel('Feature Importance', fontsize=20, fontweight='bold')
            plt.title('Box Plots of Each Feature', fontsize=24, fontweight='bold')
            plt.xticks(rotation=45, ha="right", fontsize=16, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')

            plt.tight_layout()
            if save:
                base_directory = r"C:\Users\woute\PycharmProjects\ThesisTesting\Results\GNNExplainer"
                parent_directory = os.path.join(base_directory, "Features")
                histogram_directory = return_type + 's'
                child_directory = os.path.join(parent_directory, histogram_directory)
                model_name = self.trainer.model.__class__.__name__ + self.trainer.criterion.__class__.__name__
                child_directory2 = os.path.join(child_directory, model_name)
                os.makedirs(child_directory2, exist_ok=True)
                filename = self.trainer.city_data.city_name + ".png"
                # Full path to save the plot
                full_path = os.path.join(child_directory2, filename)
                fig.savefig(full_path)
            plt.show()
        else:
            raise ValueError("Pleas input a valid retun type. Either histogram of boxplot")


""" Everything below is old, I used the subplot creator  from visualisations"""


def combined_network_subplots(explainers_list, based='counts', save=False, model_names=None, save_name=None):
    """
    Generates a combined subplot visualization of multiple explainers.

    :param explainers_list: List of GNNExplainers objects to visualize.
    :param based: Attribute to base the visualization on (e.g., 'counts').
    :param save: If True, saves the generated plot.
    :param model_names: (Optional) List of model names to use as titles for the subplots.
    :param save_name: (Optional) Custom name for the saved plot file.
    """
    use_model_names = False
    if len(explainers_list) == 4:
        row = 2
        size = (20, 16)
    else:
        row = 1
        size = (12, 8)
        use_model_names = True
    fig, axes = plt.subplots(nrows=row, ncols=2, figsize=size, constrained_layout=True)
    axes = axes.flatten()  # Flatten for easier indexing
    vmin = float('inf')
    vmax = float('-inf')
    df_merges = []
    v_columns = []
    for explainer in explainers_list:
        gdf = explainer.trainer.city_data.gdf
        df, cc, st, path = explainer._visualize_network(based_on=based)
        value_column = df.columns[1]
        v_columns.append(value_column)
        merged_gdf = gdf.merge(df, on='NWB_ID')
        df_merges.append(merged_gdf)
        vmin = min(vmin, df[value_column].min())
        vmax = max(vmax, df[value_column].max())
    colormap = 'viridis'
    for i, m_df in enumerate(df_merges):
        ax = axes[i]
        m_df.plot(
            column=v_columns[i],
            cmap=colormap,
            legend=False,  # Disable individual legends
            ax=ax,
            vmin=vmin,  # Use global vmin
            vmax=vmax,  # Use global vmax
        )

        # Set title and remove ticks
        if use_model_names:
            city_name = model_names[i]
        else:
            city_name = explainers_list[i].trainer._city_name
        ax.set_title(city_name, fontsize=20, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_edgecolor('black')  # Define frame color
            spine.set_linewidth(2)  # Make the frame thicker

        ax.set_aspect('auto', adjustable='box')

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []  # Dummy array for the colorbar
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', shrink=1.2, fraction=0.03, pad=0.04)
    cbar.set_label(based, fontsize=16, fontweight='bold')  # Change the label as needed
    cbar.ax.tick_params(labelsize=14, width=2)  # Increase tick size and line width
    for label in cbar.ax.get_xticklabels():  # Make tick labels bold
        label.set_fontweight('bold')
    if save:
        base_directory = r"C:\Users\woute\PycharmProjects\ThesisTesting\Results\GNNExplainer"
        parent_directory = os.path.join(base_directory, "Networks")
        model_name = explainers_list[0].trainer.model.__class__.__name__ + explainers_list[
            0].trainer.criterion.__class__.__name__
        child_directory = os.path.join(parent_directory, model_name)
        child_directory2 = os.path.join(child_directory, based)
        os.makedirs(child_directory2, exist_ok=True)
        if save_name is None:
            save_name = "CombinedSubplots"
        filename = save_name + ".png"
        # Full path to save the plot
        full_path = os.path.join(child_directory2, filename)
        fig.savefig(full_path)
    plt.show()


def create_subplot_from_figures(figures, columns=2, figsize=(20, 16)):
    """
    Creates a grid of subplots from a list of figure-generating functions.

    :param figures: List of functions that generate individual plots. Each function must accept an axis object as an argument.
    :param columns: Number of columns in the subplot grid. Default is 2.
    :param figsize: Tuple specifying the overall figure size. Default is (20, 16).
    :return: None. Displays the generated subplot grid.
    """
    # Calculate rows needed for the number of figures
    total_figures = len(figures)
    rows = (total_figures // columns) + (total_figures % columns)

    # Create a subplot grid
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=figsize, constrained_layout=True)

    # Ensure axes is always a 2D array
    axes = np.atleast_2d(axes)

    # Plot each figure into the subplot
    for idx, plot_func in enumerate(figures):
        row_idx, col_idx = divmod(idx, columns)
        ax = axes[row_idx, col_idx]

        # Call the plotting function on the axis
        plot_func(ax)

    # Hide unused axes
    for idx in range(len(figures), rows * columns):
        row_idx, col_idx = divmod(idx, columns)
        axes[row_idx, col_idx].axis("off")

    # Show the figure
    plt.show()
