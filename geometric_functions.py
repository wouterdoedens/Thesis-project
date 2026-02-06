import numpy as np
import pandas as pd
import pyreadr
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.data import Data
import time
import copy


class GeometricDataTransformer:

    def __init__(self, city_object, city_df=None, dont_normalize=False):
        """
        Initializes the GeometricDataTransformer with a city's geometric and adjacency data.

        :param city_object: The city object created with the DataProcessor class
        :param city_df: A DataFrame containing city data, of the city object
                        defaults to `city_object.binary_df`.
        :param dont_normalize: If True, prevents normalization of numerical columns.
                               Default is False. Normalizes to stabilize training

        :raises ValueError: If `city_object` does not contain required attributes.
        """

        if city_df is not None:
            self.c_df = city_df  # Use provided city dataframe
        else:
            self.c_df = city_object.binary_df  # Default to binary dataset

        self.adj_df = city_object.adjacency_df  # Set adjacency matrix
        self.gdf = city_object.geometric_dataframe  # Geometric data

        # Ensure adjacency dataframe index is reset for merging
        self.adj_df.reset_index(drop=True, inplace=True)
        merged_df = pd.merge(self.adj_df.copy(), self.c_df, on='NWB_ID', how='left')
        merged_df.drop(columns=['adj_ID'], inplace=True)
        self.c_df = merged_df

        # Create edge tensors
        self.edge_tensor, self.er = self._tensor_creator()

        # Normalize data unless explicitly disabled
        if not dont_normalize:
            columns_to_normalize = self.c_df.columns.difference(['NWB_ID'])
            self.c_df[columns_to_normalize] = MinMaxScaler().fit_transform(self.c_df[columns_to_normalize])
            self.not_normalized = False  # Mark as normalized
        else:
            self.not_normalized = True  # Indicate unnormalized data

        # Placeholder for PyTorch Geometric Data object
        self.data_object_PyG = None

        # Store city name
        self.city_name = city_object.city_name

        # Placeholder for feature names
        self.feature_names = None

    @property
    def normalize_data(self):
        """
        This property method applies Min-Max normalization to the `c_df` DataFrame
        if the data has not already been normalized. The normalization scales each
        feature to a range between 0 and 1, making it suitable for algorithms
        sensitive to the scale of input data.

        Returns
        -------
        self : object
            Returns the instance with the normalized `c_df` DataFrame if the data
            was not normalized prior to this call.

        """
        if self.not_normalized:
            columns_to_normalize = self.c_df.columns.difference(['NWB_ID'])
            self.c_df[columns_to_normalize] = MinMaxScaler().fit_transform(self.c_df[columns_to_normalize])
            self.not_normalized = False
            return self
        else:
            print("The data is already normalized")
            return self

    def _tensor_creator(self):
        """
        Convert the stored DataFrame with adjacency information into a PyTorch Geometric edge tensor.

        Returns:
            torch.Tensor: Edge tensor suitable for PyTorch Geometric.
            list: List of NWB_IDs with missing information.
        """
        unique = self.adj_df['NWB_ID'].unique()  # Get the unique id's
        er = []  # Stores 'NWB_ID' that are in the adjacency matrix but that do not have road information
        c = 0  # To count which percentage of edges have been checked to be undirected
        tp = 0
        tensor_list = []
        for i in range(len(self.adj_df)):
            l = self.adj_df['adj_ID'][i].split(',')  # take out the string numbers and put them in a list
            for j in range(len(l)):
                if (l[j]) and (not l[j].isspace()):  # if the string text is not empty
                    k = int(l[j])  # Create an integer of the road segment id
                    if k in unique:  # if the road segment has information
                        # The variable i is the index of the current node, df.index[df['NWB_ID'] == k][0] gives the
                        # index of the connected node
                        tensor_list.append(torch.tensor([i, self.adj_df.index[self.adj_df['NWB_ID'] == k][0]]))
                    else:  # if the road segment id has no information
                        if k not in er:  # Make sure it is not already in the list
                            er.append(k)  # save it to this list
        edge_tensor = torch.stack(tensor_list, dim=0).to(dtype=torch.long)  # Stack the tensors for the tensor list
        print('The data has to be made undirected')
        for edge in edge_tensor:  # For each edge in the tensor
            percentage = round((c / len(edge_tensor) * 100))  # Keeps count of how far in the control process we are
            c += 1
            reverse_edge = torch.tensor([edge[1], edge[0]])  # reverse the edge in the list
            if not (
                    (edge_tensor == reverse_edge.unsqueeze(0)).all(
                        dim=1).any()):  # check if it exists somewhere in the lists
                tensor_list.append(reverse_edge)  # If not, add this reverse tensor to the tensor list
            if percentage % 10 == 0:
                if tp != percentage:
                    tp = percentage
                    print(str(tp) + " percent done")  # Prints how far the process is
        edge_tensor = torch.stack(tensor_list, dim=0).to(
            dtype=torch.long)  # stack the new tensor list and create a new tensor
        return edge_tensor, er

    def create_data_object(self, target_column=None, drop_columns= None):
        """
        Create a PyTorch Geometric `Data` object from the stored DataFrame and the provided edge tensor.

        :param target_column: The column used as the target variable for graph learning.
                              Default is `'Crash_binary'`, which indicates the presence or absence of crashes.
        :param drop_columns: List of additional feature columns to exclude from node attributes.
                             By default, `['NWB_ID', target_column]` are excluded.

        :returns: A new instance of `GeometricDataTransformer` with an attached PyTorch Geometric `Data` object, containing:
                  - `x` (torch.Tensor): Node features of shape `(num_nodes, num_features)`.
                  - `edge_index` (torch.Tensor): Edge indices of shape `(2, num_edges)`, representing graph connections.
                  - `y` (torch.Tensor): Target values for nodes of shape `(num_nodes,)`.
                  - `num_classes` (int): The number of unique classes in the target variable.
                  - `feature_names` (pd.Index): List of feature column names used in `x`.
        :rtype: GeometricDataTransformer
        """

        # Create a copy of the instance to avoid modifying the original object
        new_instance = copy.copy(self)

        if target_column is None:
            target_column = 'Crash_binary'  # Default target column

        if drop_columns is None:
            drop_list = ['NWB_ID', target_column]  # Drop only ID and target column
        else:
            drop_list = ['NWB_ID', target_column] + drop_columns  # Add additional drop columns

        if (new_instance.data_object_PyG is None) or (target_column is not None) or (drop_columns is not None):
            # Create edge index tensor
            edge_ind = torch.stack([new_instance.edge_tensor[:, 0], new_instance.edge_tensor[:, 1]], dim=0)

            # Extract target values
            crashes = new_instance.c_df[target_column].tolist()

            # Drop the specified columns from features
            df_except_id = new_instance.c_df.drop(drop_list, axis=1)
            new_instance.feature_names = df_except_id.columns

            # Convert features to NumPy and then to a PyTorch tensor
            node_features = df_except_id.to_numpy(dtype=float)
            x = torch.tensor(node_features, dtype=torch.float, requires_grad=True)
            y = torch.tensor(crashes, dtype=torch.float, requires_grad=True)

            # Create PyTorch Geometric Data object
            new_instance.data_object_PyG = Data(x=x, edge_index=edge_ind, y=y, num_classes=len(set(crashes)))

        return new_instance


