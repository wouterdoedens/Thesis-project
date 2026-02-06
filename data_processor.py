import numpy as np
import pandas as pd
import pyreadr
import geopandas as gpd  # For handling geographic data
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import geometric_functions as g
from matplotlib.colors import ListedColormap
from GNN_Explainers_class import GraphVisualisations
from typing import Literal, Union

adjacency_Rdata = pyreadr.read_r('AdjacencyList.RData')  # Read adjacency data
adjacency_data = adjacency_Rdata['AdjacencyList']  # extract data
crash_Rdata = pyreadr.read_r('DataCrash.RData')  # read crash data
crash_data = crash_Rdata['DataCrash']


class DataLoader:
    """
    A class the adjacency table, crash data table and the geometric road position data

    Attributes:
    ----------
    adjacency_data : DataFrame or None
        Class-level attribute that stores adjacency data, ensuring it is loaded only once.
    crash_data : DataFrame or None
        Class-level attribute that stores crash data, ensuring it is loaded only once.
    geometric_dataframe : GeoDataFrame
        Instance-level attribute that holds the geometric data of a specific city.

    Methods:
    -------
    _load_data()
        Loads adjacency and crash data files into class-level attributes.
    _load_geometric_dataframe()
        Loads and returns a geographic DataFrame of the 4 cities
    """

    # Class-level attributes to store adjacency and crash data, loaded once for all instances
    adjacency_data = None
    crash_data = None

    def __init__(self, city):
        """
        Initializes a DataLoader instance for a specific city. If adjacency and crash data
        have not been loaded yet, they will be loaded at the class level to ensure efficient reuse.

        :param city: str
            The name of the city for which data is being loaded. needed for the child class Dataprocessor
        """

        # Load adjacency and crash data only if they haven't been loaded before
        if DataLoader.adjacency_data is None or DataLoader.crash_data is None:
            DataLoader._load_data()

        # Load geographic data specific to the given city
        self.geometric_dataframe = self._load_geometric_dataframe()

    @classmethod
    def _load_data(cls):
        """
        Class method to load adjacency and crash data into class-level attributes.
        This ensures that the data is only loaded once, avoiding redundant data loads
        for multiple instances of the class.

        :returns: None
        """

        # Load adjacency data from an RData file and store it in the class attribute
        cls.adjacency_data = pyreadr.read_r('AdjacencyList.RData')['AdjacencyList']

        # Load crash data from an RData file and store it in the class attribute
        cls.crash_data = pyreadr.read_r('DataCrash.RData')['DataCrash']

    @staticmethod
    def _load_geometric_dataframe():
        """
        Static method to load a geographic DataFrame from a shapefile.
        This data represents the geometry of the 4 cities, using shape-based information.

        :returns: GeoDataFrame
            A GeoDataFrame containing the shape data of the 4 cities.
        """

        # Define the file path to the shapefile containing geographic data
        shapefile_path = 'Shapefiles/NWB_ID_LONG_shape.shp'

        # Load the shapefile into a GeoDataFrame using geopandas
        gdf = gpd.read_file(shapefile_path)

        return gdf


class DataProcessor(DataLoader, GraphVisualisations):
    """
    A class for processing and visualizing city-specific data, extending the functionality of both
    DataLoader (for data loading) and GraphVisualisations (for data visualization).
    """

    def __init__(self, city, remove_binaries=True):
        """
        Initializes the DataProcessor instance for a specific city, loading data as needed
        and setting up city-specific data for processing.

        :param city:str The name of the city for which data processing is performed.
        :param remove_binaries: A boolean flag indicating to remove additional binaries in the speed feature columns.
               Default is True.
        """

        # Call the parent class's constructor with the city name
        super().__init__(city)

        # Validate if the provided city exists in the crash data
        if city not in DataProcessor.crash_data['mun_fct_'].unique():
            raise ValueError(f"'{city}' is not a valid city name. Execution halted.")

        # Store the city name as a private attribute
        self._city_name = str(city)

        # Filter crash data for the specified city
        self._city_df = DataProcessor.crash_data.loc[
            DataProcessor.crash_data['mun_fct_'] == self._city_name
            ]

        # Remove additional binary columns if the flag is set to True
        if remove_binaries:
            self._city_df = self._remove_additional_binaries()

        # Filter geometric data based on NWB_ID matches with the city's crash data
        self._city_gdf = self.geometric_dataframe[
            self.geometric_dataframe['NWB_ID'].isin(self._city_df['NWB_ID'].unique())
        ]

        # Initialize placeholders for different processed datasets
        self._integer_df = None  # Placeholder for integer-encoded data
        self._spatial_df = None  # Placeholder for spatially aggregated data
        self._binary_df = None  # Placeholder for binary-encoded data
        self._adjacency_df = None  # Placeholder for adjacency matrix
        self._diff = None  # Placeholder for difference calculations

    @property
    def city_name(self):
        """Returns the name of the city being processed."""
        return self._city_name

    @property
    def city_df(self):
        """Returns the crash data DataFrame filtered for the specific city."""
        return self._city_df

    def _remove_additional_binaries(self):
        """
        Removes conflicting binary indicators in speed-related columns.
        Ensures that if a road segment is classified under one speed category, it is not flagged under others.
         :return: A new DataFrame where all roads can only be one road category
        """
        speed_columns = ['Speed_50_separated', 'Speed_30_separated',
                         'Speed_50_on_road', 'Speed_30_on_road']
        df = self.city_df.copy()
        for column in speed_columns:
            conflicting_columns = [col for col in speed_columns if col != column]
            df.loc[df['Speed_infra_fct__'] == column, conflicting_columns] = 0
        return df

    @property
    def gdf(self):
        """Returns the geographic data GeoDataFrame filtered for the specific city."""
        return self._city_gdf

    def _get_unique_ids(self):
        """
        Extracts unique road segment IDs for the selected city and retrieves the corresponding adjacency data.
        Ensures only roads with features are used.
        """
        unique_ids = self._city_df['NWB_ID'].unique()
        df = self.adjacency_data[self.adjacency_data['NWB_ID'].isin(unique_ids)]
        diff = len(self.adjacency_data) - len(df)
        return df, diff

    @property
    def adjacency_df(self):
        """Retrieves adjacency data for the city, filtered by unique road segment IDs.
        """
        if self._adjacency_df is None:
            self._adjacency_df, self._diff = self._get_unique_ids()
        return self._adjacency_df

    @property
    def diff(self):
        """Retrieves the row count difference between the full adjacency dataset and the city-specific subset."""
        if self._diff is None:
            _ = self.adjacency_df  # Ensures adjacency_df is initialized
        return self._diff

    def _keep_integer_columns(self):
        """
        Filters the city's DataFrame to retain only numeric columns.
        Converts numeric strings to floats where applicable.
        """
        df = self._city_df.copy()
        for column in df.columns:
            if df[column].dtype == object:  # Check if column is a string
                try:
                    df[column] = df[column].str.replace(',', '.').astype(float)
                except ValueError:
                    df.drop(columns=[column], inplace=True)
        return df

    @property
    def integer_df(self):
        """Retrieves or generates a DataFrame with integer-only columns for the city.
         :return: A new DataFrame with only numeric features
            """
        if self._integer_df is None:
            self._integer_df = self._keep_integer_columns()
        return self._integer_df

    def _create_spatial_df(self, data_df=None):
        """
        Creates a spatially aggregated DataFrame prepared for graph neural network analysis.

        :param data_df: (Optional) A DataFrame containing road network data.
                        If not provided, the method uses self._integer_df.
        :return: A new DataFrame with aggregated spatial data.
        """

        if data_df is None:
            integer_df_copy = self.integer_df
        else:
            integer_df_copy = data_df
        column_list = integer_df_copy.columns
        start = False
        spatial_df = integer_df_copy.groupby('NWB_ID')[
            ['Crash_freq', 'Crash_fatal', 'Crash_injury', 'Crash_binary']].sum().reset_index()

        for i in range(len(column_list)):
            if column_list[i] == 'length_km':
                start = True
            if start:
                if any(column_list[i] == s for s in column_list[10:14]):
                    spatial_df[column_list[i]] = integer_df_copy.groupby('NWB_ID')[column_list[i]].sum().reset_index()[
                        column_list[i]]
                else:
                    spatial_df[column_list[i]] = integer_df_copy.groupby('NWB_ID')[column_list[i]].first().reset_index(
                        drop=True)

        return spatial_df

    @property
    def spatial_df(self):
        """Retrieves or generates a spatial DataFrame with aggregated values for crash-related columns."""
        if self._spatial_df is None:
            self._spatial_df = self._create_spatial_df()
        return self._spatial_df

    def _create_binary_df(self):
        """
        Converts crash-related columns in the spatial DataFrame to binary indicators.
        """
        df = self.spatial_df.copy()
        for column in ['Crash_freq', 'Crash_fatal', 'Crash_injury', 'Crash_binary']:
            df[column] = df[column].apply(lambda x: 1 if x > 0 else 0)
        return df

    @property
    def binary_df(self):
        """Retrieves or generates a DataFrame with binary crash indicators for the city."""
        if self._binary_df is None:
            self._binary_df = self._create_binary_df()
        return self._binary_df

    def visualize_network(self,
                          dataframe='binary',
                          based_on='speed',
                          set_title=None,
                          show_legend=True) -> None:
        """
        Visualizes the network based on the specified DataFrame and column.

        :param dataframe: Specifies the DataFrame to use for visualization. Choose either 'binary' or 'spatial'.
                          Default is 'binary'.
        :param based_on: The criteria to visualize the network. Options include:
                         - 'speed': Visualizes based on the speed of the network.
                         - 'buildings': Visualizes the types and number of buildings.
                         - Any of the feature column values
        :param set_title: (Optional) Title of the visualization.
        :param show_legend: Whether to show the legend. Default is True.

        :raises ValueError: If an invalid DataFrame or criteria is specified.

        :returns: A visualization of the road network.
        """

        # Define color palette
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
            "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"
        ]

        # Select appropriate DataFrame
        if dataframe == 'binary':
            df = self.binary_df
        elif dataframe == 'spatial':
            df = self.spatial_df
        else:
            raise ValueError('Invalid dataframe selection. Please choose "binary" or "spatial".')

        # Validate that required columns exist
        if 'NWB_ID' not in df.columns:
            raise ValueError('The selected DataFrame must contain an "NWB_ID" column.')

        valid_column_choices = df.columns.tolist()  # Get all available columns

        # Handle visualization based on different parameters
        if based_on == 'speed':
            if 'Speed_infra_fct__' not in self.city_df.columns:
                raise ValueError('"Speed_infra_fct__" column is missing in city_df.')
            df = self.city_df.groupby('NWB_ID')['Speed_infra_fct__'].first().reset_index()
            colormapping = ListedColormap(['yellow', 'grey', 'blue', 'red'])

        elif based_on == 'buildings':
            required_columns = ['Offices_150', 'Commercial_150', 'Railway_150', 'Educ_150']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f'Missing required columns for "buildings": {missing_cols}')

            df['buildings'] = df[required_columns].astype(str).agg('-'.join, axis=1)
            unique_combinations = df['buildings'].unique()
            colormapping = ListedColormap(colors[:len(unique_combinations)])
            df = df[['NWB_ID', 'buildings']]

        elif based_on in valid_column_choices:
            if based_on not in df.columns:
                raise ValueError(f'The column "{based_on}" is not found in the selected DataFrame.')

            df = df[['NWB_ID', based_on]]
            unique_values = len(df[df.columns[1]].unique())
            colormapping = 'viridis' if unique_values > 2 else ListedColormap(['lightblue', 'red'])

        else:
            raise ValueError(f'Invalid "based_on" selection. Choose from: {valid_column_choices}')

        # Call the visualization function
        self._network_visualizer(
            name=self.city_name,
            gdf=self.geometric_dataframe,
            dataframe=df,
            colormap=colormapping,
            title=set_title,
            legend=show_legend
        )
