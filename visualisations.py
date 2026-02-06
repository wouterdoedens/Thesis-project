import matplotlib.pyplot as plt
import numpy as np
import math
import os
from matplotlib.colors import ListedColormap
from GNN_Explainers_class import GNNExplainers, GraphVisualisations
from matplotlib.colors import LinearSegmentedColormap
import torch
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from data_processor import DataProcessor
from sklearn.preprocessing import MinMaxScaler
from matplotlib.transforms import Affine2D
from sklearn.preprocessing import MinMaxScaler


class Visualiser:
    """
    A class for visualizing histograms of probability distributions for two-class datasets.

    :ivar figure_count: (int) Class-level counter to track the number of figures created.
    :ivar x: (np.ndarray) Feature data used for visualization.
    :ivar y: (np.ndarray) Class labels corresponding to the feature data.
    :ivar _city_name: (str) Name of the city associated with the dataset.
    """

    figure_count = 1  # Tracks the number of figures generated

    def __init__(self):
        """
        Initializes the Visualiser class with default values.
        """
        self.x = None  # Feature data
        self.y = None  # Class labels
        self._city_name = 'No city named'  # Default city name

    def set_city_name(self, name: str):
        """
        Sets the city name associated with the dataset.

        :param name: Name of the city.
        """
        self._city_name = name

    def histogram(self, figure_number: int = None, bins: int = 50, show: bool = False, stacked: bool = False,
                  save: bool = False, model_name: str = None, filename: str = None,
                  y_range: tuple = None, set_title: str = None):
        """
        Generates a histogram for a two-class dataset.

        :param figure_number: (int, optional) Figure number for the plot. If None, uses the internal counter.
        :param bins: (int) Number of bins for the histogram. Default is 50.
        :param show: (bool) If True, displays the histogram. Default is False.
        :param stacked: (bool) Not used in this function.
        :param save: (bool) If True, saves the histogram to a specified directory. Default is False.
        :param model_name: (str, optional) Name of the model used. Default is 'Test'.
        :param filename: (str, optional) Name of the file to save the histogram.
        :param y_range: (tuple, optional) Tuple specifying y-axis limits for both histograms.
        :param set_title: (str, optional) Title of the figure. If None, defaults to 'Model for City'.
        :return: Tuple containing counts for each class and bin edges.
        :rtype: tuple (np.ndarray, np.ndarray, np.ndarray)
        """
        if figure_number is None:
            figure_number = self.figure_count  # Assign a figure number if not provided

        class_list = np.unique(self.y)  # Get unique class labels
        if len(class_list) != 2:
            raise ValueError("This function is designed for exactly two classes.")

        # Assign class labels and colors
        labels = class_list.astype(str)
        colors = ['blue', 'red']

        x_range = (0, 1)  # Set the x-axis range

        # Split feature data based on class labels
        class_1_data = self.x[self.y == class_list[0]]
        class_2_data = self.x[self.y == class_list[1]]

        if model_name is None:
            model_name = 'Test'

        if set_title is None:
            set_title = model_name + ' for ' + self._city_name  # Default title if not provided

        # Create a figure with two subplots
        fig, axs = plt.subplots(2, 1, figsize=(8, 10), num=figure_number, sharex=True)

        # Histogram for class 1
        counts0, bin_edges, _ = axs[0].hist(
            class_1_data, bins=bins, range=x_range, color=colors[0], edgecolor='black', label=f'Class {class_list[0]}'
        )
        axs[0].set_title(f'Histogram for Class {int(class_list[0])}', fontsize=20, fontweight='bold')
        axs[0].set_ylabel('Frequency', fontsize=20, fontweight='bold')
        axs[0].tick_params(axis='both', labelsize=16, labelcolor='black')

        if y_range is not None:
            axs[0].set_ylim((0, y_range[0]))

        for label in axs[0].get_yticklabels():
            label.set_fontweight('bold')

        # Histogram for class 2
        counts1, bin_edges, _ = axs[1].hist(
            class_2_data, bins=bins, range=x_range, color=colors[1], edgecolor='black', label=f'Class {class_list[1]}'
        )
        axs[1].set_title(f'Histogram for Class {int(class_list[1])}', fontsize=20, fontweight='bold')
        axs[1].set_xlabel('Probability', fontsize=20, fontweight='bold')
        axs[1].set_ylabel('Frequency', fontsize=20, fontweight='bold')
        axs[1].tick_params(axis='both', labelsize=16, labelcolor='black')

        if y_range is not None:
            axs[1].set_ylim((0, y_range[1]))

        for label in axs[1].get_xticklabels():
            label.set_fontweight('bold')
        for label in axs[1].get_yticklabels():
            label.set_fontweight('bold')

        fig.suptitle(set_title, fontsize=24, fontweight='bold')

        # Adjust layout
        plt.tight_layout()

        # Increment figure counter
        self.figure_count += 1

        if save:
            parent_directory = r"C:\Users\woute\PycharmProjects\ThesisTesting\Results\Histograms"
            directory = os.path.join(parent_directory, model_name)
            os.makedirs(directory, exist_ok=True)
            filenaam = filename + "Histogram.png"

            # Full path to save the plot
            full_path = os.path.join(directory, filenaam)
            fig.savefig(full_path)

        if show:
            plt.show()  # Only display the plot if printing is True

        return counts0, counts1, bin_edges


import torch


class SubplotCreator:
    """
    A class that handles user input selection for cities, models, losses, and other variables,
    and retrieves corresponding trainer data.

    :ivar data: Dictionary loaded from 'newtrainersdictionary.pth' containing trainer information.
    :ivar dataprocessor_dictionary: Placeholder dictionary for storing data processors.
    :ivar explainers_dictionary: Placeholder dictionary for storing explainers.
    """

    def __init__(self):
        """Initializes the SubplotCreator class by loading trainer data and setting up dictionaries."""
        self.data = torch.load('newtrainersdictionary.pth')
        self.dataprocessor_dictionary = {}
        self.explainers_dictionary = {}

    def input_city(self, city):
        """
        Validates and maps user-input city names to corresponding dataset keys.

        :param city: The name of the city input by the user.
        :return: The corresponding dataset key for the selected city, or re-prompts for a valid input.
        """
        if city == "Amsterdam":
            return "Amsterdam_Data_crash_binary"
        elif city == "Utrecht":
            return "Utrecht_Data_crash_binary"
        elif city == "Rotterdam":
            return "Rotterdam_Data_crash_binary"
        elif city == "The Hague":
            return "TheHague_Data_crash_binary"
        elif city == "Break":
            return "break"
        else:
            return self.input_city(input(
                "Invalid city. Please choose from: Amsterdam, Utrecht, Rotterdam, The Hague, or break."
            ).strip().title())

    def input_model(self, model):
        """
        Validates and maps user-input model names to corresponding model keys.

        :param model: The name of the model input by the user.
        :return: The corresponding model key, or re-prompts for a valid input.
        """
        model = model.strip().upper()
        if model == "GCN":
            return "GCNModel"
        elif model == "MLP":
            return "MLPModel"
        elif model == "BREAK":
            return "break"
        else:
            return self.input_model(input("Invalid model. Please choose from: 'GCN', 'MLP' or 'break'")
                                    .strip().upper())

    def input_loss(self, loss):
        """
        Validates and maps user-input loss function names to corresponding loss keys.

        :param loss: The name of the loss function input by the user.
        :return: The corresponding loss key, or re-prompts for a valid input.
        """
        loss = loss.strip().upper()
        if loss == "BCE":
            return "BCELoss"
        elif loss == "CB":
            return "ClassBalancedLoss"
        elif loss == "BREAK":
            return "break"
        else:
            return self.input_loss(input("Invalid Loss. Please choose from: 'BCE', 'CB' or 'break'")
                                   .strip().upper())

    def input_binary(self, binary):
        """
        Validates and maps user-input binary condition names to corresponding dataset keys.

        :param binary: The binary condition input by the user.
        :return: The corresponding key for the dataset, or re-prompts for a valid input.
        """
        binary = binary.strip().upper()
        mapping = {
            "ORIGINAL": "Original",
            "SPEED_50_SEPARATED = 1": "Speed_50_separated = 1",
            "SPEED_30_SEPARATED = 1": "Speed_30_separated = 1",
            "SPEED_50_ON_ROAD = 1": "Speed_50_on_road = 1",
            "SPEED_30_ON_ROAD = 1": "Speed_30_on_road = 1",
            "GRADE_SEPARATED = 1": "Grade_separated = 1",
            "OFFICES_150 = 1": "Offices_150 = 1",
            "COMMERCIAL_150 = 1": "Commercial_150 = 1",
            "RAILWAY_150 = 1": "Railway_150 = 1",
            "EDUC_150 = 1": "Educ_150 = 1",
            "BREAK": "break"
        }
        if binary in mapping:
            return mapping[binary]
        else:
            return self.input_binary(input(
                "Invalid binary key. Please choose from: 'Original', 'Speed_50_separated = 1', "
                "'Speed_30_separated = 1', 'Speed_50_on_road = 1', 'Speed_30_on_road = 1', 'Grade_separated = 1', "
                "'Offices_150 = 1', 'Commercial_150 = 1', 'Railway_150 = 1', 'Educ_150 = 1', or 'break': "
            ).strip().upper())

    def input_variables(self, variable):
        """
        Validates and maps user-input variable names to corresponding dataset keys.

        :param variable: The variable key input by the user.
        :return: The corresponding dataset key, or re-prompts for a valid input.
        """
        variable = variable.strip().upper()
        mapping = {
            "BIC_EXP": "Bic_exp",
            "MV_EXP": "MV_exp",
            "BETWEENNESS_NORM": "Betweenness_norm",
            "TRAF_LIGHT_DENS_SCALED": "Traf_light_dens_scaled",
            "ROUNDABOUT_DENS_SCALED": "Roundabout_dens_scaled",
            "UNSIGNALISED_DENS_SCALED": "Unsignalised_dens_scaled",
            "BREAK": "break"
        }
        if variable in mapping:
            return mapping[variable]
        else:
            return self.input_variables(input(
                "Invalid variable key. Please choose from: 'Bic_exp', 'MV_exp', 'Betweenness_norm', "
                "'Traf_light_dens_scaled', 'Roundabout_dens_scaled', 'Unsignalised_dens_scaled', or 'break': "
            ).strip().upper())

    def label_name(self, label=None):
        """
        Returns a label name, prompting the user if no label is provided.

        :param label: The label name input by the user.
        :return: The label name.
        """
        if label is None:
            label = input("Please, input the label name: ")
        return label

    def get_trainer(self, city=None, model=None, loss=None):
        """
        Retrieves a trainer based on user inputs for city, model, and loss.
        If 'break' is encountered, the process restarts from the beginning.

        :param city: The city dataset name.
        :param model: The model type.
        :param loss: The loss function type.
        :return: A tuple containing the selected trainer object, city, model, and loss.
        """
        while True:
            try:
                # Step 1: Get city input
                if city is None:
                    city = self.input_city(input(
                        "Enter the city (Amsterdam, Utrecht, Rotterdam, The Hague, or 'break' to restart): "
                    ).strip().title())
                else:
                    city = self.input_city(city.strip().title())
                if city == "break":
                    print("Restarting all steps...")
                    city, model, loss = None, None, None
                    continue

                # Step 2: Get model input
                if model is None:
                    model = self.input_model(input("Enter the model (GCN, MLP, or 'break' to restart): ").strip().upper())
                else:
                    model = self.input_model(model.strip().upper())
                if model == "break":
                    print("Restarting all steps...")
                    city, model, loss = None, None, None
                    continue

                # Step 3: Get loss input
                if loss is None:
                    loss = self.input_loss(input("Enter the loss (BCE, CB, or 'break' to restart): ").strip().upper())
                else:
                    loss = self.input_loss(loss.strip().upper())
                if loss == "break":
                    print("Restarting all steps...")
                    city, model, loss = None, None, None
                    continue

                # Retrieve the trainer from the data dictionary
                modelloss = model + loss
                trainer = self.data[city][modelloss]
                return trainer, city, model, loss

            except KeyError:
                print("The specified combination of inputs does not exist in the data. Restarting all steps...")
                city, model, loss = None, None, None

    def city_selector(self, city=None):
        """
        Validates and returns the standardized city name.

        :param city: The user-provided city name. If None, prompts the user.
        :return: A valid city name or 'break' if the user chooses to exit.
        """
        if city is None:
            city = input("Enter a city (Amsterdam, Utrecht, Rotterdam, The Hague, or 'break'): ").strip().title()
        else:
            city = city.strip().title()

        if city in ["Amsterdam", "Utrecht", "Rotterdam", "The Hague"]:
            return city
        elif city == "Break":
            return "break"
        else:
            return self.input_city(input("Invalid city. Please enter a valid option: ").strip().title())

    def df_selector(self, dataprocessor):
        """
        Prompts the user to select a DataFrame type.

        :param dataprocessor: An object containing 'binary_df' and 'spatial_df'.
        :return: The selected DataFrame (binary or spatial).
        """
        chosen_df = input("Select a DataFrame ('binary' or 'spatial', or 'exit' to quit): ").strip().lower()

        if chosen_df == "binary":
            return dataprocessor.binary_df
        elif chosen_df == "spatial":
            return dataprocessor.spatial_df
        else:
            return self.df_selector(input("Invalid choice. Enter 'binary', 'spatial', or 'exit': ").strip().lower())

    def plot_type_selector(self, plot_type=None):
        """
        Asks the user to select a plot type.

        :param plot_type: The preferred plot type. If None, prompts the user.
        :return: 'histogram' or 'boxplot'.
        """
        if plot_type is None:
            plot_type = input("Choose a plot type ('histogram' or 'boxplot'): ").strip().lower()

        if plot_type in ['histogram', 'boxplot']:
            return plot_type
        else:
            print("Invalid input. Please enter 'histogram' or 'boxplot'.")
            return self.plot_type_selector()

    def get_histogram_info(self, city=None, model=None, loss=None, labelname=None):
        """
        Retrieves histogram-related data from the stored dataset.

        :param city: The city name.
        :param model: The model name.
        :param loss: The loss function.
        :param labelname: The label name for the histogram.
        :return: A dictionary with histogram data or None if retrieval fails.
        """
        attempts = 0
        while True:
            try:
                trainer, city, model, loss = self.get_trainer(city, model, loss)

                return {
                    "bin_edges": trainer.bin_edges,
                    "counts0": trainer.counts0,
                    "counts1": trainer.counts1,
                    "label": self.label_name(labelname)
                }

            except KeyError:
                print("Invalid combination of inputs. Restarting the process...")
                city, model, loss, labelname = None, None, None, None
                attempts += 1

                if attempts > 2:  # Limit retries to prevent infinite loops
                    print("Maximum attempts reached. Exiting...")
                    return None

    def next_multiple_of_5(self, n):
        """
        Rounds a given number up to the next multiple of 5.

        :param n: The number to round up.
        :return: The nearest multiple of 5.
        """
        return math.ceil(n / 5) * 5

    def create_histogram_subplots(self, cities=None, models=None, losses=None, labels=None, titles=None, save=None):
        """
        Creates and displays multiple histogram subplots for different cities, models, and loss functions.

        :param cities: List of city names corresponding to the data to be plotted. If None, the user is prompted for input.
        :param models: List of models corresponding to each city's data.
        :param losses: List of loss functions corresponding to each city's data.
        :param labels: List of labels for each histogram.
        :param titles: List of custom titles for the subplots. If None, default titles are generated.
        :param save: If provided, saves the figure to the specified location.
        :return: None
        """

        # Determine the number of subplots
        if cities is None:
            number_subplots = int(input("How many subplots do you want?"))
        else:
            number_subplots = len(cities)

        subplot_list = []  # Stores histogram data for each subplot

        # Ask the user if they want two columns in the layout
        num_col_bool = input("Do you wish to have 2 columns, 'Y' or 'N'").strip().upper()
        if num_col_bool == 'Y':
            num_cols = 2
            # Generates index values for subplots arranged in two columns
            values = [i + j for i in range(0, (4 * number_subplots), 4) for j in range(2)]
        else:
            num_cols = 1
            values = range(number_subplots)

        # Calculate the required number of rows for the subplot grid
        num_rows = int(np.ceil(number_subplots / num_cols))
        fig, axes = plt.subplots((2 * num_rows), num_cols, figsize=(12, 8 * num_rows))

        # Flatten the axes array if there are multiple subplots
        axes = axes.flatten() if number_subplots > 1 else [axes]

        max_0, max_1 = -1000, -1000  # Initialize max values for class 0 and class 1

        # Loop to gather histogram data for each city
        for i in range(len(cities)):
            if cities is None:
                dictionary = self.get_histogram_info()
            else:
                dictionary = self.get_histogram_info(city=cities[i], model=models[i], loss=losses[i],
                                                     labelname=labels[i])

            # Update max values for better scaling of y-axis
            max_0 = max(max_0, max(dictionary['counts0']))
            max_1 = max(max_1, max(dictionary['counts1']))

            subplot_list.append(dictionary)

        bin_edges = subplot_list[0]['bin_edges']  # Get bin edges from the first dictionary

        # Plot histograms
        for idx, i in enumerate(values[:number_subplots]):

            # Plot histogram for class 0
            axes[i].bar(bin_edges[:-1], subplot_list[idx]['counts0'], width=np.diff(bin_edges), color='blue',
                        edgecolor='black')

            # Set title
            title = titles[idx] if titles else f"{models[idx]} model with {losses[idx]} loss"
            axes[i].set_title(title, fontsize=20, fontweight='bold')
            axes[i].set_ylabel("Frequency", fontsize=16, fontweight='bold')
            axes[i].grid(True, axis='y', linestyle='--', linewidth=0.5, color='black')

            # Remove x-axis tick labels and adjust tick parameters
            axes[i].set_xticklabels([])
            axes[i].tick_params(axis='both', labelsize=14, labelcolor='black')
            axes[i].set_ylim((0, self.next_multiple_of_5(max_0) + 5))

            # Add legend for class 0
            legend = axes[i].legend(loc='upper right', title="0 class", title_fontsize=14, fontsize=14)
            legend.get_title().set_fontweight('bold')
            for label in legend.get_texts():
                label.set_fontweight('bold')

            # Plot histogram for class 1
            axes[i + 2].bar(bin_edges[:-1], subplot_list[idx]['counts1'], width=np.diff(bin_edges), color='red',
                            edgecolor='black')
            axes[i + 2].set_xlabel("Probability", fontsize=16, fontweight='bold')
            axes[i + 2].set_ylabel("Frequency", fontsize=16, fontweight='bold')
            axes[i + 2].grid(True, axis='y', linestyle='--', linewidth=0.5, color='black')

            # Add legend for class 1
            legend = axes[i + 2].legend(loc='upper right', title="1 class", title_fontsize=14, fontsize=14)
            legend.get_title().set_fontweight('bold')
            for label in legend.get_texts():
                label.set_fontweight('bold')

            # Adjust y-axis limits
            axes[i + 2].set_ylim((0, self.next_multiple_of_5(max_1) + 5))

            # Format tick labels
            axes[i + 2].tick_params(axis='both', labelsize=14, labelcolor='black')
            for label in axes[i + 2].get_xticklabels():
                label.set_fontweight('bold')
            for label in axes[i + 2].get_yticklabels():
                label.set_fontweight('bold')

            plt.tight_layout()  # Adjust layout for better spacing

        # Save the figure if 'save' is provided
        if save is not None:
            parent_directory = r"C:\Users\woute\PycharmProjects\ThesisTesting\Results\Histograms\Subplots"
            model_name = f"{cities[0]}_{models[0]}.png"
            directory = os.path.join(parent_directory, model_name)
            fig.savefig(directory)

        plt.show()  # Display the plot

    def create_city_dict(self, cities, models, losses, binaries, variables, labels):
        """
        Creates a dictionary containing city-related data for analysis.

        :param cities: List of city names.
        :param models: List of model names corresponding to each city.
        :param losses: List of loss functions used.
        :param binaries: List of binary categories.
        :param variables: List of variables used in analysis.
        :param labels: List of label names.
        :return: Dictionary mapping keys to respective lists.
        """
        return {
            "cities": cities,
            "models": models,
            "losses": losses,
            "binaries": binaries,
            "variables": variables,
            "labelnames": labels,
        }

    def get_line(self, city=None, model=None, loss=None, binary=None, variable=None, labelname=None):
        """
        Retrieves a specific line of data from a nested dictionary based on user inputs.
        Calls `get_trainer()` to fetch trainer details before extracting binary and variable data.

        :param city: The city for which data is retrieved.
        :param model: The model associated with the data.
        :param loss: The loss function used.
        :param binary: The binary category (e.g., 'Original', 'Speed_50_separated').
        :param variable: The variable key (e.g., 'Bic_exp', 'MV_exp').
        :param labelname: Custom label name for the retrieved data.
        :return: A dictionary containing the selected line of data and label.
        """
        attempts = 0  # Track the number of retry attempts
        while True:
            try:
                # Step 1-3: Retrieve trainer using `get_trainer()`
                trainer, city, model, loss = self.get_trainer(city, model, loss)

                # Step 4: Get binary input
                if binary is None:
                    binary_input = input(
                        "Enter the binary key (Original, Speed_50_separated = 1, etc., or 'break' to restart): ").strip().upper()
                    binary = self.input_binary(binary_input)
                else:
                    binary = self.input_binary(binary.strip().upper())

                # Restart process if user inputs 'break'
                if binary == "break":
                    print("Restarting all steps...")
                    continue

                # Step 5: Get variable input
                if variable is None:
                    variable_input = input(
                        "Enter the variable key (Bic_exp, MV_exp, etc., or 'break' to restart): ").strip().upper()
                    variable = self.input_variables(variable_input)
                else:
                    variable = self.input_variables(variable.strip().upper())

                # Restart process if user inputs 'break'
                if variable == "break":
                    print("Restarting all steps...")
                    continue

                # Step 6: Retrieve the final dictionary using the selected binary and variable
                result2 = trainer.pdp[binary]
                dictionary = result2[variable]

                # Assign label name to the dictionary
                dictionary["label"] = self.label_name(labelname)

                return dictionary

            except KeyError:
                # Handle errors due to missing keys in the dictionary
                print("The specified combination of inputs does not exist in the data. Restarting all steps...")
                city, model, loss, binary, variable, labelname = None, None, None, None, None, None
                attempts += 1

                # Limit retries to avoid infinite loops
                if attempts > 2:
                    print("Maximum attempts reached. Exiting...")
                    return None

    def create_pdp(self, cities=None, models=None, losses=None, binaries=None, variables=None, labelnames=None,
                   number_of_lines=None):
        """
        Creates a Partial Dependence Plot (PDP) by fetching multiple lines of data using `get_line()`.

        :param cities: List of city names for filtering data.
        :param models: List of models corresponding to each city.
        :param losses: List of loss functions used.
        :param binaries: List of binary categories.
        :param variables: List of variables for PDP plotting.
        :param labelnames: List of custom label names for the plots.
        :param number_of_lines: Number of lines to fetch if `cities` is None.
        :return: A dictionary containing PDP data for visualization.
        """
        php_dict = {}  # Dictionary to store PDP data

        if cities is None:
            # If no cities are provided, get user input for the number of lines
            for i in range(number_of_lines):
                line_key = f"line{i}"
                label_key = f"Label{i}"
                min_key = f"min{i}"
                max_key = f"max{i}"

                # Fetch PDP data using `get_line()`
                line_data = self.get_line()
                predictions = line_data['predictions']

                # Store results in the dictionary
                php_dict[line_key] = predictions
                php_dict[label_key] = line_data['label']
                php_dict[min_key] = min(predictions)
                php_dict[max_key] = max(predictions)

        else:
            # If cities are provided, loop through and fetch corresponding PDP data
            for i in range(len(cities)):
                line_key = f"line{i}"
                label_key = f"Label{i}"
                min_key = f"min{i}"
                max_key = f"max{i}"

                # Fetch PDP data for the specified parameters
                line_data = self.get_line(cities[i], models[i], losses[i], binaries[i], variables[i], labelnames[i])
                predictions = line_data['predictions']

                # Store results in the dictionary
                php_dict[line_key] = predictions
                php_dict[label_key] = line_data['label']
                php_dict[min_key] = min(predictions)
                php_dict[max_key] = max(predictions)

        return php_dict



    def pdp_subplot_creator(self, data_dictionary=None, number_of_subplots=None, titles=None, labelfontsize=12,
                            save_name=None, city_name=None, set_ylim=None):
        """
        Creates a subplot of Partial Dependence Plots (PDP) with multiple subplots.

        :param data_dictionary: A dictionary containing precomputed PDP data.
        :param number_of_subplots: Number of subplots to generate (if `data_dictionary` is None).
        :param titles: List of titles for each subplot.
        :param labelfontsize: Font size for the labels.
        :param save_name: File name to save the figure (if provided).
        :param city_name: City name to create a directory for saving the plot.
        :param set_ylim: Tuple (min, max) to set y-axis limits.
        :return: None (displays the plot and optionally saves it).
        """
        subplot_list = []
        label_list = []
        total_min = float('inf')  # Initialize with a very high value
        total_max = float('-inf')  # Initialize with a very low value

        # Case 1: No pre-existing data, create PDPs from scratch
        if data_dictionary is None:
            for i in range(number_of_subplots):
                number_lines = int(input("How many lines does this PDP have? "))  # Get user input for line count
                new_pdp = self.create_pdp(number_of_lines=number_lines)
                subplot_list.append(new_pdp)

                # Extract min/max values from predictions
                new_labels = []
                for j in range(len(new_pdp) // 4):  # Each subplot has four elements per line
                    test_min = new_pdp[f"min{j}"]
                    test_max = new_pdp[f"max{j}"]
                    total_min = min(total_min, test_min)
                    total_max = max(total_max, test_max)
                    new_labels.append(new_pdp[f"Label{j}"])

                label_list.append(new_labels)

        # Case 2: Using provided data dictionary
        else:
            number_of_subplots = len(data_dictionary)  # Define subplot count
            for i in range(number_of_subplots):
                from_dictionary = data_dictionary[i]
                new_pdp = self.create_pdp(**from_dictionary)
                subplot_list.append(new_pdp)

                new_labels = []
                for j in range(len(new_pdp) // 4):
                    test_min = new_pdp[f"min{j}"]
                    test_max = new_pdp[f"max{j}"]
                    total_min = min(total_min, test_min)
                    total_max = max(total_max, test_max)
                    new_labels.append(new_pdp[f"Label{j}"])

                label_list.append(new_labels)

        # Define line styles for the plots
        linestyles = ['solid', (0, (5, 5)), 'dashdot', (0, (1, 10)), (0, (1, 5)), (0, (1, 1))]
        num_cols = 2  # Define number of columns for subplot arrangement
        num_rows = int(np.ceil(number_of_subplots / num_cols))  # Compute number of rows
        x_vals = np.linspace(5, 95, 100)  # Define x-axis values

        # Adjust y-axis limits if provided, else ask user for input
        if set_ylim is not None:
            total_min, total_max = set_ylim
        else:
            total_min = float(input(f"The total min is now {total_min}. What should it become? "))
            total_max = float(input(f"The total max is now {total_max}. What should it become? "))

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows))
        axes = axes.flatten() if number_of_subplots > 1 else [axes]  # Flatten for consistent indexing

        # Loop through each subplot
        for k in range(len(subplot_list)):
            subplot = subplot_list[k]
            for l in range(len(subplot) // 4):
                linestyle = linestyles[l % len(linestyles)]  # Cycle through linestyles
                line_key = f"line{l}"
                label_key = f"Label{l}"
                axes[k].plot(
                    x_vals, subplot[line_key], label=subplot[label_key],
                    linestyle=linestyle, linewidth=4
                )

            # Configure axes labels and limits
            axes[k].set_xlabel('Percentile of Features', fontsize=16, fontweight='bold')
            axes[k].set_ylabel('Model Prediction Average', fontsize=16, fontweight='bold')
            axes[k].set_xlim(x_vals[0], x_vals[-1])
            axes[k].set_ylim(total_min, total_max)
            axes[k].tick_params(axis='both', labelsize=14, labelcolor='black')
            axes[k].grid(True, axis='y', linestyle='--', linewidth=1, color='black')

            # Set title for each subplot
            title = titles[k] if titles else input("Please input a title: ")
            axes[k].set_title(title, fontsize=20, fontweight='bold')

            # Make tick labels bold
            for label in axes[k].get_xticklabels() + axes[k].get_yticklabels():
                label.set_fontweight('bold')

            # Add legend
            axes[k].legend(loc='upper left', prop={'size': labelfontsize, 'weight': 'bold'})

        plt.tight_layout()
        plt.show()

        # Save the figure if a save path is provided
        if save_name is not None:
            parent_directory = r"C:\Users\woute\PycharmProjects\ThesisTesting\Results\PDP"
            directory = os.path.join(parent_directory, city_name)
            os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
            filename = f"{save_name}.png"
            full_path = os.path.join(directory, filename)
            fig.savefig(full_path)

    def explainer_features(self, city, p_type=None, loss=None):
        """
        Loads and processes GNN explainer feature importance.

        :param city: City name for data retrieval.
        :param p_type: Type of plot (e.g., histogram or boxplot).
        :param loss: Loss function type.
        :return: Dictionary containing feature importance data for visualization.
        """
        trainer, city, model, loss = self.get_trainer(city, 'GCN', loss)

        # Load explainer dictionary from file
        path_name = f"{city}{model}{loss}GNNExplainer.pth"
        print(f"Path 1: {path_name}")
        explainer_dict = torch.load(path_name)

        # Initialize explainer model with loaded dictionary
        The_Explainer = GNNExplainers(trainer, set_Dictionary=explainer_dict)
        plot_type = self.plot_type_selector(p_type)

        # Compute feature importance
        values = The_Explainer.feature_importance.sum(axis=0)
        top_k_indices = np.argsort(-values)  # Get top-k feature indices

        # If plot type is histogram
        if plot_type == 'histogram':
            return {
                "return_type": plot_type,
                'values': values,
                "top_k_indices": top_k_indices,
                "feature_names": trainer.city_data.feature_names
            }

        # If plot type is boxplot
        elif plot_type == 'boxplot':
            df = pd.DataFrame(The_Explainer.feature_importance, columns=trainer.city_data.feature_names)
            return {
                "return_type": plot_type,
                'df': df,
                "top_k_indices": top_k_indices
            }

    def explainer_features_subplots(self, cities=None, return_types=None, losses=None, number_of_subplots=None,
                                    num_cols=None, titles=None, set_angles=False):
        """
        Generates and displays subplots for feature explanations.

        :param cities: List of city names for which explanations are generated. If None, cities are selected dynamically.
        :param return_types: List of return types corresponding to each city’s explanation.
        :param losses: List of loss values corresponding to each city’s explanation.
        :param number_of_subplots: Number of subplots to generate. Required if cities are None.
        :param num_cols: Number of columns in the subplot grid. Defaults to 2 if None.
        :param titles: List of titles for each subplot. If None, user input is requested for each title.
        :param set_angles: Boolean flag to set angles for plots. Currently unused.
        """
        total_indexing = None
        same_indexing = 0
        subplot_list = []

        if cities is None:
            for i in range(number_of_subplots):
                city = self.city_selector()  # Assuming this function selects a city
                new_subplot = self.explainer_features(city)  # Fetch explanation data

                if new_subplot['return_type'] == 'histogram':
                    if i == 0:
                        same_indexing = int(
                            input("Do you want to keep the indexing of the first graph? (0 = No, 1 = Yes): "))

                    if same_indexing:
                        total_indexing = new_subplot['top_k_indices']
                        same_indexing = 0

                subplot_list.append(new_subplot)
        else:
            number_of_subplots = len(cities)
            for i in range(len(cities)):
                new_subplot = self.explainer_features(cities[i], return_types[i], losses[i])
                if new_subplot['return_type'] == 'histogram':
                    if i == 0:
                        same_indexing = int(
                            input("Do you want to keep the indexing of the first graph? (0 = No, 1 = Yes): "))
                    if same_indexing:
                        total_indexing = new_subplot['top_k_indices']
                        same_indexing = 0
                subplot_list.append(new_subplot)

        if num_cols is None:
            num_cols = 2

        num_rows = int(np.ceil(number_of_subplots / num_cols))  # Arrange in a grid with 2 columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))  # Increase width
        axes = axes.flatten() if number_of_subplots > 1 else [axes]  # Flatten if there's more than one subplot

        for j in range(len(subplot_list)):
            if subplot_list[j]['return_type'] == 'boxplot':
                ax = axes[j]
                df = subplot_list[j]['df']
                new_order = subplot_list[j]['top_k_indices']
                df = df.iloc[:, new_order]
                sns.boxplot(data=df, ax=ax)
                ax.set_xlabel('Features', fontsize=16, fontweight='bold')
                ax.set_ylabel('Feature Importance', fontsize=16, fontweight='bold')

                if titles is None:
                    ax.set_title(input("Please input a title"), fontsize=20, fontweight='bold')
                else:
                    ax.set_title(titles[j], fontsize=20, fontweight='bold')

                ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', labelsize=14, labelcolor='black')
                ax.tick_params(axis='y', labelsize=14, labelcolor='black')

                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')

                legend = ax.legend(loc='upper right', title=cities[j], title_fontsize=18, fontsize=14)
                legend.get_title().set_fontweight('bold')
                for label in legend.get_texts():
                    label.set_fontweight('bold')

            else:
                if total_indexing is None:
                    the_index = subplot_list[j]['top_k_indices']
                else:
                    the_index = total_indexing

                top_k_labels = subplot_list[j]['feature_names'][the_index]
                top_k_values = subplot_list[j]['values'][the_index]
                min_val = np.min(top_k_values)
                max_val = np.max(top_k_values)

                if max_val != min_val:
                    top_k_values_normalized = (top_k_values - min_val) / (max_val - min_val)
                else:
                    top_k_values_normalized = np.zeros_like(top_k_values)

                axes[j].bar(range(len(top_k_labels)), top_k_values_normalized, color='skyblue')
                axes[j].set_xlabel("Feature", fontsize=16, fontweight='bold')
                axes[j].set_ylabel("Feature Importance", fontsize=16, fontweight='bold')

                if titles is None:
                    axes[j].set_title(input("Please input a title"), fontsize=20, fontweight='bold')
                else:
                    axes[j].set_title(titles[j], fontsize=20, fontweight='bold')

                axes[j].set_xticks(range(len(top_k_labels)))
                axes[j].set_xticklabels(top_k_labels, rotation=45, ha="right", fontsize=14, fontweight='bold')
                axes[j].tick_params(axis='y', labelsize=14, labelcolor='black')
                axes[j].grid(True, axis='y', linestyle='--', linewidth=0.5, color='black')

                for label in axes[j].get_yticklabels():
                    label.set_fontweight('bold')

                legend = axes[j].legend(loc='upper right', title=cities[j], title_fontsize=18, fontsize=14)
                legend.get_title().set_fontweight('bold')
                for label in legend.get_texts():
                    label.set_fontweight('bold')

        plt.tight_layout()
        plt.show()

    def based_on_df_original(self, dataprocessor, based_on=None):
        """
        Selects a dataframe and applies color mapping based on the chosen feature.

        :param dataprocessor: Data processor object containing city data.
        :param based_on: Feature name to base the color mapping on. Defaults to user input if None.
        :return: Tuple (dataframe, colormap) for visualization.
        """
        chosen_df = self.df_selector(dataprocessor)
        valid_column_choices = chosen_df.columns.unique()[1:23].tolist()

        if based_on is None:
            based_on = input("Please select 'speed' or any valid variable name")

        if based_on == 'speed':
            # Select speed-based data and apply a predefined color mapping
            df = dataprocessor.city_df.groupby('NWB_ID')['Speed_infra_fct__'].first().reset_index()
            colormapping = ListedColormap(['yellow', 'grey', 'blue', 'red'])
            return df, colormapping
        elif based_on in valid_column_choices:
            # Select user-defined feature and apply a colormap based on uniqueness of values
            df = chosen_df[['NWB_ID', based_on]]
            colormapping = 'viridis_r' if len(df[df.columns[1]].unique()) > 2 else ListedColormap(['lightblue', 'red'])
            return df, colormapping
        else:
            # Retry with correct input
            return self.based_on_df_original(dataprocessor)

    def trainer_df(self, trainer, based_on=None):
        """
        Creates a dataframe based on prediction or ground truth data.

        :param trainer: Trainer object containing city data.
        :param based_on: The type of data to include ('prediction' or 'ground_truth').
        :return: Tuple (dataframe, colormap) for visualization.
        """
        df = trainer.city_data.c_df[['NWB_ID']].copy()

        if based_on is None:
            based_on = input("Please select 'prediction' or 'ground_truth'")

        if based_on == 'prediction':
            df['prediction'] = trainer.x.flatten()
            colormapping = 'viridis_r'
            return df, colormapping
        elif based_on == 'ground_truth':
            df['ground_truth'] = trainer.y.flatten()
            colormapping = ListedColormap(
                ['green', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'blue'])
            return df, colormapping
        else:
            return self.trainer_df(trainer)

    def based_on_explainer(self, trainer, based_on=None):
        """
        Determines the explanation basis for the model.

        :param trainer: Trainer object containing city data.
        :param based_on: Feature to base the explanation on ('counts', 'neighbours', or a feature name).
        :return: Selected explanation feature.
        """
        if based_on is None:
            based_on = input(
                "Please input based on. Can be 'counts', 'neighbours' or " + trainer.city_data.feature_names)

        if based_on == 'counts':
            return 'counts'
        elif based_on == 'neighbours':
            return 'neighbours'
        elif based_on in trainer.city_data.feature_names:
            return based_on
        else:
            return self.based_on_explainer(trainer)

    def type_selector(self, city, network_type=None, themodel=None, theloss=None, based_on=None):
        """
        Selects the type of network for visualization and returns the corresponding dataframe and colormap.

        :param city: City name for the data selection.
        :param network_type: Type of network to use ('Original', 'Predictions', 'Explainer', 'Break').
        :param themodel: Model name for predictions (used in 'Predictions' and 'Explainer' cases).
        :param theloss: Loss function used in training.
        :param based_on: Feature to base the visualization on.
        :return: Tuple (dataframe, geopandas dataframe, colormap) or a string for 'Break'.
        """
        if network_type is None:
            network_type = input(
                "Please select a network type: 'Original', 'Predictions', 'Explainer', or 'Break'").strip().title()
        else:
            network_type = network_type.strip().title()

        if network_type == "Original":
            # Retrieve or initialize data processor
            dataprocessor_city = self.dataprocessor_dictionary.get(city, DataProcessor(city))
            self.dataprocessor_dictionary[city] = dataprocessor_city

            gdf = dataprocessor_city.gdf
            df, colormapping = self.based_on_df_original(dataprocessor_city, based_on)
            return df, gdf, colormapping

        elif network_type == "Predictions":
            # Fetch trainer and extract prediction-based data
            trainer, city, model, loss = self.get_trainer(city, themodel, theloss)
            gdf = trainer.city_data.gdf
            df, colormapping = self.trainer_df(trainer, based_on)
            return df, gdf, colormapping

        elif network_type == "Explainer":
            # Load GNNExplainer model and visualize network
            trainer, city, model, loss = self.get_trainer(city, 'GCN', theloss)
            gdf = trainer.city_data.gdf
            path_name = f"{city}{model}{loss}GNNExplainer.pth"
            print(f"Path: {path_name}")

            if path_name in self.explainers_dictionary:
                The_Explainer = self.explainers_dictionary[path_name]
            else:
                explainer_dict = torch.load(path_name)
                The_Explainer = GNNExplainers(trainer, set_Dictionary=explainer_dict)
                self.explainers_dictionary[path_name] = The_Explainer

            basing_on = self.based_on_explainer(trainer, based_on)
            df, colormap, save, path = The_Explainer._visualize_network(based_on=basing_on)
            colormapping = 'viridis_r'
            return df, gdf, colormapping

        elif network_type == "Break":
            return "break"

        else:
            # Retry selection if input is invalid
            return self.type_selector(city, input(
                "Invalid selection. Choose 'Original', 'Predictions', 'Explainer', or 'Break': ").strip().title())

    def network_subplot_creator(self, cities, network_type, models, losses, based_ons, titles, num_cols=2,
                                normalize=True,
                                same_legends=False, legend_title=None, city_name=None, save_name=None):
        """
        Creates a subplot of network visualizations for different cities and conditions.

        :param cities: List of city names to be visualized.
        :param network_type: List specifying the type of network visualization for each city.
        :param models: List of models corresponding to each city.
        :param losses: List of loss functions used for training each model.
        :param based_ons: List of feature names used to color the networks.
        :param titles: List of subplot titles corresponding to each city.
        :param num_cols: Number of columns in the subplot grid (default is 2).
        :param normalize: Whether to normalize values before plotting (default is True).
        :param same_legends: Whether to use a single legend for all subplots (default is False).
        :param legend_title: Custom title for the legend (default is None, uses first `based_on`).
        :param city_name: Name of the city directory for saving results (default is None).
        :param save_name: Name of the file to save the figure (default is None, does not save).
        """
        merged_df_list = []
        value_columns_list = []
        colormap_list = []

        if num_cols is None:
            num_cols = 2
        number_of_subplots = len(cities)
        num_rows = int(np.ceil(number_of_subplots / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8 * num_rows), constrained_layout=True)
        axes = axes.flatten() if number_of_subplots > 1 else [axes]

        # Determine global min and max for scaling if `same_legends=True`
        global_vmin = float('inf')
        global_vmax = float('-inf')

        # Process each city and prepare data for plotting
        for i in range(number_of_subplots):
            df, gdf, colormap = self.type_selector(cities[i], network_type[i], models[i], losses[i], based_ons[i])
            merged_gdf = gdf.merge(df, on='NWB_ID')
            value_column = df.columns[1]
            if normalize:
                scaler = MinMaxScaler()
                merged_gdf[value_column] = scaler.fit_transform(merged_gdf[[value_column]])
            merged_df_list.append(merged_gdf)
            value_columns_list.append(value_column)
            colormap_list.append(colormap)

            if same_legends:
                global_vmin = min(global_vmin, df[value_column].min())
                global_vmax = max(global_vmax, df[value_column].max())

        # Generate subplots for each city
        for j, m_df in enumerate(merged_df_list):
            ax = axes[j]
            value_column = value_columns_list[j]
            colormap = colormap_list[j]

            # Determine value range for color mapping
            if same_legends:
                vmin, vmax = global_vmin, global_vmax
            else:
                vmin, vmax = m_df[value_column].min(), m_df[value_column].max()

            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            linewidths = np.interp(m_df[value_column], (m_df[value_column].min(), m_df[value_column].max()), (0.5, 2.5))

            # Plot network on the subplot
            m_df.plot(
                column=value_column,
                cmap=colormap,
                legend=False,
                ax=ax,
                linewidth=linewidths,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(titles[j], fontsize=24, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#F2F2F2')
            ax.set_aspect('auto', adjustable='box')

            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

            # Add individual colorbars for each subplot if `same_legends=False`
            if not same_legends:
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.9, fraction=0.05, pad=0.1)
                cbar.ax.tick_params(labelsize=16, width=2, length=6, direction='inout', labelcolor='black')
                for label in cbar.ax.get_xticklabels():
                    label.set_fontweight('bold')
                cbar.outline.set_edgecolor('black')
                cbar.outline.set_linewidth(1)

        # Add a global colorbar if `same_legends=True`
        if same_legends:
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
            sm._A = []
            cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', shrink=0.9, fraction=0.05, pad=0.08)
            if legend_title is None:
                legend_title = based_ons[0]
            cbar.set_label(legend_title, fontsize=20, fontweight='bold')
            cbar.ax.tick_params(labelsize=20, width=2, length=8, direction='inout', labelcolor='black')
            for label in cbar.ax.get_xticklabels():
                label.set_fontweight('bold')
            cbar.outline.set_edgecolor('black')
            cbar.outline.set_linewidth(2)

        # Save the figure if `save_name` is provided
        if save_name is not None:
            parent_directory = r"C:\\Users\\woute\\PycharmProjects\\ThesisTesting\\Results\\GNNExplainer\\Networks"
            directory = os.path.join(parent_directory, city_name)
            os.makedirs(directory, exist_ok=True)
            filename = save_name + ".pdf"
            full_path = os.path.join(directory, filename)
            fig.savefig(full_path, format="pdf", bbox_inches="tight")

        plt.show()




    
