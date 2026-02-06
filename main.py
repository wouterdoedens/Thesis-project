import time
from models import Trainer, GCNModel, MLPModel
import torch
from data_processor import DataLoader, DataProcessor
from torch_geometric.explain import Explainer, CaptumExplainer
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from visualisations import SubplotCreator

# List of dataset names
string_names = [
    "Amsterdam_Data_crash_binary", "Utrecht_Data_crash_binary", "Rotterdam_Data_crash_binary",
    "TheHague_Data_crash_binary",
    "Amsterdam_Data_crash_injury", "Utrecht_Data_crash_injury", "Rotterdam_Data_crash_injury",
    "TheHague_Data_crash_injury",
    "Amsterdam_Data_crash_fatal", "Utrecht_Data_crash_fatal", "Rotterdam_Data_crash_fatal", "TheHague_Data_crash_fatal"
]

# Load the pre-trained trainer dictionary
# the_dict = torch.load('trainer_dictionary2.pth')

# Subset only binary crash datasets


# Set the device (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# Dictionary to hold GNNExplainers


# Create the object

tester = SubplotCreator()

# Vasious input lists

cities = ['Amsterdam', 'Amsterdam']
cities2 = ['Utrecht', 'Utrecht']
cities3 = ['Rotterdam', 'Rotterdam']
cities4 = ['The Hague', 'The Hague']
titles = ['test1', 'test2']
models = ['GCN', 'GCN']
models2 = ['MLP', 'MLP']
losses = ['BCE', 'CB']
labels = losses
network_types = ['Original', 'Predictions', 'Explainer']
based_on_types = ['speed', 'prediction', 'ground_truth', 'counts', 'neighbours']
features_list = ['Day', 'Hour', 'Daytime', 'Night_time', 'Day_night_fct_',
                 'Crash_freq', 'Crash_fatal', 'Crash_injury', 'Crash_binary',
                 'length_km', 'Bic_exp', 'MV_exp', 'Bic_exp_log', 'MV_exp_log',
                 'Betweenness_norm', 'Speed_50_separated', 'Speed_30_separated',
                 'Speed_50_on_road', 'Speed_30_on_road', 'Speed_infra_fct__',
                 'Grade_separated', 'Traf_light_dens_scaled', 'Roundabout_dens_scaled',
                 'Unsignalised_dens_scaled', 'Offices_150', 'Commercial_150',
                 'Railway_150', 'Educ_150', 'UTR_mun', 'AMS_mun', 'ROT_mun', 'TH_mun',
                 'mun_fct_']
loss_list = ['CB'] * 4
gcn_list = ['GCN'] * 4
cities5 = ['Amsterdam', 'Utrecht', 'Rotterdam', 'The Hague']
explainer_list = ['Explainer'] * 4
counts_list = ['counts'] * 4
subplot_network_types = ['Predictions', 'Predictions']
subplot_network_types2 = ['Original', 'Predictions']
subplot_network_types3 = ['Original', 'Explainer']
subplot_based_on_types1 = ['Bic_exp', 'Bic_exp']
subplot_based_on_types2 = ['MV_exp', 'MV_exp']
subplot_based_on_types3 = ['Betweenness_norm', 'Betweenness_norm']
subplot_based_on_types = ['prediction', 'ground_truth']
subplot_based_on_types5 = ['Betweenness_norm', 'counts']
titles = cities5  # ['Prediction', 'Ground truth']
# tester.network_subplot_creator(cities, subplot_network_types, gcn_list, loss_list, subplot_based_on_types, titles)
titles2 = ['Centrality Feature', 'Road Return Frequency']
titles3 = ['Bic_exp Feature', 'Bic_exp Feature Importance']
titles4 = ['MV_exp Feature', 'MV_exp Feature Importance']
titles5 = ['Centrality Feature', 'Centrality Feature Importance']
titles6 = ['Predictions', 'Ground Truth']

## Used to create the network subplots. Remove savename so the data is not saved


"""
# Example
# save_name='predictioncomparison'
tester.network_subplot_creator(cities, subplot_network_types, gcn_list, loss_list, subplot_based_on_types, titles6, normalize=False, city_name='Amsterdam')


tester.network_subplot_creator(cities5, explainer_list, gcn_list, loss_list, counts_list, titles, normalize=False, same_legends=True, legend_title='Road Return Frequency', city_name='Allcities', save_name='counts')
tester.network_subplot_creator(cities, subplot_network_types3, gcn_list, loss_list, subplot_based_on_types5, titles2, normalize=False, city_name='Amsterdam', save_name='CountComparison')
tester.network_subplot_creator(cities, subplot_network_types3, gcn_list, loss_list, subplot_based_on_types1, titles3, city_name='Amsterdam', save_name='Bic_exp')
tester.network_subplot_creator(cities, subplot_network_types3, gcn_list, loss_list, subplot_based_on_types2, titles4, city_name='Amsterdam', save_name='MV_exp')
tester.network_subplot_creator(cities, subplot_network_types3, gcn_list, loss_list, subplot_based_on_types3, titles5, city_name='Amsterdam', save_name='Centrality')
 """

## Creates Histogram subplots.
# Set save to true to save
"""
tester.create_histogram_subplots(cities=cities, models=models, losses=losses, labels=labels, save=False)
tester.create_histogram_subplots(cities=cities, models=models2, losses=losses, labels=labels, save=False)
tester.create_histogram_subplots(cities=cities2, models=models, losses=losses, labels=labels, save=False)
tester.create_histogram_subplots(cities=cities2, models=models2, losses=losses, labels=labels, save=False)
tester.create_histogram_subplots(cities=cities3, models=models, losses=losses, labels=labels, save=False)
tester.create_histogram_subplots(cities=cities3, models=models2, losses=losses, labels=labels, save=False)
tester.create_histogram_subplots(cities=cities4, models=models, losses=losses, labels=labels, save=False)
tester.create_histogram_subplots(cities=cities4, models=models2, losses=losses, labels=labels, save=False)
"""

## Some more list initializationn

cities5 = ['Amsterdam', 'Utrecht', 'Rotterdam', 'The Hague']
r_types = ['histogram', 'histogram', 'histogram', 'histogram']
r_types2 = ['boxplot', 'boxplot', 'boxplot', 'boxplot']

losses = ['bce', 'bce', 'bce', 'bce']
losses2 = ['cb', 'cb', 'cb', 'cb']
losses3 = ['bce', 'cb']

titles = ['15 most important features', '15 most important features', '15 most important features',
          '15 most important features']
titles2 = ['GCN with BCE Loss', 'GCN with CB loss']

## creates explainer feature importance subplots, both the histograms and boxplots
"""
tester.explainer_features_subplots(cities5, r_types, losses, titles=titles, set_angles=True)
tester.explainer_features_subplots(cities5, r_types, losses2, titles=titles, set_angles=True)

tester.explainer_features_subplots(cities, r_types2, losses3, titles=titles2, set_angles=True)
tester.explainer_features_subplots(cities2, r_types2, losses3, titles=titles2, set_angles=True)
tester.explainer_features_subplots(cities3, r_types2, losses3, titles=titles2, set_angles=True)
tester.explainer_features_subplots(cities4, r_types2, losses3, titles=titles2, set_angles=True)

"""

## Some more list initialization
"""

binaries = ["ORIGINAL", "Speed_30_on_road = 1", "Speed_50_separated = 1", "COMMERCIAL_150 = 1", "EDUC_150 = 1"]
labels = ["original", "on30=1", "sep50=1", "com150=1", "edu150=1"]
BIC_list = ["BIC_EXP"] * 5
MV_list = ["MV_EXP"] * 5
BETWEENNESS_list = ["BETWEENNESS_NORM"] * 5
TRAF_LIGHT_list = ["TRAF_LIGHT_DENS_SCALED"] * 5
UNSIGNALISED_list = ["UNSIGNALISED_DENS_SCALED"] * 5
GCN_list = ["GCN"] * 5
MLP_list = ["MLP"] * 5
cb_list = ["cb"] * 5
bce_list = ["bce"] * 5
amsterdam_list = ['Amsterdam'] * 5
utrecht_list = ['Utrecht'] * 5
rotterdam_list = ['Rotterdam'] * 5
thehague_list = ['The Hague'] * 5
titles = ['Bic_exp for\n GCN Model with CB Loss', 'Bic_exp for\n MLP Model with BCE Loss', 'MV_exp for\n GCN Model '
                                                                                           'with CB Loss',
          'MV_exp for\n MLP Model with BCE Loss', 'Betweenness_norm for\n GCN Model with CB Loss', 'Betweenness_norm '
                                                                                                   'for\n MLP Model '
                                                                                                   'with BCE Loss',
          'Traf_light_dense_scaled for\n GCN Model with CB Loss', 'Traf_light_dense_scaled for\n MLP Model with BCE '
                                                                  'Loss', 'Unsignalised_dens_scaled for\n GCN Model '
                                                                          'with CB Loss', 'Unsignalised_dens_scaled '
                                                                                          'for\n MLP Model with BCE '
                                                                                          'Loss']
# Bicycle Exposure
bic_titles = [
    "Bic_exp for\n GCN Model with CB Loss",
    "Bic_exp for\n MLP Model with BCE Loss"
]

# Motor Vehicle Exposure
mv_titles = [
    "MV_exp for\n GCN Model with CB Loss",
    "MV_exp for\n MLP Model with BCE Loss"
]

# Betweenness Centrality
bet_titles = [
    "Betweenness_norm for\n GCN Model with CB Loss",
    "Betweenness_norm for\n MLP Model with BCE Loss"
]

# Traffic Light Density
traf_titles = [
    "Traf_light_dense_scaled for\n GCN Model with CB Loss",
    "Traf_light_dense_scaled for\n MLP Model with BCE Loss"
]

# Unsignalized Intersection Density
unsig_titles = [
    "Unsignalised_dens_scaled for\n GCN Model with CB Loss",
    "Unsignalised_dens_scaled for\n MLP Model with BCE Loss"
]
"""
"""
## Creates a big list for the partial dependence plot generation
def create_list(cities):
    bic1 = tester.create_city_dict(cities, GCN_list, cb_list, binaries, BIC_list, labels=labels)
    bic2 = tester.create_city_dict(cities, MLP_list, bce_list, binaries, BIC_list, labels=labels)
    mv1 = tester.create_city_dict(cities, GCN_list, cb_list, binaries, MV_list, labels=labels)
    mv2 = tester.create_city_dict(cities, MLP_list, bce_list, binaries, MV_list, labels=labels)
    bet1 = tester.create_city_dict(cities, GCN_list, cb_list, binaries, BETWEENNESS_list, labels=labels)
    bet2 =  tester.create_city_dict(cities, MLP_list, bce_list, binaries, BETWEENNESS_list, labels=labels)
    traf1 = tester.create_city_dict(cities, GCN_list, cb_list, binaries, TRAF_LIGHT_list, labels=labels)
    traf2 =  tester.create_city_dict(cities, MLP_list, bce_list, binaries, TRAF_LIGHT_list, labels=labels)
    unsig1 = tester.create_city_dict(cities, GCN_list, cb_list, binaries, UNSIGNALISED_list, labels=labels)
    unsig2 =  tester.create_city_dict(cities, MLP_list, bce_list, binaries, UNSIGNALISED_list, labels=labels)
    the_list = [bic1, bic2, mv1, mv2, bet1, bet2, traf1, traf2, unsig1, unsig2]

    # Categorizing different groups
    bic_list = [bic1, bic2]  # Bicycle-related
    mv_list = [mv1, mv2]  # Motor vehicle-related
    bet_list = [bet1, bet2]  # Possibly betting or some metric
    traf_list = [traf1, traf2]  # Traffic-related
    unsig_list = [unsig1, unsig2]  # Unsignalized intersections or similar
    return the_list, bic_list, mv_list, bet_list, traf_list, unsig_list
"""
## Change the name of the city list and city nmae for the results of different cities
"""
city_list = thehague_list
city_name = "The Hague"
ylimits = [0, 0.6]
the_list, bic_list, mv_list, bet_list, traf_list, unsig_list  =create_list(city_list)


tester.pdp_subplot_creator(the_list, titles=titles, save_name="Total", city_name=city_name, set_ylim=ylimits)

tester.pdp_subplot_creator(bic_list, titles=bic_titles, save_name="Bic_exp", city_name=city_name, set_ylim=ylimits)
tester.pdp_subplot_creator(mv_list, titles=mv_titles, save_name="MV_exp", city_name=city_name, set_ylim=ylimits)
tester.pdp_subplot_creator(bet_list, titles=bet_titles, save_name="betweenness", city_name=city_name, set_ylim=ylimits)
tester.pdp_subplot_creator(traf_list, titles=traf_titles, save_name="traf_dense", city_name=city_name, set_ylim=ylimits)
tester.pdp_subplot_creator(unsig_list, titles=unsig_titles, save_name="unsig_dense", city_name=city_name, set_ylim=ylimits)
"""
