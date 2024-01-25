# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python [conda env:poseidon]
#     language: python
#     name: conda-env-poseidon-py
# ---

# %%
from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_J, R_Sun

R_s = 0.42*R_Sun
T_s = 3600
log_g_s = 4.85
Met_s = -0.5

star = create_star(R_s, T_s, log_g_s, Met_s)

planet_name = 'TOI-1266 c'
R_p = 0.139*R_J
T_eq = 345.87
g_p = 8.87769

planet = create_planet(planet_name, R_p, gravity=g_p, T_eq=T_eq)

# %%
from POSEIDON.core import define_model
import numpy as np

model_name = '95%CO2 5% H2O'

bulk_species = ['CO2']
param_species = ['H2O']

# Create the model object
model = define_model(model_name, bulk_species, param_species,
                     PT_profile = 'isotherm', cloud_model = 'cloud-free')

# %%
from POSEIDON.core import load_data, wl_grid_constant_R
from POSEIDON.visuals import plot_data

wl_min = 0.5
wl_max = 5.5

R = 10000

wl = wl_grid_constant_R(wl_min, wl_max, R)

data_dir = './data/' + planet_name

datasets_new = ['TOI-1266 c_SYNTHETIC_JWST_NIRSpec_Prism_95%CO2 5% H2O_N_trans_10.dat']
#This datasetsnew is obtained from function generate synthetic file

instruments = ['JWST_NIRSpec_Prism']

data_new = load_data(data_dir, datasets_new, instruments, wl)

fig_data = plot_data(data_new, planet_name, wl_min, wl_max, wl_axis='linear',
                     data_labels = ['NIRSpec_Prism'], 
                     plt_label = 'PandExo Output')


# %%
from POSEIDON.core import set_priors

prior_types = {}
prior_ranges = {}

prior_types['T'] = 'uniform'
prior_types['R_p_ref'] = 'uniform'
prior_types['log_X'] = 'uniform'

prior_ranges['T'] = [100, 1000]
prior_ranges['R_p_ref'] = [0.85*R_p, 1.15*R_p]
prior_ranges['log_X'] = [-12,-1] 

priors = set_priors(planet, star, model, data_new, prior_types, prior_ranges)

# %%
from POSEIDON.core import wl_grid_constant_R, read_opacities

#***Reading Opacities***#

opacity_treatment = 'opacity_sampling'

# Define fine temperature grid (K)
T_fine_min = 100     # Same as prior range for T
T_fine_max = 1000    # Same as prior range for T
T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM

T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

# Define fine pressure grid (log10(P/bar))
log_P_fine_min = -6   # 1 ubar is the lowest pressure in the opacity database
log_P_fine_max = 2    # 100 bar is the highest pressure in the opacity database
log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step),
                       log_P_fine_step)

# Pre-interpolate the opacities (note: model wavelength range was initialised in cell above)
opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

# %%
# Specify the pressure grid of the atmosphere
P_min = 1.0e-7    # 0.1 ubar
P_max = 100       # 100 bar
N_layers = 100    # 100 layers

P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify the reference pressure and radius
P_ref = 1         # 1 bar
R_p_ref = R_p         # Radius at reference pressure

# %%
from POSEIDON.retrieval import run_retrieval

run_retrieval(planet, star, model, opac, data_new, priors, wl, P, P_ref, 
             R=R, spectrum_type='transmission', sampling_algorithm=
              'MultiNest', N_live = 400, verbose=True, resume = False)

# %%
#pwd

# %%
from POSEIDON.utility import read_retrieved_spectrum, plot_collection
from POSEIDON.visuals import plot_spectra_retrieved
from POSEIDON.corner import generate_cornerplot

wl, spec_low2, spec_low1, spec_median, spec_high1, spec_high2 = read_retrieved_spectrum(planet_name, model_name)

spectra_median = plot_collection(spec_median, wl, collection = [])
spectra_low1 = plot_collection(spec_low1, wl, collection = [])
spectra_low2 = plot_collection(spec_low2, wl, collection = [])
spectra_high1 = plot_collection(spec_high1, wl, collection = [])
spectra_high2 = plot_collection(spec_high2, wl, collection = [])

fig_spec = plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1,
                                  spectra_high1, spectra_high2, planet_name,
                                  data_new, R_to_bin = 100,
                                  data_labels = ['NIRSpec Prism'],
                                  figure_shape = 'wide', wl_axis = 'linear',
                                  plt_label = 'Simulated JWST Retrieval')

fig_corner = generate_cornerplot(planet, model)#, true_vals = [R_p/R_J, PT_params[0],
                                                #             log_X_params[0], log_X_params[1]])


# %%
#***** Define new model *****#
#####MODEL-2#########
model_name_2 = '95%CO2 5%H2'

bulk_species_2 = ['CO2']
param_species_2 = ['H2']  

# Create the model object
model_2 = define_model(model_name_2, bulk_species_2, param_species_2,
                       PT_profile = 'isotherm', cloud_model = 'cloud-free')

#####MODEL-3#########
model_name_3 = '95%H2O 5%H2'

bulk_species_3 = ['H2O']
param_species_3 = ['H2']  

# Create the model object
model_3 = define_model(model_name_3, bulk_species_3, param_species_3,
                       PT_profile = 'isotherm', cloud_model = 'cloud-free')

#####MODEL-4#########
model_name_4 = '89%H2O 10%CO2 1%H2'

bulk_species_4 = ['H2O']
param_species_4 = ['CO2','H2'] 

# Create the model object
model_4 = define_model(model_name_4, bulk_species_4, param_species_4,
                       PT_profile = 'isotherm', cloud_model = 'cloud-free')

#####MODEL-5#########
model_name_5 = '96.5%CO2 3.5%N2'

bulk_species_5 = ['CO2']
param_species_5 = ['N2']  

# Create the model object
model_5 = define_model(model_name_5, bulk_species_5, param_species_5,
                       PT_profile = 'isotherm', cloud_model = 'cloud-free')

#***** Read opacity data *****#

# Pre-interpolate the opacities
opac_2 = read_opacities(model_2, wl, opacity_treatment, T_fine, log_P_fine)
opac_3 = read_opacities(model_3, wl, opacity_treatment, T_fine, log_P_fine)
opac_4 = read_opacities(model_4, wl, opacity_treatment, T_fine, log_P_fine)
opac_5 = read_opacities(model_5, wl, opacity_treatment, T_fine, log_P_fine)

# %%
# import os
# os.chdir("/home/swaroop/nb/TOI-1266 c/New/")

# %%
#pwd

# %%
#***** Set priors for new retrieval *****#

# Initialise prior type dictionary
prior_types_2 = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types_2['T'] = 'uniform'
prior_types_2['R_p_ref'] = 'uniform'
prior_types_2['log_X'] = 'uniform'    # 'log_X' sets the same prior for all mixing ratios

# Initialise prior range dictionary
prior_ranges_2 = {}

# Specify prior ranges for each free parameter
prior_ranges_2['T'] = [100, 1000]
prior_ranges_2['R_p_ref'] = [0.85*R_p, 1.15*R_p]
prior_ranges_2['log_X'] = [-12, -1]   # 'log_X' sets the same prior for all mixing ratios

# Create prior object for retrieval
priors_2 = set_priors(planet, star, model_2, data_new, prior_types_2, prior_ranges_2)

# %%
#***** Run atmospheric retrieval *****#

run_retrieval(planet, star, model_2, opac_2, data_new, priors_2, wl, P, P_ref, 
             R=R, spectrum_type='transmission', sampling_algorithm=
              'MultiNest', N_live = 400, verbose=True, resume = False)

# %%
#***** Set priors for new retrieval *****#

# Initialise prior type dictionary
prior_types_3 = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types_3['T'] = 'uniform'
prior_types_3['R_p_ref'] = 'uniform'
prior_types_3['log_X'] = 'uniform'    # 'log_X' sets the same prior for all mixing ratios

# Initialise prior range dictionary
prior_ranges_3 = {}

# Specify prior ranges for each free parameter
prior_ranges_3['T'] = [100, 1000]
prior_ranges_3['R_p_ref'] = [0.85*R_p, 1.15*R_p]
prior_ranges_3['log_X'] = [-12, -1]   # 'log_X' sets the same prior for all mixing ratios

# Create prior object for retrieval
priors_3 = set_priors(planet, star, model_3, data_new, prior_types_3, prior_ranges_3)

# %%
run_retrieval(planet, star, model_3, opac_3, data_new, priors_3, wl, P, P_ref, R = R,
              spectrum_type = 'transmission', sampling_algorithm = 'MultiNest',
              N_live = 400, verbose = True)

# %%
#***** Set priors for new retrieval *****#

# Initialise prior type dictionary
prior_types_4 = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types_4['T'] = 'uniform'
prior_types_4['R_p_ref'] = 'uniform'
prior_types_4['log_X'] = 'uniform'    # 'log_X' sets the same prior for all mixing ratios

# Initialise prior range dictionary
prior_ranges_4 = {}

# Specify prior ranges for each free parameter
prior_ranges_4['T'] = [100, 1000]
prior_ranges_4['R_p_ref'] = [0.85*R_p, 1.15*R_p]
prior_ranges_4['log_X'] = [-12, -1]   # 'log_X' sets the same prior for all mixing ratios

# Create prior object for retrieval
priors_4 = set_priors(planet, star, model_4, data_new, prior_types_4, prior_ranges_4)

# %%
run_retrieval(planet, star, model_4, opac_4, data_new, priors_4, wl, P, P_ref, R = R,
              spectrum_type = 'transmission', sampling_algorithm = 'MultiNest',
              N_live = 400, verbose = True)

# %%
#***** Set priors for new retrieval *****#

# Initialise prior type dictionary
prior_types_5 = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types_5['T'] = 'uniform'
prior_types_5['R_p_ref'] = 'uniform'
prior_types_5['log_X'] = 'uniform'    # 'log_X' sets the same prior for all mixing ratios

# Initialise prior range dictionary
prior_ranges_5 = {}

# Specify prior ranges for each free parameter
prior_ranges_5['T'] = [100, 1000]
prior_ranges_5['R_p_ref'] = [0.85*R_p, 1.15*R_p]
prior_ranges_5['log_X'] = [-12, -1]   # 'log_X' sets the same prior for all mixing ratios

# Create prior object for retrieval
priors_5 = set_priors(planet, star, model_5, data_new, prior_types_5, prior_ranges_5)

# %%
run_retrieval(planet, star, model_5, opac_5, data_new, priors_5, wl, P, P_ref, R = R,
              spectrum_type = 'transmission', sampling_algorithm = 'MultiNest',
              N_live = 400, verbose = True)

# %%
from POSEIDON.retrieval import Bayesian_model_comparison

model_ref = model
model_steamh2 = model_3 
model_co2h2 = model_2
model_venus = model_5

Bayesian_model_comparison(planet_name, model_ref, model_steamh2)

# %%
Bayesian_model_comparison(planet_name, model_co2h2, model_ref)

# %%
Bayesian_model_comparison(planet_name, model_ref, model_venus)

# %%
Bayesian_model_comparison(planet_name, model_3, model_4)

# %%
# from POSEIDON.utility import read_retrieved_spectrum, plot_collection
# from POSEIDON.visuals import plot_spectra_retrieved
# from POSEIDON.corner import generate_cornerplot

# wl, spec_low2, spec_low1, spec_median, spec_high1, spec_high2 = read_retrieved_spectrum(planet_name, model_name_2)

# spectra_median = plot_collection(spec_median, wl, collection = spectra_median)
# spectra_low1 = plot_collection(spec_low1, wl, collection = spectra_low1)
# spectra_low2 = plot_collection(spec_low2, wl, collection = spectra_low2)
# spectra_high1 = plot_collection(spec_high1, wl, collection = spectra_high1)
# spectra_high2 = plot_collection(spec_high2, wl, collection = spectra_high2)

# fig_spec = plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1,
#                                   spectra_high1, spectra_high2, planet_name,
#                                   data_new, R_to_bin = 100, spectra_labels=['Reference', 'CO2-H2'],
#                                   data_labels = ['NIRSpec Prism'],
#                                   figure_shape = 'wide', wl_axis = 'linear',
#                                   plt_label = 'Simulated JWST Retrieval')

# fig_corner = generate_cornerplot(planet, model)

# %%
