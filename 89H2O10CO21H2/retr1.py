from POSEIDON.core import create_star, create_planet, define_model, load_data, wl_grid_constant_R, set_priors, read_opacities
from POSEIDON.retrieval import run_retrieval
from POSEIDON.visuals import plot_data, plot_spectra_retrieved
from POSEIDON.constants import R_J, R_Sun
from POSEIDON.utility import read_retrieved_spectrum, plot_collection
from POSEIDON.corner import generate_cornerplot
import numpy as np

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


model_name = '95%CO2 5% H2O'

bulk_species = ['CO2']
param_species = ['H2O']

# Create the model object
model = define_model(model_name, bulk_species, param_species,
                     PT_profile = 'isotherm', cloud_model = 'cloud-free')

wl_min = 0.5
wl_max = 5.5

R = 10000

wl = wl_grid_constant_R(wl_min, wl_max, R)

data_dir = './data/' + planet_name

datasets_new = ['TOI-1266 c_SYNTHETIC_JWST_NIRSpec_Prism_H2O 10%CO2 1% H2_N_trans_10.dat']
#This datasetsnew is obtained from function generate synthetic file

instruments = ['JWST_NIRSpec_Prism']

data_new = load_data(data_dir, datasets_new, instruments, wl)

fig_data = plot_data(data_new, planet_name, wl_min, wl_max, wl_axis='linear',
                     data_labels = ['NIRSpec_Prism'], 
                     plt_label = 'PandExo Output')

prior_types = {}
prior_ranges = {}

prior_types['T'] = 'uniform'
prior_types['R_p_ref'] = 'uniform'
prior_types['log_X'] = 'uniform'

prior_ranges['T'] = [100, 1000]
prior_ranges['R_p_ref'] = [0.85*R_p, 1.15*R_p]
prior_ranges['log_X'] = [-12,-1] 

priors = set_priors(planet, star, model, data_new, prior_types, prior_ranges)

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

# Specify the pressure grid of the atmosphere
P_min = 1.0e-7    # 0.1 ubar
P_max = 100       # 100 bar
N_layers = 100    # 100 layers

P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify the reference pressure and radius
P_ref = 1         # 1 bar
R_p_ref = R_p         # Radius at reference pressure

run_retrieval(planet, star, model, opac, data_new, priors, wl, P, P_ref, 
             R=R, spectrum_type='transmission', sampling_algorithm=
              'MultiNest', N_live = 400, verbose=True, resume = False)

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

fig_corner = generate_cornerplot(planet, model)