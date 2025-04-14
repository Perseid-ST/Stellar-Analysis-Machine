
import pandas as pd
from astroquery.simbad import Simbad
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import numpy as np
import sys

# Ensure a cluster name is provided as a command-line argument
if len(sys.argv) < 2:
    print("Error: No cluster name provided.")
    sys.exit(1)

# Retrieve the cluster name from the command-line arguments
cluster_name = sys.argv[1]

#cluster_name = input("Enter the name of the stellar cluster: ")
#removed for processing clusters from csv
result_table = Simbad.query_object(cluster_name)

Gaia.ROW_LIMIT = 10000  # Ensure the default row limit. This has to be high because of fastmp
#most objects will end up filtering out anyway
right_ascension = result_table['ra'][0]
declination = result_table['dec'][0]

print("Right Ascension:",right_ascension, "Declination:", declination)

coord = SkyCoord(ra=right_ascension, dec=declination, unit=(u.degree, u.degree), frame='icrs')

#get the data from Gaia
j = Gaia.cone_search_async(coord, radius=u.Quantity(1.0, u.deg)) #radius of 1 degree
r = j.get_results()

df = r.to_pandas()

sigmaG_0 = 0.0027553202
sigmaGBP_0 = 0.0027901700
sigmaGRP_0 = 0.007793818

#adding error columns
df['e_Gmag'] = np.sqrt((-2.5/np.log(10)*df['phot_g_mean_flux_error']/df['phot_g_mean_flux'])**2 + sigmaG_0**2)

df['e_BP_RP'] = np.sqrt((-2.5/np.log(10)*df['phot_bp_mean_flux_error']/df['phot_bp_mean_flux'])**2 + (-2.5/np.log(10)*df['phot_rp_mean_flux_error']/df['phot_rp_mean_flux'])**2 + 2*0.00289092**2)

df = df.dropna(subset=['e_Gmag', 'e_BP_RP']) #will hopefully remove rows with NaN values in these columns

file = df.to_csv(cluster_name + ".csv", index=False)

# Load the data
file_name = cluster_name + ".csv" #file must be located inside of this directory

df = pd.read_csv(file_name)

import asteca

#define columns of the dataframe
my_cluster = asteca.cluster(
    obs_df = df,
    ra = "ra",
    dec = "dec",
    magnitude="phot_g_mean_mag",
    e_mag="e_Gmag",
    color="bp_rp",
    e_color="e_BP_RP",
    pmra="pmra",
    e_pmra="pmra_error",
    pmde="pmdec",
    e_pmde="pmdec_error",
    plx="parallax",
    e_plx="parallax_error",
)
import os 
os.environ["XDG_SESSION_TYPE"] = "xcb"
import matplotlib.pyplot as plt

#plots the cluster on a color-magnitude diagram
ax =  plt.subplot()
asteca.plot.cluster(my_cluster,ax)
plt.title("Color-Magnitude Diagram")
#plt.show()
#will have to exit first for code to continue running

my_cluster.get_center()

#plots the cluster in ra and dec with the cluster center marked with a red x
ax = plt.subplot(221)
asteca.plot.radec(my_cluster, ax)
plt.scatter(my_cluster.radec_c[0], my_cluster.radec_c[1], marker='x', s=25, c='r')
plt.xlabel("ra")
plt.ylabel("dec")
plt.title("RA and Dec")

#plots the cluster in pmra and pmde with the cluster center marked with a red x
ax = plt.subplot(222)
plt.scatter(my_cluster.pmra_v, my_cluster.pmde_v, c='k', alpha=.15, s=5)
plt.scatter(my_cluster.pms_c[0], my_cluster.pms_c[1], marker='x', s=25, c='r')
plt.xlabel("pmra")
plt.ylabel("pmde")
plt.title("Proper Motions")

#plots the cluster in paralax with the cluster center marked with a red dashed line
ax = plt.subplot(223)
plt.hist(my_cluster.plx_v, 30)
plt.axvline(my_cluster.plx_c, c='r', ls=':')
plt.xlabel("plx")
plt.title("Parallax")
#plt.show()

# Estimate the cluster's center coordinates
my_cluster.get_center()

# Add a radius attribute, required for the ``bayesian`` method
my_cluster.radius = 0.05

# Estimate the number of cluster members
my_cluster.get_nmembers()

# Define a ``membership`` object
memb = asteca.membership(my_cluster)

# Run ``fastmp`` method
probs_fastmp = memb.fastmp()

# Run ``bayesian`` method
#probs_bayes = memb.bayesian()

# fastMP membership
plt.subplot(121)
plt.title("fastMP")
plt.scatter(df['bp_rp'], df['phot_g_mean_mag'], c='grey', alpha=.25)
msk = probs_fastmp > 0.8
plt.scatter(df['bp_rp'][msk], df['phot_g_mean_mag'][msk], c=probs_fastmp[msk], ec='k', lw=.5, vmin=0.5, vmax=1)
plt.gca().invert_yaxis()
plt.xlim(0, 2.5)
plt.colorbar()
plt.title("fastMP")

# Bayesian memberhsip # will not work for some reason??
# log of clusters bayesian membership is not working:
#m15--> it was zero but now its too big *shrug*

#plt.subplot(122)
#plt.title("Bayesian")
#plt.scatter(df['bp_rp'], df['phot_g_mean_mag'], c='grey', alpha=.25)
#msk = probs_bayes > 0.9
#plt.scatter(df['bp_rp'][msk], df['phot_g_mean_mag'][msk], c=probs_bayes[msk], ec='k', lw=.5, vmin=0.5, vmax=1)
#plt.gca().invert_yaxis()
#plt.xlim(0, 2.5)
#plt.colorbar()
#plt.title("Bayesian")

#plt.show()



isochs = asteca.isochrones(
    model="PARSEC",
    isochs_path="isochrones/",# Isochrones need to be saved in the project file
    magnitude="Gmag",
    color=("G_BPmag", "G_RPmag"),
    magnitude_effl= 5822.39,#6390.7 #wavelength V filter in Angstroms
    z_to_FeH=0.0152,
    color_effl=(5035.75, 7619.96), #(5182.58, 7825.08) #wavelengths of the BP and RP filters in Angstroms
)

my_cluster_filtered = asteca.cluster(
    obs_df = df[msk],
    ra = "ra",
    dec = "dec",
    magnitude="phot_g_mean_mag",
    e_mag="e_Gmag",
    color="bp_rp",
    e_color="e_BP_RP",
    pmra="pmra",
    e_pmra="pmra_error",
    pmde="pmdec",
    e_pmde="pmdec_error",
    plx="parallax",
    e_plx="parallax_error",
)

# Synthetic cluster parameters
synthc1 = asteca.synthetic( isochs, seed=457304)#457304

# Calibrate the synthetic cluster synthcl1
fix_params = {
    "alpha":0.0, #binary fraction
    "beta":1.0, #binary fraction, if you want only singular stars both must be set to 0.0
    "Rv":3.1, 
    "DR":0.0,
    #"dm": 11.0,
    #"Av": .4285,#1328
    #"met": 0.3,
    #"loga": 7
    }
#alpha is binary fraction
#beta is binary fraction 
#Rv is selective extinction 3.1
#DR is differential reddening
synthc1.calibrate(my_cluster_filtered,fix_params)

#isoch_arr = asteca.plot.get_isochrone(synthc1, fix_params)

#ax = plt.subplot()
#asteca.plot.cluster(my_cluster_filtered, ax)
#asteca.plot.synthetic(synthc1, ax, fix_params, isoch_arr)
#plt.title(f"Synthetic Cluster with Isochrone: {cluster_name}")
#plt.gca().invert_yaxis()
#plt.show()

# Instantiat the likelihood
likelihood = asteca.likelihood(my_cluster)

def model(fit_params):
    """Generate synthetic cluster. pyABC expects a dictionary from this function, so we return a dictionary with a single element
    """
    #print(f"fit_params: {fit_params}")  # Debugging
    synth_clust = synthc1.generate(fit_params)#loga is determined before generation
    #print(f"synthetic cluster: {synth_clust}")  # Debugging
    synth_dict = {"data":synth_clust}
    #print(f"synthetic cluster: {synth_dict}")  # Debugging
    return synth_dict

def distance(syth_dict, _):
    """The likelihood returned works as a distance whihc means that the optimal value is 0.0
    """
    return likelihood.get(syth_dict["data"])

import pyabc

#print(synthc1.met_age_dict)
met_min, met_max = [0.0005, 0.0305] #metallicity
loga_min, loga_max = [7.0, 10.1] #because of m15 which is almost as old as the universe, this gets upped to 10.1 from 9.5

# Define a pyABC Distribution(). Uniform distributions are employed for all the parameters
# here but the user can of course change this as desired. See the pyABC docs for more
# information.
priors = pyabc.Distribution(
    {
        "met": pyabc.RV("uniform", met_min, met_max - met_min),
        "loga": pyabc.RV("uniform", loga_min, loga_max - loga_min),
        "dm": pyabc.RV("uniform", 7, 12 - 7), #Distance Modulus
        "Av": pyabc.RV("uniform", 0.0, 1) #Total Extinction
    }
)

# Define pyABC parameters
pop_size = 100
abc = pyabc.ABCSMC(
    model,
    priors,
    distance,
    population_size=pop_size
)
#print(f"ABC: {abc}")
# This is a temporary file required by pyABC
import os
import tempfile
db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "pyABC.db")
abc.new(db_path)

history = abc.run(minimum_epsilon=0.01, max_nr_populations=10)

final_dist = pyabc.inference_util.eps_from_hist(history)
print("Final minimized distance: {:.2f} ({:.0f}%)".format(final_dist, 100*final_dist))

# Extract last iteration and weights
df, w = history.get_distribution()

ESS = pyabc.weighted_statistics.effective_sample_size(w)
print("Effective sample size: {:.0f}".format(ESS))

print("\nParameters estimation:")
print("----------------------")

parameter_stats = []

for k in df.keys():
    _median = pyabc.weighted_statistics.weighted_median(df[k].values, w)
    _std = pyabc.weighted_statistics.weighted_std(df[k].values, w)
    print("{:<5}: {:.3f}".format(k, _median))
    print("{:<5}_error: {:.3f}".format(k, _std))
    parameter_stats.append(f"{k:<5}: {_median:.3f} ± {_std:.3f}") #seperated for process cluster.py
    #original:
    #print("{:<5}: {:.3f} +/- {:.3f}".format(k, _median, _std))

#print(df.head())  # Inspect the posterior distributions

#print(synthc1.met_age_dict)

pyabc.settings.set_figure_params("pyabc")  # for beautified plots

# Matrix of 1d and 2d histograms over all parameters
#pyabc.visualization.plot_histogram_matrix(history)

# Credible intervals over time
#pyabc.visualization.plot_credible_intervals(history)

# Extract medians for the fitted parameters
fit_params = {
    k: pyabc.weighted_statistics.weighted_median(df[k].values, w) for k in df.keys()
}
print(fit_params)
#fit_params['met'] = 0.0152 #how to get from mH value to 
import matplotlib.pyplot as plt
#fig, (ax1, ax2) = plt.subplots(1,2)

# Extract medians and standard deviations

isoch_arr = asteca.plot.get_isochrone(synthc1, fit_params)

ax = plt.subplot()
asteca.plot.cluster(my_cluster_filtered, ax)
asteca.plot.synthetic(synthc1, ax, fit_params, isoch_arr)

# Add priors as text annotations on the plot
x_text = 0.05  # X-coordinate for the text (relative to the plot)
y_text = 0.95  # Y-coordinate for the text (relative to the plot)
for i, stat in enumerate(parameter_stats):
    ax.text(
        x_text, y_text - i * 0.05,  # Adjust Y-coordinate for each line
        stat,
        transform=ax.transAxes,  # Use relative coordinates
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)  # Optional: Add a background box
    )

plt.title(f"Synthetic Cluster with Isochrone: {cluster_name}")
plt.gca().invert_yaxis()
#plt.show()

#asteca.plot.cluster(my_cluster_filtered, ax1)
#asteca.plot.synthetic(synthc1, ax2, fit_params, isoch_arr)


#asteca.plot.synthetic(synthc1, ax, fit_params, isoch_arr)
#plt.title("Synthetic Cluster with Isochrone")
#plt.show()

os.remove(file_name) #deletes the csv file after the code is done running
