
import pandas as pd
from astroquery.simbad import Simbad
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import tempfile
import pyabc

import asteca

#End of Import Statements-------------------------------------------------

def model(fit_params):
    """Generate a synthetic cluster."""
    # Call generate method with the fit_params dictionary
    synth_clust = synthcl.generate(fit_params)

    # pyABC expects a dictionary from this function, so we return a
    # dictionary with a single element.
    return {"data": synth_clust}

def distance(synth_dict, _):
    """The likelihood returned works as a distance which means that the optimal value is 0.0.
    """
    return likelihood.get(synth_dict["data"])

#--------------------------------------------------------------------------------------------
# Plot the CMD for the observed and synthetic clusters
# Function to generate a CMD plot

def cmd_plot(color, mag, label, ax=None):
    """Function to generate a CMD plot"""
    if ax is None:
        ax = plt.subplot(111)
    label = label + f", N={len(mag)}"
    ax.scatter(color, mag, alpha=0.25, label=label)
    ax.legend()
    ax.set_ylim(mag.max() + 1, mag.min() - 1)  # Invert y axis

# Set up Simbad to use the default name resolver
# Ensure a cluster name is provided as a command-line argument
if len(sys.argv) < 2:
    print("Error: No cluster name provided.")
    sys.exit(1)


def main():
    # Retrieve the cluster name from the command-line arguments
    cluster_name = sys.argv[1]

    #cluster_name = "NGC 6441"

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
    df['e_Gmag'] = np.hypot(-2.5/np.log(10)*df['phot_g_mean_flux_error']/df['phot_g_mean_flux'], sigmaG_0)

    df['e_BP_RP'] = np.sqrt((-2.5/np.log(10)*df['phot_bp_mean_flux_error']/df['phot_bp_mean_flux'])**2 + (-2.5/np.log(10)*df['phot_rp_mean_flux_error']/df['phot_rp_mean_flux'])**2 + 2*0.00289092**2)

    df = df.dropna(subset=['e_Gmag', 'e_BP_RP']) #will hopefully remove rows with NaN values in these columns

    file = df.to_csv(cluster_name + ".csv", index=False)

    # Load the data
    file_name = cluster_name + ".csv" #file must be located inside of this directory

    df = pd.read_csv(file_name)

    isochs = asteca.Isochrones(
        model='parsec',
        isochs_path="docs/_static/parsec/",
        magnitude="Gmag",
        color=("G_BPmag", "G_RPmag"),
        magnitude_effl=5822.39,
        color_effl=(5035.75, 7619.96),
        verbose=2
    )

    synthcl = asteca.Synthetic(isochs, seed=457304, verbose=2)

    my_cluster = asteca.Cluster(
        ra = df["ra"],
        dec = df["dec"],
        magnitude=df["phot_g_mean_mag"],
        e_mag=df["e_Gmag"],
        color=df["bp_rp"],
        e_color=df["e_BP_RP"],
        pmra=df["pmra"],
        e_pmra=df["pmra_error"],
        pmde=df["pmdec"],
        e_pmde=df["pmdec_error"],
        plx=df["parallax"],
        e_plx=df["parallax_error"],
        verbose=2
    )

    #using FastMP
    # Estimate the cluster's center coordinates, use the default algorithm
    my_cluster.get_center()

    # Estimate the number of cluster members, use the default algorithm
    my_cluster.get_nmembers()

    # Define a `membership` object
    memb = asteca.Membership(my_cluster, verbose=2)

    # Run `fastmp` method
    probs_fastmp = memb.fastmp()

    #plot FastMP results... Not required for the final product
    plt.title(f"fastMP Membership Probabilities for {cluster_name}")
    plt.scatter(df["bp_rp"], df["phot_g_mean_mag"], c='grey', alpha=.25)
    msk = probs_fastmp > 0.75
    plt.scatter(df["bp_rp"][msk], df["phot_g_mean_mag"][msk], c=probs_fastmp[msk], ec='k', lw=.5)
    plt.gca().invert_yaxis()
    plt.xlim(0, 2.5)
    plt.colorbar()

    plt.xlabel("G_BP - G_RP")
    plt.ylabel("Gmag")
    # Define the folder name
    output_folder = "FastMP_75"

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the plot in the specified folder
    output_path = os.path.join(output_folder, f"fastmp_{cluster_name}.png")
    plt.savefig(output_path, dpi=300)

    print(f"Figure saved to {output_path}")

    filtered_df = df[msk]

    # Create a new cluster object with the filtered data
    filtered_cluster = asteca.Cluster(
        ra=filtered_df["ra"],
        dec=filtered_df["dec"],
        magnitude=filtered_df["phot_g_mean_mag"],
        e_mag=filtered_df["e_Gmag"],
        color=filtered_df["bp_rp"],
        e_color=filtered_df["e_BP_RP"],
        pmra=filtered_df["pmra"],
        e_pmra=filtered_df["pmra_error"],
        pmde=filtered_df["pmdec"],
        e_pmde=filtered_df["pmdec_error"],
        plx=filtered_df["parallax"],
        e_plx=filtered_df["parallax_error"],
        verbose=2
    )

    # Calibrate the `synthcl` object
    synthcl.calibrate(filtered_cluster)

    # Instantiate the likelihood
    likelihood = asteca.Likelihood(filtered_cluster)

    met_min, met_max = 0.01, 0.02 #originally 0.01 to 0.02
    loga_min, loga_max = 7.0, 10.1 # originally 7.0 to 9.5
    dm_min, dm_max = 7.0, 17.0 # originally 7.0 to 10.5
    Av_min, Av_max = 0.0, 2.0 # originally 0.0 to 2.0

    # Define a pyABC Distribution(). Uniform distributions are employed for all the parameters here but the user can of course change this as desired. See the pyABC docs for more information.
    priors = pyabc.Distribution(
        {
            "met": pyabc.RV("uniform", met_min, met_max - met_min),
            "loga": pyabc.RV("uniform", loga_min, loga_max - loga_min),
            "dm": pyabc.RV("uniform", dm_min, dm_max - dm_min),
            "Av": pyabc.RV("uniform", Av_min, Av_max - Av_min)
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

    # Define a temporary file required by pyABC
    db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "pyABC.db")
    abc.new(db_path)

    history = abc.run(minimum_epsilon=0.01, max_nr_populations=20)

    final_dist = pyabc.inference_util.eps_from_hist(history)
    print("Final minimized distance: {:.2f} ({:.0f}%)".format(final_dist, 100*final_dist))

    # Extract last iteration and weights
    df, w = history.get_distribution()

    ESS = pyabc.weighted_statistics.effective_sample_size(w)
    print("Effective sample size: {:.0f}".format(ESS))

    print()
    print("Parameters estimation:")
    print("----------------------")
    parameter_stats = []
    fit_params = {}
    for k in df.keys():
        # Extract medians for the fitted parameters
        _median = pyabc.weighted_statistics.weighted_median(df[k].values, w)
        fit_params[k] = _median
        # Extract STDDEV for the fitted parameters
        _std = pyabc.weighted_statistics.weighted_std(df[k].values, w)
        print("{:<5}: {:.3f}".format(k, _median))
        print("{:<5}_error: {:.3f}".format(k, _std))
        #print("{:<5}: {:.3f} +/- {:.3f}".format(k, _median, _std))
        parameter_stats.append(f"{k}: {round(_median, 3)} +/- {round(_std, 3)}")

    pyabc.settings.set_figure_params("pyabc")  # for beautified plots


    #---------------------------------------------------------------------------------------------
    # Credible intervals over time
    plt.figure()
    pyabc.visualization.plot_credible_intervals(history)

    # Define the folder name
    output_folder = "credible_intervals_75"

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the plot in the specified folder
    output_path = os.path.join(output_folder, f"credible_intervals_{cluster_name}.png")
    plt.savefig(output_path, dpi=300)

    print(f"Figure saved to {output_path}")

    # Generate the "best fit" synthetic cluster using these parameters
    synth_arr = synthcl.generate(fit_params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Observed cluster
    cmd_plot(filtered_cluster.color, filtered_cluster.mag, "Observed stars", ax1)

    # Synthetic cluster
    # Boolean mask identifying the binary systems
    binary_msk = ~np.isnan(synth_arr[-1])
    # Extract magnitude and color
    mag, color = synth_arr[0], synth_arr[1]
    # Plot single systems
    cmd_plot(color[~binary_msk], mag[~binary_msk], "Single systems", ax2)
    ## Plot binary systems
    cmd_plot(color[binary_msk], mag[binary_msk], "Binary systems", ax2)

    # Get isochrone associated to the synthetic cluster
    isoch_arr = synthcl.get_isochrone(fit_params)
    # Plot the isochrone
    plt.plot(isoch_arr[1], isoch_arr[0], c="k")
    #plt.savefig(f"cmd_plot_{cluster_name}.png", dpi=300)

    #-------------------------------------------------------------------------------------------
    # Plot the filtered cluster and isochrone
    plt.figure(figsize=(8, 6))

    # Plot the filtered cluster
    plt.scatter(
        filtered_cluster.color,
        filtered_cluster.magnitude,
        c='blue',
        alpha=0.5,
        label="Filtered Cluster"
    )

    # Overlay the synthetic cluster as triangles
    plt.scatter(
        synth_arr[1],  # Synthetic cluster color
        synth_arr[0],  # Synthetic cluster magnitude
        c='green',
        alpha=0.7,
        marker='^',  # Triangle marker
        label="Synthetic Cluster"
    )

    # Plot the isochrone
    plt.plot(
        isoch_arr[1],  # Isochrone color
        isoch_arr[0],  # Isochrone magnitude
        c="red",
        label="Isochrone"
    )

    # Add priors as text annotations on the plot
    x_text = 0.05  # X-coordinate for the text (relative to the plot)
    y_text = 0.95  # Y-coordinate for the text (relative to the plot)
    for i, stat in enumerate(parameter_stats):
        plt.text(
            x_text, y_text - i * 0.05,  # Adjust Y-coordinate for each line
            stat,
            transform=plt.gca().transAxes,  # Use relative coordinates
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)  # Optional: Add a background box
        )

    # Invert the y-axis (as is standard for CMDs)
    plt.gca().invert_yaxis()

    # Add labels, legend, and title
    plt.xlabel("G_BP - G_RP")
    plt.ylabel("Gmag")
    plt.title(f"CMD with Isochrone for {cluster_name}")
    plt.legend()

    output_folder = "figures_75"

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the plot in the specified folder
    output_path = os.path.join(output_folder, f"cmd_with_isochrone_{cluster_name}.png")
    plt.savefig(output_path, dpi=300)

    print(f"Figure saved to {output_path}")

    os.remove(file_name) #deletes the csv file after the code is done running

if __name__ == '__main__':
    main()
