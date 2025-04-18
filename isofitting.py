
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




#asteca.plot.synthetic(synthc1, ax, fit_params, isoch_arr)
#plt.title("Synthetic Cluster with Isochrone")
#plt.show()

os.remove(file_name) #deletes the csv file after the code is done running
