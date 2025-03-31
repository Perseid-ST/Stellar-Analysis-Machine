import pandas as pd
from astroquery.simbad import Simbad
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import numpy as np

cluster_name = input("Enter the name of the stellar cluster: ")

result_table = Simbad.query_object(cluster_name)

right_ascension = result_table['ra'][0]
declination = result_table['dec'][0]

print("Right Ascension:",right_ascension, "Declination:", declination)