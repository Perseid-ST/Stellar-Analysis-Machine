# This is SAM (Stellar Analysis Machine) #
Color-Magnitude Diagrams (CMD) are graphs of stellar brightness vs. color, which translate to graphs of luminosity vs. temperature. Isochrone model curves can be fitted to a CMD to determine the cluster’s distance, age, number of heavy elements (metallicity), and amount of light being blocked by dust (extinction). There are two parts to fitting an isochrone to a cluster of stars. The first is to compare characteristics of the stars (proper motion and distance) to determine which stars are members of the cluster, and the second is to properly fit the isochrone line to the distribution or remaining stars. The process is usually manual, which can be error-prone and time-consuming. Using a Python process limits human error and accelerates the process. The results are output faster than it would be if it were done by hand; however, the results are neither accurate nor precise. This process was tested using a sampling of clusters that had already been measured by Cluster Pro Plus. The results show that this process is not yet accurate.
## Setup ##
### Environment Used: ###
The main Python packages used were Asteca (https://github.com/asteca/ASteCA) and Astroquery (https://github.com/astropy/astroquery). There is an "Enviroment.txt" file with all the Python packages used.

### Running ###
This takes a minute to run
#### Linux ####

1) Create an environment (I used venv) containing the libraries listed in "Enviroment.txt" 
    - This will only have to be done once
2) Open Terminal and enter your environment
3) Change your directory to the project folder

#### For single cluster analysis ####
   
4) Type "python isofitting.py" or "python3 isofitting.py"
5) Enter the name of the cluster when prompted

#### For multiple cluster analysis ####
   
6) Have the list of clusters in a .csv file
7) Type "python process_cluster.py" or "python3 process_cluster.py"
8) Enter the name of the .csv file when prompted
9) Use "Graphing.py" to compare data

## Acknowledgements ##
This work has made use of data from the European Space Agency (ESA) mission
Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis
Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC
has been provided by national institutions, in particular the institutions participating in
the Gaia Multilateral Agreement.​

This research has made use of the SIMBAD database, operated at CDS, Strasbourg, France​
