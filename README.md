# This is SAM (Stellar Analysis Machine) #
This project was inspired by Skynet UNC and it's Astromancer Program. Astromancer's Isochrone Fitting is done manually. This project is an attempt to limit user input as much as possible.
As of March 18th, 2025, all that is required for the user to do after everything is set up, is give the program a name of a cluster. The Caldwell names and common names do not work for this. What I mean by common names is that the program won't accept Pleiades but will accept M45. This creates a synthetic cluster based off the data from Gaia GDR3 and fits an isochrone to the synthetic cluster. There are stops along the process for you to save any of the figures and it prints out the estimation of Total Extinction (Av), Distance Modulus (dm), Log(age) (loga), and Metallicity (met).
This code does not seem to be accurate.
## Setup ##
 Make sure the isofitting.py file is in the same folder as the folder that contains the isochrones being used.
### Enviroment Used: ###
The main python packages used were Asteca (https://github.com/asteca/ASteCA), Astroquery (https://github.com/astropy/astroquery), and Ezpadova-2 (https://github.com/asteca/ezpadova-2). There is a Enviroment.txt file with all the python packages used. The version of Asteca that I used was 0.5.8. While there is a version 0.5.9 that holds the isochrones and the code for pulling those isochrones, a bug was stopping me from being able to update Asteca. If you experience this bug, you will have to use Ezpadova-2 to import the isochrones. Or you can use the isochrones that are part of this project. You will also have to update the plot.py (virtualenvs/asteca/lib/python3.11/site-packages/asteca/plot.py) file to contain the following code from version 0.5.9:

```python
def get_isochrone(
    synth: Synthetic,
    fit_params: dict,
    color_idx: int = 0,
) -> np.ndarray:
    """Generate an isochrone for plotting.

    The isochrone is generated using the fundamental parameter values
    given in the ``fit_params`` dictionary.

    :param synth: :py:class:`Synthetic <asteca.synthetic.Synthetic>` object with the
        data required to generate synthetic clusters
    :type synth: Synthetic
    :param fit_params: Dictionary with the values for the fundamental parameters
        that were **not** included in the ``fix_params`` dictionary when the
        :py:class:`Synthetic` object was calibrated (:py:meth:`calibrate` method).
    :type fit_params: dict
    :param color_idx: Index of the color to plot. If ``0`` (default), plot the
        first color. If ``1`` plot the second color. Defaults to ``0``
    :type color_idx: int

    :raises ValueError: If either parameter (met, age) is outside of allowed range

    :return: Array with the isochrone data to plot
    :rtype: np.ndarray
    """
    # Generate displaced isochrone
    fit_params_copy = dict(fit_params)

    # Check isochrones ranges
    for par in ("met", "loga"):
        try:
            pmin, pmax = min(synth.met_age_dict[par]), max(synth.met_age_dict[par])
            if fit_params_copy[par] < pmin or fit_params_copy[par] > pmax:
                raise ValueError(f"Parameter '{par}' out of range: [{pmin} - {pmax}]")
        except KeyError:
            pass

    # Generate physical synthetic cluster to extract the max mass
    isochrone_full = synth.generate(fit_params_copy, full_arr_flag=True)
    # Extract max mass
    max_mass = isochrone_full[synth.m_ini_idx].max()

    # Generate displaced isochrone
    fit_params_copy["DR"] = 0.0
    isochrone = synth.generate(fit_params_copy, plot_flag=True)

    # Apply max mass filter to isochrone
    msk = isochrone[synth.m_ini_idx] < max_mass

    # Generate proper array for plotting
    isochrone = np.array([isochrone[0], isochrone[color_idx + 1]])[:, msk]

    return isochrone
```
### Running ###
This takes a minute to run
#### Linux ####

1) Create an Enviroment (I used venv) containing the libraries listed in Enviroment.txt 
    - This will only have to be done once
2) Open Terminal and enter your enviroment
    - "source virtualenvs/asteca/bin/activate"
3) Change your directory to the project folder
4) Type "python isofitting.py" or "python3 isofitting.py"
5) Enter the name of the cluster when prompted
