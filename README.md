## This is my Physics Capstone ##
This project was inspired by Skynet UNC and it's Astromancer Program. Astromancer's Isochrone Fitting is done manually. This project is an attempt to limit user input as much as possible.
As of March 18th, 2025, all that is required for the user to do after everything is set up, is give the program a name of a cluster. The Caldwell names and common names do not work for this. What I mean by common names is that the program won't accept Pleiades but will accept M45.
This code does not seem to be accurate.
# Enviroment Used: #
The main python packages used were Asteca,(https://github.com/asteca/ASteCA) Astroquery (https://github.com/astropy/astroquery), and Ezpadova-2 (https://github.com/asteca/ezpadova-2). The version of Asteca that I used what 0.5.8. While there is a version 0.5.9 that holds the isochrones and the code for pulling those isochronse, a bug was stopping me from being able to update Asteca. If you experience this bug you will have to use Ezpadova-2 to import the isochrones. Or you can use the isochrones that are part of this project. You will also have to update plot.py (virtualenvs/asteca/lib/python3.11/site-packages/asteca/plot.py) file to contain the following from version 0.5.9:
'''python
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
'''
