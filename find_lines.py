def _consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def find_lines_threshold(spectrum, noise_factor=1):
    """
    Find the emission and absorption lines in a spectrum. The method
    here is based on deviations larger than the spectrum's uncertainty by the
    ``noise_factor``.

    This method only works with continuum-subtracted spectra and the uncertainty
    must be defined on the spectrum. To add the uncertainty, one could use
    `~specutils.manipulation.noise_region_uncertainty` to add the uncertainty.

    Parameters
    ----------
    spectrum : `~specutils.Spectrum1D`
        The spectrum object in which the lines will be found.

    noise_factor : float
       ``noise_factor`` multiplied by the spectrum's``uncertainty``, used for
        thresholding.

    Returns
    -------
    qtable: `~astropy.table.QTable`
        Table of emission and absorption lines. Line center (``line_center``),
        line type (``line_type``) and index of line center (``line_center_index``)
        are stored for each line.
    """

    # Threshold based on noise estimate and factor.
    uncertainty = spectrum.uncertainty
    inds = np.where(np.abs(spectrum.flux) > (noise_factor*uncertainty.array)*spectrum.flux.unit)[0]
    pos_inds = inds[spectrum.flux.value[inds] > 0]
    line_inds_grouped = _consecutive(pos_inds, stepsize=1)
    
    if len(line_inds_grouped[0]) > 0:
        emission_inds = [inds[np.argmax(spectrum.flux.value[inds])] for inds in line_inds_grouped]
    else:
        emission_inds = []

    #
    # Find the absorption lines
    #

    neg_inds = inds[spectrum.flux.value[inds] < 0]
    line_inds_grouped = _consecutive(neg_inds, stepsize=1)

    if len(line_inds_grouped[0]) > 0:
        absorption_inds = [inds[np.argmin(spectrum.flux.value[inds])] for inds in line_inds_grouped]
    else:
        absorption_inds = []

    #
    # Create the QTable to return the lines
    #

    qtable = QTable()
    qtable['line_center'] = list(itertools.chain(*[spectrum.spectral_axis.value[emission_inds],
                                             spectrum.spectral_axis.value[absorption_inds]]))*spectrum.spectral_axis.unit
    qtable['line_type'] = ['emission']*len(emission_inds) + ['absorption']*len(absorption_inds)
    qtable['line_center_index'] = list(itertools.chain(*[emission_inds, absorption_inds]))

    return qtable

