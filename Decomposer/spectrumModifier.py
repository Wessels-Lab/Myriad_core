# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:09:36 2021

@author: Gad.Armony
"""

from bdal.paser.IonFinder.utils import get_mass_index
import numpy as np
import pyopenms as oms


class SpectrumModifier(object):
    """
        This class modified the a spectrum to remove oxonium ions.
        It looks only for 1+ and assumes the monoisotopic peak is the most intense.

        __init__ Parameters
        ----------
        ions : tuple
            The ion masses ([M+H]1+) to remove
        mass_error : float, optional
            The mass tolerance for finding oxonium ions. Half the width of the window to look around the masses.
            The default is 0.02.
        isotope_mass_error : float, optional
            The mass tolerance for oxonium ions isotope peaks. Half the width of the window to look around the masses.
            when looking for consecutive isotopes.
            The default is 0.02.
        mass_error_unit : str, optional
            The units for mass_error and isotope_mass_error. Either 'Da' or 'ppm'
            The default is 'Da'.
        number_of_isotopes : int, optional
            How many isotopes to look for, inclusing the ion mass.
            The default is 5.
        min_int : float, optional
            The minimal  intensity for a peak to be considered an ion hit.
            The default is 0.
        min_int_thresholds tuple, optional
            The individual thresholds (minimal intensity) for a peak to be considered an ion for removal.
        int_type: str, optional
            The type of min_int either 'relative' or 'absolute'.
            The default is 'relative

        """

    def __init__(self, ions: tuple,
                 mass_error: float = 0.02,
                 isotope_mass_error: float = 0.02,
                 mass_error_unit: str = 'Da',
                 number_of_isotopes: int = 3,
                 min_int: float = None,
                 min_int_thresholds: tuple = None,
                 int_type: str = 'relative',
                 deconvolution_params=None):
        """
        Refer to the class documentation
        """

        self.ions = ions
        self.mass_error = mass_error
        self.mass_error_unit = mass_error_unit
        assert mass_error_unit in ['Da', 'ppm'], f"mass_error_unit must be either 'Da' or 'ppm', not {mass_error_unit}"
        self.isotope_mass_error = isotope_mass_error
        self.number_of_isotopes = number_of_isotopes
        if min_int is None and min_int_thresholds is None:
            min_int = 0
        assert (min_int is None) ^ (min_int_thresholds is None), \
            f"Can use only one. Either min_int ({min_int}) or min_int_thresholds ({min_int_thresholds})"
        self.min_int = min_int
        if min_int_thresholds is not None:
            assert (len(min_int_thresholds) == len(ions)), (f"min_int_thresholds must have same length as ions."
                                                            f"ions is length {len(ions)} and min_int_thresholds "
                                                            f"is length {len(min_int_thresholds)}")
            assert int_type == 'relative', "With min_int_thresholds, intensity type must be relative"
            assert int_type in ['relative', 'absolute'], f"int_type must be either 'relative' or 'absolute', not {int_type}"
            self.min_int_thresholds = {k: v for k, v in zip(ions, min_int_thresholds)}
        else:
            self.min_int_thresholds = min_int_thresholds
        self.int_type = int_type

        if deconvolution_params is None:
            deconvolution_params = {'fragment_tolerance': 20,
                                    'fragment_unit_ppm': True,
                                    'number_of_final_peaks': 0,
                                    'min_charge': 1,
                                    'keep_only_deisotoped': False,
                                    'min_isopeaks': 3,
                                    'max_isopeaks': 10,
                                    'make_single_charged': True,
                                    'annotate_charge': False,
                                    'annotate_iso_peak_count': False,
                                    'add_up_intensity': True}

        self.deconvolution_params = deconvolution_params

    def remove_mz(self, spectrum: np.ndarray,
                  mz_to_remove: float):
        """
        Removes an mz and its isotopes from a spectrum

        Parameters
        ----------
        spectrum : numpy array of shape (#,2)
            spec[:,0] are the m/z
            spec[:,1] are the intensities
        mz_to_remove : float
            the m/z value to remove.

        Returns
        -------
         numpy array of shape (#,2)
            spectrum with removed peaks.

        """

        # make sure that the spectrum has 2 columns
        assert spectrum.shape[1] == 2, f'spec has {spectrum.shape[1]} columns, 2 expected.'

        peaks_to_remove = self.get_mass_and_isotopes_index(spectrum, mz_to_remove)
        if np.isnan(peaks_to_remove).all():
            return spectrum
        else:
            return np.delete(spectrum, peaks_to_remove, axis=0)

    def get_mass_and_isotopes_index(self, spectrum: np.ndarray,
                                    mono_mass: float):
        """
        Find the index of a monoisotopic mass and it's isotopes.
        A peak is considered the next isotope if it has +1 m/z (+/-mass_error) and
            its intensity is lower than the previous isotope peak.

        Parameters
        ----------
        spectrum : numpy array of shape (#,2)
            spec[:,0] are the m/z
            spec[:,1] are the intensities
        mono_mass : float
            the m/z of the monoisotopic mass to look for.

        Returns
        -------
        mass_and_isotopes_index : list
            a list of indexes (rows) in spectrum of the monoisotopic mass and its isotopes.
            returns np.nan if the monoisotopic mass was not found

        """

        # make sure that the spectrum has 2 columns
        assert spectrum.shape[1] == 2, f'spec has {spectrum.shape[1]} columns, 2 expected.'

        mass_index = get_mass_index(spectrum, mono_mass, self.mass_error, self.mass_error_unit)
        if np.isnan(mass_index):
            return np.nan
        found_mass = spectrum[mass_index, 0]

        # check intensity of the monoisotopic peak
        if self.min_int is not None:
            if self.int_type == 'absolute' and (spectrum[mass_index, 1] < self.min_int):
                return np.nan
            if self.int_type == 'relative' and (spectrum[mass_index, 1]/spectrum[:, 1].max() < self.min_int):
                return np.nan
        elif self.min_int_thresholds is not None:
            if spectrum[mass_index, 1]/spectrum[:, 1].max() < self.min_int_thresholds[mono_mass]:
                return np.nan

        # setup output
        mass_and_isotopes_index = [mass_index]

        # look for isotope peaks untill the intensity is not decending or
        # the end of the spectrum is reached
        isotope_offset = 1
        while ((mono_mass + isotope_offset <= spectrum[spectrum.shape[0]-1, 0]) and
               (isotope_offset <= self.number_of_isotopes)):
            isotope_index = get_mass_index(spectrum, found_mass + isotope_offset, self.isotope_mass_error, self.mass_error_unit)
            # If an isotope peak was not found, stop looking further
            if np.isnan(isotope_index):
                break
            # If an isotope peak is higher in intensity don't include it and stop looking further
            elif spectrum[isotope_index, 1] >= spectrum[mass_and_isotopes_index[len(mass_and_isotopes_index)-1], 1]:
                break
            else:
                mass_and_isotopes_index.append(isotope_index)
            isotope_offset += 1
        return mass_and_isotopes_index

    def remove_ions(self, spectrum: np.ndarray,
                    ions: list = None,
                    return_removed_ions: bool = False):
        """
        Removes oxonium ions and their isotope peaks from the spectrum

        Parameters
        ----------
        spectrum : numpy array of shape (#,2)
            spec[:,0] are the m/z
            spec[:,1] are the intensities
        ions : list-like, optional
            An iterable, e.g. a list containing the m/z values to remove
            The default is self.ions.values()
        return_removed_ions : bool, optional
            If True, returns also an array of the monoisotopic masses (from <ions>) that were removed.
            The default is False.

        Returns
        -------
        numpy array of shape (#,2)
            spectrum with removed peaks.
        optional: also a 1D numpy array
            an array of monoisotopic masses that were removed.

        """

        # make sure that the spectrum has 2 columns
        assert spectrum.shape[1] == 2, f'spec has {spectrum.shape[1]} columns, 2 expected.'

        if ions is None:
            ions = self.ions

        removed_ions = []
        for ion_mass in ions:
            mod_spec = self.remove_mz(spectrum, ion_mass)
            if mod_spec.shape[0] < spectrum.shape[0]:
                removed_ions.append(ion_mass)
            elif mod_spec.shape[0] == spectrum.shape[0]:
                pass
            else:
                raise RuntimeError(f'Something went wrong. Removal of ion: {ion_mass} increased the number of peaks')
            spectrum = mod_spec
        if return_removed_ions:
            return spectrum, np.array(removed_ions)
        else:
            return spectrum

    def remove_peaks_over_mass(self, spectrum, cutoff_mass):
        peaks_to_keep = spectrum[:, 0] < cutoff_mass
        return spectrum[peaks_to_keep, :]

    def set_diff_upto_mass(self, base_spectrum, remove_peaks, cutoff_mass):
        idx = (~np.isin(base_spectrum[:, 0], remove_peaks[:, 0]) |
               (base_spectrum[:, 0] >= cutoff_mass))
        return base_spectrum[idx]

    def deiso_spec(self, spectrum, max_charge):
        spec = oms.MSSpectrum()
        deiso = oms.Deisotoper()
        spec.set_peaks((spectrum[:, 0], spectrum[:, 1]))
        deiso.deisotopeWithAveragineModel(spec, max_charge=max_charge, **self.deconvolution_params)
        return np.array(spec.get_peaks()).T