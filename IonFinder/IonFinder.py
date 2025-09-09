"""
Created on Wed Aug  4 2021

@author: Gad.Armony
"""

from .utils import get_mass_intensity_sorted
import numpy as np
from typing import Tuple

class IonFinder(object):
    """
    This class finds ions in a spectrum

    __init__ Parameters
    ----------
    ions : tuple of floats
        The ion masses ([M+H]1+) to look for
        The default is None.
    mass_error : float, optional
        Value in Da or ppm. Half the width of the window to look around the masses.
        The default is 0.02.
    mass_error_unit : str, optional
        The units for mass_error. Either 'Da' or 'ppm'
        The default is 'Da'.
    min_int : float, optional
        The minimal  intensity for a peak to be considered an ion hit.
        The default is 0.
    min_int_thresholds tuple, optional
        The individual thresholds (minimal intensity) for a peak to be considered an ion hit.
    int_type: str, optional
        The type of min_int and returned intensity either 'relative' or 'absolute'.
        The default is 'relative'

    """

    def __init__(self, ions: Tuple[float,...] = None,
                 mass_error: float = 0.02,
                 mass_error_unit: str = 'Da',
                 min_int: float = None,  # default is 0
                 min_int_thresholds: tuple = None,
                 int_type: str = 'relative'):
        """
        Refer to the class documentation
        """ 
        self.ions = ions
        self.mass_error = mass_error
        assert mass_error_unit in ['Da', 'ppm'], f"mass_error_unit must be either 'Da' or 'ppm', not {mass_error_unit}"
        self.mass_error_unit = mass_error_unit
        if min_int is None and min_int_thresholds is None:
            min_int = 0
        assert (min_int is None) or (min_int_thresholds is None), \
            f"Can use only one. Either min_int ({min_int}) or min_int_thresholds ({min_int_thresholds})"
        self.min_int = min_int
        if min_int_thresholds is not None:
            assert (len(min_int_thresholds) == len(ions)), (f"min_int_thresholds must have same length as ions."
                                                            f"ions is length {len(ions)} and min_int_thresholds "
                                                            f"is length {len(min_int_thresholds)}")
            assert int_type == 'relative', "With min_int_thresholds, intensity type must be relative"
        self.min_int_thresholds = min_int_thresholds
        assert int_type in ['relative', 'absolute'], f"int_type must be either 'relative' or 'absolute', not {int_type}"
        self.int_type = int_type

    def find_ions(self, spec: np.ndarray,
                  ions: tuple = None):
        """
        Parameters
        ----------
        spec : numpy array of shape (#,2)
            spec[:,0] is the m/z
            spec[:,1] is the intensities
        ions : tuple of floats
        The ion masses ([M+H]1+) to look for
        The default is self.ions.

        Returns
        -------
        np.array
            The intensity of the self.ions in the corresponding order.
            Type (relative vs absolute) is determined by int_type

        """
        if ions is None:
            ions = self.ions
        # make sure that the spectrum has 2 columns
        assert isinstance(spec, np.ndarray), f'spec must to be an numpy.ndarray, not {type(spec)}'
        assert spec.shape[1] == 2, f'spec has {spec.shape[1]} columns, 2 expected.'
        # make sure that ions is a iterable
        assert hasattr(ions, '__iter__'), f'ions must be iterable but it is {type(ions)}'
        assert len(ions) > 0, f'ions must not be empty'
        # initialize the output array
        ions_int = np.empty(len(ions))
        # get the intensity of the ions
        for idx, ion_mass in enumerate(ions):
            ions_int[idx] = get_mass_intensity_sorted(spec, ion_mass, self.mass_error, self.mass_error_unit)
        # filter minimum intensity
        if self.min_int_thresholds is None:
            if self.int_type == 'absolute':
                ions_int[ions_int < self.min_int] = np.nan

            elif self.int_type == 'relative':
                ions_int = ions_int/spec[:, 1].max()
                ions_int[ions_int < self.min_int] = np.nan
        else:
            ions_int = ions_int / spec[:, 1].max()
            ions_int[ions_int < self.min_int_thresholds] = np.nan

        return ions_int
