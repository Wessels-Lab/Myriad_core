# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 06:59:01 2021

@author: Gad.Armony
"""

import numpy as np
import numba


def get_mass_index(spec: np.ndarray,
                   mass: float,
                   mass_error: float,
                   mass_error_unit: str):

    """
    Finds a mass +/- a mass error in a spectrum

    Parameters
    ----------
    spec : nupy array of shape (#,2)
        spec[:,0] is the m/z
        spec:,[1] is the intensities
    mass : float
        The mass to look for.
    mass_error : float
        The mass error for looking for mass
    mass_error_unit: str ['Da', 'ppm']
        The units of mass_error

    Returns
    -------
    float
        The index of the most intense peak within the mass range.
    np.nan
        If the mass is not found

    """
    # make sure that the spectrum has 2 columns
    assert spec.shape[1] == 2, f'spec has {spec.shape[1]} columns, 2 expected.'

    # The numeric index of massed that are within the mass range.
    # The [0] is to extract the array from the returned tuple
    if mass_error_unit == 'Da':
        index = np.nonzero((spec[:, 0] >= mass-mass_error) &
                           (spec[:, 0] <= mass+mass_error))[0]
    elif mass_error_unit == 'ppm':
        index = np.nonzero((spec[:, 0] >= mass*(1-mass_error/1e6)) &
                           (spec[:, 0] <= mass*(1+mass_error/1e6)))[0]
    else:
        raise ValueError(f'mass_error_unit must be either Da or ppm, not {mass_error_unit}')

    if len(index) > 0:
        return index[spec[index, 1].argmax()]
    else:
        return np.nan


def get_mass_intensity(spec: np.ndarray,
                       mass: float,
                       mass_error: float,
                       mass_error_unit: str):

    """
    Finds a mass +/- a mass error in a spectrum

    Parameters
    ----------
    spec : numpy array of shape (#,2)
        spec[:,0] is the m/z
        spec:,[1] is the intensities
    mass : float
        The mass to look for.
    mass_error : float
        The mass error for looking for mass
    mass_error_unit: str ['Da', 'ppm']
        The units of mass_error

    Returns
    -------
    float
        The intensity of the most intense peak within the mass range.
    np.nan
        If the mass is not found

    """
    # make sure that the spectrum has 2 columns
    assert spec.shape[1] == 2, f'spec has {spec.shape[1]} columns, 2 expected.'

    if mass_error_unit == 'Da':
        index = (spec[:, 0] >= mass-mass_error) &\
                (spec[:, 0] <= mass+mass_error)
    elif mass_error_unit == 'ppm':
        index = (spec[:, 0] >= mass*(1-mass_error/1e6)) & \
                (spec[:, 0] <= mass*(1+mass_error/1e6))
    else:
        raise ValueError(f'mass_error_unit must be either Da or ppm, not {mass_error_unit}')

    if index.any():
        return spec[index, 1].max()
    else:
        return np.nan

def get_mass_intensity_sorted(spec: np.ndarray,
                              masses: np.ndarray,
                              mass_error: float,
                              mass_error_unit: str):

    """
    Finds a mass +/- a mass error in a spectrum

    Parameters
    ----------
    spec : SORTED np.ndarray of shape (#,2)
        spec[:,0] is the m/z
        spec:,[1] is the intensities
        spec is sorted by spec[:,0], ascending
    masses : np.ndarray
        The masses to look for. Works best when it is sorted ascending.
    mass_error : float
        The mass error for looking for mass
    mass_error_unit: str ['Da' ,'ppm']
        The units of mass_error

    Returns
    -------
    np.ndarray[float]
        The intensities of the most intense peak within the mass range.
        np.nan is placed where masses were not found


    """
    # make sure that the spectrum has 2 columns
    assert spec.shape[1] == 2, f'spec has {spec.shape[1]} columns, 2 expected.'
    # handle single values either as float or as ndarray.
    if isinstance(masses, float) or isinstance(masses, int):
        return get_mass_intensity_sorted_single_mass(spec, masses, mass_error, mass_error_unit)
    if masses.ndim == 0:
        masses = np.expand_dims(masses, 0)




    if mass_error_unit == 'Da':
        mass_min_error = masses - mass_error
        mass_plus_error = masses + mass_error
    elif mass_error_unit == 'ppm':
        mass_min_error = masses * (1 - mass_error / 1e6)
        mass_plus_error = masses * (1 + mass_error / 1e6)
    else:
        raise ValueError(f'mass_error_unit must be either Da or ppm, not {mass_error_unit}')

    index_min = np.searchsorted(spec[:, 0], mass_min_error, side='left')
    index_min_min = index_min.min()
    index_max = np.searchsorted(spec[index_min_min:, 0], mass_plus_error, side='right') + index_min_min

    ## prepare output
    output = set_output(index_min, index_max, spec)
    return output

@numba.njit
def set_output(index_min, index_max, spec):
    output = np.full(index_min.shape, np.nan, dtype=spec.dtype)
    for i , (idx_min, idx_max) in enumerate(zip(index_min,index_max)):
        diff = idx_max - idx_min
        # no peaks - np.nan - no need for this since there is already np.nan there.
        # if diff == 0:
        #     output[i]=np.nan
        if diff == 1:
            output[i] = spec[index_min[i], 1]
        elif diff >= 2:
            output[i] = spec[index_min[i]:index_max[i], 1].max()
    return output

def get_mass_intensity_sorted_single_mass(spec: np.ndarray,
                              mass: float,
                              mass_error: float,
                              mass_error_unit: str):

    """
    Finds a mass +/- a mass error in a spectrum

    Parameters
    ----------
    spec : SORTED numpy array of shape (#,2)
        spec[:,0] is the m/z
        spec:,[1] is the intensities
        spec is sorted by spec[:,0], ascending
    mass : float
        The mass to look for.
    mass_error : float
        The mass error for looking for mass
    mass_error_unit: str ['Da', 'ppm']
        The units of mass_error

    Returns
    -------
    float
        The intensity of the most intense peak within the mass range.
    np.nan
        If the mass is not found

    """
    # make sure that the spectrum has 2 columns
    assert spec.shape[1] == 2, f'spec has {spec.shape[1]} columns, 2 expected.'

    if mass_error_unit == 'Da':
        index_min = np.searchsorted(spec[:, 0], mass-mass_error, side='left')
        index_max = np.searchsorted(spec[index_min:, 0], mass+mass_error, side='right')+index_min
    elif mass_error_unit == 'ppm':
        index_min = np.searchsorted(spec[:, 0], mass*(1-mass_error/1e6), side='left')
        index_max = np.searchsorted(spec[index_min:, 0], mass*(1+mass_error/1e6), side='right') + index_min
    else:
        raise ValueError(f'mass_error_unit must be either Da or ppm, not {mass_error_unit}')

    if index_max > index_min:
        if spec[index_min:index_max, 1].shape == 1:
            return spec[index_min:index_max, 1]
        else:
            return spec[index_min:index_max, 1].max()
    else:
        return np.nan


def get_mass_rel_int(spec: np.ndarray,
                     mass: float,
                     mass_error: float,
                     mass_error_unit: str):

    """
    Finds a mass +/- a mass error in a spectrum

    Parameters
    ----------
    spec : numpy array of shape (#,2)
        spec[:,0] is the m/z
        spec:,[1] is the intensities
    mass : float
        The mass to look for.
    mass_error : float
        The mass error for looking for mass
    mass_error_unit: str ['Da', 'ppm']
        The units of mass_error

    Returns
    -------
    float
        The relative intensity of the most intense peak within the mass range.
    np.nan
        If the mass is not found

    """

    # make sure that the spectrum has 2 columns
    assert spec.shape[1] == 2, f'spec has {spec.shape[1]} columns, 2 expected.'

    if mass_error_unit == 'Da':
        index = (spec[:, 0] >= mass-mass_error) &\
                (spec[:, 0] <= mass+mass_error)
    elif mass_error_unit == 'ppm':
        index = (spec[:, 0] >= mass*(1-mass_error/1e6)) & \
                (spec[:, 0] <= mass*(1+mass_error/1e6))
    else:
        raise ValueError(f'mass_error_unit must be either Da or ppm, not {mass_error_unit}')

    if index.any():
        return spec[index,1].max()/spec[:,1].max()
    else:
        return np.nan


def mass_from_mz(mz_array, charge):
    """
    Parameters
    ----------
    mz_array : float or np.array
        the mz(s) to be converted to mass(s).
    charge : same as mz_array
        the charge(s) of the mz(s).

    Returns
    -------
    same as mz_array
        the mz(s) converted to mass based on the charge.
    """
    return mz_array*charge - charge*1.007276


def mz_from_mass(mass_array, charge):
    """
    Parameters
    ----------
    mass_array : float or np.array
        the mass(s) to be converted to mz(s).
    charge : same as mass_array
        the charge(s) of the mz(s).

    Returns
    -------
    same as mass_array
        the mass(s) converted to mass based on the charge.
    """
    return (mass_array + charge*1.007276)/charge


def read_ions_from_txt(path, sep='\t'):
    """

    Parameters
    ----------
    path
        path to the txt file
    sep
        The separator between the two fields in the txt file.
        The default is TAB - \t

    Returns
    -------

    """

    with open(path) as f:
        return {k: float(v) for line in f for (k, v) in [line.strip().split(sep, 1)]}
