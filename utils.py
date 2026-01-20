import re

import numba
import numpy as np
import pandas as pd


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



def str_to_comp_dict(comp_str) -> dict:
    """
    Parameters
    ----------
    comp_str: str
        a string of composition with one-letter code. e.g. H5N4F0S2

    Returns
    -------
    dict
        dictionary of this composition. e.g. {'H':5, 'N':4, 'F':0, 'S': 2}

    """
    return {b: int(cnt) for b, cnt in re.findall('([A-z]+)(\\d+)', comp_str)}


def comp_dict_to_str(comp_dict: dict[str, int]) -> str:
    """
    serializes a composition dictionary to a string, skipping 0 values
    Parameters
    ----------
    comp_dict: dict
        dictionary of this composition. e.g. {'H':5, 'N':4, 'F':0, 'S': 2}

    Returns
    -------
    str
        a string of composition with one-letter code. e.g. H5N4S2
    """
    return  ''.join([f'{k}{v}' for k, v in comp_dict.items() if v > 0])


def apply_bb_codes(comp_dict: dict[str, int], bb_codes:dict[str,str]) -> dict:
    """
    apply sugar building block one-letter codes

    Parameters
    ----------
    comp_dict: dict
        dictionary of this composition. e.g. {'Hex':5, 'HexNAc':4, 'dHex':0, 'NeuAc': 2}
    bb_codes: dict
        dictionary of the one-letter codes. e.g. {'Hex':'H', 'HexNAc':'N, 'dHex':'F', 'NeuAc': 'S'}

    Returns
    -------
    dict
        A modified comp_dict where the keys are mapped with bb_codes, the values are the same
    """
    return {bb_codes[bb]: comp_dict[bb] for bb in comp_dict}


def check_min_comp(comp: dict[str, int], minimum_composition: dict[str, int]) -> bool:
    """

    Parameters
    ----------
    comp: dict
        The composition to check
    minimum_composition: dict
        The minimum composition to check against

    Returns
    -------
    Bool
        True if comp is larger than minimum_composition, False if it is not
    """
    return all([comp[bb] >= minimum_composition[bb] for bb in minimum_composition])


def comp_to_type_comp(composition: dict[str,int], building_blocks: pd.DataFrame) -> dict[str, int]:
    """convert a glycan composition to a building block type composition, summing all building blocks of the same type"""
    bb_type_names = building_blocks.groupby('type', observed=True).apply(lambda x: x.index.to_list(), include_groups=False)
    comp_type_dict = {k:0 for k in bb_type_names.index}
    for bb_type in comp_type_dict:
        for bb in bb_type_names[bb_type]:
            if bb in composition:
                comp_type_dict[bb_type] += composition[bb]

    return comp_type_dict


def assemble_ox_ions(ions_type_comp_df: pd.DataFrame, comp_assembler: "GlyCompAssembler") -> pd.DataFrame:
    """assembles oxonium ions compositions from ions_df using comp_assembler"""
    # table to composition dicts
    ions_type_comp_dicts = ions_type_comp_df.apply(
        lambda x: {k: v for k, v in x.to_dict().items() if (pd.notna(v) and (v > 0))}, axis='columns')
    # assemble type compositions into
    assembled_ions = []
    for type_comp in ions_type_comp_dicts:
        curr_assembled_compositions = comp_assembler.assemble_composition(type_comp)
        for curr_comp in curr_assembled_compositions:
            # add mass of a proton since this is an ion
            curr_comp['mass'] += 1.007276
            assembled_ions.append(curr_comp)
    return pd.DataFrame(assembled_ions).sort_values('mass')


def assemble_Y1_offset_ions(ions_type_comp_df: pd.DataFrame, comp_assembler: "GlyCompAssembler") -> pd.DataFrame:
    """assembles Y ions offsets compositions from ions_df using comp_assembler"""
    # table to composition dicts
    ions_type_comp_dicts = ions_type_comp_df.apply(
        lambda x: {k: v for k, v in x.to_dict().items() if (pd.notna(v) and (v > 0))}, axis='columns')
    # assemble type compositions into
    assembled_ions = []
    for type_comp in ions_type_comp_dicts:
        curr_assembled_compositions = comp_assembler.assemble_composition(type_comp)
        for curr_comp in curr_assembled_compositions:
            assembled_ions.append(curr_comp)
    return pd.DataFrame(assembled_ions).sort_values('mass')


def add_dHex_to_compositions(ions_compositions: pd.DataFrame, comp_assembler: "GlyCompAssembler") -> dict[str,pd.DataFrame]:
    """Add a dHex building block to the ions_compositions for every building block of type dHex"""
    def add_dHex(dHex_name) -> pd.DataFrame:
        compositions = ions_compositions['composition'].apply(lambda x: ({**x, **{dHex_name:1}}))
        names = compositions.apply(comp_assembler.name_from_composition)
        names = 'pep+' + names
        masses = compositions.apply(comp_assembler.calcualte_composition_mass)
        out_df = pd.DataFrame({'name': names, 'composition': compositions, 'mass': masses}).sort_values('mass')
        return out_df

    dHex_names = comp_assembler.building_blocks.index[comp_assembler.building_blocks['type'] == 'Deoxy-hexose']
    out_dict = {dHex_name:add_dHex(dHex_name) for dHex_name in dHex_names}
    return out_dict
