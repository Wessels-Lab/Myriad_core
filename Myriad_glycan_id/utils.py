import re


def str_to_comp_dict(comp_str):
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
    return {b: int(cnt) for b, cnt in re.findall('([A-z]+)(\d+)', comp_str)}


def comp_dict_to_str(comp_dict):
    """

    Parameters
    ----------
    comp_dict: dict
        dictionary of this composition. e.g. {'H':5, 'N':4, 'F':0, 'S': 2}

    Returns
    -------
    str
        a string of composition with one-letter code. e.g. H5N4F0S2
    """
    return ''.join([f'{k}{v}' for k, v in comp_dict.items()])


def apply_bb_codes(comp_dict, bb_codes):
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


def check_min_comp(comp: dict, minimum_composition: dict):
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
