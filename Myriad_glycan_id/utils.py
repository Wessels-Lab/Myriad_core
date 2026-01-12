import re
import pandas as pd


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
