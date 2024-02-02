import dataclasses

import pandas as pd
import numpy as np
import re
from bdal.paser.ionFinder import ionFinder
from dataclasses import dataclass


def str_to_comp_dict(comp_str):
    return {b: int(cnt) for b, cnt in re.findall('([A-z]+)(\d+)', comp_str)}

def comp_dict_to_str(comp_dict):
    return ''.join([f'{k}{v}' for k, v in comp_dict.items()])



def check_min_comp(comp: dict, minimum_composition: dict):
    return all([comp[bb] >= minimum_composition[bb] for bb in minimum_composition])


@dataclass
class SpectrumProperties:
    oxonium_ions_intensity: np.ndarray = None
    fucose_ions_intensity: np.ndarray = None
    Y5Y1_ions_int: np.ndarray = None
    extra_ions_int: np.ndarray = None

    total_oxonium_count: int = None
    oxonium_relative_intensity: float = None
    fucose_shadow_count: int = None
    fucose_shadow_intensity: float = None


@dataclass
class CompositionProperties:
    composition_mass: float = None
    glycan_ppm_error: float = None
    bb_ox_count: dict = dataclasses.field(default_factory=dict)
    bb_ox_intensity: dict = dataclasses.field(default_factory=dict)
    has_core: int = None

class glycanCompRanker:

    def __init__(self, building_blocks: dict[str, float],
                 building_block_codes: dict[str, str],
                 ox_ions: pd.DataFrame,
                 mass_error: float = 20,
                 mass_error_unit: str = 'ppm',
                 minimum_ion_intensity: float = 100,
                 minimum_intensity_type: str = 'absolute',
                 extra_ions: tuple = (568.2116, 1013.3434, 1054.37, 1095.396)  # bisecting HexNAc, Y5(HH, HN, NN)
                 ):
        self.building_blocks = building_blocks
        assert building_block_codes.keys() == building_blocks.keys(), \
            'building_blocks and building_block_codes must have the same keys'
        # swap key and value since the compositions come with the one-letter code
        self.bb_codes = building_block_codes#
        self.ox_ions = ox_ions
        reverse_bb_codes = {k: v for v, k in building_block_codes.items()}
        self.ox_ions['composition'] = self.ox_ions['name'].apply(str_to_comp_dict)\
                                                          .apply(lambda x: {reverse_bb_codes[k]: v for k, v in x.items()})
        self.mass_error = mass_error
        self.mass_error_unit = mass_error_unit
        self.minimum_ion_intensity = minimum_ion_intensity
        self.minimum_intensity_type = minimum_intensity_type
        self.extra_ions = extra_ions

        self._fucose_mass = 146.05791
        for k in building_blocks:
            self._fucose_name = k if (building_blocks[k] - self._fucose_mass) < 0.001 else None
        # pep+HexNAc, pep+2HexNAc, pep+Hex2HexNAc, pep+2Hex2HexNAc, pep+3Hex2HexNAc
        self._pattern_fucose_shadow = (0.0, 203.0794, 365.1322, 527.185, 689.2378)
        self._min_fucose_shadow_count = 2  # minimum number of fucose shadow peaks to consider present
        # core+Hex, core+HexNAc, core+2Hex, core+HexHexNAc, core+2HexNAc
        H = self.bb_codes['Hex']
        N = self.bb_codes['HexNAc']
        self._Y5Y1_ions_compositions = pd.Series({203.0794: {H: 0, N: 2},
                                                  365.1322: {H: 1, N: 2},
                                                  527.1850: {H: 2, N: 2},
                                                  568.2116: {H: 1, N: 3},
                                                  689.2378: {H: 3, N: 2},
                                                  851.2060: {H: 4, N: 2},
                                                  892.3172: {H: 3, N: 3}})
        self.ox_finder = ionFinder(ions=tuple(self.ox_ions['mass']),
                                   mass_error=self.mass_error,
                                   mass_error_unit=self.mass_error_unit,
                                   min_int=self.minimum_ion_intensity,
                                   int_type=self.minimum_intensity_type)
        self.ion_finder = ionFinder(ions=tuple(),
                                    mass_error=self.mass_error,
                                    mass_error_unit=self.mass_error_unit,
                                    min_int=self.minimum_ion_intensity,
                                    int_type=self.minimum_intensity_type)

    def rank_compositions(self, compositions, spectrum, pep_HexNAc_mass, glycan_mass):
        spec_properties = SpectrumProperties()
        self.get_ion_intensities(pep_HexNAc_mass, spectrum, spec_properties)
        self.calculate_spec_properties(spectrum, spec_properties)

        ranking = pd.DataFrame({'composition': compositions, 'ox_score': None, 'bb_coverage': None, 'fucose_score': None,
                                'Y5Y1_score': None, 'comp_properties': CompositionProperties(), 'rank': None})
        #if there are no compositions return a None dataframe
        if ranking['composition'].iloc[0] is None:
            return ranking
        # calculate scores for the compositions
        for i, comp_row in ranking.iterrows():
            self.calculate_comp_properties(comp_row['composition'], glycan_mass, spec_properties.oxonium_ions_intensity,
                                           comp_row['comp_properties'])
            ranking.loc[i, 'ox_score'] = self.calculate_oxonium_score(comp_row['composition'], spec_properties.oxonium_ions_intensity)
            ranking.loc[i, 'bb_coverage'] = self.calculate_bb_coverage(comp_row['comp_properties'])
            if self._fucose_name:
                ranking.loc[i, 'fucose_score'] = self.calculate_fucose_score(comp_row['composition'], comp_row['comp_properties'])
            else:
                ranking.loc[i, 'fucose_score'] = None
            ranking.loc[i, 'Y5Y1_score'] = self.calculate_Y5Y1_score(comp_row['composition'], spec_properties)
        ranking['rank'] = ranking[['ox_score', 'bb_coverage', 'fucose_score', 'Y5Y1_score']].\
            rank(method='dense', ascending=False).sum(axis='columns')
        ranking = ranking.sort_values('rank', ascending=True)
        return ranking, spec_properties

    def get_ion_intensities(self, pep_HexNAc_mass, spectrum, data_class):
        # get ox ions intensity
        ox_ions = self.ox_finder.find_ions(spectrum)
        if self.minimum_intensity_type == 'relative':
            ox_ions = ox_ions / spectrum[:, 1].max()
        # don't use thresholds
        # # apply ions_thresholds
        # ox_ions[ox_ions < self.ox_ions['BP_intensity_threshold']] = np.nan
        data_class.oxonium_ions_intensity = ox_ions

        # get fucose shadow intensities
        pattern_masses = tuple(x + pep_HexNAc_mass for x in self._pattern_fucose_shadow)
        pattern_ions = self.ion_finder.find_ions(spectrum, pattern_masses)
        fucose_masses = tuple(x + self._fucose_mass for x in pattern_masses)
        fucose_ions = self.ion_finder.find_ions(spectrum, fucose_masses)
        # if there is no pattern ion, don't consider a fucose shadow ion
        fucose_ions[np.isnan(pattern_ions)] = np.nan
        data_class.fucose_ions_intensity = fucose_ions

        # get Y5Y1 ions intensities
        Y5Y1_masses = tuple(x + pep_HexNAc_mass for x in self._Y5Y1_ions_compositions.keys())
        Y5Y1_ions = self.ion_finder.find_ions(spectrum, Y5Y1_masses)
        data_class.Y5Y1_ions_int = Y5Y1_ions

        # get extra ions intensities
        extra_masses = tuple(x + pep_HexNAc_mass for x in self.extra_ions)
        extra_ions = self.ion_finder.find_ions(spectrum, extra_masses)
        data_class.extra_ions_int = extra_ions

    def calculate_oxonium_score(self, composition: dict, ox_ions_intensities: np.ndarray):

        good_ions = self.ox_ions['composition'].apply(lambda x: check_min_comp(composition, x))
        comp_ox_count = np.count_nonzero(~np.isnan(ox_ions_intensities[good_ions]))
        comp_ox_int = np.nansum(ox_ions_intensities[good_ions])

        return comp_ox_count*comp_ox_int

    def calculate_bb_coverage(self, comp_properties: CompositionProperties):
        bb_with_ox_count = sum([comp_properties.bb_ox_count[k] > 0 for k in comp_properties.bb_ox_count])
        bb_in_comp_count = sum([~np.isnan(comp_properties.bb_ox_count[k]) for k in comp_properties.bb_ox_count])
        return bb_with_ox_count / bb_in_comp_count

    def calculate_fucose_score(self, composition: dict, composition_properties: CompositionProperties):
        fucose_in_composition = self.bb_codes[self._fucose_name] in composition
        fucose_in_spectrum = composition_properties.bb_ox_count[self._fucose_name] >= self._min_fucose_shadow_count

        # both True or False is good, only one is bad
        return 0 if fucose_in_composition ^ fucose_in_spectrum else 1

    def calculate_Y5Y1_score(self, composition: dict, spec_properties: SpectrumProperties):
        # check only compositions of ions that are present in the spectrum
        compositions_to_check = self._Y5Y1_ions_compositions[~np.isnan(spec_properties.Y5Y1_ions_int)]
        composition_larger_than_all = compositions_to_check.apply(lambda x: check_min_comp(composition, x)).all()

        # composition should be larger than all
        return 1 if composition_larger_than_all else 0



    def calculate_spec_properties(self, spectrum, spec_properties):
        spec_properties.total_oxonium_count = np.count_nonzero(~np.isnan(spec_properties.oxonium_ions_intensity))
        spec_properties.oxonium_relative_intensity = np.nansum(spec_properties.oxonium_ions_intensity) / spectrum[:,
                                                                                                         1].sum()
        spec_properties.fucose_shadow_count = np.count_nonzero(~np.isnan(spec_properties.fucose_ions_intensity))
        spec_properties.fucose_shadow_intensity = np.nansum(spec_properties.fucose_ions_intensity)

    def calculate_comp_properties(self, comp: dict,
                                  glycan_mass: float,
                                  ox_ion_int: np.ndarray,
                                  comp_properties: CompositionProperties):
        comp_properties.composition_mass = sum([self.building_blocks[bb] * comp[bb] for bb in comp])
        comp_properties.glycan_ppm_error = (abs(comp_properties.composition_mass - glycan_mass) /
                                                comp_properties.composition_mass * 1e6)
        comp_properties.has_core = check_min_comp(comp, {'Hex': 3, 'HexNAc': 2}) | \
                                   (glycan_mass < self.building_blocks['Hex']*3 + self.building_blocks['HexNAc']*2)
        if 'NeuAc' in self.building_blocks:
            HN_pairs = min(comp[self.bb_codes['Hex']], comp[self.bb_codes['HexNAc']])
            comp_properties.S_smaller_HN = 1 if HN_pairs > comp[self.bb_codes['NeuAc']] else 0


        def relevant_ions_per_bb_per_comp(building_block, composition):
            # ions relevant for the composition - the oxonium ion is contained within the composition
            good_ions = self.ox_ions['composition'].apply(lambda x: check_min_comp(composition, x))

            # keep only ox ions that have bb
            bb_idx = self.ox_ions['composition'].apply(lambda x: building_block in x)

            return good_ions & bb_idx

        for bb in self.building_blocks:
            relevant_ions = relevant_ions_per_bb_per_comp(bb, comp)
            if relevant_ions.sum() > 0:
                comp_properties.bb_ox_count[bb] = np.count_nonzero(~np.isnan(ox_ion_int[relevant_ions]))
                comp_properties.bb_ox_intensity[bb] = np.nansum(ox_ion_int[relevant_ions])
            else:
                comp_properties.bb_ox_count[bb] = np.nan
                comp_properties.bb_ox_intensity[bb] = np.nan



