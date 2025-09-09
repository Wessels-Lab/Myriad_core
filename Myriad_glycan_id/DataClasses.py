
from dataclasses import dataclass
import numpy as np
import pandas as pd

from Myriad_glycan_id.utils import check_min_comp


@dataclass
class SpectrumProperties:
    """
    Dataclass to store properties of a spectrum for glycan composition ranking
    Attributes
    ----------
    oxonium_ions_intensity : np.ndarray
        Relative intensity of the oxonium ions found in the spectrum
    pattern_ions_intensity : np.ndarray
        Relative intensity of the N-glycan core pattern (that is not in Y5Y1) found in the spectrum
    fucose_ions_intensity : np.ndarray
        Relative intensity of the fucose shadow found in the spectrum
    Y5Y1_ions_intensity : np.ndarray
        Relative intensity of the Y1-Y5 ions found in the spectrum
    extra_ions_intensity : np.ndarray
        Relative intensity of the extra Y ions defined by the user found in the spectrum
    spectrum_oxonium_count : int
        The number of oxonium ions found in the spectrum (count of oxonium_ions_intensity)
    oxonium_relative_intensity_sum: float
        The sum of relative intensity of the oxonium ions found in the spectrum (sum of oxonium_ions_intensity)
    fucose_shadow_count: int
        The number of fucose shadow ions found in the spectrum (count of fucose_ions_intensity)
    fucose_shadow_intensity_sum: float
        The sum of relative intensity of the fucose shadow ions found in the spectrum (sum of fucose_ions_intensity)
    """
    oxonium_ions_intensity: np.ndarray = None
    pattern_ions_intensity: np.ndarray = None
    fucose_ions_intensity: np.ndarray = None
    Y5Y1_ions_intensity: np.ndarray = None
    extra_ions_intensity: np.ndarray = None

    spectrum_oxonium_count: int = None
    oxonium_relative_intensity_sum: float = None
    fucose_shadow_count: int = None
    fucose_shadow_intensity_sum: float = None


class CompositionProperties(object):

    def __init__(self,
                 glycan_composition: dict[str, int],
                 building_blocks: dict[str, float],
                 building_block_codes: dict[str, str],
                 corrected_glycan_mr: float,
                 isotope_offset: float):
        """
        Dataclass to store properties of a glycan composition in the context of a spectrum, for glycan composition
        ranking

        Parameters
        ----------
        glycan_composition: dict[str, int]
            The amounts of the building blocks in the composition
        building_blocks: dict[str, float]
            The masses of the building blocks
        building_block_codes: dict[str, str]
            single letter codes for the building blocks
        corrected_glycan_mr: float
            The calculated mass of the glycan (considering the isotope offset)
        isotope_offset: int
            The isotope offset for the glycan mass. (from the mass calculated by the decomposer)

        """

        self.glycan_composition = glycan_composition
        self.building_blocks = building_blocks
        self.bb_codes = building_block_codes
        self.glycan_isotope_offset = isotope_offset
        self.corrected_glycan_mr = corrected_glycan_mr

        # derived properties
        self.glycan_composition_mass = sum([building_blocks[bb] * glycan_composition[bb] for bb in glycan_composition])
        if self.glycan_composition_mass != 0:
            self.glycan_ppm_error = (abs(self.glycan_composition_mass - corrected_glycan_mr) / self.glycan_composition_mass * 1e6)
        else:
            self.glycan_ppm_error = np.nan

        # blank properties
        self.composition_oxonium_count = None
        self.composition_oxonium_intensity = None
        self.bb_ox_count = {bb: None for bb in building_blocks}
        self.bb_ox_intensity = {bb: None for bb in building_blocks}
        self.has_core = None
        self.sia_smaller_hn = None
        self.oxonium_evidence = None
        self.building_blocks_coverage = None
        self.fucose_evidence = None
        self.Y5Y1_evidence = None
        self.glycan_rank = None
        self.filtered_glycan_rank = None

    #TODO: split into multiple functions, now that it is here, it doesn't make sense to bundle
    def calculate_comp_properties(self, ox_ions: pd.DataFrame, ox_ion_int: np.ndarray):
        """
        calculate the composition properties.

        Parameters
        ----------
        ox_ions: pd.DataFrame
            A table of oxonium ions to use for the composition ranking.
            Include columns: name, composition, mass
        ox_ion_int: np.ndarray
            The relative intensities of the oxonium ions that were found in the spectrum (SpectrumProperties.oxonium_ions_intensity)
        """

        # filters for common glycans
        self.has_core = check_min_comp(self.glycan_composition, {'Hex': 3, 'HexNAc': 2}) | \
                                   (self.corrected_glycan_mr < self.building_blocks['Hex'] * 3 + self.building_blocks[
                                       'HexNAc'] * 2)
        has_ac = 'NeuAc' in self.building_blocks
        has_gc = 'NeuGc' in self.building_blocks
        HN_pairs = max(min(self.glycan_composition['Hex'] - 3, self.glycan_composition['HexNAc'] - 2), 0)
        if has_ac and not has_gc:
            self.sia_smaller_hn = True if HN_pairs >= self.glycan_composition['NeuAc'] else False
        elif has_gc and not has_ac:
            self.sia_smaller_hn = True if HN_pairs >= self.glycan_composition['NeuGc'] else False
        elif has_gc and has_ac:
            self.sia_smaller_hn = True if HN_pairs >= (self.glycan_composition['NeuAc'] +
                                                       self.glycan_composition['NeuGc']) else False
        else:
            self.sia_smaller_hn = True



        def relevant_ions_per_bb_per_comp(building_block, composition):
            # ions relevant for the composition - the oxonium ion is contained within the composition
            good_ions = ox_ions['composition'].apply(lambda x: check_min_comp(composition, x))

            # keep only ox ions that have bb
            bb_idx = ox_ions['composition'].apply(lambda x: building_block in x)

            return good_ions & bb_idx

        for bb in self.building_blocks:
            relevant_ions_bb = relevant_ions_per_bb_per_comp(bb, self.glycan_composition)
            if relevant_ions_bb.sum() > 0:
                self.bb_ox_count[bb] = np.count_nonzero(~np.isnan(ox_ion_int[relevant_ions_bb]))
                self.bb_ox_intensity[bb] = np.nansum(ox_ion_int[relevant_ions_bb])
            else:
                self.bb_ox_count[bb] = np.nan
                self.bb_ox_intensity[bb] = np.nan






