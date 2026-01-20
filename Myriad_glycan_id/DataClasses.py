import dataclasses
from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils import check_min_comp, comp_to_type_comp


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
    """
    oxonium_ions_intensity: np.ndarray = None
    pattern_ions_intensity: np.ndarray = None
    fucose_ions_intensity: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    Y5Y1_ions_intensity: np.ndarray = None
    extra_ions_intensity: np.ndarray = None

    spectrum_oxonium_count: int = None
    oxonium_relative_intensity_sum: float = None


class CompositionProperties(object):

    def __init__(self,
                 glycan_composition: dict[str, int],
                 building_blocks: pd.DataFrame,
                 corrected_glycan_mr: float,
                 isotope_offset: int):
        """
        Dataclass to store properties of a glycan composition in the context of a spectrum, for glycan composition
        ranking

        Parameters
        ----------
        glycan_composition: dict[str, int]
            The amounts of the building blocks in the composition
        building_blocks: pd.DataFrame
            The building blocks from the input parameters, parsed into a DataFrame
                index:
                    'name': str,
                columns:
                    'mass':float,
                    'code': str, len == 1,
                    'min_comp': int,
                    'max_comp': int,
                    'type':str, in ('Hexose', 'Hexose-NAc', 'Deoxy-hexose', 'Sialic-acid')
        corrected_glycan_mr: float
            The calculated mass of the glycan (considering the isotope offset)
        isotope_offset: int
            The isotope offset for the glycan mass. (from the mass calculated by the decomposer)

        """

        self.glycan_composition = glycan_composition
        self.building_blocks = building_blocks
        self.glycan_isotope_offset = isotope_offset
        self.corrected_glycan_mr = corrected_glycan_mr

        # derived properties
        self.glycan_composition_mass = sum(
            [building_blocks.loc[bb, 'mass'] * glycan_composition[bb] for bb in glycan_composition])
        self.glycan_ppm_error = (
                                            corrected_glycan_mr - self.glycan_composition_mass) / self.glycan_composition_mass * 1e6
        self.glycan_bb_type_composition = comp_to_type_comp(glycan_composition, building_blocks)
        self.non_empty_comp = any([pd.notna(v) for v in self.glycan_composition.values()])
        # filters for common glycans
        if self.non_empty_comp:
            min_core_mass = (building_blocks.loc[building_blocks['type'] == 'Hexose', 'mass'].min() * 3 +
                             building_blocks.loc[building_blocks['type'] == 'Hexose-NAc', 'mass'].min() * 2)
            self.has_core = check_min_comp(self.glycan_bb_type_composition, {'Hexose': 3, 'Hexose-NAc': 2}) | \
                            (self.corrected_glycan_mr < min_core_mass)
            HN_pairs = max(
                min(self.glycan_bb_type_composition['Hexose'] - 3, self.glycan_bb_type_composition['Hexose-NAc'] - 2),
                0)
            if self.glycan_bb_type_composition['Sialic-acid'] > 0:
                if HN_pairs >= self.glycan_bb_type_composition['Sialic-acid']:
                    self.sia_smaller_hn = True
                else:
                    self.sia_smaller_hn = False
            else:
                self.sia_smaller_hn = True
        else:
            self.has_core = None
            self.sia_smaller_hn = None

        # blank properties
        # composition_used_oxonium_mask: np.ndarray
        #     A boolean mask for the SpectrumProperties.oxonium_ions_intensity. Leaves ions supporting for this composition
        # composition_relevant_oxonium_mask: np.ndarray
        #     A boolean mask for the SpectrumProperties.oxonium_ions_intensity. Leaves relevant ions (ions that could be supporting this composition)
        # composition_oxonium_count: int
        #     The number of supporting oxonium ions (nansum of composition_used_oxonium_mask)
        # composition_oxonium_intensity: float
        #     The intensity sum of the supporting oxonium ions (the ones in the mask)
        # fucose_shadow_used_mask: np.ndarray
        #     A boolean mask for the SpectrumProperties.fucose_ions_intensity. Leaves ions supporting for this composition
        # fucose_shadow_relevant_mask: np.ndarray
        #     A boolean mask for the SpectrumProperties.fucose_ions_intensity. Leaves relevant ions (ions that could be supporting this composition)
        # fucose_shadow_count: int
        #     The number of supporting fucose shadow ions (nansum of fucose_shadow_count)
        # fucose_shadow_intensity_sum: float
        #     The intensity sum of the supporting fucose shadow ions (the ones in the mask)
        # composition_used_Y5Y1_mask: np.ndarray
        #     A boolean mask for the SpectrumProperties.Y5Y1_ions_intensity. Leaves ions supporting for this composition
        # composition_relevant_Y5Y1_mask: np.ndarray
        #     A boolean mask for the SpectrumProperties.Y5Y1_ions_intensity. Leaves relevant ions (ions that could be supporting this composition)
        # bb_ox_count: dict
        #     The number of oxonium ions supporting the presence of a building block
        # bb_ox_intensity
        #     The intensity of oxonium ions supporting the presence of a building block
        # has_core: bool
        #     Does the candidate composition have the N-glycan core (H3N2)
        # sia-smaller-hn: bool
        #     Does the composition has less or equal S than NH pairs after removing N-glycan core (N2H3) from full composition
        # oxonium_evidence
        #     Evidence for oxonium ions supporting the composition, ox_count * ox_intensity
        # building_blocks_coverage
        #     Fraction of building blocks present in the composition that are covered by the oxonium ions
        # fucose_evidence
        #     Evidence for the presence of a fucose. are there Y ions with a fucose (Y1-Y4) with a fucose containing composition
        # Y5Y1_evidence
        #     Evidence for match with found Y ions. is the composition larger than the glycan moiety of the Y ion
        # glycan_rank
        #     glycan composition rank
        # filtered_glycan_rank
        #     Glycan composition rank after applying filters (has_core and S_smaller_HN)

        self.composition_used_oxonium_mask: np.ndarray = None
        self.composition_relevant_oxonium_mask: np.ndarray = None
        self.composition_oxonium_count: int = None
        self.composition_oxonium_intensity: float = None
        self.fucose_shadow_used_mask: dict[str, np.ndarray] = {}
        self.fucose_shadow_relevant_mask: dict[str, np.ndarray] = {}
        self.fucose_shadow_count: int = None
        self.fucose_shadow_intensity_sum: float = None
        self.composition_used_Y5Y1_mask: np.ndarray = None
        self.composition_relevant_Y5Y1_mask: np.ndarray = None

        self.bb_ox_count: dict = {bb: None for bb in building_blocks.index}
        self.bb_ox_intensity: dict = {bb: None for bb in building_blocks.index}

        self.oxonium_evidence: float = None
        self.building_blocks_coverage: float = None
        self.fucose_evidence: int = None
        self.Y5Y1_evidence: float = None
        self.glycan_rank: int = None
        self.filtered_glycan_rank: int = None

    # TODO: split into multiple functions, now that it is here, it doesn't make sense to bundle
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

        def relevant_ions_per_bb_per_comp(building_block, composition):
            # ions relevant for the composition - the oxonium ion is contained within the composition
            good_ions = ox_ions['composition'].apply(lambda x: check_min_comp(composition, x))

            # keep only ox ions that have bb
            bb_idx = ox_ions['composition'].apply(lambda x: building_block in x)

            return good_ions & bb_idx

        for bb in self.building_blocks.index:
            relevant_ions_bb = relevant_ions_per_bb_per_comp(bb, self.glycan_composition)
            if relevant_ions_bb.sum() > 0:
                self.bb_ox_count[bb] = np.count_nonzero(~np.isnan(ox_ion_int[relevant_ions_bb]))
                self.bb_ox_intensity[bb] = np.nansum(ox_ion_int[relevant_ions_bb])
            else:
                self.bb_ox_count[bb] = None
                self.bb_ox_intensity[bb] = None
