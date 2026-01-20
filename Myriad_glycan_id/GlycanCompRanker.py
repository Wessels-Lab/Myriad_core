

import pandas as pd
import numpy as np

from common import IonFinder
from common import GlyCompAssembler

from utils import check_min_comp, add_dHex_to_compositions
from Myriad_glycan_id.DataClasses import SpectrumProperties, CompositionProperties


class GlycanCompRanker:
    def __init__(self, building_blocks: pd.DataFrame,
                 ox_ions: pd.DataFrame,
                 Y5Y1_evidence_ions: pd.DataFrame,
                 fucose_evidence_ions: pd.DataFrame,
                 Y0Y1_ions: pd.DataFrame,
                 extra_Y_ions: pd.DataFrame,
                 minimum_fucose_count: int,  # 2
                 mass_error: float, # 20
                 mass_error_unit: str, # 'ppm'
                 minimum_ion_intensity: float, # 0.01
                 minimum_intensity_type: str,  # 'relative'
                 filters_to_apply: list
                 ):
        """
        This class ranks glycan compositions according the ions found in a spectrum

            Parameters
            ----------
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
            ox_ions: pd.DataFrame
                A table of oxonium ions to use for the composition ranking.
                    columns:
                        'name': str
                        'compossition': dict[str,int]
                        'mass': float
            Y5Y1_evidence_ions: pd.DataFrame
                A table of Y ions to use for the composition ranking, compare compositions. columns like ox_ions.
             fucose_evidence_ions: dict[str,pd.DataFrame]
                A dictionary of tables of fucosylated Y ions to use for the composition ranking, presence is fucose evidence.
                 Keys are Deoxy-Hexose building block name, values are data frames with columns like ox_ions.
             Y0Y1_ions: pd.DataFrame
                A table of Y0- and Y1-related ions NOT used in ranking but are reported as Y ions. columns like ox_ions.
             extra_Y_ions: pd.DataFrame
                A table of extra Y ions NOT used in ranking but are reported alogside the Y ions. columns like ox_ions.
            minimum_fucose_count: int,
                The minimum count of fucose_evidence_ions to be considered as evidence for the presence of a fucose-like sugar
            mass_error: float
                The mass tolerance. Half the width of the window to look around the masses.
            mass_error_unit: str
                The units for mass_error. Either 'Da' or 'ppm'
            minimum_ion_intensity: float
                The minimum relative intensity for an ion to be considered for ranking a composition.
            minimum_intensity_type: str
                The type of minimum_ion_intensity either 'relative' or 'absolute'.
            filters_to_apply: list
                What boolean columns to use for filtered_glycan_rank
        """
        self.building_blocks = building_blocks
        assert ~ox_ions['name'].duplicated().any(), \
            f"oxonium ios names contain duplicates: {ox_ions.loc[ox_ions['name'].duplicated(False), 'name']}"
        self.ox_ions = ox_ions.copy()
        self.Y5Y1_evidence_ions = Y5Y1_evidence_ions.copy()
        self.fucose_evidence_ions_no_fuc = fucose_evidence_ions.copy()
        self.fucose_evidence_ions = add_dHex_to_compositions(fucose_evidence_ions, GlyCompAssembler(building_blocks))
        self.Y0Y1_ions = Y0Y1_ions.copy()
        self.extra_Y_ions = extra_Y_ions.copy()
        self.mass_error = mass_error
        self.mass_error_unit = mass_error_unit
        self.minimum_ion_intensity =minimum_ion_intensity
        self.minimum_intensity_type = minimum_intensity_type
        self.minimum_fucose_count = minimum_fucose_count
        self.filters_to_apply = filters_to_apply


        # set up two ions finders
        # for oxonium ions
        self.ox_finder = IonFinder(ions=self.ox_ions['mass'].to_numpy(),
                                   mass_error=self.mass_error,
                                   mass_error_unit=self.mass_error_unit,
                                   min_int=self.minimum_ion_intensity,
                                   int_type=self.minimum_intensity_type)
        # for all other ions
        self.ion_finder = IonFinder(ions=np.array([]),
                                    mass_error=self.mass_error,
                                    mass_error_unit=self.mass_error_unit,
                                    min_int=self.minimum_ion_intensity,
                                    int_type=self.minimum_intensity_type)


    def rank_compositions(self, compositions: list[CompositionProperties], spectrum: np.ndarray,
                          Y1_mz:float, Y1_charge: int) -> tuple[list[CompositionProperties], SpectrumProperties]:
        """
        Rank a list of compositions (CompositionProperties objects) according to a spectrum.
        This is THE function to call from this object.

        Parameters
        ----------
        compositions: list[CompositionProperties],
            The compositions to be ranked. Coming from GlycanCompositionGenerator.generate_composition()
        spectrum: np.ndarray
            The spectrum to use for ranking the compositions
        Y1_mz: float
            The m/z value for the peptide + HexNAc (the reference mass from the decomposer)
        Y1_charge: int
            The charge for Y1_mz

        Returns
        -------
        list[CompositionProperties]
            The input list of compositions (CompositionProperties objects) with rank populated (and other attributes too)
            SpectrumProperties with the spectrum properties used for the ranking

        SpectrumProperties
            Spectrum level properties that are used for the composition ranking.

        """
        spec_properties = SpectrumProperties()
        self.get_ion_intensities(Y1_mz, Y1_charge, spectrum, spec_properties)
        self.calculate_spec_properties(spectrum, spec_properties)

        # calculate evidences for the compositions
        for comp in compositions:
            # calculate composition properties required for evidence calculations
            comp.calculate_comp_properties(self.ox_ions, spec_properties.oxonium_ions_intensity)
            # calcualte evidence components (and store data in the composition object)
            comp.oxonium_evidence = self.calculate_oxonium_evidence(comp, spec_properties.oxonium_ions_intensity)
            comp.building_blocks_coverage = self.calculate_bb_coverage(comp)
            if (self.building_blocks['type'] == 'Deoxy-hexose').any():
                comp.fucose_evidence = self.calculate_fucose_evidence(comp, spec_properties)
            else:
                comp.fucose_evidence = None
            comp.Y5Y1_evidence = self.calculate_Y5Y1_evidence(comp, spec_properties)

        if len(compositions) > 0:
            # rank the compositions
            # TODO: if taking a lot of time, do not use pandas to do the ranking.
            ranking = pd.DataFrame([comp.__dict__ for comp in compositions])
            evidence_columns = ['oxonium_evidence', 'building_blocks_coverage', 'fucose_evidence', 'Y5Y1_evidence']
            # rank each component on its own (larger is better), sum the rankings, and rank again (smaller is better)
            ranking['glycan_rank'] = ranking[evidence_columns].rank(method='dense', ascending=False)\
                .sum(axis='columns').rank(method='dense', ascending=True).astype('Int32')
            # apply filters and then rerank
            ranking['filtered_glycan_rank'] = ranking.loc[ranking[self.filters_to_apply].all(axis='columns'), evidence_columns]\
                .rank(method='dense', ascending=False).sum(axis='columns').rank(method='dense', ascending=True).astype('Int32')
            for comp, rank, filtered_rank in zip(compositions, ranking['glycan_rank'], ranking['filtered_glycan_rank']):
                comp.glycan_rank = rank if pd.notna(rank) else None
                comp.filtered_glycan_rank = filtered_rank if pd.notna(filtered_rank) else None

        return compositions, spec_properties

    # spectrum methods
    def get_ion_intensities(self, Y1_mz: float,
                            Y1_charge: int,
                            spectrum: np.ndarray,
                            spec_properties: SpectrumProperties):
        """
        Extract the ion intensities from the spectrum, oxonium ions and Y ions.
        """
        # get ox ions intensity
        ox_ions = self.ox_finder.find_ions(spectrum)
        if self.minimum_intensity_type != 'relative':
            ox_ions = ox_ions / spectrum[:, 1].max()
        spec_properties.oxonium_ions_intensity = ox_ions

        # get fucose shadow intensities
        masses_without_fucose = tuple(Y1_mz + x / Y1_charge for x in self.fucose_evidence_ions_no_fuc['mass'])
        ions_without_fucose = self.ion_finder.find_ions(spectrum, masses_without_fucose)
        for dHex_name in self.fucose_evidence_ions:
            fucose_masses = tuple(Y1_mz + x/Y1_charge for x in self.fucose_evidence_ions[dHex_name]['mass'])
            fucose_ions = self.ion_finder.find_ions(spectrum, fucose_masses)
            # if there is no pattern ion, don't consider a fucose shadow ion
            fucose_ions[np.isnan(ions_without_fucose)] = np.nan
            spec_properties.fucose_ions_intensity[dHex_name] = fucose_ions

        # get Y0Y1 intensities
        pattern_masses = tuple(Y1_mz + x/Y1_charge for x in self.Y0Y1_ions['mass'])
        pattern_ions = self.ion_finder.find_ions(spectrum, pattern_masses)
        spec_properties.pattern_ions_intensity = pattern_ions

        # get Y5Y1 ions intensities
        Y5Y1_masses = tuple(Y1_mz + x/Y1_charge for x in self.Y5Y1_evidence_ions['mass'])
        Y5Y1_ions = self.ion_finder.find_ions(spectrum, Y5Y1_masses)
        spec_properties.Y5Y1_ions_intensity = Y5Y1_ions

        # get extra ions intensities
        extra_masses = tuple(Y1_mz + x/Y1_charge for x in self.extra_Y_ions['mass'])
        extra_ions = self.ion_finder.find_ions(spectrum, extra_masses)
        spec_properties.extra_ions_intensity = extra_ions

    def calculate_spec_properties(self, spectrum, spec_properties):
        """
        Calculates spectrum properties after the ion intensities were extracted
        """
        spec_properties.spectrum_oxonium_count = np.count_nonzero(~np.isnan(spec_properties.oxonium_ions_intensity))
        spec_properties.oxonium_relative_intensity_sum = np.nansum(spec_properties.oxonium_ions_intensity) / spectrum[:,
                                                                                                         1].sum()

    # composition evidence methods
    def calculate_oxonium_evidence(self, composition: CompositionProperties, ox_ions_intensities: np.ndarray):
        """
        Calculates the oxonium evidence = oxonium count * oxonium intensity
        """
        relevant_ions = self.ox_ions['composition'].apply(lambda x: check_min_comp(composition.glycan_composition, x))
        comp_ox_ions_mask = relevant_ions.to_numpy() & ~np.isnan(ox_ions_intensities)
        comp_ox_count = np.count_nonzero(comp_ox_ions_mask)
        comp_ox_int = np.nansum(ox_ions_intensities[relevant_ions])

        composition.composition_used_oxonium_mask = comp_ox_ions_mask
        composition.composition_relevant_oxonium_mask = relevant_ions
        composition.composition_oxonium_count = comp_ox_count
        composition.composition_oxonium_intensity = comp_ox_int

        return round(comp_ox_count*comp_ox_int, 6)

    def calculate_bb_coverage(self, comp_properties: CompositionProperties) -> float:
        """
        Calculates the building block coverage - what fraction of the building blocks in the composition
        are also in the oxonium ions.
        """
        bb_with_ox_count = sum([comp_properties.bb_ox_count[k] > 0 for k in comp_properties.bb_ox_count
                                if pd.notna(comp_properties.bb_ox_count[k])])
        bb_in_comp_count = sum([pd.notna(comp_properties.bb_ox_count[k]) for k in comp_properties.bb_ox_count])
        return bb_with_ox_count / bb_in_comp_count

    def calculate_fucose_evidence(self, composition: CompositionProperties, spec_properties: SpectrumProperties) -> int:
        """
        Calculates if there is evidence for a fucose according to the fucose shadow.

        for each dHex building block:
            The evidence is 1 if there is this dHex building block in the composition and there are fucose shadow peaks with this building block
                or if this dHex in not the composition and there are no fucose shadow peaks with this building block
            The evidence is 0 if this dHex building block is in the composition and there are no fucose shadow peaks, or vice versa
        """
        def calculate_single_dHex_evidence(dHex_name: str) -> int:
            ## is there a fucose in the composition?
            dHex_in_composition = composition.glycan_composition[dHex_name] > 0

            ## is there evidence for a fucose in the spectrum?
            # Get fucose count
            # Check if the fucose_shadow_ions are contained in the composition
            composition_larger_than_fuc_shadow = self.fucose_evidence_ions[dHex_name]['composition'].apply(lambda x: check_min_comp(composition.glycan_composition, x))
            # The fuc_shadow ions that are present in the spectrum
            fuc_ions_in_spec = ~np.isnan(spec_properties.fucose_ions_intensity[dHex_name])
            # count fuc_shadow ions found in the spectrum contained in the composition
            fucose_count = composition_larger_than_fuc_shadow[fuc_ions_in_spec].sum()

            fucose_in_spectrum = fucose_count >= self.minimum_fucose_count

            # store composition properties in composition
            fucose_used_mask = fuc_ions_in_spec & composition_larger_than_fuc_shadow.to_numpy()
            composition.fucose_shadow_used_mask[dHex_name] = fucose_used_mask
            composition.fucose_shadow_relevant_mask[dHex_name] = composition_larger_than_fuc_shadow.to_numpy()
            if composition.fucose_shadow_count is None:
                composition.fucose_shadow_count = 0
            composition.fucose_shadow_count += fucose_count
            if composition.fucose_shadow_intensity_sum is None:
                composition.fucose_shadow_intensity_sum = 0
            composition.fucose_shadow_intensity_sum += spec_properties.fucose_ions_intensity[dHex_name][fucose_used_mask].sum()
            # both True or False is good, only one is bad
            return 0 if dHex_in_composition ^ fucose_in_spectrum else 1

        dHex_evideces = []
        for dHex_name in self.building_blocks.index[self.building_blocks['type'] == 'Deoxy-hexose']:
            dHex_evideces.append(calculate_single_dHex_evidence(dHex_name))

        return sum(dHex_evideces)/len(dHex_evideces)

    def calculate_Y5Y1_evidence(self, composition: CompositionProperties, spec_properties: SpectrumProperties) -> float:
        """
        Calculate the evidence for the Y5 Y1 ions

        returns the fraction of ions found in the spectrum that support the composition.
        """
        # Check if the Y ions are contained in the composition
        composition_larger_than_Y5Y1 = self.Y5Y1_evidence_ions['composition'].apply(lambda x: check_min_comp(composition.glycan_composition, x))
        # The Y ions that are present in the spectrum
        Y5Y1_ions_in_spec = ~np.isnan(spec_properties.Y5Y1_ions_intensity)
        # What fraction of Y ions found in the spectrum are contained in the composition
        # If there are no ions present then "all" of them support this composition
        composition_larger_than_Y_fraction = composition_larger_than_Y5Y1[Y5Y1_ions_in_spec].mean() if Y5Y1_ions_in_spec.sum() > 0 else 1.0
        # Which Y5Y1 ions are supporting this compositions
        composition.composition_used_Y5Y1_mask = Y5Y1_ions_in_spec & composition_larger_than_Y5Y1.to_numpy()
        composition.composition_relevant_Y5Y1_mask = composition_larger_than_Y5Y1.to_numpy()

        # composition should be larger than all
        return composition_larger_than_Y_fraction