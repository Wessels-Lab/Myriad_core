

import pandas as pd
import numpy as np
from bdal.paser.IonFinder import IonFinder

from bdal.paser.pe_g8s_py_myriad_glycan_id.utils import str_to_comp_dict, check_min_comp
from bdal.paser.pe_g8s_py_myriad_glycan_id.DataClasses import SpectrumProperties, CompositionProperties

class GlycanCompRanker:
    def __init__(self, building_block_masses: dict[str, float],
                 building_block_codes: dict[str, str],
                 ox_ions: pd.DataFrame,
                 mass_error: float = 20,
                 mass_error_unit: str = 'ppm',
                 minimum_ion_intensity: float = 0.01,
                 minimum_intensity_type: str = 'relative',
                 extra_ions: tuple = (1013.3434, 1054.37, 1095.396),  # Y5(HH, HN, NN)
                 extra_ions_names: tuple = ('pep+H5N2', 'pep+H4N3', 'pep+H3N4')
                 ):
        """
        This class ranks glycan compositions according the ions found in a spectrum

            Parameters
            ----------
            building_block_masses: dict[str, float]
                The masses of the glycan building blocks (sugars)
            building_block_codes: dict[str, float]
                single letter codes for the building blocks
            ox_ions: pd.DataFrame
                A table of oxonium ions to use for the composition ranking.
                Include columns: name, composition, mass
            mass_error: float, optional
                The mass tolerance. Half the width of the window to look around the masses.
                The default is 20.
            mass_error_unit: str, optional
                The units for mass_error. Either 'Da' or 'ppm'
                The default is 'ppm'.
            minimum_ion_intensity: float, optional
                The minimum relative intensity for an ion to be considered for ranking a composition.
                The default is 0.01 (1%).
            minimum_intensity_type: str, optional
                The type of minimum_ion_intensity either 'relative' or 'absolute'.
                The default is 'relative'
            extra_ions: tuple[float], optional
                extra Y ions offsets (from peptide + HexNAc) to look for in the spectrum, these are NOT used for
                 the ranking, only their intensity is reported.
                 The default is (1013.3434, 1054.37, 1095.396) which are (Y5: HH, HN, NN).
            extra_ions_names, tuple[str]
        """
        self.building_blocks = building_block_masses
        assert building_block_codes.keys() == building_block_masses.keys(), \
            'building_blocks and building_block_codes must have the same keys'
        self.bb_codes = building_block_codes
        # swap key and value since the compositions come with the one-letter code
        reverse_bb_codes = {k: v for v, k in building_block_codes.items()}
        assert ~ox_ions['name'].duplicated().any(), \
            f"oxonium ios names contain duplicates: {ox_ions.loc[ox_ions['name'].duplicated(False), 'name']}"
        self.ox_ions = ox_ions.copy()
        # convert str to dict
        # TODO: move this to main.py
        self.ox_ions['composition'] = self.ox_ions['composition'].apply(str_to_comp_dict)\
                                          .apply(lambda x: {reverse_bb_codes[k]: v for k, v in x.items()})
        self.mass_error = mass_error
        self.mass_error_unit = mass_error_unit
        self.minimum_ion_intensity = minimum_ion_intensity
        self.minimum_intensity_type = minimum_intensity_type
        self.extra_ions = extra_ions
        self.extra_ions_names = extra_ions_names

        # hard-coded attributes
        self.fucose_mass = 146.0579
        fucose_name = [k for k in building_block_masses if abs(building_block_masses[k] - self.fucose_mass) < 0.001]
        self._fucose_name = fucose_name[0] if len(fucose_name) > 0 else None
        self.pattern_fucose_shadow_names = ('pep+N1F1', 'pep+N2F1', 'pep+H1N2F1', 'pep+H2N2F1', 'pep+H3N2F1')
        self.pattern_fucose_shadow = (0.0, 203.0794, 365.1322, 527.185, 689.2378)
        self._min_fucose_shadow_count = 2  # minimum number of peaks to consider the fucose shadow present

        self.Y5Y1_ions_names = ('pep+N2', 'pep+H1N2', 'pep+H2N2', 'pep+H1N3', 'pep+H3N2', 'pep+H4N2', 'pep+H3N3')
        self.Y5Y1_ions_compositions = pd.Series({203.0794: {'Hex': 0, 'HexNAc': 2},
                                                 365.1322: {'Hex': 1, 'HexNAc': 2},
                                                 527.1850: {'Hex': 2, 'HexNAc': 2},
                                                 568.2116: {'Hex': 1, 'HexNAc': 3},
                                                 689.2378: {'Hex': 3, 'HexNAc': 2},
                                                 851.2060: {'Hex': 4, 'HexNAc': 2},
                                                 892.3172: {'Hex': 3, 'HexNAc': 3}})
        # N-glycan core pattern peaks that are not part of Y5Y1
        self.pattern_names = ('pep-OH', 'pep', 'pep+N_frag', 'pep+N1')
        self.pattern = (-220.0821, -203.0794, -120.0423, 0.0)

        # set up two ions finders
        # for oxonium ions
        self.ox_finder = IonFinder(ions=tuple(self.ox_ions['mass']),
                                   mass_error=self.mass_error,
                                   mass_error_unit=self.mass_error_unit,
                                   min_int=self.minimum_ion_intensity,
                                   int_type=self.minimum_intensity_type)
        # for all other ions
        self.ion_finder = IonFinder(ions=tuple(),
                                    mass_error=self.mass_error,
                                    mass_error_unit=self.mass_error_unit,
                                    min_int=self.minimum_ion_intensity,
                                    int_type=self.minimum_intensity_type)

    def rank_compositions(self, compositions: list[CompositionProperties], spectrum: np.ndarray,
                          Y1_mz:float, Y1_charge: int):
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

        """
        spec_properties = SpectrumProperties()
        self.get_ion_intensities(Y1_mz, Y1_charge, spectrum, spec_properties)
        self.calculate_spec_properties(spectrum, spec_properties)

        # calculate evidences for the compositions
        for comp in compositions:
            # calculate composition properties required for evidence calculations
            comp.calculate_comp_properties(self.ox_ions, spec_properties.oxonium_ions_intensity)
            # calcualte evidence components
            comp.oxonium_evidence = self.calculate_oxonium_evidence(comp, spec_properties.oxonium_ions_intensity)
            comp.building_blocks_coverage = self.calculate_bb_coverage(comp)
            if self._fucose_name:
                comp.fucose_evidence = self.calculate_fucose_evidence(comp.glycan_composition,
                                                                      spec_properties.fucose_shadow_count)
            else:
                comp.fucose_evidence = None
            comp.Y5Y1_evidence = self.calculate_Y5Y1_evidence(comp.glycan_composition, spec_properties)

        if len(compositions) > 0:
            # rank the compositions
            # TODO: if taking a lot of time, do not use pandas to do the ranking.
            ranking = pd.DataFrame([comp.__dict__ for comp in compositions])
            # rank each component on its own (larger is better), sum the rankings, and rank again (smaller is better)
            ranking['glycan_rank'] = ranking[['oxonium_evidence', 'building_blocks_coverage', 'fucose_evidence', 'Y5Y1_evidence']].\
                rank(method='dense', ascending=False).sum(axis='columns').rank(method='dense', ascending=True)
            # apply filters and then rerank
            ranking['filtered_glycan_rank'] = ranking.loc[ranking['has_core'] & ranking['sia_smaller_hn'], 'glycan_rank']\
                .rank(method='dense', ascending=True)
            for comp, rank, filtered_rank in zip(compositions, ranking['glycan_rank'], ranking['filtered_glycan_rank']):
                comp.glycan_rank = rank
                comp.filtered_glycan_rank = filtered_rank

        return compositions, spec_properties

    # spectrum methods
    def get_ion_intensities(self, Y1_mz: float,
                            Y1_charge: int,
                            spectrum: np.ndarray,
                            data_class: SpectrumProperties):
        """
        Extract the ion intensities from the spectrum, oxonium ions and Y ions.
        """
        # get ox ions intensity
        ox_ions = self.ox_finder.find_ions(spectrum)
        if self.minimum_intensity_type != 'relative':
            ox_ions = ox_ions / spectrum[:, 1].max()
        data_class.oxonium_ions_intensity = ox_ions

        # get fucose shadow intensities
        pattern_masses_for_fuc = tuple(Y1_mz + x/Y1_charge for x in self.pattern_fucose_shadow)
        pattern_ions_for_fuc = self.ion_finder.find_ions(spectrum, pattern_masses_for_fuc)
        fucose_masses = tuple(x + self.fucose_mass / Y1_charge for x in pattern_masses_for_fuc)
        fucose_ions = self.ion_finder.find_ions(spectrum, fucose_masses)
        # if there is no pattern ion, don't consider a fucose shadow ion
        fucose_ions[np.isnan(pattern_ions_for_fuc)] = np.nan
        data_class.fucose_ions_intensity = fucose_ions

        # get pattern intensities
        pattern_masses = tuple(Y1_mz + x/Y1_charge for x in self.pattern)
        pattern_ions = self.ion_finder.find_ions(spectrum, pattern_masses)
        data_class.pattern_ions_intensity = pattern_ions

        # get Y5Y1 ions intensities
        Y5Y1_masses = tuple(Y1_mz + x/Y1_charge for x in self.Y5Y1_ions_compositions.keys())
        Y5Y1_ions = self.ion_finder.find_ions(spectrum, Y5Y1_masses)
        data_class.Y5Y1_ions_intensity = Y5Y1_ions

        # get extra ions intensities
        extra_masses = tuple(Y1_mz + x/Y1_charge for x in self.extra_ions)
        extra_ions = self.ion_finder.find_ions(spectrum, extra_masses)
        data_class.extra_ions_intensity = extra_ions

    def calculate_spec_properties(self, spectrum, spec_properties):
        """
        Calculates spectrum properties after the ion intensities were extracted
        """
        spec_properties.spectrum_oxonium_count = np.count_nonzero(~np.isnan(spec_properties.oxonium_ions_intensity))
        spec_properties.oxonium_relative_intensity_sum = np.nansum(spec_properties.oxonium_ions_intensity) / spectrum[:,
                                                                                                         1].sum()
        spec_properties.fucose_shadow_count = np.count_nonzero(~np.isnan(spec_properties.fucose_ions_intensity))
        spec_properties.fucose_shadow_intensity_sum = np.nansum(spec_properties.fucose_ions_intensity)

    # composition evidence methods
    def calculate_oxonium_evidence(self, composition: CompositionProperties, ox_ions_intensities: np.ndarray):
        """
        Calculates the oxonium evidence - oxonium count * oxonium intensity
        """
        good_ions = self.ox_ions['composition'].apply(lambda x: check_min_comp(composition.glycan_composition, x))
        comp_ox_count = np.count_nonzero(~np.isnan(ox_ions_intensities[good_ions]))
        comp_ox_int = np.nansum(ox_ions_intensities[good_ions])

        composition.composition_oxonium_count = comp_ox_count
        composition.composition_oxonium_intensity = comp_ox_int

        return round(comp_ox_count*comp_ox_int, 6)

    def calculate_bb_coverage(self, comp_properties: CompositionProperties):
        """
        Calculates the building block coverage - what fraction of the building blocks in the composition
        are also in the oxonium ions.
        """
        bb_with_ox_count = sum([comp_properties.bb_ox_count[k] > 0 for k in comp_properties.bb_ox_count])
        bb_in_comp_count = sum([~np.isnan(comp_properties.bb_ox_count[k]) for k in comp_properties.bb_ox_count])
        return bb_with_ox_count / bb_in_comp_count

    def calculate_fucose_evidence(self, composition: dict, fucose_count: int):
        """
        Calculates if three is evidence for a fucose according to the fucose shadow.

        returns 1 if there is a fucose in the composition and there are fucose shadow peaks
            or if there is no fucose in the composition and there are no fucose shadow peaks
        returns 0 if there is a fucose in the composition and there are no fucose shadow peaks, or vice versa
        """
        fucose_in_composition = composition[self._fucose_name] > 0
        fucose_in_spectrum = fucose_count >= self._min_fucose_shadow_count

        # both True or False is good, only one is bad
        return 0 if fucose_in_composition ^ fucose_in_spectrum else 1

    def calculate_Y5Y1_evidence(self, composition: dict, spec_properties: SpectrumProperties):
        """
        Calculate the evidence for the Y5 Y1 ions

        returns 1 if all found Y ions are contained in the composition, 0 if they are not.
        """
        # check only compositions of ions that are present in the spectrum
        compositions_to_check = self.Y5Y1_ions_compositions[~np.isnan(spec_properties.Y5Y1_ions_intensity)]
        composition_larger_than_all = compositions_to_check.apply(lambda x: check_min_comp(composition, x)).all()

        # composition should be larger than all
        return 1 if composition_larger_than_all else 0
