from pathlib import Path
import numpy as np
import pandas as pd

from common import IonFinder, GlyCompAssembler
from Decomposer import PatternFinder, SpectrumModifier
from Myriad_glycan_id import GlycanCompositionGenerator, GlycanCompRanker
from utils import mass_from_mz, mz_from_mass, comp_dict_to_str, apply_bb_codes, assemble_ox_ions, assemble_Y1_offset_ions
from parameter_tables import (ions_type_composition_table_ox_finder, oxonium_ions_type_composition_table, Y0Y1_ions,
                              Y5Y1_evidence_compositions, fucose_evidence_compositions, extra_Y_ions)

## load spectrum
spectrum_file = Path(__file__).parent / "spectrum_GTAGNALMDGASQLMGENR_H5N4S2.txt"
spectrum = np.genfromtxt(spectrum_file.as_posix(), delimiter='\t')
precursor_mz = 1025.1607096436799
precursor_charge = 4

# setup parameters
mass_error = 20
mass_error_unit = 'ppm'

building_blocks = pd.DataFrame(index = pd.Series(['Hex', 'HexNAc', 'dHex', 'NeuAc', 'NeuGc'], name='name'),
                               data = {'mass':[162.05282, 203.07937, 146.05791, 291.09542, 307.0903],
                                       'code':['H', 'N', 'F', 'S', 'G'],
                                       'min_comp':[0, 1, 0, 0, 0],
                                       'max_comp':[12, 7, 2, 4, 4],
                                       'type':['Hexose', 'Hexose-NAc', 'Deoxy-hexose', 'Sialic-acid', 'Sialic-acid']})

comp_assembler = GlyCompAssembler(building_blocks)

assembled_oxonium_ions = assemble_ox_ions(oxonium_ions_type_composition_table, comp_assembler)
assembled_oxonium_ions = assembled_oxonium_ions.sort_values(by='mass', ascending=True) # sort by mass for faster performance in finding these ions

# ion finder
min_oxonium_intensity = 0.01
min_oxonium_intensity_type = 'relative'
assembled_ions_ox_finder = assemble_ox_ions(ions_type_composition_table_ox_finder, comp_assembler)
assembled_ions_ox_finder = assembled_ions_ox_finder.sort_values(by='mass', ascending=True) # sort by mass for faster performance in finding these ions
ions_mass_ox_finder = assembled_ions_ox_finder['mass'].to_numpy()

# pattern finder
min_oxonium_count = 1
min_relative_oxonium_intensity = 0.0047
pattern = (-220.0821, -203.0794, -120.0423, 0, 203.0794, 365.1322, 527.185, 689.2378)
min_ref_mass = 850
min_ref_relative_int = 0.1
min_patterns_matches = 0

# spectrum modifier
min_intensity = 0.0
min_intensity_unit = 'relative'
ions_mass_spec_modifier = assembled_oxonium_ions['mass']

# composition generator
building_block_masses  ={'Hex': 162.05282, 'HexNAc': 203.07937, 'dHex': 146.05791, 'NeuAc': 291.09542, 'NeuGc': 307.0903}
building_block_codes ={ 'Hex': 'H', 'HexNAc': 'N', 'dHex': 'F', 'NeuAc': 'S', 'NeuGc': 'G'}
min_composition = {'Hex': 0, 'HexNAc': 1, 'dHex': 0, 'NeuAc': 0, 'NeuGc': 0}
max_composition = {'Hex': 12, 'HexNAc': 7, 'dHex': 2, 'NeuAc': 4, 'NeuGc': 4}

# composition ranker
min_fuc_shdaow_count = 2
min_ion_int = 0.01
min_ion_int_unit = 'relative'
filter_columns = ['has_core', 'sia_smaller_hn'] # composition must have the N-glycan core, Number of Sialic acids is smaller than Hex-HexNAc pairs
max_isotope_offset = 2 # look for +/- compositions matching +/- this isotope
assembled_Y5Y1_ions = assemble_Y1_offset_ions(Y5Y1_evidence_compositions, comp_assembler)
# The mass offset is from Y1, but the composition is from Y0
assembled_Y5Y1_ions['mass'] = assembled_Y5Y1_ions['mass'] - building_blocks.loc['HexNAc', 'mass']
assembled_Y5Y1_ions['name'] = 'pep+' + assembled_Y5Y1_ions['name']
assembled_fucose_ions = assemble_Y1_offset_ions(fucose_evidence_compositions, comp_assembler)

## instantiate objects
ion_finder = IonFinder(ions=ions_mass_ox_finder, mass_error=mass_error, mass_error_unit=mass_error_unit,
                       min_int=min_oxonium_intensity, int_type=min_oxonium_intensity_type)

pattern_finder = PatternFinder(pattern=pattern, mass_error=mass_error, mass_error_unit=mass_error_unit,
                               minimum_reference_mass=min_ref_mass,
                               minimum_reference_relative_intensity=min_ref_relative_int,
                               min_pattern_matches=min_patterns_matches)

spectrum_modifier = SpectrumModifier(ions=ions_mass_spec_modifier, mass_error=mass_error, isotope_mass_error=mass_error,
                                     mass_error_unit=mass_error_unit,min_int=min_intensity, int_type=min_intensity_unit)

glycan_composition_generator = GlycanCompositionGenerator(building_blocks=building_blocks,
                                                          glycan_mass_error=mass_error,
                                                          glycan_mass_error_unit=mass_error_unit)

glycan_composition_ranker = GlycanCompRanker(building_blocks=building_blocks,
                                             ox_ions=assembled_oxonium_ions,
                                             Y5Y1_evidence_ions=assembled_Y5Y1_ions,
                                             fucose_evidence_ions=assembled_fucose_ions,
                                             Y0Y1_ions=Y0Y1_ions,
                                             extra_Y_ions=extra_Y_ions,
                                             minimum_fucose_count=min_fuc_shdaow_count,
                                             mass_error=mass_error,
                                             mass_error_unit=mass_error_unit,
                                             minimum_ion_intensity=min_ion_int,
                                             minimum_intensity_type=min_ion_int_unit,
                                             filters_to_apply=filter_columns)

# process spectrum
found_ions = ion_finder.find_ions(spectrum)
num_oxonium = np.count_nonzero(~np.isnan(found_ions))
rel_ox_int_sum = np.nansum(found_ions) / (spectrum[:, 1] / spectrum[:, 1].max()).sum()

print(f'found {num_oxonium} oxonium ions with relative intensity sum {rel_ox_int_sum}\n')

# if there are enough oxonioum ions look for a pattern
if (num_oxonium >= min_oxonium_count) and (rel_ox_int_sum >= min_relative_oxonium_intensity):
    found_pattern = pattern_finder.find_rank1_pattern(spectrum,
                                                  precursor_mz=precursor_mz,
                                                  precursor_charge=precursor_charge)
    # convert to a series if a pattern was found
    found_pattern = found_pattern.iloc[0] if found_pattern.shape[0] > 0 else found_pattern

    print(f'found a pattern of length {found_pattern.match_count}. ref_mz: {found_pattern.ref_mz}, charge: {found_pattern.charge}\n')

    # if there is a pattern, modify the spectrum
    if len(found_pattern) > 0:
        Y1_mass = mass_from_mz(found_pattern['ref_mz'], found_pattern['charge'])
        modified_precursor_mz = mz_from_mass(Y1_mass, 1)
        # Prepare peptide spectrum
        pep_spectrum = spectrum_modifier.remove_peaks_over_mass(
            spectrum_modifier.deiso_spec(
                spectrum_modifier.remove_ions(spectrum),
                max_charge=precursor_charge),
            Y1_mass + 10)
        # calculate the glycan mass
        experimental_glycan_mr = (mass_from_mz(precursor_mz, precursor_charge) -  # precursor mass
                    Y1_mass +  # pep+HexNAc mass
                    building_block_masses['HexNAc'])
        print(f'modified the spectrum, spectrum length went from {spectrum.shape[0]} to {pep_spectrum.shape[0]}.\n'
              f'Y1_mass: {Y1_mass}, modified_precursor_mz: {modified_precursor_mz}, glycan mass: {experimental_glycan_mr}\n')

        # generate and rank the compositions
        compositions = []
        isotope_offsets = np.arange(-max_isotope_offset,
                                    max_isotope_offset + 1)
        for mass_offset in isotope_offsets:
            compositions = compositions + glycan_composition_generator.generate_composition(experimental_glycan_mr,
                                                                            mass_offset)

        ranked_compositions, spec_properties = glycan_composition_ranker.rank_compositions(compositions, spectrum,
                                                                                           found_pattern['ref_mz'],
                                                                                           found_pattern['charge'])
        comp_strings = [comp_dict_to_str(apply_bb_codes(c.glycan_composition, building_block_codes)) for c in ranked_compositions]
        filtered_comp_strings = [comp_dict_to_str(apply_bb_codes(c.glycan_composition, building_block_codes)) for c in ranked_compositions if c.filtered_glycan_rank]
        rank1_comp_strings = [comp_dict_to_str(apply_bb_codes(c.glycan_composition, building_block_codes)) for c in ranked_compositions if c.filtered_glycan_rank == 1]
        print(f'found {len(ranked_compositions)} compositions: {comp_strings}\n'
              f'{len(filtered_comp_strings)} filtered compositions: {filtered_comp_strings}\n'
              f'top ranking compositions: {rank1_comp_strings}\n')
