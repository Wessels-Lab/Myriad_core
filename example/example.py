from pathlib import Path
import numpy as np
import pandas as pd

from IonFinder import IonFinder
from Decomposer import PatternFinder, SpectrumModifier
from Myriad_glycan_id import GlycanCompositionGenerator, GlycanCompRanker
from IonFinder.utils import mz_from_mass, mass_from_mz
from Myriad_glycan_id.utils import apply_bb_codes, comp_dict_to_str


## load spectrum
spectrum_file = Path(__file__).parent / "spectrum_GTAGNALMDGASQLMGENR_H5N4S2.txt"
spectrum = np.genfromtxt(spectrum_file.as_posix(), delimiter='\t')

# parameters
mass_error = 20
mass_error_unit = 'ppm'
oxonium_ions = pd.DataFrame({'name': ['F', 'H', 'N', 'S-H2O', 'G-H2O', 'S', 'G', 'HF', 'H2', 'NF', 'HN', 'N2', 'HS', 'HG', 'H3', 'NS', 'NG', 'HNF', 'H2N', 'N2F', 'HN2', 'H4', 'HNS', 'HNG', 'H2NF', 'H3N', 'N2F2', 'HN2F', 'H2N2', 'HNFS', 'H5', 'HNFG', 'H2NS', 'H2NG', 'H3NF', 'HN2F2', 'H2N2F', 'H3N2', 'H2NFS', 'H6', 'H2NFG', 'H2N2F2', 'H3N2F', 'H3N3', 'H7', 'H3N2F2', 'H3N3F', 'H8', 'H3N4', 'H3N3F2', 'H3N4F', 'H9', 'H3N4F2', 'H10', 'H11', 'H12'],
                     'composition': [{'dHex': 1}, {'Hex': 1}, {'HexNAc': 1}, {'NeuAc': 1}, {'NeuGc': 1}, {'NeuAc': 1}, {'NeuGc': 1}, {'Hex': 1, 'dHex': 1}, {'Hex': 2}, {'HexNAc': 1, 'dHex': 1}, {'Hex': 1, 'HexNAc': 1}, {'HexNAc': 2}, {'Hex': 1, 'NeuAc': 1}, {'Hex': 1, 'NeuGc': 1}, {'Hex': 3}, {'HexNAc': 1, 'NeuAc': 1}, {'HexNAc': 1, 'NeuGc': 1}, {'Hex': 1, 'HexNAc': 1, 'dHex': 1}, {'Hex': 2, 'HexNAc': 1}, {'HexNAc': 2, 'dHex': 1}, {'Hex': 1, 'HexNAc': 2}, {'Hex': 4}, {'Hex': 1, 'HexNAc': 1, 'NeuAc': 1}, {'Hex': 1, 'HexNAc': 1, 'NeuGc': 1}, {'Hex': 2, 'HexNAc': 1, 'dHex': 1}, {'Hex': 3, 'HexNAc': 1}, {'HexNAc': 2, 'dHex': 2}, {'Hex': 1, 'HexNAc': 2, 'dHex': 1}, {'Hex': 2, 'HexNAc': 2}, {'Hex': 1, 'HexNAc': 1, 'dHex': 1, 'NeuAc': 1}, {'Hex': 5}, {'Hex': 1, 'HexNAc': 1, 'dHex': 1, 'NeuGc': 1}, {'Hex': 2, 'HexNAc': 1, 'NeuAc': 1}, {'Hex': 2, 'HexNAc': 1, 'NeuGc': 1}, {'Hex': 3, 'HexNAc': 1, 'dHex': 1}, {'Hex': 1, 'HexNAc': 2, 'dHex': 2}, {'Hex': 2, 'HexNAc': 2, 'dHex': 1}, {'Hex': 3, 'HexNAc': 2}, {'Hex': 2, 'HexNAc': 1, 'dHex': 1, 'NeuAc': 1}, {'Hex': 6}, {'Hex': 2, 'HexNAc': 1, 'dHex': 1, 'NeuGc': 1}, {'Hex': 2, 'HexNAc': 2, 'dHex': 2}, {'Hex': 3, 'HexNAc': 2, 'dHex': 1}, {'Hex': 3, 'HexNAc': 3}, {'Hex': 7}, {'Hex': 3, 'HexNAc': 2, 'dHex': 2}, {'Hex': 3, 'HexNAc': 3, 'dHex': 1}, {'Hex': 8}, {'Hex': 3, 'HexNAc': 4}, {'Hex': 3, 'HexNAc': 3, 'dHex': 2}, {'Hex': 3, 'HexNAc': 4, 'dHex': 1}, {'Hex': 9}, {'Hex': 3, 'HexNAc': 4, 'dHex': 2}, {'Hex': 10}, {'Hex': 11}, {'Hex': 12}],
                     'mass': [147.065186, 163.060096, 204.086646, 274.092136, 290.087016, 292.102696, 308.097576, 309.118006, 325.112916, 350.144556, 366.139466, 407.166016, 454.155516, 470.150396, 487.165736, 495.182066, 511.176946, 512.197376, 528.192286, 553.223926, 569.218836, 649.218556, 657.234886, 673.229766, 674.250196, 690.245106, 699.281836, 715.276746, 731.271656, 803.292796, 811.271376, 819.287676, 819.287706, 835.282586, 836.303016, 861.334656, 877.329566, 893.324476, 965.345616, 973.324196, 981.340496, 1023.387476, 1039.382386, 1096.403846, 1135.377016, 1185.440296, 1242.461756, 1297.429836, 1299.483216, 1388.519666, 1445.541126, 1459.482656, 1591.599036, 1621.535476, 1783.588296, 1945.641116]})

# ion finder
oxonium_ions_finder = (366.139472, 528.192296, 657.234889, 274.092128, 292.102693,
                454.155516, 512.19793, 407.166021, 495.182615)
min_oxonium_intensity = 0.01
min_oxonium_intensity_type = 'relative'

# pattern finder
min_oxonium_count = 1
min_relative_oxonium_intensity = 0.0047
precursor_mz = 1025.1607096436799
precursor_charge = 4
pattern = (-220.0821, -203.0794, -120.0423, 0, 203.0794, 365.1322, 527.185, 689.2378)
min_ref_mass = 850
min_ref_relative_int = 0.1

# spectrum modifier
min_intensity = 0.0
min_intensity_unit = 'relative'

# composition generator
building_block_masses  ={'Hex': 162.05282, 'HexNAc': 203.07937, 'dHex': 146.05791, 'NeuAc': 291.09542, 'NeuGc': 307.0903}
building_block_codes ={ 'Hex': 'H', 'HexNAc': 'N', 'dHex': 'F', 'NeuAc': 'S', 'NeuGc': 'G'}
min_composition = {'Hex': 0, 'HexNAc': 1, 'dHex': 0, 'NeuAc': 0, 'NeuGc': 0}
max_composition = {'Hex': 12, 'HexNAc': 7, 'dHex': 2, 'NeuAc': 4, 'NeuGc': 4}

# composition ranker
min_fuc_shdaow_count = 2
min_ion_int = 0.01
min_ion_int_unit = 'relative'

## instantiate objects
ion_finder = IonFinder(ions=oxonium_ions_finder, mass_error=mass_error, mass_error_unit=mass_error_unit,
                       min_int=min_oxonium_intensity, int_type=min_oxonium_intensity_type)

pattern_finder = PatternFinder(pattern, mass_error=mass_error, mass_error_unit=mass_error_unit,
                                       minimum_reference_mass=min_ref_mass,
                                       minimum_reference_relative_intensity=min_ref_relative_int)

spectrum_modifier = SpectrumModifier(ions=tuple(oxonium_ions['mass']), mass_error=mass_error, isotope_mass_error=mass_error,
                                     mass_error_unit=mass_error_unit,min_int=min_intensity, int_type=min_intensity_unit)

glycan_composition_generator = GlycanCompositionGenerator(building_block_masses=building_block_masses,
                                                            building_block_codes=building_block_codes,
                                                            min_composition=min_composition,
                                                            max_composition=max_composition,
                                                            glycan_mass_error=mass_error,
                                                            glycan_mass_error_unit=mass_error_unit,
                                                            oxonium_mass_error=mass_error,
                                                            oxonium_mass_error_unit=mass_error_unit)

glycan_composition_ranker = GlycanCompRanker(building_block_masses=building_block_masses,
                                             building_block_codes=building_block_codes,
                                             ox_ions=oxonium_ions,
                                             mass_error=mass_error,
                                             mass_error_unit=mass_error_unit,
                                             minimum_ion_intensity=min_ion_int,
                                             minimum_intensity_type=min_ion_int_unit)

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

        #generate and reank the compositions
        compositions = glycan_composition_generator.generate_composition(experimental_glycan_mr, 0)
        ranked_compositions, spec_properties = glycan_composition_ranker.rank_compositions(compositions, spectrum,
                                                                                           found_pattern['ref_mz'],
                                                                                           found_pattern['charge'])
        comp_strings = [comp_dict_to_str(apply_bb_codes(c.glycan_composition, building_block_codes)) for c in ranked_compositions]
        rank1_comp_strings = [comp_dict_to_str(apply_bb_codes(c.glycan_composition, building_block_codes)) for c in ranked_compositions if c.filtered_glycan_rank == 1]
        print(f'found {len(ranked_compositions)} compositions: {comp_strings}\n'
              f'top ranking compositions: {rank1_comp_strings}\n')
