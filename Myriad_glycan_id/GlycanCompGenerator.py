import warnings
from bdal.paser.pe_g8s_py_myriad_glycan_id.DataClasses import CompositionProperties


class GlycanCompositionGenerator(object):

    def __init__(self, building_block_masses: dict,
                 building_block_codes: dict,
                 min_composition: dict = None,
                 max_composition: dict = None,
                 glycan_mass_error: float = 20,
                 glycan_mass_error_unit: str = 'ppm',
                 oxonium_mass_error: float = 20,
                 oxonium_mass_error_unit: str = 'ppm'
                 ):
        """
        This class generates compositions based on glycan mass and defined sugar building blocks

        Parameters
        ----------
        building_block_masses : dict
            A dictionary with sugar building blocks as keys and masses as values
        building_block_codes : dict
            A mapping from building block names to building block codes e.g. {'Hex': 'H', 'HexNAc': 'N'}
            Must have the same keys as building_blocks
            These codes are used for the oxonium ions
        min_composition : dict
            The minimum number of each building block to be in an output composition
            Must have the same keys as building_blocks
            The default is all 0
        max_composition : dict
            The maximum number of each building block to be in an output composition
            Must have the same keys as building_blocks
            The default is all float('inf')
        glycan_mass_error : float
            The mass tolerance for matching to the glycan mass. Half the width of the window to look around the masses.
            The default is 0.1 Da
        glycan_mass_error_unit : str
            The units for glycan_mass_error. Either 'Da' or 'ppm'
            The default is 'Da'.
        oxonium_mass_error
            The mass tolerance for matching to the oxonium ions. Half the width of the window to look around the masses.
            The default is 0.02 Da
        oxonium_mass_error_unit
            The units for oxonium_mass_error. Either 'Da' or 'ppm'
            The default is 'Da'.
        """
        # make sure that there is Hex and HexNAc
        assert all(('Hex', 'HexNAc' in building_block_masses)), 'the building blocks must contain "Hex" and "HexNAc"'
        assert (building_block_masses['Hex'] - 162.05282) < 0.001, f'The mass of Hex should be 162.05282, not {building_block_masses["Hex"]}'
        assert (building_block_masses['HexNAc'] - 203.07937) < 0.001, f'The mass of HexNAc should be 203.07937, not {building_block_masses["Hex"]}'

        self.isotope_mass_unit = 1.003355

        # setup min and max composition
        if min_composition is None:
            min_composition = {k: 0 for k in building_block_masses}
        if max_composition is None:
            max_composition = {k: float('inf') for k in building_block_masses}

        self.building_blocks = building_block_masses
        assert building_block_codes.keys() == building_block_masses.keys(), \
            'building_blocks and building_block_codes must have the same keys'
        self.bb_codes = building_block_codes
        assert min_composition.keys() == building_block_masses.keys(), \
            'building_blocks and min_composition must have the same keys'
        self.min_composition = min_composition
        assert max_composition.keys() == building_block_masses.keys(), \
            'building_blocks and max_composition must have the same keys'
        self.max_composition = max_composition
        self.glycan_mass_error = glycan_mass_error
        assert glycan_mass_error_unit in ('Da', 'ppm'), f"mass error unit must be either 'Da' or 'ppm' not '{glycan_mass_error_unit}'"
        self.glycan_mass_error_unit = glycan_mass_error_unit
        self.oxonium_mass_error = oxonium_mass_error
        assert glycan_mass_error_unit in ('Da', 'ppm'), f"mass error unit must be either 'Da' or 'ppm' not '{oxonium_mass_error_unit}'"
        self.oxonium_mass_error_unit = oxonium_mass_error_unit


        # for creation of tuples to loop over. padding for the nested_loops_8 method
        self._bb_names = tuple(building_block_masses.keys())

        def _make_padded_tuple(pad_val, vals_dict, tup_len, keys=self._bb_names):
            """
            Makes a left padded tuple the values of a dictionary
            A tuple of length tup_len where the first values are pad_val and the rest are the values of
            vals_dict in the order of keys
            Parameters
            ----------
            pad_val
                The value to pad with
            vals_dict
                The dictionary with the values to be padded
            tup_len
                total tuple length, incuding the padding
            keys
                The keys (in order) for vals_dict

            Returns
            -------
            tuple
                A padded tuple of length tup_len

            """
            return tuple([pad_val for _ in range(tup_len - len(keys))] + [vals_dict[k] for k in keys])

        self._make_padded_tuple = _make_padded_tuple

        # pick the algorithm for generate_composition
        if len(building_block_masses) <= 8:
            self.loops_algorithm_function = self.nested_loops_8
            # create padded tuples to loop over
            self._masses = self._make_padded_tuple(1e6, self.building_blocks, tup_len=8)
            self._max_comp = self._make_padded_tuple(0, self.max_composition, tup_len=8)
        else:
            self.loops_algorithm_function = self.nested_loops_recursive
            # no actual padding but use the same function to get a tuple
            self._masses = self._make_padded_tuple(1, self.building_blocks, tup_len=len(self._bb_names))
            self._max_comp = self._make_padded_tuple(0, self.max_composition, tup_len=len(self._bb_names))
            warnings.warn('There are more than 8 building blocks. Using a slower algorithm. What is this glycan?!')

    # wrapper function for use of the correct algorithm and setting the isotope offset
    def generate_composition(self, glycan_mr: float, mass_offset: int) -> list[CompositionProperties]:
        """
        generate compositions that fit in the glycan_mr + mass_offset

        Parameters
        ----------
        glycan_mr: float
            the mass of the glycan (ad calcualted by the decomposer)
        mass_offset: int
            isotope offset - how many isotope masses to offset the glycan_mr

        Returns
        -------
        list[CompositionProperties]
            A list of CompositionProperties objects with the found compositions


        """
        corrected_glycan_mr = glycan_mr + mass_offset * self.isotope_mass_unit
        comp_dicts = self.loops_algorithm_function(corrected_glycan_mr)
        compositions = []
        for comp in comp_dicts:
            compositions.append(CompositionProperties(glycan_composition=comp, building_blocks=self.building_blocks,
                                                      building_block_codes=self.bb_codes, corrected_glycan_mr=corrected_glycan_mr,
                                                      isotope_offset=mass_offset))
        return compositions

    def nested_loops_8(self, glycan_mr: float) -> list[dict]:
        """
        Finds all compositions of building blocks (masses) that fit in glycan_mr within mass_error
        Work on exactly 8 building blocks
        Parameters
        ----------
        glycan_mr : float
            The neutral mass of the full glycan

        Returns
        -------
        List[dict[str,int]]
            A list of compositions. Each composition is a dict with the present bb_names as keys and the count as values
        -------

        """
        # set parameters
        min_comp = self._make_padded_tuple(0, self.min_composition, tup_len=8)
        max_comp = self._max_comp
        masses = self._masses
        bb_names = self._bb_names
        mass_error = self.glycan_mass_error

        if self.glycan_mass_error_unit == 'Da':
            glycan_mr_plus_error = glycan_mr + mass_error
            glycan_mr_minus_error = glycan_mr - mass_error
        elif self.glycan_mass_error_unit == 'ppm':
            glycan_mr_plus_error = glycan_mr * (1 + mass_error/1e6)
            glycan_mr_minus_error = glycan_mr * (1 - mass_error/1e6)

        # let the loops begin
        good_compositions = []
        max_0 = min(max_comp[0], int((glycan_mr_plus_error)
                                           / masses[0] + 1))
        for i0 in range(min_comp[0], max_0 + 1):
            max_1 = min(max_comp[1], int((glycan_mr_plus_error
                                                - i0 * masses[0])
                                               / masses[1] + 1))
            for i1 in range(min_comp[1], max_1 + 1):
                max_2 = min(max_comp[2], int((glycan_mr_plus_error
                                                    - i0 * masses[0]
                                                    - i1 * masses[1])
                                                   / masses[2] + 1))
                for i2 in range(min_comp[2], max_2 + 1):
                    max_3 = min(max_comp[3], int((glycan_mr_plus_error
                                                        - i0 * masses[0]
                                                        - i1 * masses[1]
                                                        - i2 * masses[2])
                                                       / masses[3] + 1))
                    for i3 in range(min_comp[3], max_3 + 1):
                        max_4 = min(max_comp[4], int((glycan_mr_plus_error
                                                            - i0 * masses[0]
                                                            - i1 * masses[1]
                                                            - i2 * masses[2]
                                                            - i3 * masses[3])
                                                           / masses[4] + 1))
                        for i4 in range(min_comp[4], max_4 + 1):
                            max_5 = min(max_comp[5], int((glycan_mr_plus_error
                                                                - i0 * masses[0]
                                                                - i1 * masses[1]
                                                                - i2 * masses[2]
                                                                - i3 * masses[3]
                                                                - i4 * masses[4])
                                                               / masses[5] + 1))
                            for i5 in range(min_comp[5], max_5 + 1):
                                max_6 = min(max_comp[6], int((glycan_mr_plus_error
                                                                    - i0 * masses[0]
                                                                    - i1 * masses[1]
                                                                    - i2 * masses[2]
                                                                    - i3 * masses[3]
                                                                    - i4 * masses[4]
                                                                    - i5 * masses[5])
                                                                   / masses[6] + 1))
                                for i6 in range(min_comp[6], max_6 + 1):
                                    max_7 = min(max_comp[7], int((glycan_mr_plus_error
                                                                        - i0 * masses[0]
                                                                        - i1 * masses[1]
                                                                        - i2 * masses[2]
                                                                        - i3 * masses[3]
                                                                        - i4 * masses[4]
                                                                        - i5 * masses[5]
                                                                        - i6 * masses[6])
                                                                       / masses[7] + 1))
                                    for i7 in range(min_comp[7], max_7 + 1):
                                        comp_mass = (i0 * masses[0]
                                                     + i1 * masses[1]
                                                     + i2 * masses[2]
                                                     + i3 * masses[3]
                                                     + i4 * masses[4]
                                                     + i5 * masses[5]
                                                     + i6 * masses[6]
                                                     + i7 * masses[7])
                                        if glycan_mr_plus_error >= comp_mass >= glycan_mr_minus_error:
                                            good_compositions.append(
                                                {k: v for (k, v) in zip(bb_names,
                                                                        [i0, i1, i2, i3, i4, i5, i6, i7][
                                                                        -len(bb_names):])})
        return good_compositions

    # recursive with last nested loop explicit - faster that recursive all the way
    def nested_loops_recursive(self, glycan_mr: float):
        """

        Parameters
        ----------
        glycan_mr : float
            The neutral mass of the full glycan

        Returns
        -------
        List[dict[str,int]]
            A list of compositions. Each composition is a dict with the present bb_names as keys and the count as values

        """
        min_comp = self._make_padded_tuple(0, self.min_composition, tup_len=len(self._bb_names))

        if self.glycan_mass_error_unit == 'Da':
            glycan_mr_plus_error = glycan_mr + self.glycan_mass_error
        elif self.glycan_mass_error_unit == 'ppm':
            glycan_mr_plus_error = glycan_mr * (1 + self.glycan_mass_error / 1e6)

        good_compositions = []
        composition = [c for c in min_comp]
        comp_mass = sum((x * y for x, y in zip(composition, self._masses)))

        def nested_loop(composition, comp_mass, good_compositions, rec_counter=0):
            # stop condition - use the nested_loops_8 for the last 8 building blocks
            if rec_counter == (len(composition) - 8):
                # send the residual mass after the first building blocks
                glycan_mr_for_8 = (glycan_mr -
                                   sum((x * y for x, y in zip(composition[:rec_counter], self._masses[:rec_counter]))))
                # the mass error should be from the original glycan_mr
                # for ppm we need to calculate a new ppm value based on the ratio
                if self.glycan_mass_error_unit == 'ppm':
                    mass_error_for_8 = self.glycan_mass_error * glycan_mr / glycan_mr_for_8
                # for error in Da it is the same
                else:  # self.glycan_mass_error_unit == 'Da'
                    mass_error_for_8 = self.glycan_mass_error



                good_compositions_8 = self.nested_loops_8(glycan_mr=glycan_mr_for_8)
                for comp_8 in good_compositions_8:
                    good_comp = {k: v for (k, v) in zip(self._bb_names, composition[:rec_counter])}
                    for k, v in comp_8.items():
                        good_comp[k] = v
                    good_compositions.append(good_comp)
                composition[rec_counter] = 0
                comp_mass = sum((x * y for x, y in zip(composition[rec_counter + 1:], self._masses)))

            else:
                max_count = int(
                    (glycan_mr_plus_error - comp_mass) / self._masses[rec_counter] + min_comp[rec_counter] + 1)
                for i in range(min_comp[rec_counter], min(self._max_comp[rec_counter], max_count) + 1):
                    composition[rec_counter] = i
                    comp_mass = sum((x * y for x, y in zip(composition, self._masses)))
                    nested_loop(composition, comp_mass, good_compositions, rec_counter + 1)
                    composition[rec_counter] = 0
                    comp_mass = sum((x * y for x, y in zip(composition, self._masses)))

        nested_loop(composition, comp_mass, good_compositions, 0)
        return good_compositions
