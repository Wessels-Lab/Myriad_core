"""
Created on Wed Aug  4 2021

@author: Gad.Armony
"""

import itertools
import pandas as pd

class GlyCompAssembler(object):
    """
    This class assembles glycan fragment compositions based on input building blocks

    __init__ Parameters
    ----------
    building_blocks: pd.DataFrame
        The building blocks from the input parameters, parsed into a DF
        index:
            'name': str,
        columns:
            'mass':float,
            'code': str, len == 1,
            'min_comp': int,
            'max_comp': int,
            'type':str, in ('Hex', 'HexNAc', 'dHex', 'NeuAc')

    """

    def __init__(self, building_blocks: pd.DataFrame):
        """
    This class assembles glycan fragment compositions based on input building blocks

    Parameters
    ----------
    building_blocks: pd.DataFrame
        The building blocks from the input parameters, parsed into a DF
        index:
            'name': str,
        columns:
            'mass':float,
            'code': str, len == 1,
            'min_comp': int,
            'max_comp': int,
            'type':str, in ('Hex', 'HexNAc', 'dHex', 'NeuAc')

    """

        self.building_blocks = building_blocks
        self.bb_type_names = building_blocks.groupby('type', observed=True).apply(lambda x: x.index.to_list(), include_groups=False)

    def assemble_composition(self, type_composition: dict[str,int]) -> list[dict]:
        """calcualte composition from type composition. This is the entry point of the class"""
        out_compositions = []

        # Pop water loss
        water_loss = 0
        if '-H2O' in type_composition:
            type_composition = type_composition.copy()
            water_loss = type_composition.pop('-H2O')
        # assert that all building block types in the composition are in the input building blocks.
        missing_bb_type = [bb_t for bb_t in type_composition if (bb_t != self.building_blocks['type']).all()]
        assert len(missing_bb_type) == 0, f'missing sugar building block type {missing_bb_type} for oxonium ion {type_composition}'

        # calculate what combinations of building block names we need for each building block type
        bb_type_combinations = {bb_type:[] for bb_type in type_composition}
        for bb_type, count in type_composition.items():
            bb_type_combinations[bb_type] = self._bb_combinations(bb_type, count)

        # generate all compositions
        assembled_compositions = self._assemble_comp_from_bb_combinations(bb_type_combinations)
        # generate name and calcualte mass for all compositions
        for assembled_comp in assembled_compositions:
            out_composition = {}
            comp_name = self.name_from_composition(assembled_comp)
            if water_loss > 0:
                if water_loss == 1:
                    comp_name += '-H2O'
                else:
                    comp_name += f'-{water_loss}H2O'
            out_composition['name'] = comp_name

            out_composition['composition'] = assembled_comp

            comp_mass = self.calcualte_composition_mass(assembled_comp)
            # remove water loss mass
            comp_mass -= water_loss * 18.01056

            out_composition['mass'] = comp_mass

            out_compositions.append(out_composition)

        return out_compositions


    def _bb_combinations(self, bb_type: str, count: int) -> list[dict[str, int]]:
        """
        Calcualte building block combinations for all building blocks within the same type.
        For example,bb_type='Hex' with two names ['Hex', 'Hex-mod'], and count=2 will yield:
        [{'Hex':2, 'Hex-mod':0}, {'Hex':1, 'Hex-mod':1},{'Hex':0, 'Hex-mod':2}]
         """
        tuple_combinations = itertools.combinations_with_replacement(self.bb_type_names[bb_type], count)

        out_combinations = []
        for tuple_combination in tuple_combinations:
            dict_combination = {bb:tuple_combination.count(bb) for bb in self.bb_type_names[bb_type]}
            # make sure that all building blocks counts are smaller than the user denifed maximum
            # (since this is a fragment, the minimum does not apply)
            if all([(dict_combination[bb] <= self.building_blocks.loc[bb,'max_comp']) for bb in dict_combination]):
                out_combinations.append(dict_combination)

        return out_combinations

    @staticmethod
    def _assemble_comp_from_bb_combinations(bb_type_combinations: dict[str,list[dict[str, int]]]) -> list[dict[str,int]]:
        """Assemble the building block type combinations into compositions using all combinations of the type combinations"""
        # calculate all the combinations picking one combination for each type
        compositions_tuples = GlyCompAssembler._recursive_nested_loops(list(bb_type_combinations.values()))
        # consolidate the individual type combinations into one composition
        assembled_comps = []
        for comp_tup in compositions_tuples:
            new_comp = {}
            for comp in comp_tup:
                for k, v in comp.items():
                    if v>0:
                        new_comp[k] = v
            assembled_comps.append(new_comp)
        return assembled_comps

    @classmethod
    def _recursive_nested_loops(cls,lists:list[list[dict]]) -> list[tuple[dict]]:
        """Generate nested loops that loop over the input list of lists"""
        if len(lists) == 0:
            yield ()
        else:
            for x in lists[0]:
                for t in cls._recursive_nested_loops(lists[1:]):
                    yield (x,) + t


    def calcualte_composition_mass(self, composition: dict[str,int]) -> float:
        """Calcualte the composition mass based on the building blocks mass"""
        return sum([self.building_blocks.loc[bb, 'mass'] * composition[bb] for bb in composition])


    def name_from_composition(self, composition: dict[str,int]) -> str:
        """generate a short name for a composition based on the composition. Skip count=0 and do add the count for count=1"""
        name_list = []
        for bb_name, bb_count in composition.items():
            code = self.building_blocks.loc[bb_name, 'code']
            bb_cnt = bb_count
            if bb_cnt == 0:
                continue
            elif bb_cnt == 1:
                bb_cnt = ''
            name_list.append(f'{code}{bb_cnt}')

        return ''.join(name_list)
