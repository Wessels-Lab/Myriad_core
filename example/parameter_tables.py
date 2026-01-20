import pandas as pd
# This is a dataframe with the compositions of oxonium ions given in terms of building block types (not building block names) for the composition ranker.
# It is used together with the building blocks parameters to calculate the oxonium ions masses.

ions_type_composition_table_ox_finder = pd.DataFrame(
    {'Sialic-acid': [1.0, 1.0, 1.0, None, None, None],
     '-H2O': [None, 1.0, None, None, None, None],
     'Hexose': [None, None, 1.0, None, 1.0, 1.0],
     'Hexose-NAc': [None, None, 1.0, 1.0, 1.0, 1.0],
     'Deoxy-hexose': [None, None, None, None, None, 1.0]}
).astype('Int32')


oxonium_ions_type_composition_table = pd.DataFrame(
    {'Hexose': [None, 1, None, None, None, 1, 2, None, 1, None, 1, 3, None, 1, 2, None, 1, 4, 1, 2, 3, None, 1, 2, 1, 5, 2, 3, 1, 2, 3, 2, 6, 2, 3, 3, 7, 3, 3, 8, 3, 3, 3, 9, 3, 10, 11, 12],
     'Hexose-NAc': [None, None, 1, None, None, None, None, 1, 1, 2, None, None, 1, 1, 1, 2, 2, None, 1, 1, 1, 2, 2, 2, 1, None, 1, 1, 2, 2, 2, 1, None, 2, 2, 3, None, 2, 3, None, 4, 3, 4, None, 4, None, None, None],
     'Deoxy-hexose': [1, None, None, None, None, 1, None, 1, None, None, None, None, None, 1, None, 1, None, None, None, 1, None, 2, 1, None, 1, None, None, 1, 2, 1, None, 1, None, 2, 1, None, None, 2, 1, None, None, 2, 1, None, 2, None, None, None],
     'Sialic-acid': [None, None, None, 1, 1, None, None, None, None, None, 1, None, 1, None, None, None, None, None, 1, None, None, None, None, None, 1, None, 1, None, None, None, None, 1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
     '-H2O': [None, None, None, None, 1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]}
).astype('Int32')

# N-glycan core pattern peaks that are not part of Y5Y1
Y0Y1_ions = pd.DataFrame({'name' : ('pep-OH', 'pep', 'pep+N_frag', 'pep+N1'),
                          'mass':(-220.0821, -203.0794, -120.0423, 0.0)})

# the type compositions of the fucosylated Y ions without the Deoxy-hexose. Each type Deoxy-hexose will be added individually
fucose_evidence_compositions = pd.DataFrame(
    {'Hexose': [0, 0, 1, 2, 3],
     'Hexose-NAc': [1, 2, 2, 2, 2],
     'Deoxy-hexose': [None, None, None, None, None],
     'Sialic-acid': [None, None, None, None, None],
     '-H2O': [None, None, None, None, None]}
).astype('Int32')

Y5Y1_evidence_compositions = pd.DataFrame(
    {'Hexose': [0, 1, 2, 1, 3, 4, 3],
     'Hexose-NAc': [2, 2, 2, 3, 2, 2, 3],
     'Deoxy-hexose': [None, None, None, None, None, None, None],
     'Sialic-acid': [None, None, None, None, None, None, None],
     '-H2O': [None, None, None, None, None, None, None]}
).astype('Int32')

extra_Y_ions = pd.DataFrame(
    {'name': ['pep+H5N2', 'pep+H4N3', 'pep+H3N4'],
     'mass': [1013.3434, 1054.37, 1095.396]}
)