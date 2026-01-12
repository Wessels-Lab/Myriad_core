# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:21:34 2021

@author: Gad.Armony
"""

from IonFinder.utils import get_mass_index, get_mass_intensity_sorted, mass_from_mz
import numpy as np
import pandas as pd


class PatternFinder(object):
    """
    This class finds a pattern of masses in a spectrum

        __init__ Parameters
        ----------
        pattern : tuple of floats
            A list of mass offsets (mass, not m/z), including 0 for the reference mass.
        mass_error : float, optional
            The mass tolerance. Half the width of the window to look around the masses.
            The default is 0.05.
        mass_error_unit : str, optional
            The units for mass_error. Either 'Da' or 'ppm'
            The default is 'Da'.
        minimum_reference_mass : float, optional
            The minimum value for a reference mass. When looking for a pep+HexNAc reference this is the minimal peptide mass
            The default is 600. To not look for patterns in oxonium ions.
        minimum_reference_relative_intensity : float in range [0,1], optional
            The minimum relative intensity for the reference mass.
            The default is 0.1 (10%).
        min_pattern_matches : int, optional
            The minimum number of pattern matches to consider when filtering for topN.
            The default is 0. The actual minimum is 2 - ref mass and another one.

        Raises
        ------
        AttributeError
            Rais an error when the pattern is missing the reference mass (offset 0).

        """
    def __init__(self, pattern: tuple,
                 mass_error: float = 0.05,
                 mass_error_unit: str = 'Da',
                 minimum_reference_mass: float = 600,
                 minimum_reference_relative_intensity: float = 0.1,
                 min_pattern_matches: float = 0):
        """
        Refer to the class documentation

        """

        # Make sure that the pattern has the 0 offset
        assert 0 in pattern, '"pattern" does not contain the reference mass - 0 offset'
        self.pattern = np.array(pattern)
        self.mass_error = mass_error
        assert mass_error_unit in ['Da', 'ppm'], f"mass_error_unit must be either 'Da' or 'ppm', not {mass_error_unit}"
        self.mass_error_unit = mass_error_unit
        self.min_ref_mass = minimum_reference_mass
        self.min_ref_rel_int = minimum_reference_relative_intensity
        self.min_pattern_matches = min_pattern_matches

    def find_rank1_pattern(self, spec: np.ndarray,
                           precursor_mz: float,
                           precursor_charge: int):
        """
        A wrapper for finding the rank1 pattern

        Parameters
        ----------
        spec : numpy array of shape (#,2)
            spec[:,0] are the m/z
            spec[:,1] are the intensities.
        precursor_mz : float
            The m/z value of the precursor isolated to generate spec.
        precursor_charge : int
            The charge of the precursor isolated to generate spec.

        Returns
        -------
        rank1_pattern :pd.DataFrame
            Columns:
                'Charge' - the charge state the pattern was found in.
                'ref_mz' - the m/z of reference mass (0 in pattern)
                'ref_pos' - the position in sepc of reference mass (0 in pattern)
                '<pattern>' - the relative intensities in spec for the peak that match the pattern, np.nan if no match was found

        """
        # make sure that the spectrum has 2 columns
        assert spec.shape[1] == 2, f'spec has {spec.shape[1]} columns, 2 expected.'

        patterns_df = self.find_pattern(spec, precursor_mz, precursor_charge)
        rank1_pattern = self.filter_topN_patterns(patterns_df, topN=1)
        return rank1_pattern

    def find_pattern(self, spec: np.ndarray,
                     precursor_mz: float,
                     precursor_charge: int):
        """
        Search for a (partial) match of self.pattern in spec.
        Considers only peaks with:
            precursor mass > peak m/z > self.min_ref_mass
            peak relative intensity > self.min_ref_rel_int
        Considers patterns at charge 1 to precursor_charge-1

        Parameters
        ----------
        spec : numpy array of shape (#,2)
            spec[:,0] are the m/z
            spec[:,1] are the intensities
        precursor_mz : float
            The m/z value of the precursor isolated to generate spec.
        precursor_charge : int
            The charge of the precursor isolated to generate spec.

        Returns
        -------
        found_patterns : pd.DataFrame
            Columns:
                'Charge' - the charge state the pattern was found in.
                'ref_mz' - the m/z of reference mass (0 in pattern)
                'ref_pos' - the position in sepc of reference mass (0 in pattern)
                '<pattern>' - the relative intensities in spec for the peak that match the pattern, np.nan if no match was found
        """

        # make sure that the spectrum has 2 columns
        assert spec.shape[1] == 2, f'spec has {spec.shape[1]} columns, 2 expected.'

        precursor_mass = mass_from_mz(precursor_mz, precursor_charge)
        # initialize the outputs
        found_patterns = []
        # check up to precursor_charge-1 unless the charge is 1
        max_check_charge = precursor_charge-1 if precursor_charge > 1 else 1
        largest_peak = spec[:, 1].max()
        for charge in range(1, int(max_check_charge)+1):
            # initialize variables
            curr_pattern = self.pattern/charge
            curr_found_pattern = np.empty(len(curr_pattern)+4)  # +4 for charge, ref_mass, ref_int, and ref_mass_idx
            # filter the spectrum for potential reference masses
            filt_spec = self.filter_spectrum(spec, precursor_mass, charge)
            # if there are no potential reference masses, skip this charge
            # if ~filt_spec.any(): continue
            # go over reference masses (filtered spectrum)
            for ref_mass, ref_int in filt_spec:
                # wipe previous iteration
                curr_found_pattern[:] = np.nan
                # look for the pattern around the reference mass
                for idx, offset in enumerate(curr_pattern):
                    if offset == 0:
                        # for the 0 offset get the intensity of the exact mass
                        curr_found_pattern[idx+4] = (
                            get_mass_intensity_sorted(spec, np.array(ref_mass + offset), 0, self.mass_error_unit)
                            / largest_peak)[0]
                        # for the rest use the mass error
                    else:
                        curr_found_pattern[idx+4] = (
                            get_mass_intensity_sorted(spec, np.array(ref_mass + offset), self.mass_error, self.mass_error_unit)
                            / largest_peak)[0]
                # there is always 1 not-nan value - the 0 offset
                if (~np.isnan(curr_found_pattern)).sum() > 1:
                    # add charge, ref_mass, ref_int, and ref_mass_idx
                    curr_found_pattern[0] = charge
                    curr_found_pattern[1] = ref_mass
                    curr_found_pattern[2] = ref_int
                    curr_found_pattern[3] = get_mass_index(spec, ref_mass, 0, self.mass_error_unit)
                    found_patterns.append(curr_found_pattern.copy())

        found_patterns = pd.DataFrame(found_patterns,
                                      columns=['charge', 'ref_mz', 'ref_int', 'ref_pos']+self.pattern.tolist())
        return found_patterns

    def filter_spectrum(self, spec: np.ndarray,
                        precursor_mass: float,
                        charge: int):
        """
        Parameters
        ----------
        spec : numpy array of shape (#,2)
            spec[:,0] are the m/z
            spec[:,1] are the intensities.
        precursor_mass : float
            The m/z value of the precursor isolated to generate spec.
        charge : int
            The charge of the precursor isolated to generate spec.

        Returns
        -------
        numpy array of shape (#,2)
            spec filtered by precursor_mass and self.min_ref_mass, considering the charge.
        """

        # make sure that the spectrum has 2 columns
        assert spec.shape[1] == 2, f'spec has {spec.shape[1]} columns, 2 expected.'

        index = (spec[:, 0] >= self.min_ref_mass/charge) &\
                (spec[:, 0] <= precursor_mass/charge) &\
                ((spec[:, 1]/(spec[:, 1].max())) >= self.min_ref_rel_int)
        return spec[index, :]

    def filter_topN_patterns(self, pat_tbl_: pd.DataFrame,
                             topN: int,
                             min_pattern_matches: int = None,
                             sort_by=('match_count', 0.0)):
        """
        Parameters
        ----------
        pat_tbl_ : pandas DataFrame, the output of self.find_pattern()
            must have columns named after self.pattern
        topN : int
            The number of sorted patterns to return.
        min_pattern_matches : int, optional
            The minimum number of pattern matches to consider when filtering for topN.
            The default is self.min_pattern_matches which defaults to 0.
        sort_by : list-like, optional
            Which columns to use to sort (rank) the patterns, ascending.
            The default is ('match_count', 0.0).

        Returns
        -------
        pandas DataFrame
            A subset of pat_tbl_, the topN patterns ranked according to sort_by
        """
        min_pattern_matches = min_pattern_matches or self.min_pattern_matches
        # don't modify the original
        pat_tbl = pat_tbl_.copy()
        # count the number of pattern matches
        pat_tbl['match_count'] = pat_tbl.loc[:, self.pattern].count(axis='columns')
        # filter for at least min_pattern_matches
        pat_tbl = pat_tbl.loc[pat_tbl['match_count'] >= min_pattern_matches, :]
        # sort and take the topN
        pat_tbl = pat_tbl.sort_values(list(sort_by), ascending=False).head(topN)
        pat_tbl['pattern_rank'] = np.arange(1, pat_tbl.shape[0]+1)

        return pat_tbl
