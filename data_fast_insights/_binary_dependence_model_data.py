import numbers
from typing import Iterable, Optional
from itertools import combinations
import logging
from collections import OrderedDict
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BinaryDependenceModelData:
    """ Class for storing data about features and target
        that are to be used in the dependence model.

        See init docstring for parameters description.

        Attributes description is to be done.
    """

    # TODO: attributes description, since they are useful
    def __init__(self,
                 base_data: pd.DataFrame,
                 y_name: str,
                 cat_cols: Optional[Iterable[str]] = None,
                 num_cols: Optional[Iterable[str]] = None,
                 y_type: Optional[str] = "quantile",
                 exclude_zero_var: Optional[bool] = True,
                 **kwargs) -> None:
        """ Initialize object that holds all information about features and target in its attributes.
            This object is supposed to be used further in the calculations of target analysis model.

        Parameters
        ----------
        base_data
            DataFrame with raw data.
        y_name
            Name of the target variable. It must be made in a way that higher values indicate better benefit
        cat_cols
            Iterable that contains names of categorical columns in data
        num_cols
            Iterable that contains names of numeric columns in data. Types must be numeric (including NaN)
        y_type
            "quantile" - model will divide target by chosen quantile
                (default is 0.5, you can set "y_quantile" argument to a number between 0 and 1)
                and rows with target lower than this are considered to be worsening the target.
            "mean" - model will divide target by its mean, and rows with target lower than this
                are considered to be worsening the target.
            "binary" - target is used as is, meant for binary targets.
                So if you want to convert your target to binary in a custom way, you can do this before
                    and set y_type to "binary".
                0 values are considered to be worsening the target

            Defaults to "quantile" with value of 0.5
        exclude_zero_var
            If True:
                - checks categorical features and excludes those having 1 unique value
                - checks numeric features and excludes those having variance = 0
            In case you've passed a dataset with such features already excluded, you might set this to False
            for potential speed up
        """
        if not isinstance(base_data, pd.DataFrame):
            raise TypeError('base_data argument must be a DataFrame object')
        self.base_data = base_data.copy()

        # SET CATEGORICAL AND NUMERIC COLUMNS
        try:
            self.cat_cols = set(cat_cols) if cat_cols is not None else set()
            self.num_cols = set(num_cols) if num_cols is not None else set()
        except TypeError:
            raise TypeError('cat_cols, num_cols arguments must be iterables')
        self.feature_sets = [self.cat_cols, self.num_cols]

        # Intersections are searched in combinations like this because lib used to have binary_cols type
        features_intersections = [s[0].intersection(s[1]) for s in combinations(self.feature_sets, 2)
                                  if s[0].intersection(s[1]) != set()]
        if features_intersections:
            raise ValueError('cat_cols, num_cols must not have common elements')

        # SET TARGET
        if y_name not in base_data.columns:
            raise ValueError(f'y_name set as {y_name} not found in base_data')
        self.y_name = y_name

        # SET TARGET PROCESSING DATA
        self.target_processing_attrs = dict()
        if y_type == 'quantile':
            self.target_processing_attrs['y_type'] = y_type
            self.target_processing_attrs['y_quantile'] = kwargs.get('y_quantile', 0.5)
        elif y_type in ('mean', 'binary'):
            self.target_processing_attrs['y_type'] = y_type
        else:
            raise ValueError('Unknown y_type, please use one of the following: "quantile", "mean", "binary"')

        # SET OTHER
        self.exclude_zero_var = exclude_zero_var

        # Data for converted features
        self.data = self.base_data[[self.y_name]].copy()

        self.col_links = OrderedDict()
        self.y_pivot = None
        self.bins = None
        self.y_binary_name = None
        self.is_data_converted = False

        self._reset_binary_data()

    def _reset_binary_data(self):
        self.data = self.base_data[[self.y_name]].copy()

        self.col_links = OrderedDict()
        self.y_pivot = None
        self.bins = None
        self.y_binary_name = None
        self.is_data_converted = False

        self._check_columns()
        self._convert_types()
        self.add_binary_target()

    def _check_columns(self) -> None:
        unmentioned = [col for col in self.base_data.columns if not any(col in s for s in self.feature_sets)
                       and col != self.y_name]
        if unmentioned:
            raise ValueError(
                f'Found {len(unmentioned)} column(s) in data '
                + """that are not specified in either cat_cols or num_cols: """
                + f'{unmentioned}')
        if self.exclude_zero_var:
            exclude_cats = list()
            exclude_nums = list()

            for cat in self.cat_cols:
                if self.base_data[cat].nunique() < 2:
                    exclude_cats.append(cat)
            for cat in exclude_cats:
                self.cat_cols.remove(cat)
                self.base_data = self.base_data.drop(cat, 1)
                logging.warning(f"{cat} feature was removed before the analysis, because it has < 2 unique values")

            for num in self.num_cols:
                if self.base_data[num].var() == 0.0:
                    exclude_nums.append(num)
            for num in exclude_nums:
                self.num_cols.remove(num)
                self.base_data = self.base_data.drop(num, 1)
                logging.warning(f"{num} feature was removed before the analysis, because it has zero variance")

    def _convert_types(self) -> None:
        logging.info("Checking input types...")
        for c in self.num_cols:
            if np.issubdtype(self.base_data[c].dtype, np.number):
                continue
            self.base_data[c] = pd.to_numeric(self.base_data[c], errors='raise')

    def get_y_pivot(self, y_series: pd.Series) -> pd.Series:
        """ Get the value that divides objects into "bad" and "good"
        """
        if self.target_processing_attrs['y_type'] == 'mean':
            return y_series.mean()
        elif self.target_processing_attrs['y_type'] == 'quantile':
            if not isinstance(self.target_processing_attrs['y_quantile'], numbers.Number):
                raise ValueError('quantile argument must be either None or a number')
            return y_series.quantile(self.target_processing_attrs['y_quantile'])
        elif self.target_processing_attrs['y_type'] == 'binary':
            return None
        else:
            raise ValueError("Unknown y type")

    def add_binary_target(self) -> None:
        self.y_pivot = self.get_y_pivot(self.base_data[self.y_name])

        # already binary if no pivot
        if self.y_pivot is None:
            if sorted(self.base_data[self.y_name].unique()) != [0, 1]:
                raise ValueError(
                    "No pivot set, expect binary target: Binary targets must have exactly 2 unique values: 0 and 1")
            self.y_binary_name = self.y_name + '_copy'
            self.data[self.y_binary_name] = self.base_data[self.y_name].astype('category').cat.codes
            return

        self.y_binary_name = 'is_' + self.y_name + '_lt_' + self.target_processing_attrs['y_type']
        self.data[self.y_binary_name] = self.base_data[self.y_name] < self.y_pivot

    def _convert_cats(self) -> None:
        """ Converting categories to binary (one-hot encoding)
        """
        for col in self.cat_cols:
            for val in self.base_data[col].unique():
                binary_name = col + '_' + str(val)
                self.data[binary_name] = (self.base_data[col] == val).astype(int)
                self.col_links[binary_name] = col
        # self.data = self.data.drop(self.cat_cols, 1)

    def _convert_nums(self, bins: dict) -> None:
        """ Converting numeric to binary (binning)
        """
        for col in self.num_cols:
            for bin_ in bins[col]['bin']:
                if bin_ == 'missing':
                    binary_name = col + '_missing'
                    self.data[binary_name] = np.where(self.base_data[col].isnull(), 1, 0)
                else:
                    binary_name = col + '_' + bin_
                    lb, rb = (float(x) for x in bin_.strip('()[]').split(','))
                    self.data[binary_name] = np.where(
                        (self.base_data[col] >= float(lb)) & (self.base_data[col] < float(rb)), 1, 0)
                self.col_links[binary_name] = col
        # self.data = self.data.drop(self.num_cols, 1)

    def convert_to_binary(self,
                          bins: Optional[dict] = None) -> None:
        """ Convert all variables to binary format.

        Parameters
        ----------
        bins
            If there are numeric columns in class instance,
            bins argument (containing binning for every numeric column) must be specified,
            otherwise these columns are not converted.

        Returns
        -------

        """
        if self.is_data_converted:
            self._reset_binary_data()
        self.bins = dict() if bins is None else bins
        self._convert_cats()
        self._convert_nums(self.bins)
        self.is_data_converted = True

    def construct_partial_combs(self, selected_feature, consider_selected_base: bool = True):
        """ Construct binary feature combinations of the selected feature and every other one.
                These features equal 1 when all of its members equal 1.

            Example:
            We have features
            - "x1", one of its values is "green"
            - "x2", one of its values is 10
            - "x3", one of its values is 120
            After conversion we will have binary features for these values that might look like:
            - "x1_green"
            - "x2_(-inf, 20]"
            - "x3_(-inf, 500]"

            after calling "construct_partial_combs("x1", consider_base="selected")"
            the following segments will be created:
            - "x1_green_AND_x2_(-inf, 20]", which equals 1 when "x1" is "green" and "x2" <= 20
            - "x1_green_AND_x3_(-inf, 500]", <...>

        Parameters
        ----------
        selected_feature : str
        consider_selected_base : bool, optional
            Which column name will be considered as the base (parent) feature for generated binary features.
            If true, then selected_feature argument is set as base column,
                otherwise it's the second feature from the combination.

            This is important for future analysis and plots. (Functions in plotting module group data by base column)


        Returns
        -------

        """
        if not self.is_data_converted:
            raise ValueError("Can only use construct_combs_up_to() when data is converted to binary format")

        selected_binary = [binary for binary, base in self.col_links.items() if base == selected_feature]
        unwanted = [self.y_name, self.y_binary_name] + selected_binary
        other_binary = [c for c in list(self.data.columns) if c not in unwanted]

        for sel in selected_binary:
            for other in other_binary:
                binary_name = sel + '_AND_' + other
                self.data[binary_name] = np.logical_and(self.data[sel], self.data[other]).astype(int)
                self.col_links[binary_name] = sel if consider_selected_base else other

    # TODO: display progress in percentage instead of just combination levels
    def construct_combs_up_to(self, comb_max_size: int) -> None:
        """ Binary feature combinations of sizes up to comb_max_size are constructed as binary features.
                These features equal 1 when all of its members equal 1.

            Note that plotting features generated by this method is not yet supported.

            Example:
                We have features
                - "x1", one of its values is "green"
                - "x2", one of its values is 10
                - "x3", one of its values is 120
                After conversion we will have binary features for these values that might look like:
                - "x1_green"
                - "x2_(-inf, 20]"
                - "x3_(-inf, 500]"
                If comb_max_size = 2 the following segments are created as well:
                - "x1_green_AND_x2_(-inf, 20]", which equals 1 when "x1" is "green" and "x2" <= 20
                - "x1_green_AND_x3_(-inf, 500]", <...>
                - "x2_(-inf, 20]_AND_x3_(-inf, 500]", <...>

        Parameters
        ----------
        comb_max_size : int

        Returns
        -------

        """
        if not self.is_data_converted:
            raise ValueError("Can only use construct_combs_up_to() when data is converted to binary format")

        _comb_max_size = int(comb_max_size)
        if _comb_max_size < 2:
            print("comb_max_size < 2, no features will be created")
            return
        elif _comb_max_size > 5:
            logging.warning(f'Using high comb_max_size ({comb_max_size}), calculations might take some time.')

        unwanted = {self.y_name, self.y_binary_name}
        binary_features = {c for c in list(self.data.columns) if c not in unwanted}

        for comb_curr_size in range(2, _comb_max_size+1):
            logger.info(f'Working on combinations of level {comb_curr_size} of {_comb_max_size}')
            binary_combs = combinations(binary_features, comb_curr_size)

            for comb in binary_combs:
                binary_name = '_AND_'.join(comb)
                self.data[binary_name] = np.logical_and.reduce([self.data[col] for col in comb]).astype(int)

                self.col_links[binary_name] = json.dumps(sorted(comb))
