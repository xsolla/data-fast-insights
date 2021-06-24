from copy import deepcopy
from functools import reduce
import logging

import pandas as pd

from data_fast_insights import BinaryDependenceModelData
from data_fast_insights.utils import exclude_zero_var, singular_experiment
from data_fast_insights.calculations import calculate_dependence


class SplitApplyCombineModelData(BinaryDependenceModelData):
    # TODO: add checks on every step that data is ready?
    def __init__(self, total_data, y_name, cat_cols, num_cols, y_type, **kwargs):
        super().__init__(total_data, y_name, cat_cols, num_cols, y_type, **kwargs)

        self.splitted = dict()
        self.global_num_bins = None
        self.exp_data = dict()
        self.exp_data_reports = dict()

        self.all_features = None
        self.cnt_excluded_by_feat = None

        self.total_res = None

        self.default_calc = calculate_dependence(None)

    def split(self, dim_name: str):
        for p in self.base_data[dim_name].unique():
            self.splitted[p] = {'data': self.base_data[self.base_data[dim_name] == p], 'use_for_report': True}

    def multiple_singular_experiments(self, **kwargs):
        for p, part in self.splitted.items():

            # excluding zero variance feats
            res = exclude_zero_var(part['data'], self.num_cols, self.cat_cols)
            num_cols_new = res['num_cols']
            cat_cols_new = res['cat_cols']

            exp = singular_experiment(
                y_name=self.y_name, part_data=res['df'].copy(), num_feats=num_cols_new, cat_feats=cat_cols_new,
                num_bins=self.global_num_bins, **kwargs)
            if not exp:
                logging.warning(f"Singular experiment on {p} returned empty dict, skipping this part")
                exp['use_for_report'] = False
            else:
                exp['use_for_report'] = part['use_for_report']
            self.exp_data[p] = deepcopy(exp)

        self.exp_data_reports = {k: v for k, v in self.exp_data.items() if v['use_for_report']}

    def filter_transpose_results(self, params_thresholds: dict = None):
        if params_thresholds is None:
            params_thresholds = dict()

        for p, data in self.exp_data_reports.items():
            tmp_res = data['res']

            for param, value in params_thresholds.items():
                tmp_res = tmp_res[tmp_res[param] > value]

            self.exp_data_reports[p]['res'] = tmp_res.T

        self.all_features = reduce(
            lambda x, y: set(x) | set(y), [r['res'].columns for r in self.exp_data_reports.values()])

    def fill_defaults(self):
        self.cnt_excluded_by_feat = {f: 0 for f in self.all_features}
        for p, data in self.exp_data_reports.items():
            merging = data['res'].copy()

            for diff in set(self.all_features).difference(set(merging.columns)):
                # TODO inplace renaming
                default_ = self.default_calc
                default_.columns = [diff]

                merging = pd.merge(merging, default_, left_index=True, right_index=True)
                self.cnt_excluded_by_feat[diff] += 1
            self.exp_data_reports[p]['res'] = merging

    def reduce(self):
        total_res = reduce(
            lambda x, y: x + y,
            [r['res'].drop(
                ['base_col', 'base_breaks', 'base_range', 'base_cats'],
                0) for r in self.exp_data_reports.values()])

        for f in total_res.columns:
            """Since we have added zeros for features that are missing,
            we must only divide by amount of experiments in which the feature is present
            """
            total_res[f] /= (len(self.exp_data_reports) - self.cnt_excluded_by_feat[f])
        total_res = total_res.T

        # Adding number of experiments feature was in
        total_res = pd.merge(total_res,
                             pd.DataFrame.from_dict(
                                 self.cnt_excluded_by_feat, orient='index', columns=['number_of_experiments']),
                             left_index=True,
                             right_index=True)
        total_res['number_of_experiments'] = len(self.exp_data_reports) - total_res['number_of_experiments']

        # fixing base_col
        for i, row in total_res.iterrows():
            if i in self.col_links:
                total_res.at[i, 'base_col'] = self.col_links[i]

        self.total_res = total_res
