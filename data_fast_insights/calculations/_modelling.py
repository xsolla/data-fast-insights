from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from data_fast_insights import utils


if TYPE_CHECKING:
    from data_fast_insights import BinaryDependenceModelData


def calculate_dependence(model_data: 'BinaryDependenceModelData') -> pd.DataFrame:
    """ Calculate dependence on target for features in model_data

    Parameters
    ----------
    model_data

    Returns
    -------
    pd.DataFrame
        DataFrame with data about dependence between features and target
        Columns description:
            total_sum - sum of the binary feature values (size of the segment, absolute)
            low_sum - sum of the binary feature values where binary target equals 1
                (how much segment entries are lower than the selected
                target threshold (e.g. median or mean))
            low_perc - (low_sum / total_sum) * 100 (how bad the segment is, in percent).
                It represents the share of objects (rows) that are lower than the selected
                target threshold (e.g. median or mean) of all segment objects.
            high_perc - (100 - low_perc).
            perc_of_total - segment share of total data, in percent (size of the segment, relative).
                Total "perc_of_total" of all segments across one base (original) feature equals 1.
            target_delta_perc - how much target mean of this segment differs from total target mean, in percent
            base_col - parent feature for segment.
                If the binary feature is a combination of multiple binary features,
                it contains json array of parent binary features
            base_breaks - chosen breaks of intervals of the parent feature (if parent feature is numeric)
            base_range - min and max values of the parent feature (if parent feature is numeric)
            base_cats - all possible categories of the parent feature (if parent feature is categorical)
    """
    # if not model_data.is_data_converted:
    #     warnings.warn("""Features in model_data seem to not be converted to binary format yet,
    #     calculate_dependence() might return wrong output.
    #     """)
    df_features = model_data.data.drop(model_data.y_name, 1)
    res_total = pd.DataFrame(
        df_features.sum(axis=0), columns=['total_sum']).sort_values(by='total_sum', ascending=False)

    df_low = df_features[df_features[model_data.y_binary_name] == 1].drop(model_data.y_binary_name, 1).copy()
    res_low = pd.DataFrame(df_low.sum(axis=0), columns=['low_sum'])
    res_low = pd.merge(res_total, res_low, left_index=True, right_index=True)
    res_low['low_perc'] = (res_low['low_sum'] / res_low['total_sum']) * 100
    res_low['high_perc'] = 100 - res_low['low_perc']
    res_low['perc_of_total'] = (res_low['total_sum'] / model_data.data.shape[0]) * 100
    res_low['target_delta_perc'] = np.nan

    res_low['base_col'] = ''
    res_low['base_breaks'] = ''
    res_low['base_range'] = ''
    # res_low['base_central_value'] = np.nan
    # res_low['base_central_value'] = res_low['base_central_value'].astype(object)
    # res_low['base_min'] = np.nan
    # res_low['base_max'] = np.nan
    res_low['base_cats'] = ''

    # TODO: change from .iterrows() to faster type of iterations (e.g. zip() on series?)
    for i, row in res_low.iterrows():
        res_low.at[i, 'target_delta_perc'] = ((model_data.data[model_data.data[i] == 1][model_data.y_name].mean() /
                                              model_data.data[model_data.y_name].mean()) - 1) * 100
        if i in model_data.col_links:
            base_col = model_data.col_links[i]
            res_low.at[i, 'base_col'] = base_col
            # res_low.at[i, 'base_central_value'] = model_data.base_data[base_col].__getattribute__(
            #     utils.choose_central_tendency_metric(base_col, model_data))()
            if base_col in model_data.num_cols:
                res_low.at[i, 'base_range'] = str([model_data.base_data[base_col].min(),
                                                   model_data.base_data[base_col].max()])
                res_low.at[i, 'base_breaks'] = model_data.bins[base_col]['breaks'].tolist()
            elif base_col in model_data.cat_cols:
                res_low.at[i, 'base_cats'] = model_data.base_data[base_col].unique()
    res_low = res_low.sort_values(by='low_perc', ascending=False)
    return res_low


def compare_intervals(selected: str, model_data: 'BinaryDependenceModelData') -> pd.DataFrame:
    """ Compare how changing certain values to other interval of same feature would affect the target.

        Supposing model_data.data and model_data.base_data have same points on same indices
        (which unless these attributes are modified manually is true)

    Parameters
    ----------
    selected
        Segment (binary feature name) that needs to be compared to other segments
    model_data

    Returns
    -------
    pd.DataFrame
        Results of comparison
        Columns description:
        (note: <metric_name> is a metric describing the column, e.g. 'mode' for categorical, 'mean' for numeric)
            old_col - segment being compared (current segment)
            old_<metric_name> - metric of the current segment (e.g. for 'trial_period_[-inf, 1)
            old_base_<metric_name> - metric of the base feature of the current segment (e.g. for 'trial_period')
            new_col - segment that old_col is being compared to (new segment)
            new_<metric_name> - metric of the new segment
            new_base_<metric_name> - metric of the parent feature of the new segment
            total_target_change_perc - how much this substitution changes total target value (on all data), in percent
    """
    # TODO: make it so model_data.data and model_data.base_data don't have to have same points on same indices
    #  or make it explicit.

    if selected not in model_data.data.columns:
        raise ValueError(f"'{selected}' feature not found in model_data.data;"
                         + " make sure you pass a binary segment name, not the original feature name")

    sel_interval_indices = model_data.data.loc[model_data.data[selected] == 1].index
    base_col = model_data.col_links[selected]
    comparison = [binary for binary, base in model_data.col_links.items() if base == base_col and binary != selected]

    # pd_metrics_attr must be a name of a pd.DataFrame method calculating some metric.
    pd_metrics_attr = utils.choose_central_tendency_metric(base_col, model_data)
    comp_df = pd.DataFrame(columns=['old_col',
                                    'old_' + pd_metrics_attr,
                                    'old_base_' + pd_metrics_attr,
                                    'new_col',
                                    'new_' + pd_metrics_attr,
                                    'new_base_' + pd_metrics_attr,
                                    'total_target_change_perc'])
    current_comp_data = {'old_col': selected,
                         'old_' + pd_metrics_attr: model_data.base_data.loc[sel_interval_indices, :][
                             base_col].__getattribute__(pd_metrics_attr)(),
                         'old_base_' + pd_metrics_attr:
                             model_data.base_data[base_col].__getattribute__(pd_metrics_attr)()}

    for index, compare_to in enumerate(comparison):
        comp_interval_indices = model_data.data.loc[model_data.data[compare_to] == 1].index

        df_int_tmp = model_data.data.copy()
        df_int_tmp.loc[sel_interval_indices,
                       [model_data.y_name]] = df_int_tmp.loc[comp_interval_indices, :][model_data.y_name].mean()

        df_base_tmp = model_data.base_data.copy()

        df_base_tmp.loc[sel_interval_indices, [base_col]] = df_base_tmp.loc[comp_interval_indices, :][
            base_col].__getattribute__(pd_metrics_attr)()
        total_target_increase = (df_int_tmp[model_data.y_name].sum() / model_data.data[model_data.y_name].sum() - 1)

        current_comp_data['new_col'] = compare_to
        current_comp_data['new_' + pd_metrics_attr] = model_data.base_data.loc[comp_interval_indices, :][
            base_col].__getattribute__(pd_metrics_attr)()
        current_comp_data['new_base_' + pd_metrics_attr] = df_base_tmp[base_col].__getattribute__(pd_metrics_attr)()

        current_comp_data['total_target_change_perc'] = total_target_increase * 100
        comp_df = comp_df.append(pd.DataFrame(current_comp_data, index=[index]))
    return comp_df
