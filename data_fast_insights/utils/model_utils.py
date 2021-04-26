import logging

from data_fast_insights import BinaryDependenceModelData
import data_fast_insights.calculations as calc


def exclude_zero_var(df, num_cols, cat_cols):
    exclude_nums = list()
    exclude_cats = list()
    for c in df.columns:
        if c in num_cols and (df[c].nunique() < 2 or df[c].var() == 0.0):
            exclude_nums.append(c)
        elif c in cat_cols and df[c].nunique() < 2:
            exclude_cats.append(c)

    new_df = df.drop(exclude_nums, 1).drop(exclude_cats, 1)
    num_feats_new = num_cols.difference(set(exclude_nums))
    cat_feats_new = cat_cols.difference(set(exclude_cats))
    return {'df': new_df, 'num_cols': num_feats_new, 'cat_cols': cat_feats_new}


def singular_experiment(part_data, cat_feats=None, num_feats=None, y_name=None, num_bins=None, **kwargs):
    dmd = BinaryDependenceModelData(
        base_data=part_data.copy(),
        cat_cols=cat_feats,
        num_cols=num_feats,
        y_name=y_name,
        **kwargs)

    if dmd.data[dmd.y_binary_name].nunique() != 2:
        logging.warning('Skipping current experiment, number of distinctive target values is not equal 2')
        return dict()

    if num_bins is None:
        num_bins = calc.make_bins(model_data=dmd)
    dmd.convert_to_binary(bins=num_bins)
    res = calc.calculate_dependence(model_data=dmd)

    return {'data': dmd, 'res': res, 'num_bins': num_bins}
