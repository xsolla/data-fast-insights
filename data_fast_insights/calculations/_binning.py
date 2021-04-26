from typing import TYPE_CHECKING
import warnings

import scorecardpy as sc

if TYPE_CHECKING:
    from data_fast_insights import BinaryDependenceModelData


def make_bins(model_data: 'BinaryDependenceModelData', manual_breaks: dict = None) -> dict:
    """ Make bins for numeric variables of model_data, optimizing for Information Value
        based on created binary target

    Parameters
    ----------
    model_data
    manual_breaks
        Info about features that need to be separated into predefined intervals.
        If this argument is set,
            function won't calculate intervals for it and will use passed values as breaks instead.
        Format: {feature_name: [break1, break2]}

    Returns
    -------
    dict
        Info about bins, where keys are features, values are dataframes with data about bins
    """
    if not model_data.num_cols:
        warnings.warn('model_data.num_cols is not set')
        return dict()
    if model_data.is_data_converted:
        warnings.warn(
            "Features in model_data seem to be already converted to binary format, binning might be futile")
    dt = model_data.base_data[model_data.num_cols].join(model_data.data[model_data.y_binary_name])
    kwargs = {'dt': dt, 'y': model_data.y_binary_name}
    # TODO: manual breaks don't work exactly as expected. It there are no values in the interval,
    #  break would not be created
    if manual_breaks is not None and isinstance(manual_breaks, dict):
        kwargs['breaks_list'] = manual_breaks
    bins = sc.woebin(**kwargs)

    # Adjusting bins manually (rounding for representation)
    # TODO: add rounding but without rerunning binning;
    #  for now users can rerun binning with manual_breaks arg
    # breaks_adj = dict()
    # for c in bins.keys():
    #     for b in bins[c]['breaks'].tolist():
    #         if isinstance(b, str):
    #             continue
    #         breaks_adj[c] = round(float(b))
    # bins = sc.woebin(model_data.data[list(model_data.num_cols) + [model_data.y_binary_name]],
    #                  y=model_data.y_binary_name,
    #                  breaks_list=breaks_adj)
    return bins


def get_breaks(bins: dict) -> dict:
    breaks = {column: bins[column]['breaks'].tolist() for column in bins}
    return breaks
