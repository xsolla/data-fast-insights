from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from data_fast_insights import utils
from data_fast_insights.resources.literals_mapping import SHOWCASE_LITERALS_MAPPING

if TYPE_CHECKING:
    from data_fast_insights import BinaryDependenceModelData


"""
Note that the coefficients for legend placement is adjusted for Jupyter Notebook.
There might be problems with plotting in different frontends 
"""
# TODO: custom sorts (see resort argument in functions)


def plot_segments_dependence(model_data: 'BinaryDependenceModelData',
                             res_low_df: pd.DataFrame,
                             base_feature_name: str,
                             param_name: str,
                             base_feature_rename: str = None,
                             plot_mean=True,
                             unit_name: str = None,
                             resort: str = 'default',
                             ax=None):
    """

    Parameters
    ----------
    model_data
    res_low_df: pd.DataFrame
        Resulting dataframe of the experiment - output of .calculations.calculate_dependence()
    base_feature_name: str
    base_feature_rename: str, optional
        What to rename base feature to on the plot
    param_name: str
        Which resulting param (column) of res_low_df to plot
    plot_mean : bool, optional (default True)
        Whether to plot line of mean value or not.
    unit_name: str, optional
        Unit name to add to interval name
    resort: str, optional (default "default")
        If "default":
            for numeric features: sort by interval values, ascending (preserves order after binning)
            for categorical features: do not sort
        If "name_asc":
            sort by interval name, ascending
        If "name_desc":
            sort by interval name, descending
        If other string:
            sort by this column values of res_low_df, ascending
    ax: matplotlib ax, optional

    Returns
    -------
    fig: matplotlib figure
    """
    if param_name in SHOWCASE_LITERALS_MAPPING:
        # kwargs = {'target_name': f'$\\bf{{{model_data.y_name}}}$'}
        kwargs = {'target_name': model_data.y_name}
        showcase_param_name = SHOWCASE_LITERALS_MAPPING[param_name](kwargs)
    else:
        showcase_param_name = param_name

    # fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    series = res_low_df[res_low_df['base_col'] == base_feature_name][param_name].copy()

    # resorting
    feat_names = [k for k, v in model_data.col_links.items() if v == base_feature_name]
    feat_names = utils.resort_binary_names(feat_names=feat_names, by=resort, base_feature_name=base_feature_name,
                                           res_low_df=res_low_df)

    series = pd.Series({k: series.loc[k] for k in feat_names})

    series.name = showcase_param_name
    is_numeric = base_feature_name in model_data.num_cols
    series.index = [utils.get_segment_name_ready_for_plot(is_numeric, base_feature_name, segment, unit_name)
                    for segment in series.index]

    if base_feature_rename:
        ax.set_xlabel(base_feature_rename)
    else:
        ax.set_xlabel(base_feature_name)
    ax.set_ylabel(showcase_param_name)
    if plot_mean:
        _ = ax.plot(series.index,
                    [series.mean() for _ in range(len(series.index))],
                    label=f'{showcase_param_name} mean',
                    color='tab:red'
                    )
    if any(o < 0 for o in series) and any(o > 0 for o in series):
        ax.axhline(y=0.0, color='black', linestyle='-')

    if param_name == 'high_perc':
        _ = series.plot(ax=ax, kind='bar', legend=True, color='c')
    else:
        _ = series.plot(ax=ax, kind='bar', legend=True)

    ax.legend(fancybox=True, framealpha=1, loc='upper right', bbox_to_anchor=(1, 1.25))
    plt.sca(ax)
    plt.xticks(rotation=10)
    plt.subplots_adjust(hspace=0.1, bottom=0.2)
    ax.xaxis.labelpad = 20

    # source: https://stackoverflow.com/a/4701285
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.0,
                     box.width, box.height * 0.9])
    ax.get_yaxis().get_label().set_visible(False)

    # plt.close(fig)
    # return fig
    return plt.gcf()


def plot_segments_central_tendency(model_data: 'BinaryDependenceModelData',
                                   base_feature_name: str,
                                   y_name: str,
                                   base_feature_rename: str = None,
                                   unit_name: str = None,
                                   resort: str = 'default',
                                   res_low_df: pd.DataFrame = None,
                                   ax=None):
    """

    Parameters
    ----------
    model_data
    base_feature_name: str
    y_name: str
        What is the Y values for plot - target or feature itself (or any other feature).
    base_feature_rename: str, optional
        What to rename base feature to on the plot
    unit_name: str, optional
        Unit name to add to interval name
    resort: str, optional (default "default")
        If "default":
            for numeric features: sort by interval values, ascending (preserves order after binning)
            for categorical features: do not sort
        If "name_asc":
            sort by interval name, ascending
        If "name_desc":
            sort by interval name, descending
        If other string:
            sort by this column values of res_low_df, ascending. Requires res_low_df argument.
    res_low_df: pd.DataFrame, optional
        Resulting dataframe of the experiment - output of .calculations.calculate_dependence()
        Only required for some sorting methods (see resort argument)
    ax: matplotlib ax, optional

    Returns
    -------
    fig: matplotlib figure
    """
    # fig = plt.figure()
    if ax is None:
        ax = plt.gca()

    pd_metrics_attr = utils.choose_central_tendency_metric(y_name, model_data)

    if base_feature_rename:
        ax.set_xlabel(base_feature_rename)
    else:
        ax.set_xlabel(base_feature_name)
    ax.set_ylabel(y_name + ' ' + pd_metrics_attr)

    # resorting
    feat_names = [k for k, v in model_data.col_links.items() if v == base_feature_name]
    feat_names = utils.resort_binary_names(feat_names=feat_names, by=resort, base_feature_name=base_feature_name,
                                           res_low_df=res_low_df)
    segments = feat_names.copy()

    plot_data = {'x_tick': list(), 'x': list(), 'y': list()}
    is_numeric = base_feature_name in model_data.num_cols
    for i, segment in enumerate(segments):
        sel_interval_indices = model_data.data.loc[model_data.data[segment] == 1].index

        tick = utils.get_segment_name_ready_for_plot(is_numeric, base_feature_name, segment, unit_name)
        plot_data['x_tick'].append(tick)
        plot_data['x'].append(i)
        plot_data['y'].append(
            model_data.base_data.loc[sel_interval_indices, :][y_name].__getattribute__(pd_metrics_attr)())

    if pd_metrics_attr == 'mode' and isinstance(plot_data['y'], Iterable):
        raise NotImplementedError(
            "plot_segments_central_tendency is not implemented for mode values / categorical features")

    elif pd_metrics_attr == 'mean':
        ax.plot(
            plot_data['x'], plot_data['y'], label=y_name + ' ' + pd_metrics_attr, linestyle='--', marker='o', color='b')
    else:
        raise ValueError(f'Unknown pd_metrics_attr: {pd_metrics_attr}.' +
                         'Check utils.choose_central_tendency_metric for allowed metrics')

    plt.sca(ax)
    plt.xticks(ticks=plot_data['x'], labels=plot_data['x_tick'], rotation=10)
    ax.legend(fancybox=True, framealpha=1, loc='upper right', bbox_to_anchor=(1, 1.25))
    ax.xaxis.labelpad = 20
    plt.subplots_adjust(hspace=0.1, bottom=0.2)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.0,
                     box.width, box.height * 0.9])
    ax.get_yaxis().get_label().set_visible(False)
    # plt.close(fig)
    # return fig
    return plt.gcf()


def plot_segments_basic_info(model_data: 'BinaryDependenceModelData',
                             res_low_df: pd.DataFrame,
                             base_feature_name: str,
                             base_feature_rename: str = None,
                             size_plot_mean: bool = True,
                             high_perc_plot_mean: bool = False,
                             unit_name: str = None,
                             resort: str = 'default'):
    """ Plot basic info about segments of base_feature_name

    Parameters
    ----------
    model_data
    res_low_df
        Resulting dataframe of calculations.calculate_dependence()
    base_feature_name
    base_feature_rename: str, optional
        What to rename base feature to on the plot
    size_plot_mean : bool, optional (default True)
        Whether to plot line of mean value on segments size plot.
    high_perc_plot_mean : bool, optional (default False)
        Whether to plot line of mean value on segments quality plot.
    unit_name:
        Unit name to add to interval name
    resort: str, optional (default "default")
        If "default":
            for numeric features: sort by interval values, ascending (preserves order after binning)
            for categorical features: do not sort
        If "name_asc":
            sort by interval name, ascending
        If "name_desc":
            sort by interval name, descending
        If other string:
            sort by this column values of res_low_df, ascending

    Returns
    -------
    fig: matplotlib figure
    """
    fig, _ = plt.subplots(3, figsize=(10, 5.5), sharex=True)
    axs = fig.axes
    plt.sca(axs[1])

    if base_feature_rename:
        final_base_feature_name = base_feature_rename
    else:
        final_base_feature_name = base_feature_name

    plot_segments_dependence(model_data=model_data,
                             res_low_df=res_low_df,
                             base_feature_name=base_feature_name,
                             param_name='perc_of_total',
                             unit_name=unit_name,
                             resort=resort,
                             ax=axs[0],
                             plot_mean=size_plot_mean)
    plot_segments_dependence(model_data=model_data,
                             res_low_df=res_low_df,
                             base_feature_name=base_feature_name,
                             param_name='high_perc',
                             unit_name=unit_name,
                             resort=resort,
                             ax=axs[1],
                             plot_mean=high_perc_plot_mean)
    plot_segments_central_tendency(base_feature_name=base_feature_name,
                                   model_data=model_data,
                                   y_name=model_data.y_name,
                                   unit_name=unit_name,
                                   resort=resort,
                                   res_low_df=res_low_df,
                                   ax=axs[2])
    fig.suptitle(f"{final_base_feature_name}", fontsize=14)
    axs[0].set_title(f"{final_base_feature_name}: Size of Segments, %", pad=7, size=10.5)
    axs[1].set_title(f"{final_base_feature_name}: Share of Objects Better than the Total Mean, by segment, %", pad=7,
                     size=10.5)
    axs[2].set_title(f"Average {model_data.y_name} by {final_base_feature_name} Segments", pad=7, size=10.5)
    axs[2].set_xlabel(final_base_feature_name)

    # for ax in axs:
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0 + box.height * 0.0,
    #                      box.width, box.height * 0.9])
    # axs[0].legend(fancybox=True, framealpha=1, loc='upper right', bbox_to_anchor=(1, 1.44), prop={'size': 9})
    # axs[1].legend(fancybox=True, framealpha=1, loc='upper right', bbox_to_anchor=(1, 1.28), prop={'size': 9})
    # axs[2].legend(fancybox=True, framealpha=1, loc='upper right', bbox_to_anchor=(1, 1.28), prop={'size': 9})

    for ax in axs:
        ax.tick_params(axis='x', which='minor', labelsize=9)
        ax.tick_params(axis='x', which='major', labelsize=9)

        ax.get_legend().remove()

    # text_coords = (0.01, 0.55)
    # # base_feature_name_ = base_feature_name.replace('_', '\_')
    # plt.text(*text_coords,
    #          # f'Data is divided into segments\n by $\\bf{base_feature_name_}$ values',
    #          f'Data is divided into segments\n by feature value ranges',
    #          ha='left',
    #          va='top',
    #          transform=fig.transFigure)
    # a1 = patches.FancyArrowPatch((text_coords[0] + 0.1, text_coords[1] - 0.08),
    #                              (0.33, 0.16),
    #                              connectionstyle="arc3,rad=.5",
    #                              arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
    #                              color="k",
    #                              transform=fig.transFigure)
    # fig.patches.extend([a1])

    # plt.subplots_adjust(hspace=0.36, left=0.33)
    plt.subplots_adjust(hspace=0.36)

    # plt.close(fig)
    # return fig
    # zoom = 2
    # w, h = fig.get_size_inches()
    # fig.set_size_inches(w * zoom, h * zoom)
    return fig
