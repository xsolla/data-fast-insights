import pandas as pd
import matplotlib.pyplot as plt

from data_fast_insights import BinaryDependenceModelData
import data_fast_insights.calculations as calc
from data_fast_insights.plotting import (plot_segments_basic_info, plot_segments_central_tendency,
                                         plot_segments_dependence)

""" Suppose we want to know which car parameters drop the number of sales.
    Note that data is synthetic and not necessarily reflects real world dependence. So do results.
"""

cars = pd.DataFrame()
cars['color'] = ['green', 'red', 'red', 'red', 'green', 'red']
cars['age'] = [10, 20, 30, 2, 5, 15]
cars['max_speed'] = [100, 60, 80, 110, float('nan'), 80]
cars['num_of_sales'] = [45, 50, 50, 101, 99, 65]
cars['year_of_sale'] = [2000, 2000, 2000, 2000, 2000, 2000]

# Initializing model data
dmd = BinaryDependenceModelData(
    base_data=cars,
    cat_cols={'color'},
    num_cols={'age', 'max_speed', 'year_of_sale'},
    y_name='num_of_sales',
    y_type='quantile',
    y_quantile=0.5)


# Getting bins for numeric variables, optimizing for Information Value
# You can readjust bins by calling make_bins again with manual_breaks parameter: {'column': [break1, break2, ...]}
num_bins = calc.make_bins(model_data=dmd)
# Show chosen breaks of intervals
for k, v in calc.get_breaks(bins=num_bins).items():
    print(k, v)

# Converting variables
dmd.convert_to_binary(bins=num_bins)
# dmd.construct_combs_up_to(2)
dmd.construct_partial_combs("color")
print(dmd.data)

# Calculating dependence
res = calc.calculate_dependence(model_data=dmd)
print('10 Most influential segments:\n', res[:10])

print(f'y threshold: {dmd.y_pivot}')

# Seeing what should happen if we change some feature (by substitution of other interval)
comparison_example = calc.compare_intervals(selected='color_green', model_data=dmd)
print(comparison_example)


comparison_example = calc.compare_intervals(selected='max_speed_[-inf,100.0)', model_data=dmd)
print(comparison_example)


# PLOTTING
"""
Note that the coefficients for legend placement is adjusted for Jupyter Notebook.
There might be problems with plotting in different frontends 
"""

plot_segments_basic_info(model_data=dmd, res_low_df=res, base_feature_name='color',
                         base_feature_rename='Color', unit_name='km/h', resort='target_delta_perc')

OTHER_PLOTS = True

if OTHER_PLOTS:
    pass
    # SINGLE PLOT EXAMPLE
    fig, _ = plt.subplots()
    plot_segments_dependence(dmd, res, 'color_green', 'high_perc', base_feature_rename='Color: Green',
                             resort='name_asc')
    fig.tight_layout()

    # PLOTTING WITH CUSTOM ADDITIONS
    fig, _ = plt.subplots(2, sharex=True)
    axs = fig.axes
    plt.sca(axs[0])
    # plot share of objects that are higher than target threshold
    plot_segments_dependence(
        model_data=dmd, res_low_df=res, base_feature_name='max_speed', param_name='high_perc',
        plot_mean=False, ax=axs[0])
    # plot mean of the feature instead of the target mean
    plot_segments_central_tendency(
        model_data=dmd, base_feature_name='max_speed', y_name='max_speed', base_feature_rename='Maximum Speed',
        ax=axs[1])

    for ax in axs:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.0,
                         box.width, box.height * 0.8])
    axs[0].legend(fancybox=True, framealpha=1, loc='upper right', bbox_to_anchor=(1, 1.70))
    axs[1].legend(fancybox=True, framealpha=1, loc='upper right', bbox_to_anchor=(1, 1.40))

    _, _ = plt.subplots()
    plot_segments_dependence(dmd, res, 'max_speed', 'target_delta_perc')

plt.show()  # not required in Notebooks
