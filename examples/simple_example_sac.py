# split-apply-combine example. See doc/OTHER_FEATURES.md "Experimental features" section

import pandas as pd
import matplotlib.pyplot as plt

from data_fast_insights.experimental import SplitApplyCombineModelData
import data_fast_insights.calculations as calc
from data_fast_insights.plotting import plot_segments_basic_info

df_prep = pd.DataFrame()
df_prep['color'] = ['green', 'red', 'red', 'red', 'green', 'red']
df_prep['age'] = [10, 20, 30, 2, 5, 15]
df_prep['max_speed'] = [100, 60, 80, 110, float('nan'), 80]
df_prep['num_of_sales'] = [45, 50, 50, 101, 99, 65]
df_prep['year_of_sale'] = [2000, 2000, 2000, 2002, 2003, 2002]

sac_base = SplitApplyCombineModelData(
    total_data=df_prep,
    cat_cols={'color'},
    num_cols={'age', 'max_speed', 'year_of_sale'},
    y_name='num_of_sales',
    y_type='quantile')

sac_base.global_num_bins = calc.make_bins(model_data=sac_base)
sac_base.convert_to_binary(bins=sac_base.global_num_bins)

res_base = calc.calculate_dependence(model_data=sac_base)


sac_base.split('year_of_sale')

for year, part in sac_base.splitted.items():
    if part['data']['max_speed'].var() == 0:
        print(f'year {year} will be skipped')
        sac_base.splitted[year]['use_for_report'] = False

sac_base.multiple_singular_experiments()
sac_base.filter_transpose_results()
sac_base.fill_defaults()
sac_base.reduce()

print(sac_base.total_res)

f = plot_segments_basic_info(sac_base, sac_base.total_res, 'max_speed',
                             base_feature_rename='Maximum Speed')
plt.show()
