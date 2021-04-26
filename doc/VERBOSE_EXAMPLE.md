Suppose there are several car models. We know some of the technical parameters and number of sales for each of these.  
And we want to know which car parameters drop the number of sales (and what should be changed in the production of cars).  
    
### Imports and settings
``` python
import pandas as pd
import matplotlib.pyplot as plt

from data_fast_insights import BinaryDependenceModelData
import data_fast_insights.calculations as calc
from data_fast_insights.plotting import (plot_segments_basic_info, plot_segments_central_tendency,
                                                  plot_segments_dependence)

```

### Setting the data
Note that data is synthetic and not necessarily reflects real world dependence. So do results.
```python
cars = pd.DataFrame()
cars['color'] = ['green', 'red', 'red', 'red', 'green', 'red']
cars['age'] = [10, 20, 30, 2, 5, 15]
cars['max_speed'] = [100, 60, 80, 110, float('nan'), 80]
cars['num_of_sales'] = [45, 50, 50, 101, 99, 65]
```
One row in the dataframe represents one model of the car. 
Our target variable is "num_of_sales"
### Creating the data object
**dmd** will be the object that holds all information about features and target in its attributes.  
You have to specify following arguments:
- **base_data** - source dataframe.  
- **cat_cols** - categorical features.
- **num_cols** - numeric features.
- **y_name** - target variable.  
:warning: **Higher values must represent better benefit, so inverse target column if needed.**
- y_type defines the pivot value by which target is separated:
    - "quantile" (separate target by a chosen quantile - add "y_quantile" argument, defaults to 0.5) 
    - "mean" (separate by mean)
    - "binary" (use target as is)  
    
**Rows with target lower than the chosen value are considered to worsen the target.**  

``` python
dmd = BinaryDependenceModelData(
    base_data=cars,
    cat_cols={'color'},
    num_cols={'age', 'max_speed'},
    y_type='quantile',
    y_quantile=0.5)
```
The object has several attributes, the following might be useful:
- base_data - source features and target data that are used for reference;
- data - current data (transforms on some method calls, will hold binary features)
- col_links - mapping between created (later) binary features and its sources;
- y_binary_name - name of the feature that separates target into "bad" and "good" rows.
So dmd.data\[dmd.y_binary_name\] is this data.  

**Note that most attributes must not be changed manually. For instance, changing indices in data or base_data
will break some functionality and you will get incorrect results.**
### Getting bins for numeric variables (optimizing for Information Value)  
``` python
num_bins = calc.make_bins(model_data=dmd)
```
If you need to see chosen breaks for intervals:
``` python
for k, v in calc.get_breaks(bins=num_bins).items():
    print(k, v)
```
_Output_:
``` 
age ['inf']
max_speed ['missing', '100.0', 'inf']
```
For age there were no intervals created.  
For max_speed:
- "missing" means there are NaNs in the data, and separate feature 
(whether max_speed is NaN) will be created
- other values are breaks, so other binary features from max_speed are:
    - "max_speed_[-inf,100.0)" - whether max_speed is lesser than 100.0,  
    - "max_speed_[100.0,inf)" - whether max_speed is greater than or equal to 100.0

### Converting variables to binary format
``` python
dmd.convert_to_binary(bins=num_bins)
print(dmd.data)
```

_Output_:
```
   num_of_sales  is_num_of_sales_lt_quantile_0.5  color_green  color_red  \
0            45                             True            1          0   
1            50                             True            0          1   
2            50                             True            0          1   
3           101                            False            0          1   
4            99                            False            1          0   
5            65                            False            0          1   

   age_[-inf,inf)  max_speed_missing  max_speed_[-inf,100.0)  \
0               1                  0                       0   
1               1                  0                       1   
2               1                  0                       1   
3               1                  0                       0   
4               1                  1                       0   
5               1                  0                       1   
```
Now dmd.data contains binary features with name "\<base_column\>_\<suffix\>", 
where <suffix> is:
- name of the category, if the feature is categorical. E.g.: "color_green", it equals 1 where color value is "green";
- interval \[a, b), if the feature is numeric. E.g.: "max_speed_[-inf,100.0)", 
    it equals 1 where max_speed is strictly less than 100.0;
- "\_nan" for NaNs in categorical features.
- "\_missing" for NaNs in numeric features.  

### Calculating dependence  
``` python
res = calc.calculate_dependence(model_data=dmd)
print(res)
```
_Output_:
```
                         total_sum  low_sum   low_perc   high_perc  \
max_speed_[-inf,100.0)        3.0        2  66.666667   33.333333   
age_[-inf,inf)                6.0        3  50.000000   50.000000   
color_red                     4.0        2  50.000000   50.000000   
color_green                   2.0        1  50.000000   50.000000   
max_speed_[100.0,inf)         2.0        1  50.000000   50.000000   
max_speed_missing             1.0        0   0.000000  100.000000   

                        perc_of_total  target_delta_perc   base_col  \
max_speed_[-inf,100.0)      50.000000         -19.512195  max_speed   
age_[-inf,inf)             100.000000           0.000000        age   
color_red                   66.666667          -2.682927      color   
color_green                 33.333333           5.365854      color   
max_speed_[100.0,inf)       33.333333           6.829268  max_speed   
max_speed_missing           16.666667          44.878049  max_speed   

                                  base_breaks     base_range     base_cats  
max_speed_[-inf,100.0)  [missing, 100.0, inf]  [60.0, 110.0]                
age_[-inf,inf)                          [inf]        [2, 30]                
color_red                                                     [green, red]  
color_green                                                   [green, red]  
max_speed_[100.0,inf)   [missing, 100.0, inf]  [60.0, 110.0]                
max_speed_missing       [missing, 100.0, inf]  [60.0, 110.0]               
```
#### Output description: 
Index is the name of a binary variable (segment).  
Columns:
- total_sum - sum of the binary feature values (size of the segment, absolute)
- low_sum - sum of the binary feature values where binary target equals 1
    (how much segment entries are lower than the selected
                target threshold (e.g. median or mean))
- low_perc - (low_sum / total_sum) * 100 (how bad the segment is, in percent).  
    It represents the share of objects (rows) that are lower than the selected
                target threshold (e.g. median or mean) of all segment objects.  
- high_perc - (100 - low_perc).
- perc_of_total - segment share of total data, in percent (size of the segment, relative)
    Total "perc_of_total" of all segments across one base (original) feature equals 1.
- target_delta_perc - how much target mean of this segment differs from total target mean, in percent
- base_col - parent feature for segment.  
    If the binary feature is a combination of multiple binary features, it contains json array of parent binary features
- base_breaks - chosen breaks of intervals of the parent feature (if parent feature is numeric)
- base_range - min and max values of the parent feature (if parent feature is numeric)
- base_cats - all possible categories of the parent feature (if parent feature is categorical)  
> :warning: Be careful with rows for which "perc_of_total" is low, these results are statistically unstable
 
#### Interpretation
The segment with highest "low_perc" value (66%) is "max_speed_[-inf,100.0)".  
This means 66% of cars having max_speed lower than 100.0 
are below the chosen threshold of the target column (num_of_sales), which is:
```python
print(f'y threshold: {dmd.y_pivot}')
```
_Output_:
```
y threshold: 57.5
```

Observations:
- people from the synthetic dataset would rather buy a fast car than a slower one;
- since segment "max_speed_[-inf,100.0)" has the highest impact on dropping the number of sales,
    it's recommended to make changes to its base feature ("max_speed") sooner than other changes.  
  **In short, making fast cars should bring more sales in this example**
 
Other information about this segment:
- "total_sum" being equal to 3 is very low for real world data
 and such segments shouldn't be analyzed in real datasets (because they are unrepresentative/unstable)
- "perc_of_total" equals 50.0, which means other 50% of cars have "max_speed" >= 100.0
- "target_delta_perc" equals to -19.512195, which means average number of sales in this segment is
19.512195% lower than average number of sales in the whole dataset



### Seeing what should happen if we change some segment (by substitution of other segments of this feature)
```python
comparison_example = calc.compare_intervals(selected='color_green', model_data=dmd)
print(comparison_example)
```  
_Output_:
```
       old_col old_mode old_base_mode    new_col new_mode new_base_mode  \
0  color_green    green           red  color_red      red           red   

   total_target_change_perc  
0                 -2.682927  
```
#### Output description 
Columns:  
(note: \<metric_name\> is a metric describing the column, e.g. 'mode' for categorical, 'mean' for numeric)
- old_col - segment being compared (current segment)
- old_\<metric_name\> - metric of the current segment (e.g. for 'trial_period_\[-inf, 1) )
- old_base_\<metric_name\> - metric of the base feature of the current segment (e.g. for 'trial_period')
- new_col - segment that old_col is being compared to (new segment)
- new_\<metric_name\> - metric of the new segment
- new_base_\<metric_name\> - metric of the parent feature of the new segment  
- total_target_change_perc - how much this substitution changes total sum of target value (on all data), in percent.  
E.g. in this case: 
```
# target sum as is
>>> print(cars['num_of_sales'].sum())
410

# expected target sum when color changed from "green" to "red" 
>>> print(cars['num_of_sales'].sum() * (1 - 0.02682927))
398.9999993
```

#### Interpretation 
Changing color of green cars to red would decrease total target value 
(on whole dataset) by 2.682927%