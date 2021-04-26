### 0.2 - 21 Apr 2021
* Overall documentation enrichment
* Interpretability improvements 
* Overall minor fixes

### 0.1 - 21 Oct 2020
* Added split-apply-combine type experiment

### 0.0.9
* Added construct_partial_combs()

### 0.0.8
* Improvements and fixes in plotting:
    * Now plot_segments_central_tendency() can plot for any Y, including target
    * Now plot_segments_dependence() supports custom parameters plots
    * Add different literals for showcasing experiment parameters

* Fixes in experiment output: 
    * Add 'high_perc' parameter to experiment output as being easier interpretable than 'low_perc'

* Other
    * Add 'base_col' values for combined features, this is for future works on combined features

### 0.0.7
* Add more plotting:
    * plot_segments_central_tendency()
    * plot_segments_basic_info() that plots all basic info from other plotting methods

### 0.0.6
* Added plotting: 
    * plot_segments_dependence():
        ```python
        plot_segments_dependence(model_data=dmd, res_low_df=res, base_feature_name='x_cat')
        ```

### 0.0.5
 * Added dependence of combinations of features:   
    set comb_max_size in convert_to_binary() to add combinations of up to such size:
    ```python
    dmd.convert_to_binary(bins=num_bins, comb_max_size=2)
    ```