import logging


def resort_binary_names(feat_names, by, base_feature_name, res_low_df):
    if by == 'default':
        # preserve order of intervals after binning (for numeric) and categories (for categorical)
        pass
    elif by == 'name_asc':
        feat_names = sorted(feat_names, reverse=False)
    elif by == 'name_desc':
        feat_names = sorted(feat_names, reverse=True)
    elif by in res_low_df.columns:
        feat_names = res_low_df[res_low_df['base_col'] == base_feature_name][[by]].sort_values(by=by).index
    else:
        logging.warning(f"Unknown argument for resort_binary_names: {by}, preserving default order")
    return feat_names


def remove_base_name(base_name: str, binary_full_name: str):
    res = binary_full_name
    if '_AND_' in res:
        # for combined features (not supported yet)
        # res = res.replace(base_name + '_AND_', '')
        # res = res.replace('_AND_' + base_name, '')
        pass
    else:
        # for singular features
        res = res.replace(base_name + '_', '')
    return res


def change_interval_name_for_plot(is_numeric: bool, interval_string: str, unit_name: str = None):
    if isinstance(unit_name, str):
        unit_name_ = '( ' + unit_name + ')'
    else:
        unit_name_ = ''

    if not is_numeric:
        return interval_string

    if interval_string.endswith('missing'):
        return 'unknown'
    elif interval_string.count(',') == 1:
        s0, s1 = interval_string.strip('[]()').split(',')
        return f'from {s0}{unit_name_} up to {s1} {unit_name_}'
    else:
        logging.warning(f'Cannot process interval string: {interval_string}')
    return interval_string


def get_segment_name_ready_for_plot(is_numeric: bool, base_name: str, binary_full_name: str, unit_name: str = None):
    if '_AND_' in binary_full_name:
        # combined features names aren't supported yet
        return binary_full_name
    # if is_numeric:
    #     res = remove_base_name(base_name, binary_full_name)
    # else:
    #     res = binary_full_name
    res = remove_base_name(base_name, binary_full_name)
    return change_interval_name_for_plot(is_numeric, res, unit_name)
