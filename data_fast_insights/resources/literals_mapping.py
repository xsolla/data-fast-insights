SHOWCASE_LITERALS_MAPPING = {
    'total_sum': lambda kwargs: 'Segment Size, Absolute',
    'low_perc': lambda kwargs: 'Share of Objects Worse than the Total Mean, %',
    'high_perc': lambda kwargs: 'Share of Objects Better than the Total Mean (Segment Quality), %',
    'perc_of_total': lambda kwargs: 'Segment Size, %',
    'target_delta_perc': lambda kwargs: f'{kwargs["target_name"]} Segment Difference from Total Mean, %',
    'group_importance': lambda  kwargs: 'Group Importance'
}
