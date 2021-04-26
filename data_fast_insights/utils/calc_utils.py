from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_fast_insights import BinaryDependenceModelData


def choose_central_tendency_metric(col_name: str, model_data: 'BinaryDependenceModelData'):
    if col_name == model_data.y_name or col_name in model_data.num_cols:
        use_metrics = 'mean'
    elif col_name in model_data.cat_cols:
        use_metrics = 'mode'
    else:
        raise ValueError('col_name not found in num_cols or cat_cols of model_data')
    return use_metrics
