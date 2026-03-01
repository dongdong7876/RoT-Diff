# -*- coding: utf-8 -*-
import functools
import traceback
from typing import List, Tuple, Any, Union

import numpy as np
import pandas as pd

from evaluation.metrics import METRICS


def encode_params(params):
    encoded_pairs = []
    for key, value in sorted(params.items()):
        if isinstance(value, (np.floating, float)):
            value = round(value, 3)
        encoded_pairs.append(f"{key}:{repr(value)}")
    return ";".join(encoded_pairs)


class Evaluator:
    """
    Evaluator class, used to calculate the evaluation metrics of the model.
    """

    def __init__(self, metric: List[Union[dict, str]]):
        """
        Initialize the evaluator object.

        :param metric: A list containing information on evaluation metrics.
                       Can be a list of strings (e.g. ['mse', 'mae'])
                       or a list of dicts (e.g. [{'name': 'mse'}, {'name': 'mase', 'seasonality': 12}]).
        """
        self.metric = metric
        self.metric_funcs = []
        self.metric_names = []

        # Create a list of evaluation indicator functions and names
        for metric_item in self.metric:
            if isinstance(metric_item, str):
                metric_info = {"name": metric_item}
            elif isinstance(metric_item, dict):
                metric_info = metric_item
            else:
                raise TypeError(f"Metric item must be a string or a dict, got {type(metric_item)}")

            metric_info_copy = metric_info.copy()

            if "name" not in metric_info_copy:
                raise ValueError("Metric dictionary must contain a 'name' key.")

            metric_name = metric_info_copy.pop("name")
            if metric_info_copy:
                metric_name += ";" + encode_params(metric_info_copy)
            self.metric_names.append(metric_name)
            metric_name_copy = metric_info.copy()
            name = metric_name_copy.pop("name")
            fun = METRICS[name]
            if metric_name_copy:
                self.metric_funcs.append(functools.partial(fun, **metric_name_copy))
            else:
                self.metric_funcs.append(fun)

    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        hist_data: Union[np.ndarray,pd.DataFrame] = None,
        **kwargs,
    ) -> list:
        """
        Calculate the evaluation index values of the model.

        :param actual: Actual observation data.
        :param predicted: Model predicted data.
        :param scaler: Normalization.
        :param hist_data:  Historical data (optional).
        :return: Indicator evaluation result.
        """

        if actual.ndim == 3:
            actual = actual.reshape(-1, actual.shape[1])
        if predicted.ndim == 3:
            predicted = predicted.reshape(-1, predicted.shape[1])

        if hist_data is not None:
            if isinstance(hist_data, pd.DataFrame):
                hist_data = hist_data.values
            else:
                hist_data = hist_data.reshape(-1, hist_data.shape[1])

        return [
            m(actual, predicted, scaler=scaler, hist_data=hist_data)
            for m in self.metric_funcs
        ]

    def evaluate_with_log(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        hist_data: np.ndarray = None,
        **kwargs,
    ) -> Tuple[List[Any], str]:
        """
        Calculate the evaluation index values of the model.

        :param actual: Actual observation data.
        :param predicted: Model predicted data.
        :param scaler: Normalization.
        :param hist_data:  Historical data (optional).
        :return: Indicator evaluation results and log information.
        """
        evaluate_result = []
        log_info = ""
        for m in self.metric_funcs:
            try:
                evaluate_result.append(
                    m(actual, predicted, scaler=scaler, hist_data=hist_data)
                )
            except Exception as e:
                evaluate_result.append(np.nan)
                log_info += f"Error in calculating {m.__name__}: {traceback.format_exc()}\n{e}\n"
        return evaluate_result, log_info

    def default_result(self):
        """
        Return the default evaluation metric results.

        :return: Default evaluation metric result.
        """
        return len(self.metric_names) * [np.nan]
