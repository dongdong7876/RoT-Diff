from argparse import ArgumentParser

import numpy as np
import pandas as pd
import os
from utils import load_yaml_param_settings
from evaluation.evaluation import evaluate_fn


def load_args():
    parser = ArgumentParser(description="Inference: Robust Anomaly Scoring")
    parser.add_argument('--data_name', type=str, required=True, help="Dataset name (e.g., SMD, SWaT, PSM, MSL, WADI)")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device index")
    parser.add_argument('--threshold_method', type=str, default='percentile',
                        help="Method to determine threshold")

    parser.add_argument('--data_path', type=str, default=None, help="Path to dataset")

    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = os.path.join('dataset', args.data_name)
    if args.config is None:
        args.config = os.path.join('config', f'config_{args.data_name}.yaml')

    return args

if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    in_channels = config['dataset']['in_channels']
    batch_size = config['exp_params']['batch_sizes']['stage2']
    vq_strategy = config['VQ']['vq_strategy']
    results_pd = evaluate_fn(args, config, batch_size, **config['dataset'])

    result = np.array(
        [args.data_name, 'RoT-Diff'])
    column_names = ['data_name', 'algo']
    result_df = pd.DataFrame([result], columns=column_names)

    if results_pd is not None:
        results = pd.concat([result_df, results_pd.reset_index(drop=True)], axis=1)
    else:
        results = result_df
    result_path = os.path.join('results', 'results_RoT-Diff.csv')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    file_exists = os.path.exists(result_path)
    results.to_csv(result_path, mode='a', header=not file_exists, index=False)
    print(f"Results saved to {result_path}")