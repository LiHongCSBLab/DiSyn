import argparse
import itertools
import json
import os
from collections import defaultdict

import pandas as pd

import data
import path_config
import utils
from train_disyn import task_specific_train_di_adv


def main(args_dict):
    utils.set_seed(args_dict['seed'])

    pretrain_list = ['drop_out', 'nums_critic', 'nums_recon']
    task_specific_train_list = ['ft_lr']
    param_str = utils.set_to_str(args_dict, pretrain_list)
    task_param_str = utils.set_to_str(args_dict, task_specific_train_list)

    args_dict.update(
        {
            'param_str': param_str,
            'model_path': os.path.join(path_config.result_path, f'Disyn_{args_dict["seed"]}', param_str),
            'task_save_path': os.path.join(path_config.result_path, f'Disyn_{args_dict["seed"]}', 'task_specific_train_step0', args_dict['drug'], task_param_str, param_str)
        })

    os.makedirs(args_dict['task_save_path'], exist_ok=True)

    gene_expressions_df = pd.read_csv(path_config.gene_expressions_csv, index_col=0)
    args_dict.update({'input_dim': gene_expressions_df.shape[-1]})

    labeled_dataloader = data.get_labeled_dataloader_generator(gene_expressions_df=gene_expressions_df, seed=args_dict["seed"], drug=args_dict['drug'])
    source_dsn, _ = utils.load_dsn(**args_dict)

    metrics_dict = defaultdict(list)
    fold_count = 0
    for train_labeled_gdsc_dataloader, test_labeled_gdsc_dataloader, test_labeled_dataloaders in labeled_dataloader:
        metrics_dict = task_specific_train_di_adv(source_dsn, source_labeled_train_dataloader=train_labeled_gdsc_dataloader,
                                                  source_labeled_val_dataloader=test_labeled_gdsc_dataloader, target_test_dataloader=test_labeled_dataloaders,
                                                  fold_count=fold_count, metrics_dict=metrics_dict, **args_dict)
        fold_count += 1

    with open(os.path.join(args_dict['task_save_path'], f'{task_param_str}_evaluation_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=95, help='seed')
    parser.add_argument('--drug', type=str, default='tem', help='drug')
    
    # Specify the pre-trained model parameters that you want to train
    parser.add_argument('--nums_recon', type=int, default=100, help='nums_recon')
    parser.add_argument('--nums_critic', type=int, default=100, help='nums_critic')
    parser.add_argument('--drop_out', type=float, default=0.0, help='drop_out')

    # Parameters used in this stage
    parser.add_argument('--ft_lr', type=float, default=0.0001, help='ft_lr')

    args = parser.parse_args()

    main(args_dict=vars(args))
