import argparse
import itertools
import json
import os
import re
from collections import defaultdict

import pandas as pd

import data
import path_config
import utils
from train_disyn import task_specific_train_disyn_step2


def main(args_dict):
    utils.set_seed(args_dict["seed"])

    params_list = ['drop_out', 'di_scale_co', 'ft_lr']
    task_param_str = utils.set_to_str(args_dict, params_list)

    args_dict.update(
        {
            'latent_dim_task': 32,
            'linker_dim': [64, 32],
            'task_save_path': os.path.join(args_dict['model_path'], task_param_str)
        })

    os.makedirs(args_dict['task_save_path'], exist_ok=True)

    gex_features_df = pd.read_csv(path_config.gex_feature_file, index_col=0)
    args_dict.update({'input_dim': gex_features_df.shape[-1]})

    # test_df = data.get_tcga_drug_labeled_df(gex_features_df=gex_features_df, drug=args.drug)

    hybrid_labeled_dataloader = data.get_hybrid_dataloader_generator(gex_features_df=gex_features_df, seed=args.seed, drug=args.drug)

    metrics_dict = defaultdict(list)
    fold_count = 0

    for train_labeled_gdsc_dataloader, test_labeled_gdsc_dataloader, tcga_unlabeled_dataloader, test_labeled_dataloaders in hybrid_labeled_dataloader:
        source_dsn, target_dsn = utils.load_disyn(fold_count=fold_count, **args_dict)

        metrics_dict = task_specific_train_disyn_step2(dsn_source=source_dsn, dsn_target=target_dsn, s_labeled_train_dataloader=train_labeled_gdsc_dataloader,
                                                       s_labeled_val_dataloader=test_labeled_gdsc_dataloader, t_unlabeled_dataloader=tcga_unlabeled_dataloader,
                                                       t_test_dataloader=test_labeled_dataloaders, fold_count=fold_count, metrics_dict=metrics_dict, **args_dict)
        fold_count += 1

    with open(os.path.join(args_dict['task_save_path'], f'{task_param_str}_ft_evaluation_results.json'), 'w') as f:
        json.dump(metrics_dict, f)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=95, help='seed')
    parser.add_argument('--drug', type=str, default='tem', help='drug')

    #     这才是训练要用到的参数
    parser.add_argument('--drop_out', type=float, default=0.0, help='drop_out')
    parser.add_argument('--di_scale_co', type=int, default=1, help='di_scale_co')
    parser.add_argument('--ft_lr', type=float, default=0.0001, help='ft_lr')
    parser.add_argument('--step', type=int, default=2, help='train iteration')

    args = parser.parse_args()

    args.model_path = f'../results/Disyn/Disyn_{args.seed}/task_specific_train_step{args.step}/{args.drug}/nums_recon_100_nums_critic_100_drop_out_0.0/clsadv_alpha_0.3_drop_out_0.0/recon_epochs_10'

    main(vars(args))