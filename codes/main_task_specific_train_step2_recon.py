import argparse
import itertools
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import data
import path_config
import utils
from train_disyn import recon_classadv_train_di_adv


def main(args_dict):
    utils.set_seed(args_dict["seed"])

    params_list = ['clsadv_alpha', 'drop_out']
    task_param_str = utils.set_to_str(args_dict, params_list)

    args_dict.update(
        {
            'latent_dim_task': 32,
            'linker_dim': [64, 32],
            'param_str': args_dict['model_path'].split('/')[6],
            'task_save_path': os.path.join(path_config.result_path, f'Disyn_{args_dict["seed"]}', f'task_specific_train_step{args_dict["step"]}', args_dict['drug'], args_dict['model_path'].split('/')[6], task_param_str)
        })

    os.makedirs(args_dict['task_save_path'], exist_ok=True)

    gex_features_df = pd.read_csv(path_config.gex_feature_file, index_col=0)
    args_dict.update({'input_dim': gex_features_df.shape[-1]})

    fold_count = 0

    hybrid_labeled_dataloader = data.get_hybrid_dataloader_generator(gex_features_df=gex_features_df, seed=args.seed, drug=args.drug)

    for train_labeled_gdsc_dataloader, _, tcga_unlabeled_dataloader, _ in hybrid_labeled_dataloader:

        source_dsn, target_dsn = utils.build_disyn_from_classifier(args_dict['model_path'], fold_count, **args_dict)

        recon_classadv_train_di_adv(dsn_source=source_dsn, dsn_target=target_dsn, s_labeled_dataloader=train_labeled_gdsc_dataloader, t_unlabeled_dataloader=tcga_unlabeled_dataloader,
                                    fold_count=fold_count, **args_dict)

        fold_count += 1

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=95, help='seed')
    parser.add_argument('--drug', type=str, default='tem', help='drug')
    parser.add_argument('--step', type=int, default=2, help='train iteration')

    parser.add_argument('--recon_epochs', type=int, default=10, help='recon_epochs')
    parser.add_argument('--clsadv_alpha', type=float, default=0.3, help='clsadv_alpha')
    parser.add_argument('--clf_train_bf_recon', type=int, default=0, help='clf_train_bf_recon')
    parser.add_argument('--rev_back_internal', type=int, default=3, help='rev_back_internal')
    parser.add_argument('--drop_out', type=float, default=0.0, help='drop_out')

    args = parser.parse_args()

    # Implemention of di network reconstruction procedure
    # Pass in a specific model path with task specific trained
    args.model_path = f'../results/Disyn/Disyn_{args.seed}/task_specific_train_step{args.step - 1}/{args.drug}/nums_recon_100_nums_critic_100_drop_out_0.0/ft_lr_0.0001'

    main(args_dict=vars(args))
