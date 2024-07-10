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

    params_list = ['clsadv_alpha', 'drop_out', 'recon_epochs']
    task_params = utils.set_to_str(args_dict, params_list)

    args_dict.update(
    {
        'latent_dim_task': 32,
        'linker_dim': [64, 32],
        'task_save_path': os.path.join(path_config.result_path, f'Disyn_{args_dict["seed"]}', f'task_specific_train_step{args_dict["step"]}', args_dict['drug'], task_params)
    })

    os.makedirs(args_dict['task_save_path'], exist_ok=True)

    gene_expressions_df = pd.read_csv(path_config.gene_expressions_csv, index_col=0)
    args_dict.update({'input_dim': gene_expressions_df.shape[-1]})

    fold_count = 0
    
    hybrid_labeled_dataloader = data.get_hybrid_dataloader_generator(gene_expressions_df=gene_expressions_df, seed=args.seed, drug=args.drug)

    for train_labeled_gdsc_dataloader, _, tcga_unlabeled_dataloader, _ in hybrid_labeled_dataloader:

        source_dsn, target_dsn = utils.build_disyn_from_classifier(args_dict['model_path'], fold_count, **args_dict)

        recon_classadv_train_di_adv(dsn_source=source_dsn, dsn_target=target_dsn, s_labeled_dataloader=train_labeled_gdsc_dataloader, t_unlabeled_dataloader=tcga_unlabeled_dataloader,
                                    fold_count=fold_count, **args_dict)

        fold_count += 1

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Training Setting
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=95, help='seed')
    parser.add_argument('--drug', type=str, default='tem', help='drug_code')
    parser.add_argument('--step', type=int, default=1, help='the steps of iterations')
    # 'step' refers to the steps of iterations, you may need to specify which step currently at each iteration.

    # Parameters used in this stage
    parser.add_argument('--clf_train_bf_recon', type=int, default=0, help='classifier train before recon epochs')
    parser.add_argument('--rev_back_internal', type=int, default=3, help='The interval of adversarial training')
    parser.add_argument('--recon_epochs', type=int, default=100, help='model_recon_epochs')
    parser.add_argument('--clsadv_alpha', type=float, default=0.3, help='alpha in Adversarial training of task classifiers')

    # Parameters used to locate the model path of the previous stage
    parser.add_argument('--drop_out', type=float, default=0.0, help='drop_out')
    parser.add_argument('--ft_lr', type=float, default=0.0001, help='ft_lr')
    # When Step>1 ...
    parser.add_argument('--di_scale_co', type=int, default=1, help='The multiple of synthetic data compared to real world data')
    
    args = parser.parse_args()

    model_path = utils.get_model_save_path(args)

    args_dict=vars(args)
    print(model_path)
    
    for item in os.listdir(model_path):
        if os.path.isdir(os.path.join(model_path, item)) and item[0]!='.':
            args_dict['model_path'] = os.path.join(model_path, item)
            main(args_dict)
