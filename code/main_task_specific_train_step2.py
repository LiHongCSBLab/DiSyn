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

    recon_params = utils.set_to_str(args_dict, ['clsadv_alpha', 'drop_out', 'recon_epochs'])
    task_params = utils.set_to_str(args_dict, ['di_scale_co', 'drop_out', 'ft_lr'])
    
    args_dict.update(
        {
            'model_path': os.path.join(path_config.result_path, f'Disyn_{args_dict["seed"]}', f'task_specific_train_step{args_dict["step"]}', args_dict["drug"], recon_params),
            'latent_dim_task': 32,
            'linker_dim': [64, 32],
            'task_save_path': os.path.join(path_config.result_path, f'Disyn_{args_dict["seed"]}', f'task_specific_train_step{args_dict["step"]}', args_dict["drug"], task_params, recon_params)
        })

    os.makedirs(args_dict['task_save_path'], exist_ok=True)

    gene_expressions_df = pd.read_csv(path_config.gene_expressions_csv, index_col=0)
    args_dict.update({'input_dim': gene_expressions_df.shape[-1]})

    # test_df = data.get_tcga_drug_labeled_df(gene_expressions_df=gene_expressions_df, drug=args.drug)
    hybrid_labeled_dataloader = data.get_hybrid_dataloader_generator(gene_expressions_df=gene_expressions_df, seed=args.seed, drug=args.drug)

    metrics_dict = defaultdict(list)
    fold_count = 0

    for train_labeled_gdsc_dataloader, test_labeled_gdsc_dataloader, tcga_unlabeled_dataloader, test_labeled_dataloaders in hybrid_labeled_dataloader:
        source_dsn, target_dsn = utils.load_disyn(fold_count=fold_count, **args_dict)

        metrics_dict = task_specific_train_disyn_step2(dsn_source=source_dsn, dsn_target=target_dsn, s_labeled_train_dataloader=train_labeled_gdsc_dataloader,
                                                       s_labeled_val_dataloader=test_labeled_gdsc_dataloader, t_unlabeled_dataloader=tcga_unlabeled_dataloader,
                                                       t_test_dataloader=test_labeled_dataloaders, fold_count=fold_count, metrics_dict=metrics_dict, **args_dict)
        fold_count += 1

    with open(os.path.join(args_dict['task_save_path'], f'{task_params}_evaluation_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #Training Setting
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=95, help='seed')
    parser.add_argument('--drug', type=str, default='tem', help='drug')
    parser.add_argument('--step', type=int, default=1, help='train iteration')

    # Parameters used to locate the model path of the previous stage
    parser.add_argument('--recon_epochs', type=int, default=100, help='model_recon_epochs')
    parser.add_argument('--clsadv_alpha', type=float, default=0.3, help='alpha in Adversarial training of task classifiers')
    parser.add_argument('--drop_out', type=float, default=0.0, help='drop_out')
    
    # Parameters used in this stage
    parser.add_argument('--di_scale_co', type=int, default=1, help='The multiple of synthetic data compared to real world data')
    parser.add_argument('--ft_lr', type=float, default=0.0001, help='ft_lr')

    args = parser.parse_args()

    args_dict = vars(args)

    main(args_dict)
