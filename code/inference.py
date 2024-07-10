import argparse

import os

import pandas as pd

import data
import path_config
import utils

def main(args_dict):
    gene_expressions_df = pd.read_csv(path_config.gene_expressions_csv, index_col=0)
    args_dict.update({'input_dim': gene_expressions_df.shape[-1]})

    test_df = data.get_tcga_drug_labeled_df(gene_expressions_df=gene_expressions_df, drug=args_dict['drug'])

    target_classifier = utils.load_classifier_for_inference(args_dict['clf_path'], **args_dict)

    prediction_df = utils.predict_target_classification(classifier=target_classifier, test_df=test_df, device=args_dict['device'])

    os.makedirs(f'../results/inference/{args_dict["drug"]}', exist_ok=True)
    prediction_df.to_csv(f'../results/inference/{args_dict["drug"]}/{args_dict["drug"]}.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--drug', type=str, default='tem', help='drug')
    parser.add_argument('--clf_path', type=str, default='./best_models', help='clf_path')

    args = parser.parse_args()

    main(vars(args))
