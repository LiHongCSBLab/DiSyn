import json
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import re
import shutil
import path_config

from pandas import Series,DataFrame
import pandas as pd
import utils
import torch
from net import EncoderDecoder, FC
from utils import predict_target_classification
from sklearn.metrics import roc_auc_score


def get_tcga_drug_labeled_df(gene_expressions_df, drug):
    drug_mapping_df = pd.read_csv(path_config.gdsctcga_mapping_file, index_col=0)

    drug = drug_mapping_df.loc[drug, 'drug_name']

    patient_response = pd.read_csv(path_config.tcga_response).set_index(["drug.name"])
    patient_response_drug = patient_response.loc[drug].set_index(["patient"])

    response_dic = {'Non-response': 0, 'Response': 1}
    patient_label = patient_response_drug['response_RS'].map(response_dic)

    patient_labeled_feature_df = gene_expressions_df.loc[patient_response_drug.index]

    return patient_labeled_feature_df, patient_label


def load_classifier_for_inference(clf_path, **kwargs):
    shared_encoder = FC(input_dim=kwargs['input_dim'],
                        output_dim=128,
                        hidden_dims=[512, 256]).to(kwargs['device'])

    decoder = FC(input_dim=128,
                 output_dim=1,
                 hidden_dims=[64, 32]).to(kwargs['device'])

    target_classifier = EncoderDecoder(encoder=shared_encoder, decoder=decoder).to(kwargs['device'])

    target_classifier.load_state_dict(torch.load(clf_path, map_location=torch.device(kwargs['device'])))

    return target_classifier


drug_list = ['tem','bic', 'ble', 'doc', 'eto', 'met', 'pac', 'pem', 'tam', 'vinb', 'vin', 'fu', 'sor', 'gem', 'cis', 'dox']
drug_mapping_df = pd.read_csv(path_config.gdsctcga_mapping_file, index_col=0)
gene_expressions_df = pd.read_csv(path_config.gene_expressions_csv, index_col=0)
training_params = {'input_dim': gene_expressions_df.shape[-1], 'device': 'cuda:0'}

names = locals()
print("Re-evaluation of DiSyn on TCGA based on AUROC")

for metric in ['acc','f1','auroc','auprc','precision','recall','specificity']:
    names['seed_drug_best_infold_'+str(metric)+'_dict'] = dict()
    for seed in [82, 15, 4, 95, 36]:
        names['seed_drug_best_infold_'+str(metric)+'_dict'][seed] = defaultdict(dict)

for metric in ['auroc']:
    for drug_code in drug_list:
        for seed in [82, 15, 4, 95, 36]:
            test_df,patient_labels = get_tcga_drug_labeled_df(gene_expressions_df=gene_expressions_df, drug=drug_code)
            for fold_count in [0,1,2,3,4]:
                drug_name = drug_mapping_df.loc[drug_code, 'drug_name']
                target_classifier = load_classifier_for_inference(os.path.join('../data/fig3_reproduce_AUROC', drug_name, str(seed), f'classifier_{fold_count}.pt' ), **training_params)
                
                prediction_df = predict_target_classification(classifier=target_classifier, test_df=test_df, device=training_params['device'])
                
                names['seed_drug_best_infold_'+str(metric)+'_dict'][seed][drug_name][fold_count] = roc_auc_score(y_true=patient_labels, y_score=prediction_df.values)

            
df = pd.DataFrame(columns=[drug_mapping_df.loc[drug_code, 'drug_name'] for drug_code in drug_list])
for metric in ['auroc']:
    for seed in [82, 15, 4, 95, 36]:
        for fold in [0,1,2,3,4]:
            method_seed_fold_dict = { k:v[fold] for k,v in names['seed_drug_best_infold_auroc_dict'][seed].items() }
            df.loc[f'DiSyn_{seed}_{fold}'] = method_seed_fold_dict


# This table records the AUROC scores of each fold of our model on the test set.
df.to_csv('../results/re_evaluation_metric_AUROC.csv')

#print('mean')
#print(df.mean())

#print('std')
#print(df.std())
