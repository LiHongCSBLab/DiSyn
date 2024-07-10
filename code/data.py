import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import path_config


def get_unlabeled_dataloaders():
    gene_expressions_df = pd.read_csv(path_config.gene_expressions_csv, index_col=0)

    target_samples = [attr for attr in gene_expressions_df.index if attr.startswith('TCGA')]
    source_samples = gene_expressions_df.index.difference(target_samples)

    target_tensor = torch.from_numpy(gene_expressions_df.loc[target_samples].values.astype('float32'))
    source_tensor = torch.from_numpy(gene_expressions_df.loc[source_samples].values.astype('float32'))

    target_dataloader = DataLoader(TensorDataset(target_tensor),
                                   batch_size=64,
                                   shuffle=True,
                                   drop_last=True)

    source_dataloader = DataLoader(TensorDataset(source_tensor),
                                   batch_size=64,
                                   shuffle=True,
                                   drop_last=True)

    return source_dataloader, target_dataloader


def get_gdsc_labeled_dataloader_generator(gene_expressions_df, seed, drug):

    cellline_response = pd.read_csv(path_config.cellline_response).set_index(["DRUG_NAME"])
    cellline_response_drug = cellline_response.loc[drug].set_index(["cell_tissue"])

    threshold = np.median(cellline_response_drug['AUC'])
    gdsc_labels = (cellline_response_drug['AUC'] < threshold).astype('int')
    gdsc_labeled_feature_df = gene_expressions_df.loc[cellline_response_drug.index]

    drug_split_dict = np.load(f'../data/datasplitfold/seeds/drug_split_dict_{seed}.npy', allow_pickle=True).item()

    for fold, index in drug_split_dict[drug].items():
        train_index = index[0]
        test_index = index[1]
        train_labeled_gdsc_df, test_labeled_gdsc_df = gdsc_labeled_feature_df.values[train_index], \
            gdsc_labeled_feature_df.values[test_index]
        train_gdsc_labels, test_gdsc_labels = gdsc_labels.values[train_index], gdsc_labels.values[test_index]

        train_labeled_gdsc_dateset = TensorDataset(
            torch.from_numpy(train_labeled_gdsc_df.astype('float32')),
            torch.from_numpy(train_gdsc_labels))
        test_labeled_gdsc_dataset = TensorDataset(
            torch.from_numpy(test_labeled_gdsc_df.astype('float32')),
            torch.from_numpy(test_gdsc_labels))

        train_labeled_gdsc_dataloader = DataLoader(train_labeled_gdsc_dateset,
                                                   batch_size=64,
                                                   shuffle=True)

        test_labeled_gdsc_dataloader = DataLoader(test_labeled_gdsc_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

        yield train_labeled_gdsc_dataloader, test_labeled_gdsc_dataloader


def get_tcga_unlabeled_dataloader(gene_expressions_df):
    # Return all tcga unlabeled samples, shuffle=True
    tcga_samples = [attr for attr in gene_expressions_df.index if attr.startswith('TCGA')]
    tcga_df = gene_expressions_df.loc[tcga_samples]

    tcga_dataset = TensorDataset(
        torch.from_numpy(tcga_df.values.astype('float32'))
    )

    tcga_loader = DataLoader(tcga_dataset,
                             batch_size=64,
                             shuffle=True,
                             drop_last=True)

    print(f'len(tcga_df):{len(tcga_dataset)}')

    return tcga_loader


def get_TCGA_labeled_dataloaders(gene_expressions_df, drug):

    patient_response = pd.read_csv(path_config.tcga_response).set_index(["drug.name"])
    patient_response_drug = patient_response.loc[drug].set_index(["patient"])

    response_dic = {'Non-response': 0, 'Response': 1}
    patient_label = patient_response_drug['response_RS'].map(response_dic)

    patient_labeled_feature_df = gene_expressions_df.loc[patient_response_drug.index]

    assert all(patient_label.index == patient_labeled_feature_df.index)

    labeled_tcga_dateset = TensorDataset(
        torch.from_numpy(patient_labeled_feature_df.values.astype('float32')),
        torch.from_numpy(np.array(patient_label)))

    labeled_tcga_dataloader = DataLoader(labeled_tcga_dateset,
                                         batch_size=64,
                                         shuffle=True)

    return labeled_tcga_dataloader


def get_labeled_dataloader_generator(gene_expressions_df, seed, drug):
    drug_mapping_df = pd.read_csv(path_config.gdsctcga_mapping_file, index_col=0)

    gdsc_labeled_dataloader_generator = get_gdsc_labeled_dataloader_generator(gene_expressions_df=gene_expressions_df, seed=seed, drug=drug_mapping_df.loc[drug, 'gdsc_name'])

    test_labeled_dataloaders = get_TCGA_labeled_dataloaders(gene_expressions_df=gene_expressions_df,
                                                            drug=drug_mapping_df.loc[drug, 'drug_name'])

    for train_labeled_gdsc_dataloader, test_labeled_gdsc_dataloader in gdsc_labeled_dataloader_generator:
        yield train_labeled_gdsc_dataloader, test_labeled_gdsc_dataloader, test_labeled_dataloaders


def get_hybrid_dataloader_generator(gene_expressions_df, seed, drug):
    drug_mapping_df = pd.read_csv(path_config.gdsctcga_mapping_file, index_col=0)

    gdsc_labeled_dataloader_generator = get_gdsc_labeled_dataloader_generator(gene_expressions_df=gene_expressions_df, seed=seed, drug=drug_mapping_df.loc[drug, 'gdsc_name'])

    tcga_unlabeled_dataloader = get_tcga_unlabeled_dataloader(gene_expressions_df=gene_expressions_df)

    test_labeled_dataloaders = get_TCGA_labeled_dataloaders(gene_expressions_df=gene_expressions_df,
                                                            drug=drug_mapping_df.loc[drug, 'drug_name'])

    for train_labeled_gdsc_dataloader, test_labeled_gdsc_dataloader in gdsc_labeled_dataloader_generator:
        yield train_labeled_gdsc_dataloader, test_labeled_gdsc_dataloader, tcga_unlabeled_dataloader, test_labeled_dataloaders


def get_tcga_drug_labeled_df(gene_expressions_df, drug):
    drug_mapping_df = pd.read_csv(path_config.gdsctcga_mapping_file, index_col=0)

    drug = drug_mapping_df.loc[drug, 'drug_name']

    patient_response = pd.read_csv(path_config.tcga_response).set_index(["drug.name"])
    patient_response_drug = patient_response.loc[drug].set_index(["patient"])
    patient_labeled_feature_df = gene_expressions_df.loc[patient_response_drug.index]

    return patient_labeled_feature_df
