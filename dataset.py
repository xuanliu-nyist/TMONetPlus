import os
from functools import lru_cache
import numpy as np
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pandas as pd
import joblib

PanCancer = {
    'ACC': 0, 'BLCA': 1, 'CESC': 2, 'CHOL': 3,
    'COAD': 4, 'DLBC': 5, 'ESCA': 6, 'GBM': 7,
    'HNSC': 8, 'KICH': 9, 'KIRC': 10, 'KIRP': 11,
    'LGG': 12, 'LIHC': 13, 'LUAD': 14, 'LUSC': 15,
    'MESO': 16, 'OV': 17, 'PAAD': 18, 'PCPG': 19,
    'PRAD': 20, 'READ': 21, 'SARC': 22, 'SKCM': 23,
    'STAD': 24, 'TGCT': 25, 'THCA': 26, 'THYM': 27,
    'UCEC': 28, 'UCS': 29, 'UVM': 30, 'BRCA': 31
}


class CancerDataset(Dataset):
    def __init__(self, omics_paths, omics_types, clinical_data_path, index_data_path, fold, cancer_types=None,
                 Percent_Train=None, training=True):
        assert len(omics_paths) == len(omics_types), "Number of omics paths and types must match"

        self.clinical_data = pd.read_csv(clinical_data_path)
        self.clinical_data['ID'] = self.clinical_data['ID'].str[:12].str.replace('-', '.')
        self.clinical_data['OS.time'] = pd.to_numeric(
            self.clinical_data['OS.time'], errors='coerce'
        )
        self.clinical_data.dropna(subset=['Cancer_Type', 'OS.time'], inplace=True)
        self.clinical_data.set_index('ID', inplace=True)
        self.clinical_data_dict = self.clinical_data.to_dict('index')

        train_data = pd.read_csv(index_data_path)
        train_data['Sample_ID'] = train_data['Sample_ID'].str[:12].str.replace('-', '.')

        if cancer_types is None:
            cancer_types = self.clinical_data['Cancer_Type'].unique()
            print("Unique Cancer_Types:", cancer_types)

        cond = (train_data['Fold'] == fold) & (train_data['Cancer_Type'].isin(cancer_types))
        if Percent_Train is not None:
            cond &= (train_data['Percent_Train'] == Percent_Train)
        self.train_data = train_data[
            cond & train_data['Sample_ID'].isin(self.clinical_data.index)
        ].reset_index(drop=True)

        self.omics = {}
        self.omics_stats = {}
        for omics_type, path in zip(omics_types, omics_paths):
            df = pd.read_csv(path)
            df['ID'] = df['ID'].str[:12].str.replace('-', '.')
            df.set_index('ID', inplace=True)
            df = df.astype('float32')

            df = df.fillna(df.mean())
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], 0)

            mean = df.mean().values
            std = df.std().values
            std[std == 0] = 1e-6

            self.omics[omics_type] = df
            self.omics_stats[omics_type] = (mean, std)

        self.training = training

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sample_id = self.train_data.at[idx, 'Sample_ID']
        if pd.isnull(sample_id):
            raise ValueError(f"Invalid Sample_ID at index {idx}")

        omics_data = {}
        for omics_type, df in self.omics.items():
            values = df.loc[sample_id].values
            mean, std = self.omics_stats[omics_type]

            std_safe = np.where(std == 0, 1e-6, std)
            standardized = (values - mean) / std_safe
            omics_data[omics_type] = torch.nan_to_num(
                torch.from_numpy(standardized).float(),
                nan=0.0, posinf=1e6, neginf=-1e6
            )
        clinical = self.clinical_data_dict.get(sample_id)
        if clinical is None:
            raise ValueError(f"No clinical data for {sample_id}")

        OS_time = max(float(clinical['OS.time']), 1.0)
        OS_time_tensor = torch.tensor([OS_time / 30], dtype=torch.float32)

        OS = torch.tensor([clinical['OS']], dtype=torch.float32)

        cancer_type_value = clinical['Cancer_Type']
        if cancer_type_value not in PanCancer:
            raise KeyError(f"Cancer type {cancer_type_value} not in PanCancer mapping")
        cancer_type_tensor = torch.tensor([PanCancer[cancer_type_value]], dtype=torch.long)

        return OS, OS_time_tensor, omics_data, cancer_type_tensor

class MetastaticDataset(Dataset):
    def __init__(self, omics_paths_prim, omics_paths_meta, omics_types, split_path, fold, label_path, is_test=False):
        assert len(omics_paths_prim) == len(omics_types), "Number of omics paths and types must match"
        assert len(omics_paths_meta) == len(omics_types), "Number of omics paths and types must match"

        split = joblib.load(split_path)
        self.label = joblib.load(label_path)
        if not is_test:
            self.samples = split[f'fold{fold}_train']
        else:
            self.samples = split[f'fold{fold}_test']
        # Select data for the specified fold and cancer types
        self.omics_prim = {}
        for omics_type, path in zip(omics_types, omics_paths_prim):
            df = pd.read_csv(path, index_col=0)
            self.omics_prim[omics_type] = df

        self.omics_meta = {}
        for omics_type, path in zip(omics_types, omics_paths_meta):
            df = pd.read_csv(path, index_col=0)
            self.omics_meta[omics_type] = df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get the index of the sample from the training data
        sample_id = self.samples[idx]
        omics_data = {}

        if self.label[sample_id] == 0:
            for omics_type, df in self.omics_prim.items():
                ts = torch.Tensor(np.nan_to_num(np.array(df[df['ID'].str[:12] == sample_id].values.tolist()[0][1:])))
                omics_data[omics_type] = ts
            cancer_type = torch.LongTensor([0])

        if self.label[sample_id] == 1:
            for omics_type, df in self.omics_meta.items():
                ts = torch.Tensor(np.nan_to_num(np.array(df[df['ID'].str[:12] == sample_id].values.tolist()[0][1:])))
                omics_data[omics_type] = ts
            cancer_type = torch.LongTensor([1])

        if self.label[sample_id] == 2:
            for omics_type, df in self.omics_prim.items():
                ts = torch.Tensor(np.nan_to_num(np.array(df[df['ID'].str[:12] == sample_id].values.tolist()[0][1:])))
                omics_data[omics_type] = ts
            cancer_type = torch.LongTensor([1])

        return omics_data, cancer_type
