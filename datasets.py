from utils import get_clf, add_attachment_points
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os 

INTERACTION_TYPE = {'B': 1, 'F': 2, 'A': 3}
UNKNOWN_TYPE = 4
LABEL_MAP = {0: -1, 1: 1}  # negative label should not be zero due to classifier free guidance


class InteractionsDataset(Dataset):
    def __init__(self, data_dir, protein_graph_dir, tokenizer):
        self.protein_graph_dir =protein_graph_dir
        self.tokenizer = tokenizer
        # get all prot assays
        prot_assays = os.listdir(protein_graph_dir)
        prot_assays = [i.split('.')[0] for i in prot_assays]
        
        self.df = pd.read_csv(data_dir)
        chem_assays = self.df['Assay ID'].unique()

        assert len(set(chem_assays) - set(prot_assays)) == 0, 'Missing targets'
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
            row = self.df.iloc[idx]
            cur_assay = row['Assay ID']

            # Extract the specific columns
            cur_backbone = row['Backbone']
            cur_chains = row['Chains']
            cur_assay_type = row['Assay Type']
            cur_label = row['Label']

            cur_protein = torch.load(os.path.join(self.protein_graph_dir, cur_assay+'.pt'))
            cur_tok_chain = torch.tensor(self.tokenizer.encode(cur_chains).ids, dtype=torch.int32)
            cur_tok_backbone = torch.tensor(self.tokenizer.encode(cur_backbone).ids, dtype=torch.int32)

            add_info = torch.tensor([LABEL_MAP[cur_label], INTERACTION_TYPE.get(cur_assay_type, UNKNOWN_TYPE)])
            return cur_tok_backbone, cur_tok_chain, cur_protein, add_info


class FSMolDataSet(Dataset):
    def __init__(self, smiles, smiles_backbones, assay_ids, assay_types, labels, tokenizer,
                 calc_rf=False, use_backbone=False, min_clf_samples=300, min_roc_auc=0.75, num_sites=2, seed=0):
        self.assays = {}

        for cur_smiles, cur_backbone, cur_assay_id, cur_assay_type, cur_label in zip(smiles, smiles_backbones, assay_ids, assay_types, labels):
            if cur_assay_id not in self.assays:
                self.assays[cur_assay_id] = {'active': [], 'active_backbone': [], 'inactive': [],
                                             'inactive_backbone': [], 'assay_id': cur_assay_id,
                                             'active_smiles': [], 'inactive_smiles': [],
                                             'assay_type': INTERACTION_TYPE.get(cur_assay_type, UNKNOWN_TYPE)}
            if use_backbone:
                if isinstance(cur_backbone, float) and np.isnan(cur_backbone):
                    cur_backbone = 'None'

                if cur_backbone == 'None':
                    cur_backbone = add_attachment_points(cur_smiles, n=num_sites, seed=seed)
                model_input = np.array(tokenizer.encode(cur_backbone).ids)
                model_input = torch.from_numpy(model_input).long()

            if cur_label == 0:
                self.assays[cur_assay_id]['inactive'].append(model_input)
                self.assays[cur_assay_id]['inactive_backbone'].append(cur_backbone)
                self.assays[cur_assay_id]['inactive_smiles'].append(cur_smiles)
            else:
                self.assays[cur_assay_id]['active'].append(model_input)
                self.assays[cur_assay_id]['active_backbone'].append(cur_backbone)
                self.assays[cur_assay_id]['active_smiles'].append(cur_smiles)

        self.tasks = [assay for assay in self.assays.values()]
        if calc_rf:
            for task in self.tasks:
                if len(task['active']) + len(task['inactive']) > min_clf_samples:
                    positive_smiles = [cur_smile for cur_smile in task['active_smiles']]
                    negative_smiles = [cur_smile for cur_smile in task['inactive_smiles']]
                    clf, roc_auc, fpr, tpr, thresholds = get_clf(positive_smiles, negative_smiles)
                    if roc_auc > min_roc_auc:
                        task['clf'] = clf
                        task['threshold'] = thresholds[np.argmax(tpr - fpr)]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, item):
        return self.tasks[item] 


if __name__ == "__main__":
    from utils import read_csv, load_tokenizer_from_file
    tokenizer = load_tokenizer_from_file ('./data/base_tok.json')
    protein_graph_dir = './test_graphs'

    test_non_chiral_smiles, test_backbones, test_chains, test_assay_ids, test_types, test_labels = read_csv('./data/fsmol/test.csv')


    test_ds = FSMolDataSet(test_non_chiral_smiles, test_backbones, test_assay_ids, test_types, test_labels,
                        './test_graphs', tokenizer, calc_rf=True, use_backbone=True)
    
    for i in test_ds:
        pass
