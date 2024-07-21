from tokenizers import decoders, models, pre_tokenizers, processors, Tokenizer
from sklearn.metrics import roc_auc_score, roc_curve
from SmilesPE.pretokenizer import atomwise_tokenizer
from sklearn.ensemble import RandomForestClassifier
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
import scaffoldgraph as sg
import pandas as pd
import numpy as np
import itertools
import random
import torch
import re


def get_fp(smiles: str):
    fp_obj = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2, nBits=2048,
                                                   useChirality=False)
    fp = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_obj, fp)
    return fp


def get_clf(positive_smiles, negative_smiles, test_fraction=0.2, num_estimators=100, max_depth=2):
    positive_fp = [get_fp(smile) for smile in positive_smiles]
    negative_fp = [get_fp(smile) for smile in negative_smiles]
    pos_ratio = len(positive_fp) / (len(negative_fp) + len(positive_fp))
    num_test = int(test_fraction * (len(negative_fp) + len(positive_fp)))
    pos_test_fp = positive_fp[:int(pos_ratio*num_test)]
    pos_train_fp = positive_fp[int(pos_ratio * num_test):]
    neg_test_fp = negative_fp[:int((1-pos_ratio)*num_test)]
    neg_train_fp = negative_fp[int((1-pos_ratio) * num_test):]
    X = np.stack(pos_train_fp + neg_train_fp, axis=0)
    y = np.concatenate([np.ones(len(pos_train_fp)), np.zeros(len(neg_train_fp))])
    sample_weights = [1 if cur_label == 0 else len(neg_train_fp) / len(pos_train_fp) for cur_label in y]
    clf = RandomForestClassifier(max_depth=max_depth, random_state=0, n_estimators=num_estimators)
    clf.fit(X, y, sample_weight=sample_weights)
    positives = np.ones(len(pos_test_fp))
    negatives = np.zeros(len(neg_test_fp))
    labels = np.concatenate([positives, negatives], axis=0)
    test_samples = np.concatenate([pos_test_fp, neg_test_fp], axis=0)
    probs = clf.predict_proba(test_samples)
    roc_auc = roc_auc_score(labels, probs[:, 1])
    fpr, tpr, thresholds = roc_curve(labels, probs[:, 1], pos_label=1)
    return clf, roc_auc, fpr, tpr, thresholds


def read_csv(path: str):
    df = pd.read_csv(path)
    non_chiral_smiles = df.iloc[:, 1].tolist()
    backbones = df.iloc[:, 2].tolist()
    chains = df.iloc[:, 3].tolist()
    assay_ids = df.iloc[:, 4].tolist()
    types = df.iloc[:, 5].tolist()
    labels = df.iloc[:, 6].tolist()
    return non_chiral_smiles, backbones, chains, assay_ids, types, labels


def calc_tani_sim(mol1_smiles, mol2_smiles):
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048, useChirality=False)
    tani_sim = DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)
    return tani_sim


def _generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device), diagonal=0) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_target_masks(tgt, device, pad_idx):
    tgt_seq_len = tgt.shape[1]
    tgt_mask = _generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = (tgt == pad_idx)
    return tgt_mask, tgt_padding_mask


def top_p_decode(model, cond_memory, uncond_memory, max_len, start_symbol, end_symbol, device, guidance_scale=1.,
                 p=0.9, prefix=None):
    if prefix is None:
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        generation_length = max_len - 1
    else:
        ys = torch.tensor(prefix).reshape(1, -1).to(device)
        generation_length = max_len - len(prefix)
    for i in range(generation_length):
        if isinstance(cond_memory, tuple):
            cond_memory = (cond_memory[0].to(device), cond_memory[1].to(device))
        else:
            cond_memory = cond_memory.to(device)
        # uncond_memory = torch.zeros_like(cond_memory).to(device)
        cond_out = model(ys, cond_memory, target_mask=None, target_padding_mask=None)
        if guidance_scale != 1.:
            uncond_out = model(ys, uncond_memory, target_mask=None, target_padding_mask=None)
            out = uncond_out + (cond_out - uncond_out) * guidance_scale
        else:
            out = cond_out
        probs = torch.softmax(out, dim=2)
        sorted_probs, sorted_indices = torch.sort(probs[0, -1, :], descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=0)
        top_p_mask = cum_probs > p
        if torch.all(top_p_mask):
            top_p_mask[0] = False
        sorted_probs[top_p_mask] = 0
        remainder_sum = torch.sum(sorted_probs)
        sorted_probs = sorted_probs / remainder_sum
        next_sorted_token = torch.multinomial(sorted_probs, 1)
        next_token_id = sorted_indices[next_sorted_token.item()]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(ys.data).fill_(next_token_id)], dim=1)
        if next_token_id == end_symbol:
            break
    return ys


def top_k_decode(model, cond_memory, uncond_memory, max_len, start_symbol, end_symbol, device, guidance_scale=1.,
                 k=3, prefix=None):
    if prefix is None:
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        generation_length = max_len - 1
    else:
        ys = torch.tensor(prefix).reshape(1, -1).to(device)
        generation_length = max_len - len(prefix)
    for i in range(generation_length):
        cond_memory = cond_memory.to(device)
        cond_out = model(ys, cond_memory, target_mask=None, target_padding_mask=None)
        if guidance_scale != 1.:
            uncond_out = model(ys, uncond_memory, target_mask=None, target_padding_mask=None)
            out = uncond_out + (cond_out - uncond_out) * guidance_scale
        else:
            out = cond_out
        probs = torch.softmax(out, dim=2)
        sorted_probs, sorted_indices = torch.sort(probs[0, -1, :], descending=True)
        sorted_probs[k:] = 0
        remainder_sum = torch.sum(sorted_probs)
        sorted_probs = sorted_probs / remainder_sum
        next_sorted_token = torch.multinomial(sorted_probs, 1)
        next_token_id = sorted_indices[next_sorted_token.item()]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(ys.data).fill_(next_token_id)], dim=1)
        if next_token_id == end_symbol:
            break
    return ys


def create_smiles_tokenizer(all_smiles, max_num_chains=25):
    alphabet = set()
    for cur_smiles in all_smiles:
        toks = atomwise_tokenizer(cur_smiles)
        alphabet.update(toks)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    chain_sites = [f'[{i + 1}*]' for i in range(max_num_chains)]
    special_tokens = ["<pad>", "<bos>", "<eos>", "."]
    special_tokens.extend(chain_sites)
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_tokens(list(alphabet))
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        pair=None,
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")), ("<eos>", tokenizer.token_to_id("<eos>"))],
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def load_tokenizer_from_file(file_path: str):
    return Tokenizer.from_file(file_path)


def smiles_valid(smiles):
    if smiles is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def random_smiles(smiles):
    m1 = Chem.MolFromSmiles(smiles)
    m1.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
    return Chem.MolToSmiles(m1)


def get_scaffold(smiles, shuffle):
    m = Chem.MolFromSmiles(smiles)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(m)
    scaffold = Chem.MolToSmiles(scaffold_mol)
    scaffold = get_canonical_smiles(scaffold)
    if scaffold == '':
        return smiles
    if shuffle:
        scaffold = random_smiles(scaffold)
    return scaffold


def get_canonical_smiles(smiles, chirality=True):
    chiral = 1 if chirality else 0
    return Chem.CanonSmiles(smiles, useChiral=chiral)


def get_num_fused_rings(mol):
    """rings that share at least one atom with another ring"""
    num_fused_rings = set()
    atom_rings = [set(cur_ring) for cur_ring in mol.GetRingInfo().AtomRings()]
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            if not atom_rings[i].isdisjoint(atom_rings[j]):
                num_fused_rings.update([i, j])
    return len(num_fused_rings)


def get_core_and_chains(smiles, chains_weight_threshold=0.):
    m1 = Chem.MolFromSmiles(smiles)
    if m1 is None:
        return None, None
    num_rings = AllChem.CalcNumRings(m1)
    if num_rings == 0:
        return None, None
    num_fused_rings = get_num_fused_rings(m1)
    frags = sg.tree_frags_from_mol(m1)
    clean_core = frags[0]
    m1_weight = AllChem.CalcExactMolWt(m1)
    for i, cur_frag in enumerate(frags):
        try:
            frag_weight = AllChem.CalcExactMolWt(cur_frag)
        except:
            return None, None
        if (m1_weight - frag_weight) / m1_weight > chains_weight_threshold or num_fused_rings != get_num_fused_rings(cur_frag):
            break
        clean_core = cur_frag
    core = Chem.ReplaceSidechains(m1, clean_core)
    if core is None:
        return None, None
    core_smiles = Chem.MolToSmiles(core)
    sanitized_core = Chem.MolFromSmiles(core_smiles)
    if sanitized_core is None:
        return None, None
    core_clean_smiles = Chem.MolToSmiles(sanitized_core)
    side_chains = Chem.ReplaceCore(m1, clean_core)
    side_chains_smiles = Chem.MolToSmiles(side_chains)
    if core_clean_smiles == '' or side_chains_smiles == '':
        return None, None
    return core_clean_smiles, side_chains_smiles


def reconstruct_from_core_and_chains(core, chains):
    chains = Chem.MolFromSmiles(chains)
    core_clean = Chem.MolFromSmiles(core)
    if core_clean is None or chains is None:
        return None
    try:
        side_chain_mols = Chem.GetMolFrags(chains, asMols=True)
    except:
        return None
    for mol in side_chain_mols:
        if len([atom.GetSmarts() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]) == 0:
            return None
    side_chain_tags = [[atom.GetSmarts() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"][0]
                       for mol in side_chain_mols]
    side_chain_indexes = [[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"][0]
                          for mol in side_chain_mols]
    side_chain_dict = dict(zip(side_chain_tags, zip(side_chain_mols, side_chain_indexes)))

    core_side_chain_tags = [atom.GetSmarts() for atom in core_clean.GetAtoms() if atom.GetSymbol() == "*"]
    core_side_chain_tags = [re.sub(r'(\[\d+\*).*\]', r'\1]', x) for x in core_side_chain_tags]
    current_core = core_clean
    for tag in core_side_chain_tags:
        replacement = side_chain_dict.get(tag, None)
        if replacement is None:
            return None
        new_core = Chem.ReplaceSubstructs(current_core,
                                          Chem.MolFromSmiles(tag),
                                          replacement[0],
                                          replacementConnectionPoint=side_chain_dict[tag][1],
                                          useChirality=1)
        if new_core[0] is None:
            return None
        current_core = new_core[0]
    reconstructed_smiles = Chem.MolToSmiles(current_core)
    reconstructed_smiles_clean = re.sub(r'\[\d+\*\]', '', reconstructed_smiles)
    if not smiles_valid(reconstructed_smiles_clean):
        return None
    recon = Chem.MolToSmiles(Chem.MolFromSmiles(reconstructed_smiles_clean))
    canon = Chem.CanonSmiles(recon, useChiral=0)
    return canon


def isotopize_dummies(fragment, isotope):
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == "*":
            atom.SetIsotope(isotope)
    return fragment


def add_attachment_points(smiles, n, seed=None, fg_weight=0, fg_list=[]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if seed is not None:
        random.seed(seed)
    if len(fg_list) == 0:
        fg_list = [(1.0, "[c,C][H]", "C*")]
    else:
        epsilon = 1.e-10
        fg_weights = itertools.accumulate([fg_weight / len(fg_list)
                                           for i in range(len(fg_list))]
                                          + [1.0 + epsilon - fg_weight])
        fg_list = list(zip(fg_weights,
                           [x[0] for x in fg_list] + ["[c,C][H]"],
                           [x[1] for x in fg_list] + ["C*"]))

    current_mol = Chem.AddHs(mol)
    current_mol.UpdatePropertyCache()
    current_attachment_index = 1
    for i in range(n):
        next_mol = []
        max_tries = 100
        current_try = 0
        while len(next_mol) == 0:
            the_choice = [x for x in fg_list if x[0] >= random.random()][0]
            the_target = Chem.MolFromSmarts(the_choice[1])
            the_replacement = isotopize_dummies(Chem.MolFromSmiles(the_choice[2]), current_attachment_index)
            next_mol = Chem.ReplaceSubstructs(current_mol, the_target, the_replacement)
            current_try += 1
            if current_try >= max_tries:
                break  # we failed
        if current_try >= max_tries:
            continue  # skip and try again (we will return less than n attachment points)
        current_attachment_index += 1
        current_mol = random.choice(next_mol)
        current_mol.UpdatePropertyCache()

    current_mol = Chem.RemoveHs(current_mol)
    current_mol.UpdatePropertyCache()

    current_smiles = Chem.MolToSmiles(current_mol)
    return current_smiles

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 0)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 14)
