import random
from datetime import datetime
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from models import InteractionEncoder, InteractionTranslator, TransformerEncoder, TransformerDecoder
from utils import calc_tani_sim, get_fp, smiles_valid, get_canonical_smiles, read_csv, load_tokenizer_from_file, \
    top_p_decode, top_k_decode, smiles_valid, reconstruct_from_core_and_chains
from datasets import FSMolDataSet
import argparse
from tqdm import tqdm
from rdkit import RDLogger      
RDLogger.DisableLog('rdApp.*')                                                                                                                                                         

def generate_molecules(model, memory, uncond_memory, device, tokenizer, max_len, start_symbol, end_symbol,
                       guidance_scale, num_samples, sampling_method, p, k, orig_core, orig_molecule,
                       mol_backbone=True):
    generated_mols = []
    for _ in range(num_samples):
        if sampling_method == 'top_p':
            cur_sample = top_p_decode(model, cond_memory=memory, uncond_memory=uncond_memory, max_len=max_len,
                               start_symbol=start_symbol, end_symbol=end_symbol, device=device,
                               guidance_scale=guidance_scale, p=p)
        elif sampling_method == 'top_k':
            cur_sample = top_k_decode(model, cond_memory=memory, uncond_memory=uncond_memory, max_len=max_len,
                               start_symbol=start_symbol, end_symbol=end_symbol, device=device,
                               guidance_scale=guidance_scale, k=k)
        else:
            print('no support for this sampling method')
            exit()
        cur_sample = cur_sample.cpu().numpy().tolist()[0]
        if mol_backbone:
            cur_sample = tokenizer.decode(cur_sample, skip_special_tokens=False)
            cur_sample = cur_sample.replace(tokenizer.id_to_token(start_symbol), '')
            cur_sample = cur_sample.replace(tokenizer.id_to_token(end_symbol), '')
            cur_sample = reconstruct_from_core_and_chains(orig_core, cur_sample)
        else:
            cur_sample = tokenizer.decode(cur_sample)

        if smiles_valid(cur_sample):
            generated_mols.append(cur_sample)
        else:
            generated_mols.append(orig_molecule)
    return generated_mols


def evaluate_task(opt_molecules: dict, clf, threshold: float, similarity_threshold: float = 0.4, num_samples: int = 10):
    all_success_rates, all_diversities, all_similarities = [], [], []
    all_valid_samples, num_molecules = [], 0
    for orig_mol in opt_molecules.keys():
        for idx, opt_mol in enumerate(opt_molecules[orig_mol]):
            num_molecules += 1
            if smiles_valid(opt_mol) and get_canonical_smiles(opt_mol) != get_canonical_smiles(orig_mol):
                all_valid_samples.append(opt_mol)
    for _ in range(num_samples):
        chosen_mols, similarities, tot_success = [], [], 0
        for orig_mol in opt_molecules.keys():
            candidates = [opt_mol for opt_mol in opt_molecules[orig_mol] if smiles_valid(opt_mol)
                          and get_canonical_smiles(opt_mol) != get_canonical_smiles(orig_mol)]
            if len(candidates) == 0:
                continue
            chosen_candidate = random.choice(candidates)
            chosen_mols.append(chosen_candidate)
            cur_sim = calc_tani_sim(orig_mol, chosen_candidate)
            similarities.append(cur_sim)
            cur_score = clf.predict_proba(np.reshape(get_fp(chosen_candidate), (1, -1)))
            cur_score = cur_score[0][1]
            if cur_score > threshold and cur_sim > similarity_threshold:
                tot_success += 1
        all_success_rates.append(tot_success / len(opt_molecules.keys()))
        all_diversities.append(len(set(chosen_mols)) / max(1, len(chosen_mols)))
        all_similarities.append(sum(similarities) / max(1, len(chosen_mols)))
    avg_diversity, std_diversity = np.mean(all_diversities), np.std(all_diversities)
    avg_similarity, std_similarity = np.mean(all_similarities), np.std(all_similarities)
    avg_success, std_success = np.mean(all_success_rates), np.std(all_success_rates)
    validity = len(all_valid_samples) / num_molecules
    return validity, avg_diversity, std_diversity, avg_similarity, std_similarity, avg_success, std_success


if __name__ == '__main__':
    cur_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_path = "/Users/amitaflalo/Desktop/cfom/models/2023_05_24_14_46_54/epoch20"
    
    with open(f'{model_path}/hyper_params.json', 'r') as f:
        hyper_params = json.load(f)

    optimization_parameters = {
        'guidance_scale': hyper_params['guidance_scale'],
        'sampling_method': 'top_p',  # can be 'top_p' or 'top_k'
        'num_molecules_generated': 20,
        'p': 1.,
        'k': 40,
        'model_used': model_path
    }
    print(f'{optimization_parameters}')

    hyper_params.update(optimization_parameters)
    output_dir = Path(f'./results/{cur_time}')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(f'{str(output_dir)}/hyper_params.json', 'w') as f:
        json.dump(hyper_params, f, indent=4)

    train_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_train.npz')
    valid_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_valid.npz')
    test_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_test.npz')
    
    train_non_chiral_smiles, train_backbones, train_chains, train_assay_ids, train_types, train_labels = read_csv('./data/fsmol/train.csv')
    valid_non_chiral_smiles, valid_backbones, valid_chains, valid_assay_ids, valid_types, valid_labels = read_csv('./data/fsmol/valid.csv')
    test_non_chiral_smiles, test_backbones, test_chains, test_assay_ids, test_types, test_labels = read_csv('./data/fsmol/test.csv')

    tokenizer = load_tokenizer_from_file(f'{model_path}/tokenizer_object.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = InteractionTranslator(prot_encoder=InteractionEncoder(2 + 1024, hyper_params['embedding_dim']),
                                  mol_encoder=TransformerEncoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                                                 hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['encoder_n_head'],
                                                                 n_layers=hyper_params['encoder_n_layer'],
                                                                 max_length=hyper_params['max_mol_len'],
                                                                 pad_token=tokenizer.token_to_id('<pad>')),
                                  decoder=TransformerDecoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                                             hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['decoder_n_head'],
                                                             n_layers=hyper_params['decoder_n_layer'],
                                                             max_length=hyper_params['max_mol_len']))
    
    model.load_state_dict(torch.load(f'{model_path}/model.pt', map_location=device))
    model.to(device)

    # test dataset 
    test_ds = FSMolDataSet(test_non_chiral_smiles, test_backbones, test_assay_ids, test_types, test_labels,
                           test_prot_embeds, tokenizer, calc_rf=True, use_backbone=True)
    model.eval()
    with torch.no_grad():
        for task in tqdm(test_ds.tasks):
            if task.get('clf', None) is None:
                continue
            cur_interaction = torch.from_numpy(np.append(task['protein'], [1, task['assay_type']]))
            cur_interaction = cur_interaction.float().to(device)
            prot_embed = torch.reshape(model.prot_encoder(cur_interaction), (1, 1, -1))
            zero_prot_embed = torch.zeros_like(prot_embed).to(device)
            all_mol_embeds = []
            idx = 0
            while idx < len(task['inactive']):
                end = min(idx + hyper_params['bs'], len(task['inactive']))
                batch = task['inactive'][idx: end]
                batch_size = end - idx
                idx = end
                cur_mols = torch.stack(batch).to(device)
                cur_mols_embeds = model.mol_encoder(cur_mols)
                all_mol_embeds.extend([cur_mols_embeds[i, :, :].cpu() for i in range(batch_size)])

            current_sample_pairs = {}
            for orig_mol_embed, orig_mol, orig_backbone in zip(all_mol_embeds, task['inactive_smiles'],
                                                               task['inactive_backbone']):
                orig_mol_embed = orig_mol_embed.to(device)
                orig_mol_embed = torch.unsqueeze(orig_mol_embed, dim=0)
                memory = torch.concat([prot_embed, orig_mol_embed], dim=1)
                uncond_memory = torch.concat([zero_prot_embed, orig_mol_embed], dim=1)
                generated_mols = generate_molecules(model.decoder, memory, uncond_memory, device, tokenizer,
                                                    hyper_params['max_mol_len'],
                                                    tokenizer.token_to_id('<bos>'),
                                                    tokenizer.token_to_id('<eos>'),
                                                    hyper_params['guidance_scale'],
                                                    hyper_params['num_molecules_generated'],
                                                    hyper_params['sampling_method'],
                                                    hyper_params['p'],
                                                    hyper_params['k'],
                                                    orig_backbone,
                                                    orig_mol)
                current_sample_pairs[orig_mol] = generated_mols
            cur_output_dir = Path(f'{str(output_dir)}/{task["assay_id"]}')
            cur_output_dir.mkdir(parents=True, exist_ok=True)
            with open(f'{str(cur_output_dir)}/samples.txt', 'w') as f:
                lines = []
                for orig_mol in current_sample_pairs.keys():
                    for sample in current_sample_pairs[orig_mol]:
                        lines.append(f'{orig_mol} {sample}\n')
                f.writelines(lines)

    benchmarks = [str(output_dir)]
    test_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_test.npz')
    test_non_chiral_smiles, test_backbones, test_chains, test_assay_ids, test_types, test_labels = read_csv('./data/fsmol/test.csv')
    tokenizer = load_tokenizer_from_file(f'{model_path}/tokenizer_object.json')
    test_ds = FSMolDataSet(test_non_chiral_smiles, test_backbones, test_assay_ids, test_types, test_labels,
                           test_prot_embeds, tokenizer, calc_rf=True, use_backbone=True)
    for cur_method in benchmarks:
        cur_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        cur_data = {'assay_id': [], 'validity': [], 'diversity': [], 'diversity_std': [], 'similarity': [],
                    'similarity_std': [], 'success_rate': [], 'success_rate_std': []}
        all_interesting_pairs = []
        tasks = [task for task in test_ds.tasks if task.get('clf', None) is not None]
        for idx, task in enumerate(tasks):
            print(f'currently evaluating task number {idx} out of {len(tasks)} for method {cur_method}')
            with open(f'{cur_method}/{task["assay_id"]}/samples.txt', 'r') as f:
                all_pairs = [line.replace('\n', '').split(' ') for line in f.readlines()]
            opt_samples = {}
            for pair in all_pairs:
                if pair[0] in opt_samples:
                    opt_samples[pair[0]].append(pair[1])
                else:
                    opt_samples[pair[0]] = [pair[1]]
            validity, avg_diversity, std_diversity, avg_similarity, std_similarity, avg_success, std_success = \
                evaluate_task(opt_samples, task['clf'], threshold=task['threshold'], similarity_threshold=0.4)
            print(f'suc: {avg_success}')
            cur_data['assay_id'].append(task['assay_id'])
            cur_data['validity'].append(validity)
            cur_data['diversity'].append(avg_diversity)
            cur_data['diversity_std'].append(std_diversity)
            cur_data['similarity'].append(avg_similarity)
            cur_data['similarity_std'].append(std_similarity)
            cur_data['success_rate'].append(avg_success)
            cur_data['success_rate_std'].append(std_success)
        pd.DataFrame.from_dict(cur_data).to_csv(f'{cur_method}/evaluation_{cur_time}.csv')
