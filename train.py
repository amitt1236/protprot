from models import InteractionEncoder, InteractionTranslator, TransformerEncoder, TransformerDecoder
from utils import read_csv, create_target_masks, load_tokenizer_from_file
from datasets import InteractionsDataset, FSMolDataSet
from torch_geometric.loader import DataLoader
from evaluate import generate_molecules
from evaluate import evaluate_task
from datetime import datetime
from torch.optim import Adam, AdamW
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import gc
import copy
import torch
import os

def training(model, optimizer, tokenizer, loader, epochs, device, cur_epoch=0):

    print(f'model has: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')
    recon_loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in tqdm(range(cur_epoch, epochs)):
        recon_losses = []
        for cur_tok_backbone, cur_tok_chain, cur_protein, add_info in loader:

            optimizer.zero_grad()
            cur_protein_graph = cur_protein

            encoder_tokenized_in = cur_tok_backbone.to(device)

            decoder_tokenized_in, decoder_tokenized_tgt = cur_tok_backbone.to(device)[:, :-1], cur_tok_chain[:, 1:].to(device)
            target_mask, target_padding_mask = create_target_masks(decoder_tokenized_in, device, tokenizer.token_to_id('<pad>'))

            mol_embeds = model.mol_encoder(encoder_tokenized_in)
            prot_embed = torch.unsqueeze(torch.concat((model.prot_encoder(cur_protein_graph), add_info.to(device)), dim=1), dim=1)
            memory = torch.concat([prot_embed, mol_embeds], dim=1)

            logits = model.decoder(decoder_tokenized_in, memory, target_mask=target_mask,
                                    target_padding_mask=target_padding_mask)
            
            recon_loss = recon_loss_fn(logits.reshape(-1, logits.shape[-1]), decoder_tokenized_tgt.reshape(-1).to(torch.long))
            recon_loss.backward()
            optimizer.step()
            recon_losses.append(recon_loss)
        
        print(f'epoch: {epoch} loss : {sum(recon_losses) / len(recon_losses)}')
        if epoch != 0 and epoch > 10:
            cur_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            output_dir = Path(f'./models/{cur_time}/epoch{epoch + 1}')
            output_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            },f'{str(output_dir)}/model.pt')
            tokenizer.save(f'{str(output_dir)}/tokenizer_object.json')
            print("*"  * 20 + "model saved" + "*" * 20)
        
        gc.collect()

def validation_step(model, tokenizer, hyper_params, device, split=1, prot_path = './test_graphs'):
    from torch_geometric.data.batch import Batch
    if split == 0:
        valid_non_chiral_smiles, valid_backbones, valid_chains, valid_assay_ids, valid_types, valid_labels = read_csv('./data/fsmol/valid.csv')
        dataset = FSMolDataSet(valid_non_chiral_smiles, valid_backbones, valid_assay_ids, valid_types, valid_labels,
                                tokenizer, calc_rf=True, use_backbone=hyper_params['mol_backbone'])
    
    else:
        test_non_chiral_smiles, test_backbones, test_chains, test_assay_ids, test_types, test_labels = read_csv('./data/fsmol/test.csv')
        dataset = FSMolDataSet(test_non_chiral_smiles, test_backbones, test_assay_ids, test_types, test_labels,
                                tokenizer, calc_rf=True, use_backbone=hyper_params['mol_backbone'])
    model.eval()
    with torch.no_grad():
        for task in tqdm(dataset):
            if task.get('clf', None) is None:
                continue            
            
            cur_protein_graph = torch.load(os.path.join(prot_path, task['assay_id']+'.pt')).to(device)
            prot_embed = torch.unsqueeze(model.prot_encoder(Batch.from_data_list([cur_protein_graph])), dim=1)
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
                
            opt_molecules = {}
            for orig_mol_embed, orig_mol, orig_backbone in zip(all_mol_embeds, task['inactive_smiles'], task['inactive_backbone']):

                orig_mol_embed = orig_mol_embed.to(device)
                orig_mol_embed = torch.unsqueeze(orig_mol_embed, dim=0)
                memory = torch.concat([prot_embed, orig_mol_embed], dim=1)

                uncond_memory = None
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
                                                    orig_mol,
                                                    mol_backbone=hyper_params['mol_backbone'])
                opt_molecules[orig_mol] = generated_mols

            validity, avg_diversity, std_diversity, avg_similarity, std_similarity, avg_success, std_success = \
                evaluate_task(opt_molecules, task['clf'], threshold=task['threshold'],
                                similarity_threshold=hyper_params['similarity_threshold'])

            print('*'*10)
            print(task['assay_id'])
            print(f'thresh: {task["threshold"]}')
            print(f'success rate: {avg_success}, {std_success}')
            print(f'diversity: {avg_diversity}, {std_diversity}')
            print(f'sim: {avg_similarity}, {std_similarity}')
            print(f'validity: {validity}')

def get_latest_model_dir(models_dir='./models'):
    # List all directories in the models directory
    model_dirs = [d for d in Path(models_dir).iterdir() if d.is_dir()]
    
    # Sort directories by creation time
    model_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    # Get the latest directory
    if model_dirs:
        latest_dir = model_dirs[0]
        
        # List all directories in the latest directory
        inner_dirs = [d for d in latest_dir.iterdir() if d.is_dir()]
        
        # Ensure there's only one inner directory and return it
        if len(inner_dirs) == 1:
            return str(inner_dirs[0])
        else:
            return None
    else:
        return None

    
def main():
    hyper_params = {
        'bs': 2,
        'lr': 3e-4,
        'weight_decay': 0.,
        'epochs': 100,
        'max_mol_len': 128,
        'embedding_dim': 256,
        'arch_type': 'transformer', 
        'decoder_n_layer': 2,
        'decoder_n_head': 4,
        'encoder_n_layer': 2,
        'encoder_n_head': 4,
        'unconditional_percentage': 0.,
        'guidance_scale': 1.,
        'sampling_method': 'top_p',  # can be 'top_p' or 'top_k'
        'num_molecules_generated': 20,
        'p': 1.,
        'k': 40,
        'mol_backbone': True,
        'similarity_threshold': 0.4,
        'num_samples': 10
    }
    print('*' *20, 'parameters', '*' * 20)
    print(hyper_params)
    print('*' *20, 'parameters', '*' * 20)

    train_data_dir = './data/fsmol/train_data.csv'
    protein_graph_dir = './train_graphs/train_graphs'
    tokenizer = load_tokenizer_from_file ('./data/base_tok.json')

    train_ds = InteractionsDataset(train_data_dir, protein_graph_dir, tokenizer)

    model = InteractionTranslator(prot_encoder=InteractionEncoder(hyper_params['embedding_dim'],hidden=40, ns=16 ,nv=4),
                                mol_encoder=TransformerEncoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                                                hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['encoder_n_head'],
                                                                n_layers=hyper_params['encoder_n_layer'],
                                                                max_length=hyper_params['max_mol_len'],
                                                                pad_token=tokenizer.token_to_id('<pad>')),
                                decoder=TransformerDecoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                                            hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['decoder_n_head'],
                                                            n_layers=hyper_params['decoder_n_layer'],
                                                            max_length=hyper_params['max_mol_len']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # optimizer = Adam(model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay'])
    optimizer = AdamW(model.parameters(), lr=hyper_params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, fused=True)

    load_model = False
    cur_epoch = 0
    if load_model:
        model_path = get_latest_model_dir()
        loaded = torch.load(os.path.join(model_path,'model.pt'), map_location=device)
        model.load_state_dict(loaded['model_state_dict'])
        tokenizer = load_tokenizer_from_file(os.path.join(model_path,'tokenizer_object.json'))
        # optimizer = optimizer.load_state_dict(loaded['optimizer_state_dict'])
        cur_epoch = loaded['epoch']

    torch.backends.cudnn.benchmark = True
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=6)
    training(model, optimizer, tokenizer, train_loader, 50, device, cur_epoch=cur_epoch)
    
if __name__ == "__main__":
    main()
