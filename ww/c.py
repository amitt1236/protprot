import torch

his_state_dict = torch.load('model-7.pt', map_location='cpu')
my_state_dict  = torch.load('cc.pt', map_location='cpu')

keywords = ['decoder', 'mol_encoder']

for key, value in his_state_dict.items():
    if any(keyword in key for keyword in keywords):
        if key in my_state_dict:
            print(key)
            my_state_dict[key] = value
        else:
            print(f"Key {key} not found in target state dict.")

torch.save(my_state_dict, 'mmodel.pt')