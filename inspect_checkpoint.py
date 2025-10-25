# inspect_checkpoint.py
import torch

checkpoint_path = 'model_zoo/DeepFM/Avazu/DeepFM_avazu_x4_001/avazu_x4_3bbbc4c9/DeepFM_avazu_gen.model'

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

print("\nCheckpoint structure:")
if isinstance(checkpoint, dict):
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor {value.shape}")
        elif hasattr(value, 'state_dict'):
            print(f"  {key}: Model/Module")
        else:
            print(f"  {key}: {type(value)}")
