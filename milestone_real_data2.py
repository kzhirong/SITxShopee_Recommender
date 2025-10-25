import sys
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import importlib
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config
from fuxictr.pytorch.dataloaders import RankDataLoader
import torch
from transformers import AutoModel
sys.path.append('model_zoo/DeepFM/src')
from projector import FeatureProjector
from transformers import AutoTokenizer

print("="*60)
print("MILESTONE: Real Data → Trained Encoder → Projector → LLM")
print("="*60)

# 1. Load configuration
print("\n1. Loading configuration...")
config_path = 'model_zoo/DeepFM/config'
experiment_id = 'DeepFM_avazu_gen'
params = load_config(config_path, experiment_id)

# Fix paths
for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
    if key in params:
        params[key] = params[key].replace('../../', '')
        # Fix the .csv.parquet issue
        if key != 'data_root' and '.csv' in params[key]:
            params[key] = params[key].replace('.csv', '.parquet')

print(f"   Embedding dim: {params['embedding_dim']}")

# 2. Load feature map
print("\n2. Loading feature map...")
data_dir = os.path.join(params['data_root'], params['dataset_id'])
feature_map_json = os.path.join(data_dir, "feature_map.json")
feature_map = FeatureMap(params['dataset_id'], data_dir)
feature_map.load(feature_map_json, params)
print(f"   Number of features: {len(feature_map.features)}")

# 3. Load full model (but we'll only use parts of it)
print("\n3. Loading DeepFM model...")
model_src_path = "model_zoo.DeepFM.src"
src = importlib.import_module(model_src_path)
model_class = getattr(src, params['model'])
model = model_class(feature_map, **params)

checkpoint_path = 'model_zoo/DeepFM/Avazu/DeepFM_avazu_x4_001/avazu_x4_3bbbc4c9/DeepFM_avazu_gen.model'
model.load_weights(checkpoint_path)
model.eval()
print(f"   Full model loaded")

# 4. Extract only the components we need
print("\n4. Extracting components needed for POC...")
embedding_layer = model.embedding_layer
gen_encoder = model.gen
print(f"   ✓ Embedding layer extracted")
print(f"   ✓ Generative encoder (gen) extracted")
print(f"   Note: Ignoring lr_layer, mlp, linear_out (not needed for POC)")

# 5. Load real data
print("\n5. Loading real Avazu test data...")
params['num_workers'] = 0
test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
batch = next(iter(test_gen))
feature_names = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                 'site_category', 'app_id', 'app_domain', 'app_category',
                 'device_id', 'device_ip', 'device_model', 'device_type',
                 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18',
                 'C19', 'C20', 'C21', 'weekday', 'weekend']

X_sample = torch.stack([batch[feat][:2] for feat in feature_names], dim=1)
y_sample = batch['click'][:2]
print(f"   Loaded 2 real samples: {X_sample.shape}")
print(f"   Ground truth labels: {y_sample.squeeze().tolist()}")

# 6. Pipeline: Embeddings
print("\n6. Step 1: Feature IDs → Embeddings...")
with torch.no_grad():
    batch_dict_sample = {feat: batch[feat][:2] for feat in feature_names}
    embedded = embedding_layer(batch_dict_sample)
print(f"   Raw feature IDs: {X_sample.shape}")
print(f"   → Embedded:      {embedded.shape}")

# 7. Pipeline: Trained Encoder
print("\n7. Step 2: Embeddings → Trained Encoder...")
with torch.no_grad():
    encoder_output = gen_encoder(embedded)
    # Handle if it returns a tuple
    if isinstance(encoder_output, tuple):
        encoded = encoder_output[0]
    else:
        encoded = encoder_output
print(f"   Embedded:        {embedded.shape}")
print(f"   → Encoded:       {encoded.shape}")

# 8. Pipeline: Projector (NEW component)
print("\n8. Step 3: Encoded → Projector (NEW)...")
projector = FeatureProjector(feature_dim=16, llm_dim=1024)
projected = projector(encoded)
print(f"   Encoded:         {encoded.shape}")
print(f"   → Projected:     {projected.shape}")

# 8.5 Load LLM and tokenizer (NEW - load once)
print("\n8.5. Loading LLM and tokenizer...")
qwen = AutoModel.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print(f"   ✓ LLM loaded: Qwen3-0.6B")
print(f"   ✓ Tokenizer loaded")

# 9. Create text prompt and get embeddings
print("\n9. Step 4: Create text prompt...")
prompt = "Given this user's ad context, predict click probability."
tok = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    text_embeds = qwen.get_input_embeddings()(tok.input_ids)
print(f"   Prompt: '{prompt}'")
print(f"   → Text embeddings: {text_embeds.shape}")

# 10. Concatenate text + projected features
print("\n10. Step 5: Concatenate text + projected features...")
# Use only first row for clarity
projected_1row = projected[:1]  # [1, 24, 1024]
inputs_embeds = torch.cat([text_embeds, projected_1row], dim=1)
print(f"   Text embeddings:     {text_embeds.shape}")
print(f"   Projected features:  {projected_1row.shape}")
print(f"   → Combined:          {inputs_embeds.shape}")

# 11. Feed to LLM
print("\n11. Step 6: Feed combined embeddings to LLM...")
with torch.no_grad():
    llm_output = qwen(inputs_embeds=inputs_embeds, return_dict=True)
    final_hidden = llm_output.last_hidden_state[:, -1, :]  # Last token
print(f"   Combined input:      {inputs_embeds.shape}")
print(f"   → LLM final hidden:  {final_hidden.shape}")

# 12. Add simple prediction head
print("\n12. Step 7: Prediction head (untrained)...")
prediction_head = torch.nn.Linear(1024, 1)
with torch.no_grad():
    logits = prediction_head(final_hidden)
    prob = torch.sigmoid(logits)
print(f"   Final hidden:        {final_hidden.shape}")
print(f"   → Raw logit:         {logits.item():.4f}")
print(f"   → Click probability: {prob.item():.4f}")
print(f"   Ground truth:        {y_sample[0].item()}")

# 13. Summary
print("\n" + "="*60)
print("COMPLETE PIPELINE SUMMARY")
print("="*60)
print("\nFrom TRAINED model:")
print("  1. Embedding Layer:  Feature IDs → 16-dim embeddings")
print("  2. Gen Encoder:      Transform embeddings")
print("\nNEW hybrid components:")
print("  3. Projector:        16-dim → 1024-dim (LLM space)")
print("  4. Text Prompt:      Tokenize task description")
print("  5. Concatenate:      [text tokens] + [projected features]")
print("  6. LLM:              Process combined sequence")
print("  7. Prediction Head:  Final hidden → click probability")
print("\n" + "="*60)
print("✅ HYBRID PIPELINE COMPLETE")
print("="*60)
print(f"\nResult for 1 sample:")
print(f"  Predicted probability: {prob.item():.4f}")
print(f"  Ground truth:          {y_sample[0].item()}")
print(f"  Note: Prediction head is UNTRAINED (random weights)")
