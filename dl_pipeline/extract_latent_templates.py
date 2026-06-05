import os
import sys
import torch
import numpy as np
import pickle
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dl_pipeline.model import SignLanguageTransformer
from dl_pipeline.dataset import SignLanguageDataset

def extract_latent_templates():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    train_csv = os.path.join(base_dir, 'data', 'processed_csv', 'train.csv')
    models_dir = os.path.join(base_dir, 'models', 'dl_models')
    
    model_path = os.path.join(models_dir, 'best_model.pth')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    label_map_path = os.path.join(models_dir, 'label_map.pkl')
    
    # Load Scaler and Label Map
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(label_map_path, 'rb') as f:
        label_map = pickle.load(f)
        
    num_classes = len(label_map)
    inverse_label_map = {v: k for k, v in label_map.items()}
    
    # Load Model
    model = SignLanguageTransformer(input_size=168, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Dataset
    print("Loading dataset...")
    dataset = SignLanguageDataset([train_csv], max_len=64, scaler=scaler, label_map=label_map, is_train=False)
    
    templates_by_class = defaultdict(list)
    
    print("Extracting latent templates...")
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            label_str = inverse_label_map[y.item()]
            
            # Keep only 3 templates per class for speed
            if len(templates_by_class[label_str]) >= 3:
                continue
                
            x_tensor = x.unsqueeze(0).to(device) # (1, 64, 168)
            latent = model.extract_features(x_tensor) # (1, 64, 128)
            
            # Save as numpy array (64, 128)
            latent_np = latent.squeeze(0).cpu().numpy()
            templates_by_class[label_str].append(latent_np)
            
    # Save the dictionary of latent templates
    out_path = os.path.join(models_dir, 'latent_templates.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(dict(templates_by_class), f)
        
    print(f"Saved {sum(len(v) for v in templates_by_class.values())} latent templates to {out_path}.")

if __name__ == '__main__':
    extract_latent_templates()
