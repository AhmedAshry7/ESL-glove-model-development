import os
import torch
from torch.utils.data import DataLoader
from dl_pipeline.dataset import SignLanguageDataset
from dl_pipeline.model import SignLanguageTransformer
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed_csv')
    models_dir = os.path.join(base_dir, 'models', 'dl_models')
    
    test_csv = os.path.join(data_dir, 'test.csv')
    model_path = os.path.join(models_dir, 'best_model.pth')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    label_map_path = os.path.join(models_dir, 'label_map.pkl')
    
    if not os.path.exists(model_path):
        print("Model not found. Please run train.py first.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Scaler and Label Map
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(label_map_path, 'rb') as f:
        label_map = pickle.load(f)
        
    inverse_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    input_size = 56 * 3
    max_seq_len = 64
    
    print("Loading test dataset...")
    test_dataset = SignLanguageDataset(
        csv_paths=[test_csv], 
        max_len=max_seq_len, 
        scaler=scaler, 
        label_map=label_map,
        is_train=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load Model
    model = SignLanguageTransformer(input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    target_names = [inverse_label_map[i] for i in range(num_classes)]
    
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    evaluate_model()
