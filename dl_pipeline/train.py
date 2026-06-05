import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dl_pipeline.dataset import SignLanguageDataset
from dl_pipeline.model import SignLanguageTransformer
import pickle

def train_model():
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed_csv')
    models_dir = os.path.join(base_dir, 'models', 'dl_models')
    os.makedirs(models_dir, exist_ok=True)
    
    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv = os.path.join(data_dir, 'val.csv')
    
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    max_seq_len = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Datasets
    print("Loading datasets...")
    train_dataset = SignLanguageDataset(
        csv_paths=[train_csv], 
        max_len=max_seq_len, 
        is_train=True
    )
    
    val_dataset = SignLanguageDataset(
        csv_paths=[val_csv], 
        max_len=max_seq_len, 
        scaler=train_dataset.scaler, 
        label_map=train_dataset.label_map,
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Save the scaler and label map for inference
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    with open(os.path.join(models_dir, 'label_map.pkl'), 'wb') as f:
        pickle.dump(train_dataset.label_map, f)
        
    num_classes = len(train_dataset.label_map)
    # Features = 56 original columns * 3 (raw, vel, acc)
    input_size = 56 * 3
    
    print(f"Found {num_classes} classes. Input size: {input_size}")
    
    # Initialize Model
    model = SignLanguageTransformer(input_size=input_size, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_dataset)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_dataset)
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
              
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pth'))
            print("  --> Saved new best model")

    print(f"Training completed. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == '__main__':
    train_model()
