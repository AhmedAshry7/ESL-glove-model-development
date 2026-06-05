import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dl_pipeline.dataset_ctc import SyntheticCTCDataset, collate_fn_ctc
from dl_pipeline.model_ctc import SignLanguageCTCModel

def decode_greedy(log_probs):
    # log_probs shape: (T, N, C)
    # returns list of predicted sequences
    T, N, C = log_probs.shape
    preds = log_probs.argmax(dim=-1).transpose(0, 1) # (N, T)
    decoded_seqs = []
    for i in range(N):
        seq = []
        prev_idx = -1
        for t in range(T):
            idx = preds[i, t].item()
            if idx != 0 and idx != prev_idx:
                seq.append(idx)
            prev_idx = idx
        decoded_seqs.append(seq)
    return decoded_seqs

def calc_accuracy(decoded_seqs, targets, target_lengths):
    correct = 0
    total = len(decoded_seqs)
    
    ptr = 0
    for i in range(total):
        length = target_lengths[i].item()
        target_seq = targets[ptr:ptr+length].tolist()
        ptr += length
        
        if decoded_seqs[i] == target_seq:
            correct += 1
            
    return correct / total

def train_model(epochs=50, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(base_dir, 'data', 'processed_csv', 'train.csv')
    test_csv = os.path.join(base_dir, 'data', 'processed_csv', 'test.csv')
    
    # 1. Datasets
    print("Loading datasets...")
    train_dataset = SyntheticCTCDataset([train_csv], is_train=True, num_samples=2000)
    
    # Validation dataset uses exactly isolated test samples? No, let's use synthetic too so we test continuous decoding
    val_dataset = SyntheticCTCDataset([test_csv], scaler=train_dataset.scaler, label_map=train_dataset.label_map, is_train=False, num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_ctc, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ctc, num_workers=0)
    
    # 2. Model
    num_classes = len(train_dataset.label_map)
    input_size = 56 * 3 # 168
    model = SignLanguageCTCModel(input_size=input_size, num_classes=num_classes).to(device)
    
    # 3. Loss & Optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    models_dir = os.path.join(base_dir, 'models', 'dl_models')
    os.makedirs(models_dir, exist_ok=True)
    
    best_val_acc = 0.0
    print("Starting CTC training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for sequences, targets, input_lengths, target_lengths in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            log_probs = model(sequences) # (batch, time, classes)
            
            # CTCLoss expects (time, batch, classes)
            log_probs = log_probs.transpose(0, 1)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            
            # Clip gradients to prevent exploding gradients with CTC
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_decoded = []
        all_targets = []
        all_target_lengths = []
        
        with torch.no_grad():
            for sequences, targets, input_lengths, target_lengths in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                
                log_probs = model(sequences)
                log_probs_t = log_probs.transpose(0, 1)
                
                loss = criterion(log_probs_t, targets, input_lengths, target_lengths)
                val_loss += loss.item() * sequences.size(0)
                
                decoded = decode_greedy(log_probs_t)
                all_decoded.extend(decoded)
                all_targets.append(targets.cpu())
                all_target_lengths.append(target_lengths.cpu())
                
        val_loss /= len(val_dataset)
        cat_targets = torch.cat(all_targets)
        cat_lengths = torch.cat(all_target_lengths)
        val_acc = calc_accuracy(all_decoded, cat_targets, cat_lengths)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}, Val Sequence Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < train_loss):
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_model_ctc.pth'))
            print("  --> Saved new best CTC model")
            
    # Save Scaler and Label Map
    with open(os.path.join(models_dir, 'scaler_ctc.pkl'), 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    with open(os.path.join(models_dir, 'label_map_ctc.pkl'), 'wb') as f:
        pickle.dump(train_dataset.label_map, f)
        
    print(f"Training completed. Best Val Sequence Acc: {best_val_acc*100:.2f}%")

if __name__ == '__main__':
    train_model(epochs=100)
