import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

class SyntheticCTCDataset(Dataset):
    def __init__(self, csv_paths, scaler=None, label_map=None, is_train=True, num_samples=5000):
        self.is_train = is_train
        self.num_samples = num_samples
        
        self.raw_sequences, self.raw_labels = self._load_data(csv_paths)
        
        if label_map is None and is_train:
            unique_labels = sorted(list(set(self.raw_labels)))
            # CTC requires blank at index 0. Shift all other labels by 1.
            self.label_map = {lbl: i + 1 for i, lbl in enumerate(unique_labels)}
        else:
            self.label_map = label_map
            
        self.inverse_label_map = {i: lbl for lbl, i in self.label_map.items()}
        
        # Fit scaler on isolated features if training
        if scaler is None and is_train:
            self.scaler = StandardScaler()
            all_feats = []
            for seq in self.raw_sequences:
                feat = self._extract_features(seq)
                all_feats.append(feat)
            all_feats_flat = np.vstack(all_feats)
            self.scaler.fit(all_feats_flat)
        else:
            self.scaler = scaler

    def _load_data(self, csv_paths):
        raw_sequences = []
        raw_labels = []
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                continue
            sequences_dict = defaultdict(list)
            labels_dict = {}
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if not row: continue
                    sid, lbl = int(row[0]), row[1]
                    feats = [float(x) for x in row[4:]]
                    labels_dict[sid] = lbl
                    sequences_dict[sid].append(feats)
                    
            for sid in sorted(sequences_dict.keys()):
                raw_labels.append(labels_dict[sid])
                raw_sequences.append(np.array(sequences_dict[sid]))
        return raw_sequences, raw_labels

    def _extract_features(self, seq):
        vel = np.zeros_like(seq)
        vel[1:] = seq[1:] - seq[:-1]
        acc = np.zeros_like(seq)
        acc[2:] = vel[2:] - vel[1:-1]
        return np.concatenate([seq, vel, acc], axis=-1)

    def _generate_synthetic_sequence(self):
        num_signs = random.randint(1, 5) if self.is_train else random.randint(3, 6)
        
        chosen_indices = [random.randint(0, len(self.raw_sequences) - 1) for _ in range(num_signs)]
        
        target_labels = [self.label_map[self.raw_labels[idx]] for idx in chosen_indices]
        
        combined_seq = []
        for i, idx in enumerate(chosen_indices):
            seq = self.raw_sequences[idx]
            
            # Subsample randomly to vary speeds
            if self.is_train and random.random() < 0.5:
                speed = random.uniform(0.8, 1.2)
                t_old = np.linspace(0, 1, len(seq))
                t_new = np.linspace(0, 1, int(len(seq) / speed))
                if len(t_new) > 5:
                    interp = interp1d(t_old, seq, axis=0, kind='linear', fill_value='extrapolate')
                    seq = interp(t_new)
            
            combined_seq.append(seq)
            
            # Add synthetic transition
            if i < len(chosen_indices) - 1:
                next_seq = self.raw_sequences[chosen_indices[i+1]]
                num_trans_frames = random.randint(5, 20)
                start_frame = seq[-1]
                end_frame = next_seq[0]
                alphas = np.linspace(0, 1, num_trans_frames + 2)[1:-1]
                transition = np.array([(1-a)*start_frame + a*end_frame for a in alphas])
                combined_seq.append(transition)
                
        final_raw = np.vstack(combined_seq)
        feats = self._extract_features(final_raw)
        scaled_feats = self.scaler.transform(feats)
        
        return torch.tensor(scaled_feats, dtype=torch.float32), torch.tensor(target_labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples if self.is_train else 500

    def __getitem__(self, idx):
        return self._generate_synthetic_sequence()

def collate_fn_ctc(batch):
    sequences, labels = zip(*batch)
    
    input_lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Concatenate targets into 1D tensor for CTCLoss
    targets = torch.cat(labels)
    
    return padded_sequences, targets, input_lengths, target_lengths
