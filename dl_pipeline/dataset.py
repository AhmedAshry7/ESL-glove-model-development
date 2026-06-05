import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

class SignLanguageDataset(Dataset):
    def __init__(self, csv_paths, max_len=64, scaler=None, label_map=None, is_train=True):
        """
        Args:
            csv_paths (list of str): Paths to CSV files containing the data.
            max_len (int): The target length to interpolate all sequences to.
            scaler (StandardScaler): Scikit-learn scaler for normalization.
            label_map (dict): Mapping from string labels to integer indices.
            is_train (bool): Whether this is a training dataset.
        """
        self.max_len = max_len
        self.is_train = is_train
        
        # Load all sequences from CSVs
        self.sequences = []
        self.labels = []
        
        raw_sequences, raw_labels = self._load_data(csv_paths)
        
        # Build or use label map
        if label_map is None and is_train:
            unique_labels = sorted(list(set(raw_labels)))
            self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.label_map = label_map
            
        self.inverse_label_map = {i: lbl for lbl, i in self.label_map.items()}
        
        # Process sequences: Interpolate, Extract Features, Normalize
        processed_seqs = []
        for seq in raw_sequences:
            # seq shape: (T, 56)
            interpolated = self._interpolate_sequence(seq, target_len=self.max_len)
            features = self._extract_features(interpolated)
            processed_seqs.append(features)
            
        # Stack for scaling
        # processed_seqs shape: (N, max_len, num_features)
        all_features = np.array(processed_seqs) # Shape: (N, max_len, F)
        N, T, F = all_features.shape
        
        if scaler is None and is_train:
            self.scaler = StandardScaler()
            # Flatten to fit scaler
            flat_features = all_features.reshape(-1, F)
            self.scaler.fit(flat_features)
        else:
            self.scaler = scaler
            
        # Apply scaling
        flat_features = all_features.reshape(-1, F)
        scaled_features = self.scaler.transform(flat_features)
        self.features = scaled_features.reshape(N, T, F).astype(np.float32)
        
        self.labels = np.array([self.label_map[lbl] for lbl in raw_labels], dtype=np.int64)

    def _load_data(self, csv_paths):
        raw_sequences = []
        raw_labels = []
        
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file not found: {csv_path}")
                continue
                
            sequences_dict = defaultdict(list)
            labels_dict = {}
            
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if not row:
                        continue
                    sample_id = int(row[0])
                    label = row[1]
                    # We skip timestamp (idx 3) for the feature array, taking cols 4 onwards
                    features = [float(x) for x in row[4:]]
                    
                    labels_dict[sample_id] = label
                    sequences_dict[sample_id].append(features)
                    
            for sid in sorted(sequences_dict.keys()):
                seq = np.array(sequences_dict[sid])
                raw_labels.append(labels_dict[sid])
                raw_sequences.append(seq)
                
                if self.is_train:
                    # Augmentation 1: Pad beginning with first frame (simulating early Spotter start)
                    shift = int(len(seq) * 0.2) # 20% shift
                    if shift > 0:
                        pad_front = np.tile(seq[0], (shift, 1))
                        aug1 = np.vstack([pad_front, seq])
                        raw_labels.append(labels_dict[sid])
                        raw_sequences.append(aug1)
                        
                        # Augmentation 2: Pad end with last frame (simulating late Spotter stop)
                        pad_end = np.tile(seq[-1], (shift, 1))
                        aug2 = np.vstack([seq, pad_end])
                        raw_labels.append(labels_dict[sid])
                        raw_sequences.append(aug2)
                        
                        # Augmentation 3: Pad both ends (simulating loose Spotter)
                        aug3 = np.vstack([pad_front, seq, pad_end])
                        raw_labels.append(labels_dict[sid])
                        raw_sequences.append(aug3)
                        
        return raw_sequences, raw_labels

    def _interpolate_sequence(self, seq, target_len=64):
        """
        Interpolates a sequence of shape (T, F) to (target_len, F)
        """
        T, F = seq.shape
        if T == target_len:
            return seq
            
        # Original time steps
        old_indices = np.linspace(0, 1, T)
        # Target time steps
        new_indices = np.linspace(0, 1, target_len)
        
        interpolator = interp1d(old_indices, seq, axis=0, kind='linear', fill_value="extrapolate")
        new_seq = interpolator(new_indices)
        return new_seq

    def _extract_features(self, seq):
        """
        Extracts 1st and 2nd derivatives from the sequence and concatenates them.
        seq shape: (T, F)
        returns: (T, F * 3)
        """
        # Calculate velocity (1st derivative)
        vel = np.zeros_like(seq)
        vel[1:] = seq[1:] - seq[:-1]
        
        # Calculate acceleration (2nd derivative)
        acc = np.zeros_like(seq)
        acc[2:] = vel[2:] - vel[1:-1]
        
        # Concatenate raw, velocity, and acceleration
        features = np.concatenate([seq, vel, acc], axis=-1)
        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y
