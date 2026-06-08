import os
import sys
import numpy as np
import torch
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TimeFMExtractor:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        print("Loading TimesFM Model via Hugging Face Transformers...")
        from transformers import AutoModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Point to the local directory where the user downloaded the transformers version
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "timesfm-2.5-200m-transformers")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find the TimesFM model folder at: {model_path}\nPlease make sure the folder is named exactly 'timesfm-2.5-200m-transformers'.")
            
        try:
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Failed to load the model using transformers: {e}")
            raise e

    def extract_timefm_embeddings(self, values: np.ndarray) -> np.ndarray:
        if values is None or len(values) < 5:
            return None
            
        seq_len, num_channels = values.shape
        batch_input = values.T  # Shape: [num_channels, seq_len]
        
        with torch.no_grad():
            input_tensor = torch.tensor(batch_input, dtype=torch.float32).to(self.device)
            
            # TimesFM context length handling
            max_len = 512
            if input_tensor.shape[1] < max_len:
                pad_len = max_len - input_tensor.shape[1]
                input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_len))
            else:
                input_tensor = input_tensor[:, :max_len]
                
            try:
                # With AutoModel for TimesFM, we pass past_values and request hidden states
                # In latest transformers API for TimesFM:
                outputs = self.model(past_values=input_tensor, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states[-1] # [batch_size, patch_len, hidden_dim]
                
                # pool the temporal dimension (mean pooling over time/patches)
                pooled = torch.mean(hidden_states, dim=1) # [num_channels, hidden_dim]
                
                # Flatten across all channels
                feature_vec = pooled.cpu().numpy().flatten()
                return feature_vec
            except Exception as e:
                print(f"Embedding extraction failed: {e}")
                return self._fallback_stats(values)

    def extract_timefm_embeddings_batch(self, list_of_values: list) -> list:
        if not list_of_values:
            return []
            
        valid_indices = []
        batch_inputs = []
        max_len = 512
        
        for idx, values in enumerate(list_of_values):
            if values is not None and len(values) >= 5:
                # Shape: [num_channels, seq_len]
                batch_inputs.append(values.T)
                valid_indices.append(idx)
                
        if not batch_inputs:
            return [None] * len(list_of_values)
            
        # We need to pad sequences to max_len so we can batch them
        # TimeFM usually takes 512 context
        padded_inputs = []
        for b_in in batch_inputs:
            seq_len = b_in.shape[1]
            if seq_len < max_len:
                padded = np.pad(b_in, ((0,0), (0, max_len - seq_len)), mode='constant')
            else:
                padded = b_in[:, :max_len]
            padded_inputs.append(padded)
            
        # Stack all: [num_sequences, num_channels, seq_len] -> [num_sequences * num_channels, seq_len]
        stacked = np.concatenate(padded_inputs, axis=0)
        
        results = [None] * len(list_of_values)
        
        with torch.no_grad():
            input_tensor = torch.tensor(stacked, dtype=torch.float32).to(self.device)
            try:
                outputs = self.model(past_values=input_tensor, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states[-1] # [batch_size, patch_len, hidden_dim]
                pooled = torch.mean(hidden_states, dim=1) # [batch_size, hidden_dim]
                
                # Split back
                num_channels = list_of_values[valid_indices[0]].shape[1]
                split_pooled = torch.split(pooled, num_channels, dim=0)
                
                for i, idx in enumerate(valid_indices):
                    # Flatten across all channels
                    results[idx] = split_pooled[i].cpu().numpy().flatten()
            except Exception as e:
                print(f"Batch embedding extraction failed: {e}")
                for i, idx in enumerate(valid_indices):
                    results[idx] = self._fallback_stats(list_of_values[idx])
                    
        return results

    def _fallback_stats(self, values):
        """Fallback just in case TimeFM embeddings fail"""
        means = np.mean(values, axis=0)
        stds = np.std(values, axis=0)
        return np.concatenate([means, stds])

def extract_timefm_embeddings(values: np.ndarray) -> np.ndarray:
    extractor = TimeFMExtractor.get_instance()
    return extractor.extract_timefm_embeddings(values)

def extract_timefm_embeddings_batch(list_of_values: list) -> list:
    extractor = TimeFMExtractor.get_instance()
    return extractor.extract_timefm_embeddings_batch(list_of_values)
