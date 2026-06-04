import numpy as np
import os
import json

class TemplateManager:
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.templates = {}
        self.load_all()
        
    def load_all(self):

        self.templates = {}
        self.thresholds = {}
        if not os.path.exists(self.template_dir):
            return
            
        thresh_path = os.path.join(self.template_dir, 'class_thresholds.json')
        if os.path.exists(thresh_path):
            with open(thresh_path, 'r') as f:
                self.thresholds = json.load(f)
            
        for filename in sorted(os.listdir(self.template_dir)):
            if not filename.endswith('.npz'):
                continue
            non_templates = {'discriminative_weights.npz', 'normalization_stats.npz'}
            if filename in non_templates:
                continue
                
            label = filename.replace('.npz', '')
            data = np.load(os.path.join(self.template_dir, filename))
            
            templates = [data[key] for key in sorted(data.files) if key.startswith('template_')]
            if templates:
                self.templates[label] = templates
            
    def get_templates(self) -> dict:
        return self.templates
        
    def get_thresholds(self) -> dict:
        return self.thresholds
