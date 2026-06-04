import numpy as np
import os

# Files that live in the templates dir but are NOT sign templates
_RESERVED_FILES = {'discriminative_weights.npz', 'normalization_stats.npz'}

class TemplateManager:
    """Loads and manages SDTW templates."""
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.templates = {}  # {label: [template1, template2, ...]}
        self.load_all()
        
    def load_all(self):
        """Load all sign .npz template files from the directory.
        
        Only loads files whose first key is 'tmpl_0', skipping reserved files
        like normalization_stats.npz and discriminative_weights.npz.
        """
        self.templates = {}
        self.thresholds = {}
        if not os.path.exists(self.template_dir):
            return
            
        # Load thresholds if available
        import json
        thresh_path = os.path.join(self.template_dir, 'class_thresholds.json')
        if os.path.exists(thresh_path):
            with open(thresh_path, 'r') as f:
                self.thresholds = json.load(f)
            
        for filename in sorted(os.listdir(self.template_dir)):
            if not filename.endswith('.npz'):
                continue
            if filename in _RESERVED_FILES:
                continue
                
            label = filename.replace('.npz', '')
            data = np.load(os.path.join(self.template_dir, filename))
            
            # Only accept files that contain template arrays (key 'tmpl_0')
            if 'tmpl_0' not in data.files:
                continue
            
            tmpls = [data[key] for key in sorted(data.files) if key.startswith('tmpl_')]
            if tmpls:
                self.templates[label] = tmpls
            
    def get_templates(self) -> dict:
        return self.templates
        
    def get_thresholds(self) -> dict:
        return self.thresholds
