import numpy as np
import os

class TemplateManager:
    """Loads and manages SDTW templates."""
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.templates = {}  # {label: [template1, template2, ...]}
        self.load_all()
        
    def load_all(self):
        """Load all .npz template files from the directory."""
        self.templates = {}
        if not os.path.exists(self.template_dir):
            return
            
        for filename in os.listdir(self.template_dir):
            if not filename.endswith('.npz'):
                continue
                
            label = filename.replace('.npz', '')
            data = np.load(os.path.join(self.template_dir, filename))
            
            tmpls = []
            for key in sorted(data.files):
                tmpls.append(data[key])
                
            self.templates[label] = tmpls
            
    def get_templates(self) -> dict:
        return self.templates
