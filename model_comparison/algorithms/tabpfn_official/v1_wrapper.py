"""
Standalone wrapper for TabPFN v1 (Nature 2022)
This module provides isolated imports to avoid conflicts with v2
"""

import sys
import os
import importlib.util

_v1_module_cache = {}

def get_v1_classifier():
    """
    Load TabPFN v1 classifier without polluting global namespace
    """
    if 'TabPFNClassifierV1' in _v1_module_cache:
        return _v1_module_cache['TabPFNClassifierV1']
    
    v1_base_path = os.path.join(os.path.dirname(__file__), 'v1_nature2022')
    
    saved_path = sys.path.copy()
    saved_modules = set(sys.modules.keys())
    
    try:
        sys.path.insert(0, v1_base_path)
        
        spec = importlib.util.spec_from_file_location(
            "tabpfn_v1_interface",
            os.path.join(v1_base_path, 'scripts', 'transformer_prediction_interface.py')
        )
        module = importlib.util.module_from_spec(spec)
        
        sys.modules.pop('tabpfn', None)
        sys.modules.pop('tabpfn.utils', None)
        sys.modules.pop('tabpfn.scripts', None)
        
        import tabpfn.utils
        import tabpfn.scripts.model_builder
        
        spec.loader.exec_module(module)
        
        _v1_module_cache['TabPFNClassifierV1'] = module.TabPFNClassifier
        return module.TabPFNClassifier
        
    except Exception as e:
        raise ImportError(f"Failed to load TabPFN v1: {e}")
    finally:
        new_modules = set(sys.modules.keys()) - saved_modules
        for mod_name in new_modules:
            if 'tabpfn' in mod_name.lower() and 'tabpfn_v1' not in mod_name:
                sys.modules.pop(mod_name, None)
        
        sys.path = saved_path



