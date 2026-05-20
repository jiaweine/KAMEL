"""
ML Algorithms Registry for Comparison Experiments.
"""

from .catboost_classifier import CatBoostClassifier_ML
from .xgboost_classifier import XGBoostClassifier_ML
from .lightgbm_classifier import LightGBMClassifier_ML
from .random_forest import RandomForestClassifier_ML
from .logistic_regression import LogisticRegressionClassifier
from .gradient_boosting import GradientBoostingClassifier_ML
from .tabnet_classifier import TabNetClassifier_ML
from .ft_transformer_classifier import FTTransformerClassifier_ML
from .saint_classifier import SAINTClassifier_ML
from .tabicl_classifier import TabICLClassifier_ML
from .tabpfn_offline_classifier import TabPFNOfflineClassifier_ML
from .tabm_classifier import TabMClassifier_ML
from .mothernet_classifier import MotherNetClassifier_ML
from .tabr_classifier import TabRClassifier_ML
from .node_classifier import NODEClassifier_ML
from .gandalf_classifier import GANDALFClassifier_ML

# Algorithm Registry (16 core algorithms)
# Note: TALENT models (talent_modernnca, talent_resnet, talent_realmlp) 
# are available via algorithm_configs.py but not included in default registry
# TabR now uses official Yandex Research implementation (tabr_classifier.py)
ALGORITHM_REGISTRY = {
    'catboost': CatBoostClassifier_ML,
    'xgboost': XGBoostClassifier_ML,
    'lightgbm': LightGBMClassifier_ML,
    'random_forest': RandomForestClassifier_ML,
    'logistic_regression': LogisticRegressionClassifier,
    'gradient_boosting': GradientBoostingClassifier_ML,
    'tabnet': TabNetClassifier_ML,
    'ft_transformer': FTTransformerClassifier_ML,
    'saint': SAINTClassifier_ML,
    'tabicl': TabICLClassifier_ML,
    'tabpfn_offline': TabPFNOfflineClassifier_ML,
    'tabm': TabMClassifier_ML,
    'mothernet': MotherNetClassifier_ML,
    'tabr': TabRClassifier_ML,
    'node': NODEClassifier_ML,
    'gandalf': GANDALFClassifier_ML,
}


def get_algorithm(name: str):
    """Get algorithm class by name."""
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' not found. Available: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name]


__all__ = ['ALGORITHM_REGISTRY', 'get_algorithm']

