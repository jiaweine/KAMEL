"""
KAMEL: KAN-fused Adaptive Mixture-of-Experts for Tabular Learning

Package initializer kept minimal to avoid expensive imports during package load.
"""

__version__ = "1.0.0"
__author__ = "KAMEL Team"

# Intentionally avoid importing submodules here to prevent heavy dependency chains
# Import submodules directly where needed, e.g., `from kamel.data.preprocessor import TabularDataProcessor`

__all__ = [
    "__version__",
    "__author__",
]

