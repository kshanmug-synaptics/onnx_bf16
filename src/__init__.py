"""
ONNX Model Precision Conversion - Source Package

This package contains modules for converting ONNX models with various BF16 quantization strategies.

Modules:
- activation_transforms: BF16 conversion strategies (cast sandwiching, weights-only, full BF16)
- conversion_utils: Utility functions for model conversion and shape fixing
- testing_utils: Model testing and inference validation utilities
- metrics_utils: Error analysis and performance benchmarking utilities  
- main_orchestrator: High-level orchestration for batch and single model processing
"""

__version__ = "1.0.0"
__author__ = "ONNX BF16 Conversion Team"
