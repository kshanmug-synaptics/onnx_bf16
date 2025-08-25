#!/usr/bin/env python3
"""
Testing utilities for ONNX model validation and inference.

This module contains functions for:
- Creating test inputs for any ONNX model
- Running inference tests with ONNX Runtime
- Model analysis and operation counting
"""

import onnx
import onnxruntime as ort
import numpy as np
import os
import time


def create_test_inputs(session):
    """Create test inputs for any ONNX model based on its input specifications."""
    inputs_info = session.get_inputs()
    test_inputs = {}
    
    for input_info in inputs_info:
        input_name = input_info.name
        input_shape = input_info.shape
        input_type = input_info.type
        
        # Determine test shape based on input specification and model type
        test_shape = []
        for i, dim in enumerate(input_shape):
            if isinstance(dim, int) and dim > 0:
                test_shape.append(dim)
            elif input_name == 'input_values' and i == 1:
                # Special case for Moonshine encoder audio input (must come before other checks)
                test_shape.append(16000)  # 1 second of 16kHz audio
            elif 'samples' in str(dim).lower():
                # Another way to catch audio sample dimensions
                test_shape.append(16000)  # 1 second of 16kHz audio
            elif isinstance(dim, str) or 'batch' in str(dim).lower():
                test_shape.append(1)  # Use batch size 1
            elif 'sequence' in str(dim).lower() or 'time' in str(dim).lower():
                test_shape.append(100)  # Reasonable sequence length
            elif 'feature' in str(dim).lower() or 'channel' in str(dim).lower():
                test_shape.append(80)  # Reasonable feature dimension
            elif 'height' in str(dim).lower():
                test_shape.append(224)  # Common image height
            elif 'width' in str(dim).lower():
                test_shape.append(224)  # Common image width
            else:
                test_shape.append(1)  # Default
        
        # Create test data based on type
        if 'tensor(float)' in str(input_type):
            test_data = np.random.randn(*test_shape).astype(np.float32)
        elif 'tensor(int64)' in str(input_type):
            # For token IDs, use reasonable vocabulary range
            test_data = np.random.randint(0, 1000, test_shape).astype(np.int64)
        elif 'tensor(int32)' in str(input_type):
            test_data = np.random.randint(0, 1000, test_shape).astype(np.int32)
        elif 'tensor(bool)' in str(input_type):
            # For boolean flags
            test_data = np.ones(test_shape, dtype=bool)
        else:
            # Default to float32
            test_data = np.random.randn(*test_shape).astype(np.float32)
        
        test_inputs[input_name] = test_data
    
    print(f"Created {len(test_inputs)} test inputs")
    return test_inputs


def test_model(model_path, test_name="Model"):
    """Test an ONNX model with ONNXRuntime."""
    print(f"\nTesting {test_name}: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return False, None
    
    try:
        start_time = time.time()
        session = ort.InferenceSession(model_path)
        
        # Get input/output info
        inputs_info = session.get_inputs()
        outputs_info = session.get_outputs()
        
        print(f"Model has {len(inputs_info)} inputs and {len(outputs_info)} outputs")
        
        # Create test inputs
        test_inputs = create_test_inputs(session)
        
        # Run inference
        print("Running inference...")
        inference_start = time.time()
        outputs = session.run(None, test_inputs)
        inference_time = time.time() - inference_start
        
        # Analyze outputs
        print(f"Success! Model produced {len(outputs)} outputs")
        
        total_time = time.time() - start_time
        print(f"Test completed in {total_time:.2f} seconds (inference: {inference_time:.4f}s)")
        
        return True, outputs
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return False, None


def analyze_model_operations(model_path):
    """Analyze and print all operations in the model."""
    print(f"\nAnalyzing operations in: {model_path}")
    
    try:
        m = onnx.load(model_path)
        g = m.graph
        
        op_counts = {}
        
        for i, node in enumerate(g.node):
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
    except Exception as e:
        print(f"Error analyzing model: {e}")
        return {}
    
    return op_counts
