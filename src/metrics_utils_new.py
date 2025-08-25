#!/usr/bin/env python3
"""
Metrics and comparison utilities for model evaluation.

This module contains functions for:
- Comparing FP32 vs quantized model outputs
- Computing error metrics (relative error, SNR, RMSE, etc.)
- Performance analysis and benchmarking
"""

import onnxruntime as ort
import numpy as np
import os
import time
import traceback


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


def compare_models(fp32_model_path, bf16_model_path):
    """Compare FP32 vs cast-sandwiched models and compute error metrics."""
    print(f"\n{'='*20} COMPARING MODELS {'='*20}")
    
    if not os.path.exists(fp32_model_path):
        print(f"FP32 model not found: {fp32_model_path}")
        return {}
    
    if not os.path.exists(bf16_model_path):
        print(f"Cast-sandwiched model not found: {bf16_model_path}")
        return {}
    
    try:
        print("Loading models for comparison...")
        
        # Load both models
        fp32_session = ort.InferenceSession(fp32_model_path)
        bf16_session = ort.InferenceSession(bf16_model_path)
        
        # Create test inputs (same for both models)
        test_inputs = create_test_inputs(fp32_session)
        
        print("Running FP32 model...")
        fp32_start = time.time()
        fp32_outputs = fp32_session.run(None, test_inputs)
        fp32_time = time.time() - fp32_start
        
        print("Running cast-sandwiched BF16 model...")
        bf16_start = time.time()
        bf16_outputs = bf16_session.run(None, test_inputs)
        bf16_time = time.time() - bf16_start
        
        # Compare outputs (focus on the first/main output)
        if len(fp32_outputs) > 0 and len(bf16_outputs) > 0:
            fp32_output = fp32_outputs[0]
            bf16_output = bf16_outputs[0]
            
            print(f"FP32 output shape: {fp32_output.shape}, dtype: {fp32_output.dtype}")
            print(f"BF16 output shape: {bf16_output.shape}, dtype: {bf16_output.dtype}")
            
            if fp32_output.shape == bf16_output.shape and fp32_output.size > 0:
                # Compute error metrics
                print(f"\n{'='*20} ERROR ANALYSIS {'='*20}")
                
                # Absolute difference
                abs_diff = np.abs(fp32_output - bf16_output)
                max_abs_diff = np.max(abs_diff)
                mean_abs_diff = np.mean(abs_diff)
                
                # Relative error
                epsilon = 1e-12
                relative_error = np.sum(abs_diff) / (np.sum(np.abs(fp32_output)) + epsilon)
                
                # Signal-to-noise ratio
                signal_power = np.mean(fp32_output ** 2)
                noise_power = np.mean((fp32_output - bf16_output) ** 2)
                snr_db = 10 * np.log10(signal_power / (noise_power + epsilon))
                
                # Cosine similarity
                fp32_flat = fp32_output.flatten()
                bf16_flat = bf16_output.flatten()
                cosine_sim = np.dot(fp32_flat, bf16_flat) / (np.linalg.norm(fp32_flat) * np.linalg.norm(bf16_flat))
                
                # Root mean square error
                rmse = np.sqrt(np.mean((fp32_output - bf16_output) ** 2))
                
                print(f"📊 COMPARISON RESULTS:")
                print(f"-" * 40)
                print(f"Relative Error:           {relative_error:.8f} ({relative_error*100:.6f}%)")
                print(f"Signal-to-Noise Ratio:    {snr_db:.2f} dB")
                # print(f"Mean Absolute Error:      {mean_abs_diff:.8f}")
                # print(f"Max Absolute Error:       {max_abs_diff:.8f}")
                # print(f"Root Mean Square Error:   {rmse:.8f}")
                # print(f"Cosine Similarity:        {cosine_sim:.8f}")
                # print(f"FP32 Inference Time:      {fp32_time:.4f}s")
                # print(f"BF16 Inference Time:      {bf16_time:.4f}s")
                # print(f"Speed Ratio (BF16/FP32):  {bf16_time/fp32_time:.2f}x")
                
                return {
                    'relative_error': relative_error,
                    'snr_db': snr_db
                }
            else:
                print(f"Cannot compare: shape mismatch or empty arrays")
                return {}
        else:
            print("No outputs to compare")
            return {}
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return {}
