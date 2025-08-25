#!/usr/bin/env python3
"""
Conversion utilities for ONNX model processing.

This module contains utility functions for:
- BF16 weight conversion with stochastic rounding
- Opset conversion 
- Dynamic shape fixing
- Model saving with fixed shapes
"""

import onnx
from onnx import helper, TensorProto, numpy_helper, version_converter
import numpy as np
from pathlib import Path
import time

try:
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False


def round_to_nearest_even_bf16(float32_array):
    """Round to nearest even BF16 for weights."""
    original_shape = float32_array.shape
    
    float32_bytes = float32_array.astype(np.float32).tobytes()
    uint32_view = np.frombuffer(float32_bytes, dtype=np.uint32)
    
    # Round to nearest even for bfloat16 conversion
    rounding_bias = (uint32_view >> 16) & 1
    bfloat16_uint32 = (uint32_view + 0x7fff + rounding_bias) & 0xFFFF0000
    
    result = np.frombuffer(bfloat16_uint32.astype(np.uint32).tobytes(), dtype=np.float32)
    return result.reshape(original_shape)


def convert_to_opset22(model):
    """Convert ONNX model to opset 22 for better BF16 support."""
    current_opset = model.opset_import[0].version if model.opset_import else "Unknown"
    print(f"Current opset version: {current_opset}")
    
    if current_opset != 22:
        print("Converting to opset 22...")
        try:
            model = version_converter.convert_version(model, 22)
            new_opset = model.opset_import[0].version if model.opset_import else "Unknown"
            print(f"New opset version: {new_opset}")
        except Exception as e:
            print(f"Warning: Could not convert to opset 22: {e}")
    else:
        print("Model already at opset 22")
    
    return model


def fix_dynamic_input_shapes(model, model_path):
    """
    Fix dynamic input shapes to static shapes based on model type.
    This is particularly important for Moonshine encoder models to avoid IREE compilation issues.
    """
    print("Checking and fixing dynamic input shapes...")
    
    # Detect if this is a Moonshine encoder model
    is_moonshine_encoder = ('moonshine' in str(model_path).lower() and 'encoder' in str(model_path).lower()) or \
                          any(inp.name == 'input_values' for inp in model.graph.input)
    
    shapes_fixed = 0
    for input_info in model.graph.input:
        input_name = input_info.name
        current_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        
        # Check if shape has dynamic dimensions (0 or negative values)
        has_dynamic_dims = any(dim <= 0 for dim in current_shape)
        
        if has_dynamic_dims:
            print(f"  Input '{input_name}' has dynamic shape: {current_shape}")
            
            # Determine the correct static shape based on model type and input name
            if is_moonshine_encoder and input_name == 'input_values':
                # Moonshine encoder audio input should be [1, 16000]
                new_shape = [1, 16000]
                print(f"  Setting Moonshine encoder audio input to static shape: {new_shape}")
            else:
                # For other inputs, replace dynamic dimensions with reasonable defaults
                new_shape = []
                for i, dim in enumerate(current_shape):
                    if dim <= 0:
                        if i == 0:  # Batch dimension
                            new_shape.append(1)
                        elif input_name == 'input_values' and i == 1:
                            new_shape.append(16000)  # Audio sequence length
                        elif 'sequence' in input_name.lower() or 'time' in input_name.lower():
                            new_shape.append(100)  # Default sequence length
                        elif 'feature' in input_name.lower() or 'channel' in input_name.lower():
                            new_shape.append(80)  # Default feature dimension
                        else:
                            new_shape.append(1)  # Safe default
                    else:
                        new_shape.append(dim)
                print(f"  Setting input '{input_name}' to static shape: {new_shape}")
            
            # Update the input shape
            input_info.type.tensor_type.shape.ClearField("dim")
            for dim_size in new_shape:
                dim = input_info.type.tensor_type.shape.dim.add()
                dim.dim_value = dim_size
            
            shapes_fixed += 1
        else:
            print(f"  Input '{input_name}' already has static shape: {current_shape}")
    
    if shapes_fixed > 0:
        print(f"Fixed {shapes_fixed} dynamic input shapes to static shapes")
    else:
        print("All input shapes are already static")
    
    # Use onnxsim to propagate static shapes throughout the network
    print("Using onnxsim to propagate static shapes through the network...")
    if ONNXSIM_AVAILABLE:
        try:
            # Create input shapes dict for onnxsim
            input_shapes = {}
            for input_info in model.graph.input:
                shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                input_shapes[input_info.name] = shape
            
            print(f"Input shapes for onnxsim: {input_shapes}")
            
            # Use onnxsim to propagate shapes and perform minimal optimization
            model_simplified, check = simplify(
                model, 
                overwrite_input_shapes=input_shapes,
                perform_optimization=False,  # Don't optimize, just propagate shapes
                skip_shape_inference=False,  # Enable shape inference
                skip_fuse_bn=True,          # Skip batch norm fusion
                skip_constant_folding=False  # Allow constant folding for shape propagation
            )
            
            if check:
                model = model_simplified
                
                # Also fix output shapes if they are still dynamic
                print("Fixing any remaining dynamic output shapes...")
                output_shapes_fixed = 0
                for output_info in model.graph.output:
                    if output_info.type.tensor_type.shape.dim:
                        current_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
                        has_dynamic_dims = any(dim <= 0 for dim in current_shape)
                        
                        if has_dynamic_dims:
                            print(f"  Output '{output_info.name}' has dynamic shape: {current_shape}")
                            
                            # Fix dynamic output dimensions with reasonable defaults
                            new_shape = []
                            for i, dim in enumerate(current_shape):
                                if dim <= 0:
                                    new_shape.append(1)  # Safe default for dynamic outputs
                                else:
                                    new_shape.append(dim)
                            
                            print(f"  Setting output '{output_info.name}' to static shape: {new_shape}")
                            
                            # Update the output shape
                            output_info.type.tensor_type.shape.ClearField("dim")
                            for dim_size in new_shape:
                                dim = output_info.type.tensor_type.shape.dim.add()
                                dim.dim_value = dim_size
                            
                            output_shapes_fixed += 1
                
                if output_shapes_fixed > 0:
                    print(f"Fixed {output_shapes_fixed} dynamic output shapes to static shapes")
                
                # Check how many intermediate tensors have shape information
                shape_info_count = len(model.graph.value_info)
                dynamic_shapes_remaining = 0
                
                for value_info in model.graph.value_info:
                    if value_info.type.tensor_type.shape.dim:
                        for dim in value_info.type.tensor_type.shape.dim:
                            if dim.dim_value <= 0:
                                dynamic_shapes_remaining += 1
                                break
                
                print(f"onnxsim results: {shape_info_count} intermediate tensors have shape info")
                if dynamic_shapes_remaining > 0:
                    print(f"Warning: {dynamic_shapes_remaining} tensors still have dynamic dimensions")
                else:
                    print("✓ All intermediate tensors now have static shapes")
            else:
                print("onnxsim validation failed, keeping original model")
                
        except Exception as e:
            print(f"onnxsim failed: {e}")
            print("Proceeding without onnxsim shape propagation")
    else:
        print("Warning: onnxsim not available. Shapes may not be fully propagated.")
        print("Install onnxsim with: pip install onnxsim")
    
    return model


def convert_weights_to_bf16(graph):
    """
    Convert all FP32 weight initializers in a graph to BF16 using round to nearest even.
    Returns the number of weights converted.
    """
    weights_converted = 0
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            # Load the float32 data
            weights_fp32 = numpy_helper.to_array(init)
            
            # Convert to BF16 using round to nearest even
            weights_bf16_pattern = round_to_nearest_even_bf16(weights_fp32)
            
            # Update the initializer with BF16 bit pattern data
            new_init = numpy_helper.from_array(weights_bf16_pattern, name=init.name)
            new_init.data_type = TensorProto.FLOAT  # Keep as FP32 data type
            
            # Replace the initializer
            for i, existing_init in enumerate(graph.initializer):
                if existing_init.name == init.name:
                    graph.initializer[i].CopyFrom(new_init)
                    break
            
            weights_converted += 1
    
    return weights_converted


def save_static_opset22_with_fixed_shapes(model_path, output_path):
    """
    Load an ONNX model, fix dynamic input shapes to static shapes, and save it.
    
    - Model remains completely FP32 
    - Only fixes dynamic input shapes to static shapes
    - Important for IREE compilation compatibility
    """
    print(f"Loading model: {model_path} (mode: static opset22 with fixed shapes)")
    start_time = time.time()
    
    try:
        m = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Convert to opset 22 first for better compatibility
    m = convert_to_opset22(m)
    
    # Fix dynamic input shapes to static shapes (important for IREE compilation)
    m = fix_dynamic_input_shapes(m, model_path)
    
    # Ensure the model uses opset 22
    if m.opset_import:
        m.opset_import[0].version = 22
        print("Set model opset to 22")
    
    # Check and save model
    try:
        onnx.checker.check_model(m)
        print("Model validation passed")
    except Exception as e:
        print(f"Model validation warning: {e}")
    
    try:
        onnx.save(m, output_path)
        print(f"Saved static opset22 model with fixed shapes to: {output_path}")
        
        conversion_time = time.time() - start_time
        print(f"Conversion completed in {conversion_time:.2f} seconds")
        
        return m
    except Exception as e:
        print(f"Error saving model: {e}")
        return None
