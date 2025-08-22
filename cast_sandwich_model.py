#!/usr/bin/env python3
"""
Universal ONNX Cast Sandwich Converter

Convert any FP32 ONNX model to use FP32 computation with BF16<->FP32 cast sandwiching.
This keeps the model interface as FP32 but runs all ops in FP32 internally with BF16 precision loss simulation.

Features:
- Automatic opset 22 conversion for better BF16 support
- Tensor-type-based cast sandwiching (only applies to FP32 tensors)
- Preserves correct output types for operations like Shape (int64), comparisons (bool), etc.
- Round to nearest even BF16 for weights
- Comprehensive performance evaluation and comparison
"""

import onnx
from onnx import helper, TensorProto, numpy_helper, version_converter
import onnxruntime as ort
import numpy as np
import os
import argparse
import time
from pathlib import Path

try:
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    print("Warning: onnxsim not available. Some shape inference features may be limited.")

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
                                    if i == 0:  # Batch dimension
                                        new_shape.append(1)
                                    elif 'sequence' in output_info.name.lower() or 'time' in output_info.name.lower():
                                        new_shape.append(100)  # Default sequence length
                                    elif 'vocab' in output_info.name.lower() or 'logit' in output_info.name.lower():
                                        new_shape.append(51865)  # Common vocab size for Moonshine
                                    elif 'feature' in output_info.name.lower() or 'hidden' in output_info.name.lower():
                                        new_shape.append(288)  # Common hidden dimension
                                    else:
                                        new_shape.append(1)  # Safe default
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

def get_expected_output_type(op_type, input_types):
    """
    Determine the expected output type for an operation based on its type and input types.
    This helps us preserve correct types for operations like Shape (which outputs int64).
    """
    # Operations that always output specific types regardless of input
    if op_type == "Shape":
        return TensorProto.INT64
    elif op_type == "Size":
        return TensorProto.INT64
    elif op_type in ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Not", "And", "Or"]:
        return TensorProto.BOOL
    elif op_type in ["ArgMax", "ArgMin", "NonZero"]:
        return TensorProto.INT64
    elif op_type in ["Cast"]:
        # Cast operations output whatever type they're casting to
        # This will be handled separately in the cast logic
        return TensorProto.FLOAT  # Default assumption
    elif op_type in ["Constant"]:
        # Constants output their defined type
        return TensorProto.FLOAT  # Default assumption
    elif op_type in ["Split", "Slice", "Gather", "GatherElements", "Unsqueeze", "Squeeze", "Expand", "Reshape"]:
        # These ops preserve the primary input type
        if input_types and input_types[0] in [TensorProto.INT64, TensorProto.INT32, TensorProto.INT8, TensorProto.BOOL]:
            return input_types[0]
        return TensorProto.FLOAT  # Default to FP32 for float-like operations
    
    # For most other operations, if any input is FP32, output is likely FP32
    if TensorProto.FLOAT in input_types:
        return TensorProto.FLOAT
    
    # For operations with no FP32 inputs, preserve the primary input type
    if input_types:
        return input_types[0]  # Use first input type as default
    
    # Default fallback
    return TensorProto.FLOAT

def cast_sandwich_model(model_path, output_path):
    """
    Convert any FP32 ONNX model to use FP32 computation with BF16<->FP32 cast sandwiching.
    
    - Model inputs/outputs remain FP32
    - For each operation: Cast inputs through BF16 precision loss, run op in FP32, cast outputs through BF16 precision loss
    - Only applies cast sandwiching to FP32 tensors, preserves int64/bool/etc. types
    """
    print(f"Loading model: {model_path} (mode: full cast sandwiching)")
    start_time = time.time()
    
    try:
        m = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Convert to opset 22 first for better BF16 support
    m = convert_to_opset22(m)
    
    # Fix dynamic input shapes to static shapes (important for IREE compilation)
    m = fix_dynamic_input_shapes(m, model_path)
    
    g = m.graph
    
    print(f"Original model has {len(g.node)} nodes")
    
    # Analyze model inputs and outputs
    fp32_inputs = []
    fp32_outputs = []
    
    for input_info in g.input:
        if input_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            fp32_inputs.append(input_info.name)
    
    for output_info in g.output:
        if output_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            fp32_outputs.append(output_info.name)
    
    print(f"Found {len(fp32_inputs)} FP32 inputs and {len(fp32_outputs)} FP32 outputs for cast sandwiching")
    
    # Convert all FP32 weight initializers to BF16 using round to nearest even
    weights_converted = 0
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            # Load the float32 data
            weights_fp32 = numpy_helper.to_array(init)
            
            # Convert to BF16 using round to nearest even
            weights_bf16_pattern = round_to_nearest_even_bf16(weights_fp32)
            
            # Update the initializer with BF16 bit pattern data
            new_init = numpy_helper.from_array(weights_bf16_pattern, name=init.name)
            new_init.data_type = TensorProto.FLOAT  # Keep as FP32 data type
            
            # Replace the initializer
            for i, existing_init in enumerate(g.initializer):
                if existing_init.name == init.name:
                    g.initializer[i].CopyFrom(new_init)
                    break
            
            weights_converted += 1
    
    print(f"Converted {weights_converted} weight initializers to BF16 using round to nearest even")
    
    # Add cast sandwiching for all operations
    new_nodes = []
    cast_counter = 0
    
    # Track tensor types (start with inputs as FP32, initializers as FP32)
    tensor_types = {}
    for input_info in g.input:
        tensor_types[input_info.name] = input_info.type.tensor_type.elem_type
    for init in g.initializer:
        tensor_types[init.name] = init.data_type
    
    # Add cast sandwiching for inputs first (FP32 -> BF16 -> FP32)
    input_cast_nodes = []
    input_cast_counter = 0
    input_name_mapping = {}  # Map original input names to cast-sandwiched names
    
    for input_info in g.input:
        if input_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            input_name = input_info.name
            
            # Step 1: Cast FP32 input to BF16 (temporary)
            bf16_input_name = f"{input_name}_input_bf16_temp_{input_cast_counter}"
            cast_input_to_bf16 = helper.make_node(
                "Cast",
                [input_name],
                [bf16_input_name],
                to=TensorProto.BFLOAT16,
                name=f"cast_input_to_bf16_{input_cast_counter}"
            )
            input_cast_nodes.append(cast_input_to_bf16)
            input_cast_counter += 1
        
            # Step 2: Cast BF16 back to FP32
            fp32_input_name = f"{input_name}_input_fp32_{input_cast_counter}"
            cast_input_to_fp32 = helper.make_node(
                "Cast",
                [bf16_input_name],
                [fp32_input_name],
                to=TensorProto.FLOAT,
                name=f"cast_input_to_fp32_{input_cast_counter}"
            )
            input_cast_nodes.append(cast_input_to_fp32)
            tensor_types[fp32_input_name] = TensorProto.FLOAT
            input_cast_counter += 1
            
            # Keep the original input as FP32 in our tracking
            tensor_types[input_name] = TensorProto.FLOAT
            
            # Map the original input name to the cast-sandwiched name
            input_name_mapping[input_name] = fp32_input_name
    
    print(f"Added input cast sandwiching for {len(input_name_mapping)} FP32 inputs")
    
    # Start with input cast nodes
    new_nodes = input_cast_nodes
    cast_counter = input_cast_counter
    
    # Apply cast sandwiching based on tensor types, not operation types
    for i, n in enumerate(g.node):
        # if i % 50 == 0:  # Progress indicator every 50 nodes
        #     print(f"Processing node {i+1}/{len(g.node)}: {n.op_type}")
        
        # Analyze each input to determine its type and process accordingly
        processed_inputs = []
        input_types_for_op = []
        fp32_input_indices = []
        
        for j, input_name in enumerate(n.input):
            # Check if this is an original input that has been cast-sandwiched
            if input_name in input_name_mapping:
                # This is a cast-sandwiched FP32 input
                actual_input_name = input_name_mapping[input_name]
                processed_inputs.append(actual_input_name)
                fp32_input_indices.append(j)
                input_types_for_op.append(TensorProto.FLOAT)
            elif input_name in tensor_types:
                # Use the known tensor type
                tensor_type = tensor_types[input_name]
                processed_inputs.append(input_name)
                input_types_for_op.append(tensor_type)
                if tensor_type == TensorProto.FLOAT:
                    fp32_input_indices.append(j)
            else:
                # Unknown tensor - assume it's from an initializer or constant
                processed_inputs.append(input_name)
                input_types_for_op.append(TensorProto.FLOAT)  # Default assumption
                fp32_input_indices.append(j)  # Assume FP32 for safety
        
        # Determine expected output types for this operation
        expected_output_types = []
        for j, output_name in enumerate(n.output):
            expected_type = get_expected_output_type(n.op_type, input_types_for_op)
            expected_output_types.append(expected_type)
        
        # Only apply cast sandwiching if we have FP32 inputs and FP32 outputs
        has_fp32_inputs = len(fp32_input_indices) > 0
        has_fp32_outputs = any(ot == TensorProto.FLOAT for ot in expected_output_types)
        
        if has_fp32_inputs and has_fp32_outputs:
            # Create FP32 output names for cast sandwiching
            fp32_outputs = []
            for j, output_name in enumerate(n.output):
                if expected_output_types[j] == TensorProto.FLOAT:
                    fp32_output_name = f"{output_name}_fp32_{cast_counter}"
                    fp32_outputs.append(fp32_output_name)
                    tensor_types[fp32_output_name] = TensorProto.FLOAT
                    cast_counter += 1
                else:
                    # Non-FP32 output - use original name
                    fp32_outputs.append(output_name)
                    tensor_types[output_name] = expected_output_types[j]
            
            # Create operation with processed inputs
            op_with_cast_sandwich = helper.make_node(
                n.op_type,
                processed_inputs,
                fp32_outputs,
                name=f"{n.name}_fp32" if n.name else f"{n.op_type}_fp32_{i}"
            )
            
            # Copy all attributes
            for attr in n.attribute:
                op_with_cast_sandwich.attribute.append(attr)
            
            new_nodes.append(op_with_cast_sandwich)
            
            # Apply cast sandwiching only to FP32 outputs
            for j, output_name in enumerate(n.output):
                if expected_output_types[j] == TensorProto.FLOAT:
                    # Cast FP32 output through BF16 for precision loss simulation
                    # FP32 -> BF16 -> FP32
                    
                    # Step 1: Cast FP32 output to BF16 (temporary)
                    bf16_output_name = f"{output_name}_bf16_temp_{cast_counter}"
                    cast_to_bf16 = helper.make_node(
                        "Cast",
                        [fp32_outputs[j]],
                        [bf16_output_name],
                        to=TensorProto.BFLOAT16,
                        name=f"cast_to_bf16_{cast_counter}"
                    )
                    new_nodes.append(cast_to_bf16)
                    cast_counter += 1
                    
                    # Step 2: Cast BF16 back to FP32 for the actual output
                    cast_back_to_fp32 = helper.make_node(
                        "Cast",
                        [bf16_output_name],
                        [output_name],
                        to=TensorProto.FLOAT,
                        name=f"cast_back_to_fp32_{cast_counter}"
                    )
                    new_nodes.append(cast_back_to_fp32)
                    # Track the final output as FP32
                    tensor_types[output_name] = TensorProto.FLOAT
                    cast_counter += 1
                # For non-FP32 outputs, the original output name is already used and tracked
        else:
            # For operations with no FP32 inputs/outputs, just copy the operation as-is
            # Update input names if they reference cast-sandwiched inputs
            processed_inputs = []
            for input_name in n.input:
                # Check if this is an original input that has been cast-sandwiched
                if input_name in input_name_mapping:
                    actual_input_name = input_name_mapping[input_name]
                else:
                    actual_input_name = input_name
                processed_inputs.append(actual_input_name)
            
            # Copy operation as-is
            copied_op = helper.make_node(
                n.op_type,
                processed_inputs,
                list(n.output),
                name=n.name if n.name else f"{n.op_type}_{i}"
            )
            
            # Copy all attributes
            for attr in n.attribute:
                copied_op.attribute.append(attr)
            
            new_nodes.append(copied_op)
            
            # For outputs, use the expected output types
            for j, output_name in enumerate(n.output):
                tensor_types[output_name] = expected_output_types[j]
    
    print(f"Generated {len(new_nodes)} nodes (originally {len(g.node)})")
    print(f"Added {cast_counter} cast operations")
    
    # Replace nodes in graph
    g.ClearField("node")
    g.node.extend(new_nodes)
    
    # Ensure the model uses opset 22 for better BF16 support
    if m.opset_import:
        m.opset_import[0].version = 22
        print("Set model opset to 22 for BF16 support")
    
    # Clear existing value_info since we've changed the graph structure
    g.ClearField("value_info")
    
    # Check and save model
    try:
        onnx.checker.check_model(m)
        print("Model validation passed")
    except Exception as e:
        print(f"Model validation warning: {e}")
    
    try:
        onnx.save(m, output_path)
        print(f"Saved cast-sandwiched model to: {output_path}")
        
        conversion_time = time.time() - start_time
        print(f"Conversion completed in {conversion_time:.2f} seconds")
        
        return m
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def weights_only_quantize_model(model_path, output_path):
    """
    Convert an ONNX model to use BF16 weights while keeping all activations as FP32.
    
    - Model inputs/outputs remain FP32
    - Only weights are quantized to BF16 using round to nearest even
    - No cast sandwiching is applied to activations
    """
    print(f"Loading model: {model_path} (mode: weights-only BF16)")
    start_time = time.time()
    
    try:
        m = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Convert to opset 22 first for better BF16 support
    m = convert_to_opset22(m)
    
    # Fix dynamic input shapes to static shapes (important for IREE compilation)
    m = fix_dynamic_input_shapes(m, model_path)
    
    g = m.graph
    
    print(f"Original model has {len(g.node)} nodes")
    
    # Analyze model inputs and outputs
    fp32_inputs = []
    fp32_outputs = []
    
    for input_info in g.input:
        if input_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            fp32_inputs.append(input_info.name)
    
    for output_info in g.output:
        if output_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            fp32_outputs.append(output_info.name)
    
    print(f"Found {len(fp32_inputs)} FP32 inputs and {len(fp32_outputs)} FP32 outputs (keeping as FP32)")
    
    # Convert all FP32 weight initializers to BF16 using round to nearest even
    weights_converted = 0
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            # Load the float32 data
            weights_fp32 = numpy_helper.to_array(init)
            
            # Convert to BF16 using round to nearest even
            weights_bf16_pattern = round_to_nearest_even_bf16(weights_fp32)
            
            # Update the initializer with BF16 bit pattern data
            new_init = numpy_helper.from_array(weights_bf16_pattern, name=init.name)
            new_init.data_type = TensorProto.FLOAT  # Keep as FP32 data type
            
            # Replace the initializer
            for i, existing_init in enumerate(g.initializer):
                if existing_init.name == init.name:
                    g.initializer[i].CopyFrom(new_init)
                    break
            
            weights_converted += 1
    
    print(f"Converted {weights_converted} weight initializers to BF16 using round to nearest even")
    print("Keeping all graph nodes unchanged (no activation quantization)")
    
    # Ensure the model uses opset 22 for better BF16 support
    if m.opset_import:
        m.opset_import[0].version = 22
        print("Set model opset to 22 for BF16 support")
    
    # Check and save model
    try:
        onnx.checker.check_model(m)
        print("Model validation passed")
    except Exception as e:
        print(f"Model validation warning: {e}")
    
    try:
        onnx.save(m, output_path)
        print(f"Saved weights-only quantized model to: {output_path}")
        
        conversion_time = time.time() - start_time
        print(f"Conversion completed in {conversion_time:.2f} seconds")
        
        return m
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def full_bf16_model(model_path, output_path):
    """
    Convert an ONNX model to use BF16 for all weights and activations.
    
    - Model inputs/outputs are BF16
    - All weights are quantized to BF16 using round to nearest even
    - All activations are BF16 (no FP32 computation)
    - Note: This model is not compatible with ONNX Runtime (BF16 not supported)
    """
    print(f"Loading model: {model_path} (mode: full BF16)")
    start_time = time.time()
    
    try:
        m = onnx.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Convert to opset 22 first for better BF16 support
    m = convert_to_opset22(m)
    
    # Fix dynamic input shapes to static shapes (important for IREE compilation)
    m = fix_dynamic_input_shapes(m, model_path)
    
    g = m.graph
    
    print(f"Original model has {len(g.node)} nodes")
    
    # Convert all FP32 weight initializers to BF16 using round to nearest even
    weights_converted = 0
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            # Load the float32 data
            weights_fp32 = numpy_helper.to_array(init)
            
            # Convert to BF16 using round to nearest even
            weights_bf16_pattern = round_to_nearest_even_bf16(weights_fp32)
            
            # Update the initializer with BF16 bit pattern data but keep as BF16 type
            new_init = numpy_helper.from_array(weights_bf16_pattern.astype(np.float32), name=init.name)
            new_init.data_type = TensorProto.BFLOAT16  # Use actual BF16 data type
            
            # Replace the initializer
            for i, existing_init in enumerate(g.initializer):
                if existing_init.name == init.name:
                    g.initializer[i].CopyFrom(new_init)
                    break
            
            weights_converted += 1
    
    print(f"Converted {weights_converted} weight initializers to BF16")
    
    # Convert model inputs to BF16
    inputs_converted = 0
    for input_info in g.input:
        if input_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            input_info.type.tensor_type.elem_type = TensorProto.BFLOAT16
            inputs_converted += 1
    
    print(f"Converted {inputs_converted} model inputs to BF16")
    
    # Convert model outputs to BF16
    outputs_converted = 0
    for output_info in g.output:
        if output_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            output_info.type.tensor_type.elem_type = TensorProto.BFLOAT16
            outputs_converted += 1
    
    print(f"Converted {outputs_converted} model outputs to BF16")
    
    # No changes to graph nodes - they will work with BF16 tensors directly
    print("Keeping all graph nodes unchanged (native BF16 computation)")
    
    # Ensure the model uses opset 22 for better BF16 support
    if m.opset_import:
        m.opset_import[0].version = 22
        print("Set model opset to 22 for BF16 support")
    
    # Check and save model (note: this may fail validation with ONNX Runtime)
    try:
        onnx.checker.check_model(m)
        print("Model validation passed")
    except Exception as e:
        print(f"Model validation warning (expected for BF16): {e}")
    
    try:
        onnx.save(m, output_path)
        print(f"Saved full BF16 model to: {output_path}")
        print("⚠️  Note: This model is not compatible with ONNX Runtime (BF16 not supported)")
        
        conversion_time = time.time() - start_time
        print(f"Conversion completed in {conversion_time:.2f} seconds")
        
        return m
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

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

def analyze_model_operations(model_path):
    """Analyze and print all operations in the model."""
    print(f"\nAnalyzing operations in: {model_path}")
    
    try:
        m = onnx.load(model_path)
        g = m.graph
        
        op_counts = {}
        # print(f"\nModel has {len(g.node)} total nodes:")
        # print("-" * 60)
        
        for i, node in enumerate(g.node):
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
            
            # if i < 10:  # Show first 10 nodes in detail
            #     print(f"{i+1:3d}. {op_type:15s} - {node.name[:40]:40s}")
            #     print(f"     Inputs:  {[inp[:30] for inp in node.input]}")
            #     print(f"     Outputs: {[out[:30] for out in node.output]}")
        
        # if len(g.node) > 10:
        #     print(f"... and {len(g.node) - 10} more nodes")
        
        # print(f"\nOperation counts:")
        # print("-" * 30)
        # for op_type, count in sorted(op_counts.items()):
        #     print(f"{op_type:20s}: {count:3d}")
        
        # print(f"\nTotal operations: {sum(op_counts.values())}")
        
    except Exception as e:
        print(f"Error analyzing model: {e}")

def process_single_model(input_path, output_dir=None, test_original=True, weights_only=False, full_bf16=False):
    """Process a single model with cast sandwiching, weights-only quantization, or full BF16."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Input model not found: {input_path}")
        return None
    
    # Determine output directory and filename
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    model_name = input_path.stem
    if full_bf16:
        suffix = "_bf16.onnx"
        mode_desc = "full BF16"
    elif weights_only:
        suffix = "_weights_bf16.onnx"
        mode_desc = "weights-only BF16"
    else:
        suffix = "_cast_sandwich.onnx"
        mode_desc = "full cast sandwiching"
    output_path = output_dir / f"{model_name}{suffix}"
    print(f"\n{'='*60}")
    print(f"PROCESSING MODEL: {model_name} ({mode_desc})")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Analyze original model
    if test_original:
        print(f"\n{'='*20} ORIGINAL MODEL ANALYSIS {'='*20}")
        analyze_model_operations(str(input_path))
        
        print(f"\n{'='*20} TESTING ORIGINAL MODEL {'='*20}")
        original_success, _ = test_model(str(input_path), f"Original {model_name}")
    
    # Convert model with cast sandwiching, weights-only, or full BF16
    print(f"\n{'='*20} CONVERTING MODEL {'='*20}")
    if full_bf16:
        converted_model = full_bf16_model(str(input_path), str(output_path))
    elif weights_only:
        converted_model = weights_only_quantize_model(str(input_path), str(output_path))
    else:
        converted_model = cast_sandwich_model(str(input_path), str(output_path))
    
    if converted_model is None:
        print(f"Failed to convert {model_name}")
        return None
    
    # Analyze converted model
    print(f"\n{'='*20} CONVERTED MODEL ANALYSIS {'='*20}")
    analyze_model_operations(str(output_path))
    
    # Test the converted model (skip for full BF16 as it's not compatible with ONNX Runtime)
    converted_success = True
    if full_bf16:
        print(f"\n{'='*20} SKIPPING MODEL TESTING {'='*20}")
        print("⚠️  Skipping inference test: Full BF16 models are not compatible with ONNX Runtime")
        converted_success = True  # Assume success since we can't test
    else:
        print(f"\n{'='*20} TESTING CONVERTED MODEL {'='*20}")
        mode_test_name = f"Weights-only {model_name}" if weights_only else f"Cast-sandwiched {model_name}"
        converted_success, _ = test_model(str(output_path), mode_test_name)
    
    # Compare models if both work (skip for full BF16)
    results = {}
    if full_bf16:
        print("⚠️  Skipping model comparison: Full BF16 models cannot be tested with ONNX Runtime")
        results = {'skipped': True, 'reason': 'BF16 not supported by ONNX Runtime'}
    elif test_original and original_success and converted_success:
        results = compare_models(str(input_path), str(output_path))
    elif not test_original and converted_success:
        # For weights-only mode when original test is skipped, still run comparison if original model exists
        results = compare_models(str(input_path), str(output_path))
    
    return {
        'model_name': model_name,
        'input_path': str(input_path),
        'output_path': str(output_path),
        'original_success': original_success if test_original else True,
        'converted_success': converted_success,
        'comparison_results': results
    }

def save_fp32_with_fixed_shapes(model_path, output_path):
    """
    Load an FP32 ONNX model, fix dynamic input shapes to static shapes, and save it.
    
    - Model remains completely FP32
    - Only fixes dynamic input shapes to static shapes
    - Important for IREE compilation compatibility
    """
    print(f"Loading model: {model_path} (mode: FP32 with fixed shapes)")
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
        print(f"Saved FP32 model with fixed shapes to: {output_path}")
        
        conversion_time = time.time() - start_time
        print(f"Conversion completed in {conversion_time:.2f} seconds")
        
        return m
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Universal ONNX Cast Sandwich Converter")
    parser.add_argument("input_model", nargs="?", help="Path to input ONNX model (optional in batch mode)")
    parser.add_argument("-o", "--output", help="Output directory (default: same as input)")
    parser.add_argument("--no-test-original", action="store_true", help="Skip testing original model")
    parser.add_argument("--batch-mode", action="store_true", help="Test multiple models with ALL modes")
    parser.add_argument("--weights-only", action="store_true", help="Use weights-only quantization (no activation cast sandwiching)")
    parser.add_argument("--full-bf16", action="store_true", help="Create full BF16 model (all weights and activations in BF16, not testable with ONNX Runtime)")
    parser.add_argument("--fix-shapes-only", action="store_true", help="Only fix dynamic input shapes to static shapes, keep model as FP32")
    
    args = parser.parse_args()
    
    if args.batch_mode and not args.input_model:
        # Batch mode without specific model - use dummy value
        args.input_model = "batch_mode"
    
    if args.batch_mode:
        # Test multiple predefined models
        test_models = [
            "input_models/moonshine_encoder_fp32_opset22.onnx",
            "input_models/moonshine_decoder_fp32_opset22.onnx", 
            "input_models/yealink_fp32_opset22.onnx",
            "input_models/mobilenet_v2_fp32.onnx"
        ]
        
        print(f"\n{'='*60}")
        print("BATCH MODE: Testing multiple models with ALL modes")
        print(f"{'='*60}")
        
        all_results = []
        
        for model_path in test_models:
            if os.path.exists(model_path):
                print(f"\n{'='*80}")
                print(f"PROCESSING: {os.path.basename(model_path)}")
                print(f"{'='*80}")
                
                # First, create FP32 model with fixed shapes
                model_name = Path(model_path).stem
                output_dir = Path(args.output) if args.output else Path("output_models")
                fp32_fixed_path = output_dir / f"{model_name}_fp32.onnx"
                
                print(f"\n{'='*30} MODE 0: FP32 WITH FIXED SHAPES {'='*30}")
                fp32_fixed_model = save_fp32_with_fixed_shapes(model_path, str(fp32_fixed_path))
                
                if fp32_fixed_model is None:
                    print(f"Failed to create FP32 model with fixed shapes for {model_name}")
                    continue
                
                # Use the FP32 fixed shape model as input for all other conversions
                base_model_path = str(fp32_fixed_path)
                
                # Process with full cast sandwiching
                print(f"\n{'='*30} MODE 1: FULL CAST SANDWICHING {'='*30}")
                result_full = process_single_model(base_model_path, args.output, not args.no_test_original, weights_only=False, full_bf16=False)
                
                # Process with weights-only quantization
                print(f"\n{'='*30} MODE 2: WEIGHTS-ONLY QUANTIZATION {'='*30}")
                result_weights = process_single_model(base_model_path, args.output, False, weights_only=True, full_bf16=False)  # Skip original test for second run
                
                # Process with full BF16
                print(f"\n{'='*30} MODE 3: FULL BF16 {'='*30}")
                result_full_bf16 = process_single_model(base_model_path, args.output, False, weights_only=False, full_bf16=True)  # Skip original test for third run
                
                # Store all results
                if result_full and result_weights and result_full_bf16:
                    combined_result = {
                        'model_name': model_name,
                        'input_path': model_path,
                        'fp32_fixed_path': str(fp32_fixed_path),
                        'original_success': result_full['original_success'],
                        'full_cast_sandwiching': result_full,
                        'weights_only': result_weights,
                        'full_bf16': result_full_bf16
                    }
                    all_results.append(combined_result)
            else:
                print(f"Model not found: {model_path}")
        
        # Comprehensive Summary
        print(f"\n{'='*80}")
        print("COMPREHENSIVE SUMMARY: ALL QUANTIZATION MODES")
        print(f"{'='*80}")
        
        # Table header
        print(f"\n{'Model':<35} {'Mode':<25} {'Success':<8} {'Rel Error (%)':<14} {'SNR (dB)':<10}")
        print("-" * 92)
        
        for result in all_results:
            model_name = result['model_name']
            
            # Full cast sandwiching results
            full_result = result['full_cast_sandwiching']
            full_success = "✓" if full_result['converted_success'] else "✗"
            if full_result['comparison_results'] and 'relative_error' in full_result['comparison_results']:
                full_error = f"{full_result['comparison_results']['relative_error']*100:.4f}%"
                full_snr = f"{full_result['comparison_results']['snr_db']:.2f}"
            else:
                full_error = "N/A"
                full_snr = "N/A"
            
            # Weights-only results
            weights_result = result['weights_only']
            weights_success = "✓" if weights_result['converted_success'] else "✗"
            if weights_result['comparison_results'] and 'relative_error' in weights_result['comparison_results']:
                weights_error = f"{weights_result['comparison_results']['relative_error']*100:.4f}%"
                weights_snr = f"{weights_result['comparison_results']['snr_db']:.2f}"
            else:
                weights_error = "N/A"
                weights_snr = "N/A"
            
            # Full BF16 results
            bf16_result = result['full_bf16']
            bf16_success = "✓" if bf16_result['converted_success'] else "✗"
            
            # Print rows
            print(f"{model_name:<35} {'Full Cast':<25} {full_success:<8} {full_error:<14} {full_snr:<10}")
            print(f"{'':<35} {'Weights-Only':<25} {weights_success:<8} {weights_error:<14} {weights_snr:<10}")
            print(f"{'':<35} {'Full BF16':<25} {bf16_success:<8} {'N/A':<14} {'N/A':<10}")
            print("-" * 92)
        
        # Analysis summary
        print(f"\n{'='*20} ANALYSIS SUMMARY {'='*20}")
        successful_models = [r for r in all_results if r['full_cast_sandwiching']['converted_success'] and r['weights_only']['converted_success'] and r['full_bf16']['converted_success']]
        
        if successful_models:
            print(f"Successfully processed {len(successful_models)}/{len(all_results)} models in all modes")
            
            # Aggregate statistics (only for testable modes)
            full_errors = [r['full_cast_sandwiching']['comparison_results']['relative_error'] 
                          for r in successful_models if r['full_cast_sandwiching']['comparison_results'] and 'relative_error' in r['full_cast_sandwiching']['comparison_results']]
            weights_errors = [r['weights_only']['comparison_results']['relative_error'] 
                             for r in successful_models if r['weights_only']['comparison_results'] and 'relative_error' in r['weights_only']['comparison_results']]
            
            if full_errors and weights_errors:
                print(f"\nAverage Relative Error:")
                print(f"  Full Cast Sandwiching: {np.mean(full_errors)*100:.4f}%")
                print(f"  Weights-Only:          {np.mean(weights_errors)*100:.4f}%")
                
                if np.mean(weights_errors) < np.mean(full_errors):
                    print("🎯 Weights-only quantization shows better accuracy on average!")
                else:
                    print("🎯 Full cast sandwiching shows better accuracy on average!")
            
            print(f"\nFull BF16 models created but not tested (ONNX Runtime doesn't support BF16)")
        else:
            print("No models were successfully processed in all modes")
        
        print(f"\n📁 All output models saved to: {args.output or 'output_models/'}")
        print("💡 Note: FP32 models with fixed shapes are saved as *_fp32.onnx")
        print("💡 Note: Full BF16 models are created for future use with BF16-compatible runtimes")
        print("Processing complete!")
    elif args.fix_shapes_only:
        # Just fix shapes mode
        input_path = Path(args.input_model)
        if not input_path.exists():
            print(f"Input model not found: {input_path}")
            return
        
        # Determine output directory and filename
        if args.output is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = input_path.stem
        output_path = output_dir / f"{model_name}_fp32.onnx"
        
        print(f"\n{'='*60}")
        print(f"FIXING INPUT SHAPES: {model_name}")
        print(f"{'='*60}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        
        result = save_fp32_with_fixed_shapes(str(input_path), str(output_path))
        
        if result:
            print(f"\n{'='*20} SUCCESS {'='*20}")
            print(f"FP32 model with fixed shapes saved to: {output_path}")
        else:
            print(f"\n{'='*20} FAILED {'='*20}")
            print(f"Failed to fix shapes for {model_name}")
    else:
        # Process single model
        result = process_single_model(args.input_model, args.output, not args.no_test_original, args.weights_only, args.full_bf16)
        
        if result:
            print(f"\n{'='*20} FINAL SUMMARY {'='*20}")
            print(f"Model: {result['model_name']}")
            if args.full_bf16:
                mode = "Full BF16 (all weights and activations in BF16)"
            elif args.weights_only:
                mode = "Weights-only quantization"
            else:
                mode = "Full cast sandwiching"
            print(f"Mode: {mode}")
            print(f"Conversion: {'SUCCESS' if result['converted_success'] else 'FAILED'}")
            if result['comparison_results'] and 'relative_error' in result['comparison_results']:
                comp = result['comparison_results']
                print(f"Relative Error: {comp['relative_error']*100:.4f}%")
                print(f"SNR: {comp['snr_db']:.2f} dB")
            elif args.full_bf16:
                print("⚠️  Model cannot be tested with ONNX Runtime (BF16 not supported)")
                print("💡 Model created for future use with BF16-compatible runtimes")

if __name__ == "__main__":
    main()
