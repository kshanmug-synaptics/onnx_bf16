#!/usr/bin/env python3
"""
Activation transformation utilities for BF16 quantization strategies.

This module contains functions for:
- Cast sandwiching (BF16 -> FP32 -> BF16 operations)
- Weights-only quantization
- Full BF16 conversion
- Output type inference for operations
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import time
from .conversion_utils import convert_to_opset22, fix_dynamic_input_shapes, convert_weights_to_bf16


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


def create_cast_sandwiched_graph(model):
    """
    Apply cast sandwiching to a model graph.
    
    For each operation: Cast inputs through BF16 precision loss, run op in FP32, cast outputs through BF16 precision loss.
    Only applies cast sandwiching to FP32 tensors, preserves int64/bool/etc. types.
    """
    g = model.graph
    
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
    
    # Clear existing value_info since we've changed the graph structure
    g.ClearField("value_info")
    
    return model


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
    
    # Convert all FP32 weight initializers to BF16 using round to nearest even
    weights_converted = convert_weights_to_bf16(m.graph)
    print(f"Converted {weights_converted} weight initializers to BF16 using round to nearest even")
    
    # Apply cast sandwiching to all operations
    m = create_cast_sandwiched_graph(m)
    
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
    weights_converted = convert_weights_to_bf16(g)
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
            from .conversion_utils import round_to_nearest_even_bf16
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
