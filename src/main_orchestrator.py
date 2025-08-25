"""
Main orchestrator for ONNX BF16 conversion pipeline.

Handles single model processing, batch processing, and result summary generation.
Coordinates between all the modular components to provide a unified workflow.
"""

import os
import argparse
from pathlib import Path
import numpy as np

from .conversion_utils import save_fp32_with_fixed_shapes, convert_to_opset22, fix_dynamic_input_shapes
from .activation_transforms import cast_sandwich_model, weights_only_quantize_model, full_bf16_model
from .testing_utils import test_model, analyze_model_operations
from .metrics_utils import compare_models


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


def process_batch_models(model_list, output_dir=None, test_original=True):
    """Process multiple models with all quantization modes."""
    print(f"\n{'='*60}")
    print("BATCH MODE: Testing multiple models with ALL modes")
    print(f"{'='*60}")
    
    all_results = []
    
    for model_path in model_list:
        if os.path.exists(model_path):
            print(f"\n{'='*80}")
            print(f"PROCESSING: {os.path.basename(model_path)}")
            print(f"{'='*80}")
            
            # First, create FP32 model with fixed shapes
            model_name = Path(model_path).stem
            output_dir_path = Path(output_dir) if output_dir else Path("output_models")
            fp32_fixed_path = output_dir_path / f"{model_name}_fp32.onnx"
            
            print(f"\n{'='*30} MODE 0: FP32 WITH FIXED SHAPES {'='*30}")
            fp32_fixed_model = save_fp32_with_fixed_shapes(model_path, str(fp32_fixed_path))
            
            if fp32_fixed_model is None:
                print(f"Failed to create FP32 model with fixed shapes for {model_name}")
                continue
            
            # Use the FP32 fixed shape model as input for all other conversions
            base_model_path = str(fp32_fixed_path)
            
            # Process with full cast sandwiching
            print(f"\n{'='*30} MODE 1: FULL CAST SANDWICHING {'='*30}")
            result_full = process_single_model(base_model_path, output_dir, test_original, weights_only=False, full_bf16=False)
            
            # Process with weights-only quantization
            print(f"\n{'='*30} MODE 2: WEIGHTS-ONLY QUANTIZATION {'='*30}")
            result_weights = process_single_model(base_model_path, output_dir, False, weights_only=True, full_bf16=False)  # Skip original test for second run
            
            # Process with full BF16
            print(f"\n{'='*30} MODE 3: FULL BF16 {'='*30}")
            result_full_bf16 = process_single_model(base_model_path, output_dir, False, weights_only=False, full_bf16=True)  # Skip original test for third run
            
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
    
    return all_results


def generate_comprehensive_summary(all_results, output_dir=None):
    """Generate a comprehensive summary of all quantization modes."""
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
    
    print(f"\n📁 All output models saved to: {output_dir or 'output_models/'}")
    print("💡 Note: FP32 models with fixed shapes are saved as *_fp32.onnx")
    print("💡 Note: Full BF16 models are created for future use with BF16-compatible runtimes")
    print("Processing complete!")


def get_default_test_models():
    """Get default list of test models for batch processing."""
    return [
        "input_models/moonshine_encoder_fp32_opset22.onnx",
        "input_models/moonshine_decoder_fp32_opset22.onnx", 
        "input_models/yealink_fp32_opset22.onnx",
        "input_models/mobilenet_v2_fp32.onnx"
    ]
