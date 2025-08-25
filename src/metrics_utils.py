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
from .testing_utils import create_test_inputs


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
                metrics = compute_error_metrics(fp32_output, bf16_output)
                
                # Add timing information
                metrics['fp32_time'] = fp32_time
                metrics['bf16_time'] = bf16_time
                metrics['speed_ratio'] = bf16_time / fp32_time
                
                print(f"\n{'='*20} ERROR ANALYSIS {'='*20}")
                print(f"📊 COMPARISON RESULTS:")
                print(f"-" * 40)
                print(f"Relative Error:           {metrics['relative_error']:.8f} ({metrics['relative_error']*100:.6f}%)")
                print(f"Signal-to-Noise Ratio:    {metrics['snr_db']:.2f} dB")
                
                return metrics
            else:
                print(f"Cannot compare: shape mismatch or empty arrays")
                return {}
        else:
            print("No outputs to compare")
            return {}
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        traceback.print_exc()
        return {}


def compute_error_metrics(reference_output, test_output):
    """
    Compute comprehensive error metrics between reference and test outputs.
    
    Args:
        reference_output: Reference (ground truth) output array
        test_output: Test output array to compare against reference
        
    Returns:
        Dictionary containing various error metrics
    """
    # Absolute difference
    abs_diff = np.abs(reference_output - test_output)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Relative error
    epsilon = 1e-12
    relative_error = np.sum(abs_diff) / (np.sum(np.abs(reference_output)) + epsilon)
    
    # Signal-to-noise ratio
    signal_power = np.mean(reference_output ** 2)
    noise_power = np.mean((reference_output - test_output) ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + epsilon))
    
    # Cosine similarity
    ref_flat = reference_output.flatten()
    test_flat = test_output.flatten()
    cosine_sim = np.dot(ref_flat, test_flat) / (np.linalg.norm(ref_flat) * np.linalg.norm(test_flat) + epsilon)
    
    # Root mean square error
    rmse = np.sqrt(np.mean((reference_output - test_output) ** 2))
    
    # Normalized RMSE (NRMSE)
    output_range = np.max(reference_output) - np.min(reference_output)
    nrmse = rmse / (output_range + epsilon)
    
    # Mean absolute percentage error (MAPE)
    mape = np.mean(np.abs((reference_output - test_output) / (reference_output + epsilon))) * 100
    
    return {
        'relative_error': relative_error,
        'snr_db': snr_db,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'rmse': rmse,
        'nrmse': nrmse,
        'cosine_similarity': cosine_sim,
        'mape': mape
    }


def benchmark_model(model_path, num_runs=10, warmup_runs=3):
    """
    Benchmark model inference performance.
    
    Args:
        model_path: Path to ONNX model
        num_runs: Number of inference runs for timing
        warmup_runs: Number of warmup runs (not included in timing)
        
    Returns:
        Dictionary with timing statistics
    """
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return {}
    
    try:
        session = ort.InferenceSession(model_path)
        test_inputs = create_test_inputs(session)
        
        # Warmup runs
        for _ in range(warmup_runs):
            session.run(None, test_inputs)
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            session.run(None, test_inputs)
            times.append(time.time() - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times)
        }
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return {}


def generate_summary_table(results_list):
    """
    Generate a formatted summary table from a list of model results.
    
    Args:
        results_list: List of dictionaries containing model processing results
        
    Returns:
        Formatted string containing the summary table
    """
    if not results_list:
        return "No results to display"
    
    # Table header
    header = f"{'Model':<35} {'Mode':<25} {'Success':<8} {'Rel Error (%)':<14} {'SNR (dB)':<10}"
    separator = "-" * 92
    
    lines = [header, separator]
    
    for result in results_list:
        model_name = result['model_name']
        
        # Full cast sandwiching results
        if 'full_cast_sandwiching' in result:
            full_result = result['full_cast_sandwiching']
            full_success = "✓" if full_result['converted_success'] else "✗"
            if full_result['comparison_results'] and 'relative_error' in full_result['comparison_results']:
                full_error = f"{full_result['comparison_results']['relative_error']*100:.4f}%"
                full_snr = f"{full_result['comparison_results']['snr_db']:.2f}"
            else:
                full_error = "N/A"
                full_snr = "N/A"
            
            lines.append(f"{model_name:<35} {'Full Cast':<25} {full_success:<8} {full_error:<14} {full_snr:<10}")
        
        # Weights-only results
        if 'weights_only' in result:
            weights_result = result['weights_only']
            weights_success = "✓" if weights_result['converted_success'] else "✗"
            if weights_result['comparison_results'] and 'relative_error' in weights_result['comparison_results']:
                weights_error = f"{weights_result['comparison_results']['relative_error']*100:.4f}%"
                weights_snr = f"{weights_result['comparison_results']['snr_db']:.2f}"
            else:
                weights_error = "N/A"
                weights_snr = "N/A"
            
            lines.append(f"{'':<35} {'Weights-Only':<25} {weights_success:<8} {weights_error:<14} {weights_snr:<10}")
        
        # Full BF16 results
        if 'full_bf16' in result:
            bf16_result = result['full_bf16']
            bf16_success = "✓" if bf16_result['converted_success'] else "✗"
            
            lines.append(f"{'':<35} {'Full BF16':<25} {bf16_success:<8} {'N/A':<14} {'N/A':<10}")
        
        lines.append(separator)
    
    return "\n".join(lines)


def compute_aggregate_statistics(results_list):
    """
    Compute aggregate statistics across multiple model results.
    
    Args:
        results_list: List of model processing results
        
    Returns:
        Dictionary with aggregate statistics
    """
    successful_models = [r for r in results_list 
                        if r.get('full_cast_sandwiching', {}).get('converted_success', False) 
                        and r.get('weights_only', {}).get('converted_success', False) 
                        and r.get('full_bf16', {}).get('converted_success', False)]
    
    if not successful_models:
        return {}
    
    # Aggregate statistics (only for testable modes)
    full_errors = [r['full_cast_sandwiching']['comparison_results']['relative_error'] 
                  for r in successful_models 
                  if r['full_cast_sandwiching']['comparison_results'] 
                  and 'relative_error' in r['full_cast_sandwiching']['comparison_results']]
    
    weights_errors = [r['weights_only']['comparison_results']['relative_error'] 
                     for r in successful_models 
                     if r['weights_only']['comparison_results'] 
                     and 'relative_error' in r['weights_only']['comparison_results']]
    
    stats = {
        'total_models': len(results_list),
        'successful_models': len(successful_models)
    }
    
    if full_errors:
        stats['full_cast_mean_error'] = np.mean(full_errors)
        stats['full_cast_std_error'] = np.std(full_errors)
    
    if weights_errors:
        stats['weights_only_mean_error'] = np.mean(weights_errors)
        stats['weights_only_std_error'] = np.std(weights_errors)
    
    if full_errors and weights_errors:
        stats['weights_only_better'] = np.mean(weights_errors) < np.mean(full_errors)
    
    return stats
