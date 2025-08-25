#!/usr/bin/env python3
"""
Universal ONNX BF16 Precision Conversion Pipeline

A directory-based Python tool for converting ONNX models with multiple BF16 quantization strategies:
- Cast sandwiching: BF16 → FP32 → BF16 around each computational operation
- Weights-only quantization: BF16 weights with FP32 activations  
- Full BF16 conversion: Native BF16 weights and activations

Features:
- Automatic opset 22 conversion for better BF16 support
- Tensor-type-based cast sandwiching (only applies to FP32 tensors)
- Preserves correct output types for operations like Shape (int64), comparisons (bool), etc.
- Stochastic rounding for BF16 weights  
- Comprehensive performance evaluation and comparison
- Directory-based batch processing
- Detailed error analysis and summary tables
"""

import argparse
import os
from pathlib import Path

from src.main_orchestrator import (
    process_single_model, 
    process_batch_models, 
    generate_comprehensive_summary,
    get_default_test_models
)
from src.conversion_utils import save_fp32_with_fixed_shapes


def process_directory(input_dir, output_dir=None, mode="all", test_original=True):
    """
    Process all ONNX models in a directory with specified quantization modes.
    
    Args:
        input_dir: Directory containing input ONNX models
        output_dir: Output directory for converted models  
        mode: Processing mode ('all', 'cast_sandwich', 'weights_only', 'full_bf16')
        test_original: Whether to test original models
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory not found: {input_dir}")
        return
        
    if not input_path.is_dir():
        print(f"Input path is not a directory: {input_dir}")
        return
    
    # Find all ONNX models in the directory
    onnx_files = list(input_path.glob("*.onnx"))
    if not onnx_files:
        print(f"No ONNX models found in directory: {input_dir}")
        return
    
    print(f"Found {len(onnx_files)} ONNX models in {input_dir}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = input_path.parent / "output_models"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == "all":
        # Process with all modes (batch processing)
        model_paths = [str(f) for f in onnx_files]
        all_results = process_batch_models(model_paths, str(output_dir), test_original)
        generate_comprehensive_summary(all_results, str(output_dir))
    else:
        # Process with single mode
        weights_only = (mode == "weights_only")
        full_bf16 = (mode == "full_bf16")
        
        results = []
        for model_path in onnx_files:
            print(f"\n{'='*80}")
            print(f"PROCESSING: {model_path.name}")
            print(f"{'='*80}")
            
            result = process_single_model(
                str(model_path), 
                str(output_dir), 
                test_original, 
                weights_only, 
                full_bf16
            )
            
            if result:
                results.append(result)
        
        # Print summary for single mode
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE: {mode.upper()} MODE")
        print(f"{'='*60}")
        print(f"Successfully processed {len(results)}/{len(onnx_files)} models")
        print(f"Output directory: {output_dir}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Universal ONNX BF16 Precision Conversion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models with all three modes
  python main.py input_models/

  # Process with specific output directory
  python main.py input_models/ --output-dir my_output/

  # Process with only cast sandwiching
  python main.py input_models/ --mode cast_sandwich

  # Process with only weights-only quantization
  python main.py input_models/ --mode weights_only

  # Process with only full BF16
  python main.py input_models/ --mode full_bf16

  # Process single model
  python main.py input_models/model.onnx --mode cast_sandwich

  # Fix shapes only (no quantization)
  python main.py input_models/model.onnx --fix-shapes-only
        """
    )
    
    parser.add_argument(
        "input_path", 
        nargs="?",  # Make it optional
        help="Path to input ONNX model or directory containing ONNX models (optional in batch mode)"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        help="Output directory (default: adjacent output_models/ directory)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["all", "cast_sandwich", "weights_only", "full_bf16"],
        default="all",
        help="Processing mode (default: all)"
    )
    parser.add_argument(
        "--no-test-original", 
        action="store_true", 
        help="Skip testing original models (faster processing)"
    )
    parser.add_argument(
        "--batch-mode", 
        action="store_true", 
        help="Process predefined test models with all modes"
    )
    parser.add_argument(
        "--fix-shapes-only", 
        action="store_true", 
        help="Only fix dynamic input shapes to static shapes, keep model as FP32"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.batch_mode and not args.input_path:
        parser.error("input_path is required unless using --batch-mode")
    
    if args.batch_mode:
        # Use predefined test models
        test_models = get_default_test_models()
        
        print(f"\n{'='*60}")
        print("BATCH MODE: Testing predefined models with ALL modes")
        print(f"{'='*60}")
        
        all_results = process_batch_models(
            test_models, 
            args.output_dir, 
            not args.no_test_original
        )
        generate_comprehensive_summary(all_results, args.output_dir)
        
    elif args.fix_shapes_only:
        # Just fix shapes mode
        input_path = Path(args.input_path)
        if not input_path.exists():
            print(f"Input model not found: {input_path}")
            return
        
        # Determine output directory and filename
        if args.output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(args.output_dir)
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
        # Process input path (file or directory)
        input_path = Path(args.input_path)
        
        if not input_path.exists():
            print(f"Input path not found: {args.input_path}")
            return
        
        if input_path.is_dir():
            # Directory processing
            process_directory(
                str(input_path),
                args.output_dir,
                args.mode,
                not args.no_test_original
            )
        else:
            # Single file processing
            if args.mode == "all":
                print("Warning: Single file processing with --mode=all will process the file with all three modes")
                
                # Create FP32 baseline first
                model_name = input_path.stem
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                else:
                    output_dir = input_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                fp32_path = output_dir / f"{model_name}_fp32.onnx"
                fp32_model = save_fp32_with_fixed_shapes(str(input_path), str(fp32_path))
                
                if fp32_model:
                    # Process with all modes using the FP32 baseline
                    all_results = process_batch_models(
                        [str(fp32_path)], 
                        str(output_dir), 
                        not args.no_test_original
                    )
                    generate_comprehensive_summary(all_results, str(output_dir))
            else:
                # Single mode processing
                weights_only = (args.mode == "weights_only") 
                full_bf16 = (args.mode == "full_bf16")
                
                result = process_single_model(
                    str(input_path),
                    args.output_dir,
                    not args.no_test_original,
                    weights_only,
                    full_bf16
                )
                
                if result:
                    print(f"\n{'='*20} FINAL SUMMARY {'='*20}")
                    print(f"Model: {result['model_name']}")
                    if full_bf16:
                        mode = "Full BF16 (all weights and activations in BF16)"
                    elif weights_only:
                        mode = "Weights-only quantization"
                    else:
                        mode = "Full cast sandwiching"
                    print(f"Mode: {mode}")
                    print(f"Conversion: {'SUCCESS' if result['converted_success'] else 'FAILED'}")
                    if result['comparison_results'] and 'relative_error' in result['comparison_results']:
                        comp = result['comparison_results']
                        print(f"Relative Error: {comp['relative_error']*100:.4f}%")
                        print(f"SNR: {comp['snr_db']:.2f} dB")
                    elif full_bf16:
                        print("⚠️  Model cannot be tested with ONNX Runtime (BF16 not supported)")
                        print("💡 Model created for future use with BF16-compatible runtimes")


if __name__ == "__main__":
    main()
