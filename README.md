# ONNX Model Precision Conversion and Testing

A directory-based Python tool for converting ONNX models with multiple BF16 quantization strategies: cast sandwiching, weights-only quantization, and full BF16 conversion.

## Project Structure

```
convert_onnx_models/
├── main.py                        # Main directory-based processing script
├── src/                          # Source modules
│   ├── main_orchestrator.py      # Model processing orchestration
│   ├── activation_transforms.py  # BF16 conversion strategies
│   ├── testing_utils.py         # Model testing and comparison
│   ├── metrics_utils.py         # Error analysis and benchmarking
│   └── conversion_utils.py      # Conversion utilities
├── input_models/                 # Input ONNX models directory
│   ├── model1.onnx
│   ├── model2.onnx
│   └── ...
├── output_models/                # Generated models directory
    ├── {model}_static_opset22.onnx  # Static opset22 with fixed shapes
    ├── {model}_cast_sandwich.onnx   # Cast sandwiched models
    ├── {model}_weights_bf16.onnx    # Weights-only BF16 models
    └── {model}_bf16.onnx            # Full BF16 models

```

## Conversion Modes

### 1. Cast Sandwiching
- **Strategy**: BF16 → FP32 → BF16 around each computational operation
- **Input**: FP32 type but receives BF16 bit patterns
- **Weights**: BF16 bit patterns with round-to-nearest-even
- **Computation**: FP32 precision with BF16 precision loss simulation
- **Output**: FP32 type
- **Compatible**: ✅ ONNX Runtime testable

### 2. Weights-Only Quantization
- **Strategy**: BF16 weights, FP32 activations
- **Input**: FP32 type and precision
- **Weights**: BF16 bit patterns with round-to-nearest-even
- **Computation**: FP32 precision throughout
- **Output**: FP32 type and precision
- **Compatible**: ✅ ONNX Runtime testable

### 3. Full BF16
- **Strategy**: Native BF16 weights and activations
- **Input**: BF16 type
- **Weights**: BF16 type
- **Computation**: BF16 precision throughout
- **Output**: BF16 type
- **Compatible**: ❌ Not testable with current ONNX Runtime

## Usage

### Directory-Based Processing

Process all ONNX models in a directory:

```bash
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
```

### Command Line Options

```
python main.py <input_dir> [options]

Arguments:
  input_dir                 Directory containing input ONNX models

Options:
  -h, --help               Show help message
  --output-dir, -o DIR     Output directory (default: adjacent output_models/)
  --mode, -m MODE          Processing mode: all, cast_sandwich, weights_only, full_bf16 (default: all)
  --no-test-original       Skip testing original models (faster)
```

## Output Structure

For each input model, the following files are generated:

| Mode | Output File | ONNX Runtime | Testing | Description |
|------|-------------|--------------|---------|-------------|
| Baseline | `{model}_static_opset22.onnx` | ✅ | ✅ | Static opset22 with fixed shapes |
| Cast Sandwich | `{model}_cast_sandwich.onnx` | ✅ | ✅ | BF16→FP32→BF16 operations |
| Weights-Only | `{model}_weights_bf16.onnx` | ✅ | ✅ | BF16 weights, FP32 activations |
| Full BF16 | `{model}_bf16.onnx` | ❌ | ⏭️ | Native BF16 (future use) |

## Comprehensive Analysis

### Summary Tables

When processing with `--mode all`, the tool generates comprehensive comparison tables:

```
================================================================================
COMPREHENSIVE SUMMARY: ALL QUANTIZATION MODES
================================================================================

Model                               Mode                      Success  Rel Error (%)  SNR (dB)  
--------------------------------------------------------------------------------------------
mobilenet_v2                        Full Cast                 ✓        0.0012%        68.54     
                                     Weights-Only              ✓        0.0008%        72.31     
                                     Full BF16                 ✓        N/A            N/A       
--------------------------------------------------------------------------------------------
moonshine_encoder                    Full Cast                 ✓        0.0045%        58.22     
                                     Weights-Only              ✓        0.0031%        61.87     
                                     Full BF16                 ✓        N/A            N/A       
--------------------------------------------------------------------------------------------

==================== ANALYSIS SUMMARY ====================
Successfully processed 2/2 models in all modes

Average Relative Error:
  Full Cast Sandwiching: 0.0029%
  Weights-Only:          0.0020%

🎯 Weights-only quantization shows better accuracy on average!

📁 All output models saved to: output_models/
💡 Note: Static opset22 models with fixed shapes are saved as *_static_opset22.onnx
💡 Note: Full BF16 models are created for future use with BF16-compatible runtimes
Processing complete!
```

### Error Analysis

For each testable mode, the tool computes:
- **Relative Error**: Sum of absolute differences relative to static opset22 baseline
- **Signal-to-Noise Ratio (SNR)**: Quality metric in decibels

## Key Features

### 🎯 **Round-to-Nearest-Even**
- Improved numerical properties for BF16 weight conversion
- Better deterministic behavior compared to stochastic rounding

### 📊 **Comprehensive Testing**
- Automatic inference validation for ONNX Runtime compatible models
- Numerical analysis comparing all modes to static opset22 baseline
- Error metrics: relative error, SNR, normalized RMSE

### 🔧 **Robust Processing**
- Handles dynamic input shapes by fixing them to static shapes
- Compatible with IREE compilation requirements
- Opset 22 standardization for better compatibility

### 📈 **Detailed Reporting**
- Progress tracking for each model and mode
- Comprehensive summary tables
- Per-model and aggregate statistics

### 🏗️ **Modular Architecture**
- Clean separation of conversion strategies
- Reusable components for different use cases
- Easy to extend with new quantization methods

## Technical Details

### Cast Sandwiching Implementation
1. **Weights**: Convert FP32 weights to BF16 bit patterns using round-to-nearest-even
2. **Operations**: Wrap each computational operation with Cast(BF16→FP32) and Cast(FP32→BF16)
3. **Inputs**: Accept FP32 type containing BF16 bit patterns
4. **Outputs**: Produce FP32 type for compatibility

### Weights-Only Implementation
1. **Weights**: Convert FP32 weights to BF16 bit patterns using round-to-nearest-even
2. **Operations**: Preserve original FP32 computation
3. **Types**: Keep all tensor types as FP32 for ONNX Runtime compatibility

### Full BF16 Implementation
1. **Weights**: Convert to native BF16 type
2. **Operations**: Preserve original operations with BF16 types
3. **Types**: Convert all float tensors to BF16 type
4. **Note**: Generated for future use when ONNX Runtime supports BF16

## Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Example Workflow

1. **Prepare Input**: Place ONNX models in `input_models/` directory
2. **Run Processing**: Execute `python main.py input_models/`
3. **Review Results**: Check `output_models/` for converted models
4. **Analyze Summary**: Review comprehensive analysis tables
5. **Use Models**: Deploy appropriate precision variant based on requirements

## Notes

- **Full BF16 models** are generated but not tested due to ONNX Runtime limitations
- **Cast sandwiched models** require BF16 bit patterns as input for proper behavior
- **Weights-only models** offer the best compatibility while maintaining good accuracy
- **Fixed shapes** are applied to ensure IREE compilation compatibility
- **Opset 22** is used for maximum compatibility across runtime environments
