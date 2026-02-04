#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比三个版本的输出：
1. Python TFLite版本
2. Python PyTorch版本
3. C++ TFLite Micro版本
"""

import sys
import os

# Add root directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

import numpy as np
import tensorflow as tf
import torch
from ctc import CTC, KwsCtcPrefixDecoder
from funasr.utils.load_utils import load_audio_text_image_video
from a2 import extract_fbank, frontend, FsmnKWS, FSMNExport
from funasr.train_utils.load_pretrained_model import load_pretrained_model

CKPT_PATH = os.path.join(ROOT_DIR, "finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt")

CHUNK_SIZE = 10
NUM_LAYERS = 4
PROJ_DIM = 128
LEFT_CTX = 9
RIGHT_CTX = 2

def load_pt_encoder(device="cpu"):
    """加载 PyTorch encoder 作为参考"""
    from a2 import convert_conv2d_to_conv1d_weights
    
    model_kws = FsmnKWS()
    model_kws.eval()
    load_pretrained_model(
        model=model_kws,
        path=CKPT_PATH,
        ignore_init_mismatch=True,
        oss_bucket=None,
        scope_map=[],
        excludes=["ctc"],
    )
    convert_conv2d_to_conv1d_weights(model_kws, checkpoint_path=CKPT_PATH)
    model_kws.float().to(device)
    encoder_export = FSMNExport(model=model_kws.encoder)
    encoder_export.eval()
    return encoder_export, model_kws

def run_pytorch_encoder_stateful_chunked(encoder_export, speech, device="cpu"):
    """运行 PyTorch encoder（stateful chunked）"""
    B, T, D = speech.shape
    caches = [
        torch.zeros(B, PROJ_DIM, LEFT_CTX, device=device, dtype=torch.float32)
        for _ in range(NUM_LAYERS)
    ]
    
    out_list = []
    start = 0
    while start < T:
        end = min(start + CHUNK_SIZE, T)
        chunk = speech[:, start:end, :]
        
        # Pad chunk to CHUNK_SIZE if needed
        if chunk.shape[1] < CHUNK_SIZE:
            pad = torch.zeros(B, CHUNK_SIZE - chunk.shape[1], D, device=device, dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=1)
        
        # Right context
        if end < T:
            rc = speech[:, end:end+RIGHT_CTX, :]
            if rc.shape[1] < RIGHT_CTX:
                pad_rc = torch.zeros(B, RIGHT_CTX - rc.shape[1], D, device=device, dtype=rc.dtype)
                rc = torch.cat([rc, pad_rc], dim=1)
        else:
            rc = None
        
        with torch.no_grad():
            out_chunk, caches = encoder_export(chunk, *caches, right_context=rc)
        
        # Only take the actual frames (not padding)
        actual_size = end - start
        out_list.append(out_chunk[:, :actual_size, :].cpu().numpy())
        start = end
    
    output = np.concatenate(out_list, axis=1)
    return output

def run_python_tflite_encoder_stateful_chunked(model_path, speech):
    """运行 Python TFLite encoder（stateful chunked）"""
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    
    # Find tensor indices
    main_input_idx = None
    right_context_idx = None
    cache_input_indices = {}
    cache_output_indices = {}
    logits_output_idx = None
    
    for inp in input_details:
        name = inp['name']
        if 'input' in name and 'cache' not in name and 'right' not in name:
            main_input_idx = inp['index']
        elif 'right_context' in name:
            right_context_idx = inp['index']
        elif 'in_cache_0' in name:
            cache_input_indices[0] = inp['index']
        elif 'in_cache_1' in name:
            cache_input_indices[1] = inp['index']
        elif 'in_cache_2' in name:
            cache_input_indices[2] = inp['index']
        elif 'in_cache_3' in name:
            cache_input_indices[3] = inp['index']
    
    # Sort output details by name to ensure consistent ordering
    def get_output_key(out):
        name = out.get('name', '')
        if ':' in name:
            try:
                return int(name.split(':')[-1])
            except:
                return 0
        return 0
    
    sorted_output_details = sorted(output_details, key=get_output_key)
    
    for out in sorted_output_details:
        name = out.get('name', '')
        idx = get_output_key(out)
        if idx == 0:
            logits_output_idx = out['index']
        elif idx >= 1 and idx <= 4:
            cache_output_indices[idx - 1] = out['index']
    
    # Get quantization parameters
    input_quant_params = None
    output_quant_params = None
    cache_quant_params = {}
    
    for inp in input_details:
        if inp['index'] == main_input_idx:
            input_quant_params = inp.get('quantization_parameters', None)
        elif inp['index'] in cache_input_indices.values():
            idx = [k for k, v in cache_input_indices.items() if v == inp['index']][0]
            cache_quant_params[idx] = inp.get('quantization_parameters', None)
    
    for out in output_details:
        if out['index'] == logits_output_idx:
            output_quant_params = out.get('quantization_parameters', None)
    
    # Quantization helper functions
    def quantize(data, quant_params, dtype):
        """量化 float32 数据到 int16/int8"""
        if quant_params is None or len(quant_params) == 0:
            return data.astype(dtype)
        scales = quant_params.get('scales', None)
        zero_points = quant_params.get('zero_points', None)
        if scales is None or len(scales) == 0 or zero_points is None or len(zero_points) == 0:
            return data.astype(dtype)
        scale = scales[0] if isinstance(scales, (list, tuple, np.ndarray)) else scales
        zero_point = zero_points[0] if isinstance(zero_points, (list, tuple, np.ndarray)) else zero_points
        
        quantized = np.round(data / scale + zero_point).astype(dtype)
        
        if dtype == np.int8:
            quantized = np.clip(quantized, -128, 127)
        elif dtype == np.int16:
            quantized = np.clip(quantized, -32768, 32767)
        
        return quantized
    
    def dequantize(data, quant_params):
        """反量化 int16/int8 数据到 float32"""
        if quant_params is None or len(quant_params) == 0:
            return data.astype(np.float32)
        scales = quant_params.get('scales', None)
        zero_points = quant_params.get('zero_points', None)
        if scales is None or len(scales) == 0 or zero_points is None or len(zero_points) == 0:
            return data.astype(np.float32)
        scale = scales[0] if isinstance(scales, (list, tuple, np.ndarray)) else scales
        zero_point = zero_points[0] if isinstance(zero_points, (list, tuple, np.ndarray)) else zero_points
        
        return (data.astype(np.float32) - zero_point) * scale
    
    B, T, D = speech.shape
    caches = [np.zeros((1, PROJ_DIM, LEFT_CTX), dtype=np.float32) for _ in range(NUM_LAYERS)]
    output_logits = []
    
    for start in range(0, T, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, T)
        current_chunk_size = end - start
        
        # Prepare chunk
        chunk = speech[:, start:end, :].astype(np.float32)
        if current_chunk_size < CHUNK_SIZE:
            pad = np.zeros((1, CHUNK_SIZE - current_chunk_size, D), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=1)
        
        # Right context
        if end < T:
            rc = speech[:, end:end+RIGHT_CTX, :].astype(np.float32)
            if rc.shape[1] < RIGHT_CTX:
                pad_rc = np.zeros((1, RIGHT_CTX - rc.shape[1], D), dtype=np.float32)
                rc = np.concatenate([rc, pad_rc], axis=1)
        else:
            rc = np.zeros((1, RIGHT_CTX, D), dtype=np.float32)
        
        # Quantize inputs if needed
        input_dtype = input_details[main_input_idx]['dtype']
        if input_dtype == np.int8 or input_dtype == np.int16:
            chunk_quant = quantize(chunk, input_quant_params, input_dtype)
            interp.set_tensor(main_input_idx, chunk_quant)
            
            rc_quant = quantize(rc, input_quant_params, input_dtype)
            interp.set_tensor(right_context_idx, rc_quant)
            
            for i in range(NUM_LAYERS):
                if i in cache_input_indices:
                    cache_quant = quantize(caches[i], cache_quant_params.get(i, None), input_dtype)
                    interp.set_tensor(cache_input_indices[i], cache_quant)
        else:
            interp.set_tensor(main_input_idx, chunk)
            interp.set_tensor(right_context_idx, rc)
            for i in range(NUM_LAYERS):
                if i in cache_input_indices:
                    interp.set_tensor(cache_input_indices[i], caches[i])
        
        # Run inference
        interp.invoke()
        
        # Get output
        if logits_output_idx is None:
            raise ValueError("Could not find logits output tensor")
        
        output = interp.get_tensor(logits_output_idx)
        
        # Find output details for logits
        logits_output_details = None
        for out in sorted_output_details:
            if out['index'] == logits_output_idx:
                logits_output_details = out
                break
        
        # Dequantize if needed
        if logits_output_details:
            output_dtype = logits_output_details['dtype']
            if output_dtype == np.int8 or output_dtype == np.int16:
                output = dequantize(output, output_quant_params)
        
        output_logits.append(output[0, :current_chunk_size, :])
        
        # Update cache
        for i in range(NUM_LAYERS):
            if i in cache_output_indices:
                cache_out = interp.get_tensor(cache_output_indices[i])
                # Find cache output details
                cache_out_details = None
                for out in sorted_output_details:
                    if out['index'] == cache_output_indices[i]:
                        cache_out_details = out
                        break
                
                if cache_out_details:
                    cache_out_dtype = cache_out_details['dtype']
                    if cache_out_dtype == np.int8 or cache_out_dtype == np.int16:
                        caches[i] = dequantize(cache_out, cache_quant_params.get(i, None))
                    else:
                        caches[i] = cache_out
                else:
                    caches[i] = cache_out
    
    output = np.concatenate(output_logits, axis=0)
    return output[np.newaxis, :, :]

def compare_outputs(output1, output2, name1, name2, threshold=1e-2):
    """对比两个输出"""
    print(f"\n{name1} vs {name2}:")
    print(f"  Shapes: {name1}={output1.shape}, {name2}={output2.shape}")
    
    # Trim to same size
    min_T = min(output1.shape[1], output2.shape[1])
    min_D = min(output1.shape[2], output2.shape[2])
    
    output1_trimmed = output1[:, :min_T, :min_D]
    output2_trimmed = output2[:, :min_T, :min_D]
    
    diff = np.abs(output1_trimmed - output2_trimmed)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    
    # Per-chunk analysis
    print(f"  Differences per chunk:")
    for chunk_idx in range(0, min_T, CHUNK_SIZE):
        chunk_end = min(chunk_idx + CHUNK_SIZE, min_T)
        chunk_diff = diff[:, chunk_idx:chunk_end, :]
        chunk_max = np.max(chunk_diff)
        chunk_mean = np.mean(chunk_diff)
        print(f"    Chunk {chunk_idx//CHUNK_SIZE + 1} (frames {chunk_idx}-{chunk_end-1}): max={chunk_max:.6e}, mean={chunk_mean:.6e}")
    
    if max_diff < threshold:
        print(f"  ✅ PASS: Max difference < {threshold}")
    else:
        print(f"  ⚠️  WARNING: Max difference >= {threshold}")
    
    return max_diff, mean_diff

def main():
    if len(sys.argv) < 4:
        print("Usage: python compare_three_versions.py <model.tflite> <input.npy> <cpp_output.npy> [wav_file]")
        print("Example: python compare_three_versions.py ../tflite_models/fsmn_encoder_stateful_16x8.tflite ../features/test_xiaoyun_fbank.npy output_logits_16x8.npy")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_file = sys.argv[2]
    cpp_output_file = sys.argv[3]
    wav_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    print("=" * 60)
    print("Three-Version Comparison")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input: {input_file}")
    print(f"C++ TFLite Micro output: {cpp_output_file}")
    if wav_file:
        print(f"WAV file: {wav_file}")
    print()
    
    # Load input
    print("Loading input...")
    input_fbank = np.load(input_file)
    if len(input_fbank.shape) == 2:
        input_fbank = input_fbank[np.newaxis, :, :]
    print(f"  Shape: {input_fbank.shape}")
    print()
    
    # 1. Run Python TFLite
    print("=" * 60)
    print("1. Running Python TFLite encoder...")
    print("=" * 60)
    try:
        python_tflite_output = run_python_tflite_encoder_stateful_chunked(model_path, input_fbank)
        print(f"Python TFLite output shape: {python_tflite_output.shape}")
    except Exception as e:
        print(f"ERROR: Failed to run Python TFLite: {e}")
        import traceback
        traceback.print_exc()
        python_tflite_output = None
    print()
    
    # 2. Run PyTorch
    print("=" * 60)
    print("2. Running PyTorch encoder...")
    print("=" * 60)
    try:
        device = "cpu"
        encoder_export, model_kws = load_pt_encoder(device=device)
        input_tensor = torch.from_numpy(input_fbank).float().to(device)
        pytorch_output = run_pytorch_encoder_stateful_chunked(encoder_export, input_tensor, device=device)
        print(f"PyTorch output shape: {pytorch_output.shape}")
    except Exception as e:
        print(f"ERROR: Failed to run PyTorch: {e}")
        import traceback
        traceback.print_exc()
        pytorch_output = None
    print()
    
    # 3. Load C++ TFLite Micro output
    print("=" * 60)
    print("3. Loading C++ TFLite Micro output...")
    print("=" * 60)
    try:
        cpp_output = np.load(cpp_output_file)
        print(f"C++ TFLite Micro output shape: {cpp_output.shape}")
    except Exception as e:
        print(f"ERROR: Failed to load C++ output: {e}")
        import traceback
        traceback.print_exc()
        cpp_output = None
    print()
    
    # 4. Compare all versions
    print("=" * 60)
    print("4. Comparison Results")
    print("=" * 60)
    
    if python_tflite_output is not None and pytorch_output is not None:
        compare_outputs(python_tflite_output, pytorch_output, "Python TFLite", "PyTorch", threshold=1e-2)
    
    if python_tflite_output is not None and cpp_output is not None:
        compare_outputs(python_tflite_output, cpp_output, "Python TFLite", "C++ TFLite Micro", threshold=1e-2)
    
    if pytorch_output is not None and cpp_output is not None:
        compare_outputs(pytorch_output, cpp_output, "PyTorch", "C++ TFLite Micro", threshold=1e-2)
    
    # 5. CTC Decoding (always perform if outputs are available)
    if pytorch_output is not None or cpp_output is not None:
        print()
        print("=" * 60)
        print("5. CTC Decoding")
        print("=" * 60)
        
        try:
            # Load CTC decoder using AutoModel (same as verify_encoder_output.py)
            from funasr import AutoModel
            model = AutoModel(
                model="/root/volume/ctc/speech_charctc_kws_phone-xiaoyun",
                keywords="小云小云",
                output_dir="./outputs/debug",
                device="cpu",
                disable_update=True,
            )
            kwargs = model.kwargs
            tokenizer = kwargs["tokenizer"]
            keywords = kwargs["keywords"]
            
            ctc_conf = {
                "dropout_rate": 0.0,
                "ctc_type": "builtin",
                "reduce": True,
                "ignore_nan_grad": True,
                "extra_linear": False,
            }
            ctc = CTC(odim=2599, encoder_output_size=2599, **ctc_conf)
            decoder = KwsCtcPrefixDecoder(
                ctc=ctc,
                keywords=keywords,
                token_list=tokenizer.token_list,
                seg_dict=tokenizer.seg_dict,
            )
            
            # Decode PyTorch output
            if pytorch_output is not None:
                print("Decoding PyTorch output...")
                pt_logits_tensor = torch.from_numpy(pytorch_output[0]).float()
                pt_result = decoder.decode(pt_logits_tensor)
                print(f"  PyTorch Result: {pt_result}")
                if pt_result[0]:
                    print(f"    Keyword: {pt_result[1]}, Score: {pt_result[2]:.6f}")
                else:
                    print(f"    No keyword detected")
            
            # Decode Python TFLite output
            if python_tflite_output is not None:
                print("Decoding Python TFLite output...")
                python_tflite_logits_tensor = torch.from_numpy(python_tflite_output[0]).float()
                python_tflite_result = decoder.decode(python_tflite_logits_tensor)
                print(f"  Python TFLite Result: {python_tflite_result}")
                if python_tflite_result[0]:
                    print(f"    Keyword: {python_tflite_result[1]}, Score: {python_tflite_result[2]:.6f}")
                else:
                    print(f"    No keyword detected")
            
            # Decode C++ output
            if cpp_output is not None:
                print("Decoding C++ TFLite Micro output...")
                cpp_logits_tensor = torch.from_numpy(cpp_output[0]).float()
                cpp_result = decoder.decode(cpp_logits_tensor)
                print(f"  C++ TFLite Micro Result: {cpp_result}")
                if cpp_result[0]:
                    print(f"    Keyword: {cpp_result[1]}, Score: {cpp_result[2]:.6f}")
                else:
                    print(f"    No keyword detected")
            
            # Compare results
            print()
            print("CTC Decoding Comparison:")
            results = []
            if pytorch_output is not None:
                results.append(("PyTorch", pt_result))
            if python_tflite_output is not None:
                results.append(("Python TFLite", python_tflite_result))
            if cpp_output is not None:
                results.append(("C++ TFLite Micro", cpp_result))
            
            if len(results) >= 2:
                # Check if all results match
                all_match = True
                first_keyword = None
                for name, result in results:
                    if result[0]:  # If keyword detected
                        if first_keyword is None:
                            first_keyword = result[1]
                        elif result[1] != first_keyword:
                            all_match = False
                            break
                    else:
                        if first_keyword is not None:
                            all_match = False
                            break
                
                if all_match and first_keyword:
                    print(f"  ✅ PASS: All versions detected the same keyword: '{first_keyword}'")
                elif all_match and first_keyword is None:
                    print(f"  ⚠️  WARNING: No keyword detected in any version")
                else:
                    print(f"  ⚠️  WARNING: Different keywords detected:")
                    for name, result in results:
                        if result[0]:
                            print(f"    {name}: '{result[1]}' (score: {result[2]:.6f})")
                        else:
                            print(f"    {name}: No keyword detected")
            
            # Confidence Score Comparison
            print()
            print("=" * 60)
            print("Confidence Score Comparison")
            print("=" * 60)
            
            scores = {}
            if pytorch_output is not None and pt_result[0]:
                scores["PyTorch"] = pt_result[2]
            if python_tflite_output is not None and python_tflite_result[0]:
                scores["Python TFLite"] = python_tflite_result[2]
            if cpp_output is not None and cpp_result[0]:
                scores["C++ TFLite Micro"] = cpp_result[2]
            
            if len(scores) > 0:
                print("\nConfidence Scores:")
                for name, score in scores.items():
                    print(f"  {name:20s}: {score:.8f} ({score*100:.4f}%)")
                
                if len(scores) >= 2:
                    print("\nConfidence Score Differences:")
                    score_list = list(scores.items())
                    for i in range(len(score_list)):
                        for j in range(i + 1, len(score_list)):
                            name1, score1 = score_list[i]
                            name2, score2 = score_list[j]
                            diff = abs(score1 - score2)
                            rel_diff = (diff / max(score1, score2)) * 100 if max(score1, score2) > 0 else 0
                            print(f"  {name1} vs {name2}:")
                            print(f"    Absolute difference: {diff:.8f} ({diff*100:.4f}%)")
                            print(f"    Relative difference: {rel_diff:.4f}%")
                            
                            # Determine which is higher
                            if score1 > score2:
                                print(f"    → {name1} is {((score1 - score2) / score2 * 100):.4f}% higher than {name2}")
                            elif score2 > score1:
                                print(f"    → {name2} is {((score2 - score1) / score1 * 100):.4f}% higher than {name1}")
                            else:
                                print(f"    → Scores are identical")
                            print()
                    
                    # Use PyTorch as reference if available
                    if "PyTorch" in scores:
                        ref_score = scores["PyTorch"]
                        print("Comparison against PyTorch (reference):")
                        for name, score in scores.items():
                            if name != "PyTorch":
                                diff = score - ref_score
                                rel_diff = (diff / ref_score) * 100 if ref_score > 0 else 0
                                print(f"  {name}:")
                                print(f"    Difference: {diff:+.8f} ({diff*100:+.4f}%)")
                                print(f"    Relative: {rel_diff:+.4f}%")
                                if abs(rel_diff) < 1.0:
                                    print(f"    ✅ Very close to PyTorch (< 1% difference)")
                                elif abs(rel_diff) < 5.0:
                                    print(f"    ⚠️  Moderate difference (< 5% difference)")
                                else:
                                    print(f"    ⚠️  Significant difference (>= 5% difference)")
                                print()
            else:
                print("  ⚠️  No confidence scores available (no keywords detected)")
        except Exception as e:
            print(f"ERROR: Failed to decode: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Comparison completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
