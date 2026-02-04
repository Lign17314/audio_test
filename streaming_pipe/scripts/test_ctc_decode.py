#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 streaming_pipe 输出的 CTC 解码

用法:
    python test_ctc_decode.py [streaming_fbank.npy]
"""

import sys
import os

# Add parent directories to path to import modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMING_FBANK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(STREAMING_FBANK_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from ctc import CTC, KwsCtcPrefixDecoder
from a2 import FsmnKWS, FSMNExport, convert_conv2d_to_conv1d_weights
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr import AutoModel

CKPT_PATH = os.path.join(PROJECT_ROOT, "finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt")

def load_pt_encoder(device="cpu"):
    """加载 PyTorch encoder"""
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
    return encoder_export

def run_encoder_chunked(encoder, input_fbank, device="cpu"):
    """以分块方式运行encoder（匹配C++实现）"""
    input_tensor = torch.from_numpy(input_fbank).float().to(device)
    
    B, T, D = input_tensor.shape
    PROJ_DIM = 128
    LEFT_CTX = 9
    RIGHT_CTX = 2
    NUM_LAYERS = 4
    CHUNK_SIZE = 10
    
    # Initialize caches
    caches = [
        torch.zeros(B, PROJ_DIM, LEFT_CTX, device=device, dtype=torch.float32)
        for _ in range(NUM_LAYERS)
    ]
    
    out_list = []
    start = 0
    while start < T:
        end = min(start + CHUNK_SIZE, T)
        chunk = input_tensor[:, start:end, :]
        
        # Pad chunk to CHUNK_SIZE if needed
        if chunk.shape[1] < CHUNK_SIZE:
            pad_size = CHUNK_SIZE - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad_size))
        
        # Prepare right_context
        if end < T:
            rc = input_tensor[:, end:end + RIGHT_CTX, :]
            if rc.shape[1] < RIGHT_CTX:
                rc = torch.nn.functional.pad(rc, (0, 0, 0, RIGHT_CTX - rc.shape[1]))
        else:
            rc = torch.zeros(B, RIGHT_CTX, D, device=device, dtype=torch.float32)
        
        # Run encoder
        with torch.no_grad():
            out_chunk, caches = encoder(chunk, *caches, right_context=rc)
        
        # Only save the actual frames
        actual_frames = end - start
        out_list.append(out_chunk[:, :actual_frames, :].cpu().numpy())
        start = end
    
    # Concatenate all chunks
    return np.concatenate(out_list, axis=1)

def main():
    # 默认使用build目录下的输出
    if len(sys.argv) > 1:
        fbank_file = sys.argv[1]
    else:
        fbank_file = os.path.join(STREAMING_FBANK_DIR, "build/streaming_fbank_output.npy")
    
    print("=" * 60)
    print("Streaming FBank CTC Decoding Test")
    print("=" * 60)
    print(f"Input: {fbank_file}")
    print()
    
    # 1. Load streaming fbank
    print("Loading streaming FBank features...")
    streaming_fbank = np.load(fbank_file)
    print(f"  Shape: {streaming_fbank.shape}")
    print(f"  First frame (first 5 values): {streaming_fbank[0, 0, :5]}")
    
    # Check data quality
    if np.any(np.isnan(streaming_fbank)) or np.any(np.isinf(streaming_fbank)):
        print("  ⚠️  WARNING: Input contains NaN or Inf!")
        return
    else:
        print("  ✅ Input is valid (no NaN or Inf)")
    print()
    
    # 2. Run PyTorch encoder
    print("Running PyTorch encoder (stateful chunked)...")
    device = "cpu"
    encoder = load_pt_encoder(device=device)
    
    pt_logits = run_encoder_chunked(encoder, streaming_fbank, device=device)
    print(f"  Encoder output shape: {pt_logits.shape}")
    print(f"  First logit frame (first 5 values): {pt_logits[0, 0, :5]}")
    
    # Check encoder output quality
    if np.any(np.isnan(pt_logits)) or np.any(np.isinf(pt_logits)):
        print("  ⚠️  WARNING: Encoder output contains NaN or Inf!")
        return
    else:
        print("  ✅ Encoder output is valid (no NaN or Inf)")
    print()
    
    # 3. CTC Decoding
    print("=" * 60)
    print("CTC Decoding")
    print("=" * 60)
    
    try:
        # Load model for CTC decoder
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
        kws_decoder = KwsCtcPrefixDecoder(
            ctc=ctc,
            keywords=keywords,
            token_list=tokenizer.token_list,
            seg_dict=tokenizer.seg_dict,
        )
        
        # Decode
        pt_logits_tensor = torch.from_numpy(pt_logits[0, :, :]).float()
        result = kws_decoder.decode(pt_logits_tensor)
        
        hit, keyword, score = result
        
        print()
        if hit:
            print(f"✅ SUCCESS: Keyword detected!")
            print(f"  Keyword: '{keyword}'")
            print(f"  Confidence: {score:.4f} ({score*100:.2f}%)")
        else:
            print(f"❌ FAILED: No keyword detected")
        
        print()
        print(f"Full result: {result}")
        
    except Exception as e:
        print(f"⚠️  ERROR: Failed to perform CTC decoding: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
