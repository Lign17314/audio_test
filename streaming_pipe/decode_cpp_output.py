#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解码 C++ encoder 的输出 logits

用法:
    python decode_cpp_output.py <cpp_output_logits.npy>
"""

import sys
import os
import numpy as np
import torch
from ctc import CTC, KwsCtcPrefixDecoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "../..")

# Add root directory to path to import ctc module
sys.path.insert(0, ROOT_DIR)
from ctc import CTC, KwsCtcPrefixDecoder


def main():
    if len(sys.argv) < 2:
        print("Usage: python decode_cpp_output.py <cpp_output_logits.npy>")
        print("Example: python decode_cpp_output.py output_logits.npy")
        sys.exit(1)
    
    cpp_output_file = sys.argv[1]
    
    print("=" * 60)
    print("C++ Output Decoder")
    print("=" * 60)
    print(f"C++ output file: {cpp_output_file}")
    print()
    
    # 1. Load C++ output
    print("Loading C++ encoder output...")
    cpp_logits = np.load(cpp_output_file)
    print(f"  Shape: {cpp_logits.shape}")
    
    # Handle different shapes
    if len(cpp_logits.shape) == 2:
        # (T, D) -> (1, T, D)
        cpp_logits = cpp_logits[np.newaxis, :, :]
    
    T = cpp_logits.shape[1]
    D = cpp_logits.shape[2]
    print(f"  Frames: {T}")
    print(f"  Features: {D}")
    print()
    
    # 2. CTC Decoding
    print("=" * 60)
    print("CTC Decoding")
    print("=" * 60)
    
    try:
        # Load model for CTC decoder
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
        kws_decoder = KwsCtcPrefixDecoder(
            ctc=ctc,
            keywords=keywords,
            token_list=tokenizer.token_list,
            seg_dict=tokenizer.seg_dict,
        )
        
        # Decode C++ output
        print("Decoding C++ output...")
        cpp_logits_tensor = torch.from_numpy(cpp_logits[0, :, :]).float()
        cpp_decoded = kws_decoder.decode(cpp_logits_tensor)
        
        # Display result
        cpp_hit, cpp_keyword, cpp_score = cpp_decoded
        
        print()
        print("=" * 60)
        print("Decoding Result")
        print("=" * 60)
        
        if cpp_hit:
            print(f"✅ Keyword detected: {cpp_keyword}")
            print(f"   Confidence score: {cpp_score:.6f}")
        else:
            print("❌ No keyword detected")
        
        print()
        print("Detailed result:")
        print(f"  Hit: {cpp_hit}")
        print(f"  Keyword: {cpp_keyword}")
        print(f"  Score: {cpp_score}")
        
    except Exception as e:
        print(f"⚠️  ERROR: Failed to perform CTC decoding: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Please ensure:")
        print("  1. funasr is installed: pip install funasr")
        print("  2. Model path exists: /root/volume/ctc/speech_charctc_kws_phone-xiaoyun")
        print("  3. ctc.py is in the root directory")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("Decoding completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
