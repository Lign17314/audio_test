#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取 fbank 特征并保存，供 C++ 版本使用

用法:
    python prepare_fbank_features.py <wav_file> [output_dir]

参考 a2.py 的方式加载音频和处理特征
"""

import sys
import os
import numpy as np
import torch
from funasr import AutoModel
from funasr.utils.load_utils import load_audio_text_image_video
from a2 import extract_fbank, frontend

def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_fbank_features.py <wav_file> [output_dir]")
        print("Example: python prepare_fbank_features.py test.wav ./features")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./features"
    
    # 检查文件是否存在
    if not os.path.isfile(wav_file):
        print(f"ERROR: WAV file not found: {wav_file}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(wav_file))[0]
    fbank_file = os.path.join(output_dir, f"{base_name}_fbank.npy")
    info_file = os.path.join(output_dir, f"{base_name}_info.txt")
    
    print(f"Processing: {wav_file}")
    print(f"Output directory: {output_dir}")
    
    # 参考 a2.py 的方式：使用 AutoModel 获取 tokenizer 和 frontend
    print("Loading AutoModel to get tokenizer and frontend...")
    model = AutoModel(
        model="/root/volume/ctc/speech_charctc_kws_phone-xiaoyun",
        keywords="小云小云",
        output_dir="./outputs/debug",
        device="cpu",
        disable_update=True,
    )
    kwargs = model.kwargs
    tokenizer = kwargs["tokenizer"]
    frontend_from_model = kwargs["frontend"]  # 使用模型中的 frontend（与 a2.py 一致）
    
    print("Loading audio file...")
    # 使用与 a2.py 完全相同的方式加载音频
    audio_sample_list = load_audio_text_image_video(
        wav_file,
        fs=frontend_from_model.fs,
        audio_fs=kwargs.get("fs", 16000),
        data_type="sound",
        tokenizer=tokenizer,
    )
    
    # 提取 fbank 特征（使用模型中的 frontend）
    print("Extracting fbank features...")
    speech, speech_lengths = extract_fbank(
        audio_sample_list,
        data_type="fbank",
        frontend=frontend_from_model,
    )
    
    # 统一 dtype=float32（与 a2.py 一致）
    speech = speech.to(device="cpu", dtype=torch.float32)
    speech_lengths = speech_lengths.to(device="cpu")
    
    # 转换为 numpy 并保存
    speech_np = speech.cpu().numpy().astype(np.float32)
    speech_lengths_np = speech_lengths.cpu().numpy().astype(np.int32)
    
    # 保存特征数据
    np.save(fbank_file, speech_np)
    print(f"Saved fbank features: {fbank_file}")
    print(f"  Shape: {speech_np.shape}")
    print(f"  Dtype: {speech_np.dtype}")
    print(f"  Length: {speech_lengths_np[0]} frames")
    
    # 保存信息文件（供 C++ 读取）
    with open(info_file, 'w') as f:
        f.write(f"shape: {speech_np.shape}\n")
        f.write(f"dtype: float32\n")
        f.write(f"length: {speech_lengths_np[0]}\n")
        f.write(f"chunk_size: 10\n")
    
    print(f"Saved info file: {info_file}")
    print("\n✅ Feature extraction completed!")
    print(f"\nNext steps:")
    print(f"  1. Run C++ encoder: ./tflite_micro_cpp/run_encoder <model.tflite> {fbank_file} <output.npy>")
    print(f"  2. Verify results: python verify_encoder_output.py {fbank_file} <output.npy>")

if __name__ == "__main__":
    main()
