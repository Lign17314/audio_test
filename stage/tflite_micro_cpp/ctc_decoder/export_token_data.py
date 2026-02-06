#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 token_list 和 seg_dict 到文件，供 C++ CTC 解码器使用

用法:
    python export_token_data.py [output_dir]

输出文件:
    - token_list.txt: 每行一个 token
    - seg_dict.txt: 格式: word token1 token2 ...
"""

import sys
import os

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    
    print("Loading AutoModel...")
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
    token_list = tokenizer.token_list
    seg_dict = tokenizer.seg_dict
    
    print(f"Token list size: {len(token_list)}")
    print(f"Seg dict size: {len(seg_dict) if seg_dict else 0}")
    
    # Export token_list
    token_list_file = os.path.join(output_dir, "token_list.txt")
    print(f"\nExporting token_list to: {token_list_file}")
    with open(token_list_file, 'w', encoding='utf-8') as f:
        for token in token_list:
            f.write(f"{token}\n")
    print(f"  Exported {len(token_list)} tokens")
    
    # Export seg_dict
    seg_dict_file = os.path.join(output_dir, "seg_dict.txt")
    print(f"\nExporting seg_dict to: {seg_dict_file}")
    with open(seg_dict_file, 'w', encoding='utf-8') as f:
        if seg_dict:
            for word, tokens in seg_dict.items():
                # Write: word token1 token2 ...
                tokens_str = ' '.join(tokens)
                f.write(f"{word} {tokens_str}\n")
            print(f"  Exported {len(seg_dict)} entries")
        else:
            print("  Seg dict is empty, creating empty file")
    
    print("\nExport completed!")
    print(f"  Token list: {token_list_file}")
    print(f"  Seg dict: {seg_dict_file}")

if __name__ == "__main__":
    main()
