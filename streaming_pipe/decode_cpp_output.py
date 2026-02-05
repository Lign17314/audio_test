#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解码 C++ encoder 的输出 logits

这个脚本用于对 C++ 端生成的 encoder 输出进行 CTC 解码，
检测是否包含特定的关键词（如"小云小云"）。

用法:
    python decode_cpp_output.py <cpp_output_logits.npy>
    
示例:
    python decode_cpp_output.py output_logits.npy
"""

import sys
import os
import numpy as np
import torch
import math
import torch.nn.functional as F
from ctc import CTC, KwsCtcPrefixDecoder
from collections import defaultdict
# 获取脚本所在目录和根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "../..")

# 将根目录添加到 Python 路径，以便导入 ctc 模块
sys.path.insert(0, ROOT_DIR)
from ctc import CTC, KwsCtcPrefixDecoder

def is_sublist(main_list, check_list):
    """检查 check_list 是否是 main_list 的子列表
    
    Args:
        main_list: 主列表（被检查的列表）
        check_list: 要查找的子列表
        
    Returns:
        int: 如果找到子列表，返回其在主列表中的起始索引；否则返回 -1
        
    示例:
        is_sublist([1, 2, 3, 4, 5], [3, 4]) -> 2
        is_sublist([1, 2, 3], [4, 5]) -> -1
    """
    # 如果主列表比子列表短，不可能包含
    if len(main_list) < len(check_list):
        return -1
    
    # 如果长度相等，直接比较
    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1
    
    # 遍历主列表，查找子列表
    for i in range(len(main_list) - len(check_list)):
        if main_list[i] == check_list[0]:
            # 找到第一个元素匹配，检查后续元素
            for j in range(len(check_list)):
                if main_list[i + j] != check_list[j]:
                    break
            else:
                # 所有元素都匹配，返回起始索引
                return i
    else:
        return -1


def main():
    # 检查命令行参数
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

    # ========== 步骤 1: 加载 C++ 输出的 logits ==========
    print("Loading C++ encoder output...")
    cpp_logits = np.load(cpp_output_file)
    print(f"  Shape: {cpp_logits.shape}")

    # 处理不同的输入形状
    if len(cpp_logits.shape) == 2:
        # (T, D) -> (1, T, D) 添加 batch 维度
        cpp_logits = cpp_logits[np.newaxis, :, :]

    T = cpp_logits.shape[1]  # 时间帧数
    D = cpp_logits.shape[2]  # 特征维度（词汇表大小）
    print(f"  Frames: {T}")
    print(f"  Features: {D}")
    print()

    # ========== 步骤 2: CTC 解码 ==========
    print("=" * 60)
    print("CTC Decoding")
    print("=" * 60)

    try:
        # 准备输入数据
        print("Decoding C++ output...")
        # 转换为 PyTorch tensor 并添加 batch 维度
        cpp_logits_tensor = torch.from_numpy(cpp_logits[0, :, :]).float().unsqueeze(0)
        
        # 对 logits 进行 softmax，得到概率分布
        raw_logp = F.softmax(cpp_logits_tensor, dim=2).detach().squeeze(0)
        xlen = torch.tensor([raw_logp.size(1)])

        # ========== CTC Beam Search 参数设置 ==========
        maxlen = raw_logp.size(0)  # 最大时间帧数
        ctc_probs = raw_logp  # CTC 概率分布 (T, vocab_size)
        
        # 初始假设：空序列，概率为 (pb=1.0, pnb=0.0)
        # cur_hyps 格式: [(prefix, (pb, pnb, nodes))]
        # - prefix: 当前的 token 序列（tuple）
        # - pb: blank 结尾的路径概率
        # - pnb: non-blank 结尾的路径概率
        # - nodes: 记录每个 token 的详细信息 [{'token': id, 'frame': t, 'prob': p}]
        cur_hyps = [(tuple(), (1.0, 0.0, []))]
        
        score_beam_size = 3  # 每帧保留的 top-k 候选数
        path_beam_size = 20  # 保留的路径数
        
        # 关键词相关的 token ID 集合（用于过滤）
        keywords_tokenset = {0, 1462, 976}  # 0=blank, 1462=小, 976=云
        
        # 关键词定义
        keywords_token = {
            "小云小云": {
                "token_id": (1462, 976, 1462, 976),  # "小云小云" 对应的 token 序列
                "token_str": "1462 976 1462 976 ",
            }
        }
        # ========== CTC Beam Search 主循环 ==========
        # 逐帧处理，每一帧更新所有候选路径
        print("maxlen" , maxlen)
        for t in range(0, maxlen):
            probs = ctc_probs[t]  # 当前帧的概率分布 (vocab_size,)
            
            # next_hyps 用于存储当前帧的所有候选路径
            # key: prefix (token序列), value: (pb, pnb, nodes)
            next_hyps = defaultdict(lambda: (0.0, 0.0, []))

            # ===== 2.1 第一次剪枝：选择 top-k 最高概率的 token =====
            top_k_probs, top_k_index = probs.topk(score_beam_size)  # (score_beam_size,)
            # print(top_k_probs.tolist(), top_k_index.tolist())
            # 过滤掉概率太小的 token 和不在关键词集合中的 token
            filter_probs = []
            filter_index = []
            for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
                if keywords_tokenset is not None:
                    # 只保留概率 > 0.05 且在关键词集合中的 token
                    if prob > 0.05 and idx in keywords_tokenset:
                        filter_probs.append(prob)
                        filter_index.append(idx)
                else:
                    if prob > 0.05:
                        filter_probs.append(prob)
                        filter_index.append(idx)
            # print(filter_probs, filter_index)
            # 如果没有符合条件的 token，跳过当前帧
            if len(filter_index) == 0:
                continue

            # ===== 遍历所有候选 token，更新路径 =====
            for s in filter_index:
                ps = probs[s].item()  # 当前 token 的概率
                if s != 0:
                    print(f"frame:{t}, token:{s}, score:{ps}")

                # 遍历上一帧的所有候选路径
                for prefix, (pb, pnb, cur_nodes) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None  # 路径的最后一个 token
                    
                    if s == 0:  # 当前 token 是 blank
                        # blank 不改变路径，只更新概率
                        n_pb, n_pnb, nodes = next_hyps[prefix]
                        n_pb = n_pb + pb * ps + pnb * ps  # 累加概率
                        nodes = cur_nodes.copy()
                        next_hyps[prefix] = (n_pb, n_pnb, nodes)
                        
                    elif s == last:  # 当前 token 与路径最后一个 token 相同
                        # 情况 1: *ss -> *s (non-blank 结尾，重复 token 不添加)
                        if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                            n_pb, n_pnb, nodes = next_hyps[prefix]
                            n_pnb = n_pnb + pnb * ps
                            nodes = cur_nodes.copy()
                            # 更新最后一个 token 的概率和帧位置（取最大概率）
                            if ps > nodes[-1]["prob"]:
                                nodes[-1]["prob"] = ps
                                nodes[-1]["frame"] = t
                            next_hyps[prefix] = (n_pb, n_pnb, nodes)

                        # 情况 2: *s-s -> *ss (blank 结尾，可以添加重复 token)
                        if not math.isclose(pb, 0.0, abs_tol=0.000001):
                            n_prefix = prefix + (s,)  # 添加新 token
                            n_pb, n_pnb, nodes = next_hyps[n_prefix]
                            n_pnb = n_pnb + pb * ps
                            nodes = cur_nodes.copy()
                            nodes.append(
                                dict(token=s, frame=t, prob=ps)
                            )  # 记录 token 信息
                            next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                            
                    else:  # 当前 token 与路径最后一个 token 不同
                        n_prefix = prefix + (s,)  # 添加新 token
                        n_pb, n_pnb, nodes = next_hyps[n_prefix]
                        if nodes:
                            # 如果已经有这个路径，更新最后一个 token 的概率
                            if ps > nodes[-1]["prob"]:
                                nodes[-1]["prob"] = ps
                                nodes[-1]["frame"] = t
                        else:
                            # 新路径，添加 token 信息
                            nodes = cur_nodes.copy()
                            nodes.append(
                                dict(token=s, frame=t, prob=ps)
                            )
                        n_pnb = n_pnb + pb * ps + pnb * ps
                        next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

            # ===== 2.2 第二次剪枝：保留概率最高的 path_beam_size 条路径 =====
            next_hyps = sorted(
                next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True
            )

            cur_hyps = next_hyps[:path_beam_size]

        # 最终的候选路径
        hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in cur_hyps]
        print("hyps", hyps)
        # ========== 步骤 3: 检测关键词 ==========
        hit_keyword = None  # 检测到的关键词
        hit_score = 1.0     # 关键词的置信度分数
        
        # 遍历所有候选路径，查找是否包含关键词
        for one_hyp in hyps:
            prefix_ids = one_hyp[0]      # token ID 序列
            # path_score = one_hyp[1]    # 路径总概率（未使用）
            prefix_nodes = one_hyp[2]    # token 详细信息列表
            
            assert len(prefix_ids) == len(prefix_nodes)
            
            # 检查每个关键词
            for word in keywords_token.keys():
                lab = keywords_token[word]["token_id"]  # 关键词的 token ID 序列
                
                # 检查关键词序列是否是当前路径的子序列
                offset = is_sublist(prefix_ids, lab)
                
                if offset != -1:  # 找到关键词
                    hit_keyword = word
                    
                    # 计算关键词的置信度：所有相关 token 概率的乘积
                    for idx in range(offset, offset + len(lab)):
                        hit_score *= prefix_nodes[idx]["prob"]
                    break
                    
            if hit_keyword is not None:
                # 对置信度取平方根（几何平均）
                hit_score = math.sqrt(hit_score)
                break
        
        print("hit_keyword", hit_keyword, "hit_score", hit_score)

    except Exception as e:
        print(f"⚠️  ERROR: Failed to perform CTC decoding: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("Please ensure:")
        print("  1. funasr is installed: pip install funasr")
        print(
            "  2. Model path exists: /root/volume/ctc/speech_charctc_kws_phone-xiaoyun"
        )
        print("  3. ctc.py is in the root directory")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Decoding completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
