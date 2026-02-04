# -*- coding: utf-8 -*-
"""
Encoder 后端对比：PT / ONNX / TFLite(float32, float16, 16x8, int8, int8_no_calib)
与 a2.py 相同流程：同一批 wav → 同一 frontend → 不同 encoder → 对比 logits 并做 CTC 解码。
- PT / ONNX：整段 (1, T, 400) 一次推理。
- TFLite：输入固定 (1, 10, 400)，按 10 帧分块推理后拼接（每块零 cache，与导出设定一致）。
- int8: 使用校准数据的全整数量化版本
- int8_no_calib: 不使用校准数据的权重量化版本（激活保持 float32）
"""

from __future__ import print_function

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from funasr import AutoModel
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from ctc import CTC, KwsCtcPrefixDecoder

# 仅需模型定义与导出结构，不执行 a2 的 __main__
from a2 import FsmnKWS, FSMNExport

# 路径（与 a2 / export_tflite 一致）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "../res/finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt")
TFLITE_MODELS_DIR = os.path.join(SCRIPT_DIR, "tflite_models")
ONNX_STATEFUL_PATH = os.path.join(TFLITE_MODELS_DIR, "fsmn_encoder_stateful.onnx")
TFLITE_STATEFUL_PATHS = {
    "float32": os.path.join(TFLITE_MODELS_DIR, "fsmn_encoder_stateful_float32.tflite"),
    "float16": os.path.join(TFLITE_MODELS_DIR, "fsmn_encoder_stateful_float16.tflite"),
    "16x8": os.path.join(TFLITE_MODELS_DIR, "fsmn_encoder_stateful_16x8.tflite"),
    "int8": os.path.join(TFLITE_MODELS_DIR, "fsmn_encoder_stateful_int8.tflite"),
    "int8_no_calib": os.path.join(TFLITE_MODELS_DIR, "fsmn_encoder_stateful_int8_no_calib.tflite"),
}

CHUNK_SIZE = 10
NUM_LAYERS = 4
PROJ_DIM = 128
LEFT_CTX = 9
RIGHT_CTX = 2
DELAY_FRAMES = 2  # 与 a2.py 一致：流式输出延迟 2 帧再与整段对齐、解码
# WavFrontend 配置：frame_shift=10ms, lfr_m=5, lfr_n=3
# LFR后帧率 = frame_shift * lfr_n = 10ms * 3 = 30ms/帧
FRAME_RATE_MS = 30.0  # LFR后帧率：30ms/帧（原始fbank 10ms/帧，lfr_n=3降采样）


def load_pt_encoder(device="cpu"):
    """PT encoder（FSMNExport + 零 cache），与 a2 一致。"""
    model_kws = FsmnKWS()
    model_kws.eval()
    load_pretrained_model(
        model=model_kws,
        path=CKPT_PATH,
        ignore_init_mismatch=True,  # 允许 Conv2d->Conv1d 的形状不匹配
        oss_bucket=None,
        scope_map=[],
        excludes=["ctc"],
    )
    # 自动转换 Conv2d 权重到 Conv1d（从 checkpoint 加载）
    from a2 import convert_conv2d_to_conv1d_weights
    convert_conv2d_to_conv1d_weights(model_kws, checkpoint_path=CKPT_PATH)
    model_kws.float().to(device)
    encoder_export = FSMNExport(model=model_kws.encoder)
    encoder_export.eval()
    return encoder_export


def run_pt_encoder(encoder_export, speech, device="cpu"):
    """speech: (1, T, 400). 返回 (1, T, 2599) numpy。"""
    B = speech.shape[0]
    # Conv1d 版本：cache 形状为 (B, PROJ_DIM, LEFT_CTX)，无最后一维
    zero_caches = [
        torch.zeros(B, PROJ_DIM, LEFT_CTX, device=device, dtype=speech.dtype)
        for _ in range(NUM_LAYERS)
    ]
    with torch.no_grad():
        out, _ = encoder_export(speech, *zero_caches, right_context=None)
    return out.cpu().numpy()


def run_pt_encoder_stateful_chunked(encoder_export, speech, device="cpu", chunk_size=CHUNK_SIZE, measure_latency=False):
    """与 a2.py 流式完全一致：不 pad，最后一块可不足 10 帧，末块传 right_context=None。返回 (1, T, 2599) numpy。"""
    B, T, D = speech.shape
    # Conv1d 版本：cache 形状为 (B, PROJ_DIM, LEFT_CTX)，无最后一维
    caches = [
        torch.zeros(B, PROJ_DIM, LEFT_CTX, device=device, dtype=speech.dtype)
        for _ in range(NUM_LAYERS)
    ]
    out_list = []
    latencies = []
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        chunk = speech[:, start:end, :]
        if end < T:
            rc = speech[:, end : end + RIGHT_CTX, :]
            if rc.shape[1] < RIGHT_CTX:
                rc = F.pad(rc, (0, 0, 0, RIGHT_CTX - rc.shape[1]))
        else:
            rc = None  # 与 a2 一致：末块不传 right_context
        if measure_latency:
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
        with torch.no_grad():
            out_chunk, caches = encoder_export(chunk, *caches, right_context=rc)
        if measure_latency:
            torch.cuda.synchronize() if device == "cuda" else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
        out_list.append(out_chunk.cpu().numpy())
        start = end
    out = np.concatenate(out_list, axis=1)[:, :T, :]
    if measure_latency:
        return out, latencies
    return out


def collect_stateful_repr_chunks(encoder_export, speech, device="cpu", chunk_size=CHUNK_SIZE):
    """
    与 run_pt_encoder_stateful_chunked 一致地流式跑一遍，但每步 yield (input_chunk, right_context, c0,c1,c2,c3)
    即「进入该块时的输入 + right_context + 当时的 in_caches」，用于 TFLite 量化校准（真实流式激活分布）。
    right_context 为 None 的末块会 pad 为零 (1,2,D) 以便保存。
    """
    B, T, D = speech.shape
    # Conv1d 版本：cache 形状为 (B, PROJ_DIM, LEFT_CTX)，无最后一维
    caches = [
        torch.zeros(B, PROJ_DIM, LEFT_CTX, device=device, dtype=speech.dtype)
        for _ in range(NUM_LAYERS)
    ]
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        chunk = speech[:, start:end, :]
        if chunk.shape[1] < chunk_size:
            chunk = F.pad(chunk, (0, 0, 0, chunk_size - chunk.shape[1]))
        if end < T:
            rc = speech[:, end : end + RIGHT_CTX, :]
            if rc.shape[1] < RIGHT_CTX:
                rc = F.pad(rc, (0, 0, 0, RIGHT_CTX - rc.shape[1]))
        else:
            rc = torch.zeros(B, RIGHT_CTX, D, device=speech.device, dtype=speech.dtype)
        # 记录进入该块时的 (chunk, rc, caches)
        chunk_np = chunk.cpu().numpy()
        rc_np = rc.cpu().numpy()
        c0, c1, c2, c3 = [c.cpu().numpy() for c in caches]
        yield chunk_np, rc_np, c0, c1, c2, c3
        with torch.no_grad():
            _, caches = encoder_export(chunk, *caches, right_context=rc if end < T else None)
        start = end


def apply_delay_frames(logits_np, delay=DELAY_FRAMES):
    """流式输出延迟 delay 帧再与整段对齐，与 a2 的 ref_stream 一致。logits_np: (1, T, D)。"""
    if delay <= 0 or logits_np.shape[1] <= delay:
        return logits_np
    # delayed[t] = raw[t-delay] (t>=delay), 前 delay 帧用首帧复制
    B, T, D = logits_np.shape
    out = np.empty_like(logits_np)
    out[:, :delay, :] = logits_np[:, :1, :]
    out[:, delay:, :] = logits_np[:, : T - delay, :]
    return out


def load_onnx_encoder_stateful():
    """ONNX encoder（带 cache 输入/输出），用于分块时传递 cache。"""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("pip install onnxruntime")
    return ort.InferenceSession(ONNX_STATEFUL_PATH, providers=["CPUExecutionProvider"])


def run_onnx_encoder_stateful_chunked(session, speech, chunk_size=CHUNK_SIZE, measure_latency=False):
    """speech: (1, T, 400)。分块推理，块间传递 cache + right_context，与 PT 流式一致。"""
    B, T, D = speech.shape
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        speech = np.concatenate([speech, np.zeros((B, pad, D), dtype=speech.dtype)], axis=1)
    # 检查 ONNX 模型的 cache 输入形状（可能是旧的 Conv2d 版本 (B,D,9,1) 或新的 Conv1d 版本 (B,D,9)）
    cache_input_shape = None
    for inp in session.get_inputs():
        if "in_cache_0" in inp.name:
            cache_input_shape = inp.shape
            break
    # ONNX shape 可能是动态的（如 ['batch', 128, 9, 1]），需要检查最后一个元素是否为 1 或检查长度
    if cache_input_shape is None:
        # 默认使用旧格式（兼容性）
        use_conv2d_format = True
    elif len(cache_input_shape) == 4:
        # 旧 Conv2d 版本：cache 形状为 (B, PROJ_DIM, LEFT_CTX, 1)
        cache_shape = (B, PROJ_DIM, LEFT_CTX, 1)
        use_conv2d_format = True
    else:
        # 新 Conv1d 版本：cache 形状为 (B, PROJ_DIM, LEFT_CTX)
        cache_shape = (B, PROJ_DIM, LEFT_CTX)
        use_conv2d_format = False
    caches = [np.zeros(cache_shape, dtype=np.float32) for _ in range(NUM_LAYERS)]
    out_list = []
    latencies = []
    for start in range(0, speech.shape[1], chunk_size):
        end = start + chunk_size
        chunk = speech[:, start:end, :].astype(np.float32)
        if end < speech.shape[1]:
            rc = speech[:, end : end + RIGHT_CTX, :].astype(np.float32)
            if rc.shape[1] < RIGHT_CTX:
                rc = np.concatenate([rc, np.zeros((B, RIGHT_CTX - rc.shape[1], D), dtype=np.float32)], axis=1)
        else:
            rc = np.zeros((B, RIGHT_CTX, D), dtype=np.float32)
        # 根据模型格式转换 right_context
        # 检查 right_context 的期望形状
        rc_input_shape = None
        for inp in session.get_inputs():
            if "right_context" in inp.name:
                rc_input_shape = inp.shape
                break
        if rc_input_shape and len(rc_input_shape) == 4:
            # 旧格式：需要 permute 到 (B, D, T_rc, 1)
            rc_for_model = rc.transpose(0, 2, 1)[:, :, :, np.newaxis]  # (B, 2, D) -> (B, D, 2, 1)
        else:
            # 新格式：保持 (B, T_rc, D)
            rc_for_model = rc
        feeds = {
            "input": chunk,
            "in_cache_0": caches[0],
            "in_cache_1": caches[1],
            "in_cache_2": caches[2],
            "in_cache_3": caches[3],
            "right_context": rc_for_model,
        }
        if measure_latency:
            t0 = time.perf_counter()
        outs = session.run(
            ["output", "out_cache_0", "out_cache_1", "out_cache_2", "out_cache_3"],
            feeds,
        )
        if measure_latency:
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
        out_list.append(outs[0])
        cache_out = [outs[1], outs[2], outs[3], outs[4]]
        # 如果模型输出是旧格式 (B, D, 9, 1)，需要 squeeze 最后一维
        if use_conv2d_format and cache_out[0].ndim == 4:
            caches = [c.squeeze(-1) for c in cache_out]  # (B, D, 9, 1) -> (B, D, 9)
        else:
            caches = cache_out
    out = np.concatenate(out_list, axis=1)
    if measure_latency:
        return out[:, :T, :], latencies
    return out[:, :T, :]


def run_tflite_encoder_stateful_chunked(interp, speech, chunk_size=CHUNK_SIZE, measure_latency=False):
    """speech: (1, T, 400)。分块、块间传递 cache + right_context，与 ONNX/PT 流式一致。TFLite 固定 10 帧/块，不足则 pad。"""
    B, T, D = speech.shape
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        speech = np.concatenate([speech, np.zeros((B, pad, D), dtype=speech.dtype)], axis=1)
    inp_details = {d.get("name", ""): d for d in interp.get_input_details()}
    out_details_list = interp.get_output_details()
    # 按名称中编号排序：StatefulPartitionedCall:0=logits, :1..:4=out_cache_0..3
    def out_key(d):
        name = d.get("name", "")
        if ":" in name:
            return int(name.split(":")[-1])
        return 0
    out_details_list = sorted(out_details_list, key=out_key)
    # 检查 TFLite 模型的 cache 输入形状和数据类型（可能是旧的 Conv2d 版本 (B,D,9,1) 或新的 Conv1d 版本 (B,D,9)）
    cache_input_shape = None
    cache_dtype = None
    input_dtype = None
    right_context_dtype = None
    input_quant_params = None
    right_context_quant_params = None
    output_quant_params = None
    
    # 遍历所有输入以获取类型信息和量化参数
    for k, d in inp_details.items():
        if "in_cache_0" in k:
            cache_input_shape = d["shape"]
            cache_dtype = d["dtype"]
        elif "input" in k and "cache" not in k and "right" not in k:
            input_dtype = d["dtype"]
            input_quant_params = d.get("quantization_parameters", None)
        elif "right_context" in k:
            right_context_dtype = d["dtype"]
            right_context_quant_params = d.get("quantization_parameters", None)
    
    # 获取输出量化参数
    if len(out_details_list) > 0:
        output_quant_params = out_details_list[0].get("quantization_parameters", None)
    
    # 调试信息：打印量化参数（仅对量化模型）
    if input_quant_params is not None and len(input_quant_params) > 0:
        scales = input_quant_params.get('scales', None)
        zero_points = input_quant_params.get('zero_points', None)
        if scales is not None and len(scales) > 0:
            print("  [量化调试] input scale=%.6f, zero_point=%.1f" % (scales[0] if isinstance(scales, (list, tuple, np.ndarray)) else scales, 
                                                                      zero_points[0] if isinstance(zero_points, (list, tuple, np.ndarray)) else zero_points))
    if output_quant_params is not None and len(output_quant_params) > 0:
        scales = output_quant_params.get('scales', None)
        zero_points = output_quant_params.get('zero_points', None)
        if scales is not None and len(scales) > 0:
            print("  [量化调试] output scale=%.6f, zero_point=%.1f" % (scales[0] if isinstance(scales, (list, tuple, np.ndarray)) else scales,
                                                                       zero_points[0] if isinstance(zero_points, (list, tuple, np.ndarray)) else zero_points))
    
    # 如果没有找到 cache，使用 input 的 dtype 作为参考
    if cache_dtype is None:
        cache_dtype = input_dtype if input_dtype is not None else np.float32
    if right_context_dtype is None:
        right_context_dtype = input_dtype if input_dtype is not None else np.float32
    
    # 量化辅助函数
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
        
        # 量化：q = round(x / scale + zero_point)
        quantized = np.round(data / scale + zero_point).astype(dtype)
        
        # 对于 int8，限制范围 [-128, 127]
        if dtype == np.int8:
            quantized = np.clip(quantized, -128, 127)
        # 对于 int16，限制范围 [-32768, 32767]
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
        
        # 反量化：x = (q - zero_point) * scale
        return (data.astype(np.float32) - zero_point) * scale
    
    if cache_input_shape is None or len(cache_input_shape) == 4:
        # 旧 Conv2d 版本：cache 形状为 (B, PROJ_DIM, LEFT_CTX, 1)
        cache_shape = (B, PROJ_DIM, LEFT_CTX, 1)
        use_conv2d_format = True
    else:
        # 新 Conv1d 版本：cache 形状为 (B, PROJ_DIM, LEFT_CTX)
        cache_shape = (B, PROJ_DIM, LEFT_CTX)
        use_conv2d_format = False
    
    # 根据模型输入类型创建 cache（16x8 模型需要 int16，int8 模型需要 int8，float 模型需要 float32）
    caches = [np.zeros(cache_shape, dtype=cache_dtype) for _ in range(NUM_LAYERS)]
    out_list = []
    latencies = []
    for start in range(0, speech.shape[1], chunk_size):
        end = start + chunk_size
        # 准备 chunk（float32），然后根据模型类型量化
        chunk_float = speech[:, start:end, :].astype(np.float32)
        if input_quant_params is not None and len(input_quant_params) > 0:
            scales = input_quant_params.get('scales', None)
            if scales is not None and len(scales) > 0:
                # 量化模型：需要量化输入
                chunk = quantize(chunk_float, input_quant_params, input_dtype if input_dtype is not None else np.float32)
            else:
                # 非量化模型：直接转换类型
                chunk = chunk_float.astype(input_dtype if input_dtype is not None else np.float32)
        else:
            # 非量化模型：直接转换类型
            chunk = chunk_float.astype(input_dtype if input_dtype is not None else np.float32)
        
        if end < speech.shape[1]:
            rc_float = speech[:, end : end + RIGHT_CTX, :].astype(np.float32)
            if rc_float.shape[1] < RIGHT_CTX:
                rc_float = np.concatenate([rc_float, np.zeros((B, RIGHT_CTX - rc_float.shape[1], D), dtype=np.float32)], axis=1)
            if right_context_quant_params is not None and len(right_context_quant_params) > 0:
                scales = right_context_quant_params.get('scales', None)
                if scales is not None and len(scales) > 0:
                    # 量化模型：需要量化 right_context
                    rc = quantize(rc_float, right_context_quant_params, right_context_dtype)
                else:
                    # 非量化模型：直接转换类型
                    rc = rc_float.astype(right_context_dtype)
            else:
                # 非量化模型：直接转换类型
                rc = rc_float.astype(right_context_dtype)
        else:
            rc_float = np.zeros((B, RIGHT_CTX, D), dtype=np.float32)
            if right_context_quant_params is not None and len(right_context_quant_params) > 0:
                scales = right_context_quant_params.get('scales', None)
                if scales is not None and len(scales) > 0:
                    # 量化模型：需要量化 right_context
                    rc = quantize(rc_float, right_context_quant_params, right_context_dtype)
                else:
                    # 非量化模型：直接转换类型
                    rc = rc_float.astype(right_context_dtype)
            else:
                # 非量化模型：直接转换类型
                rc = rc_float.astype(right_context_dtype)
        for k, d in inp_details.items():
            if "input" in k and "cache" not in k and "right" not in k:
                interp.set_tensor(d["index"], chunk)
            elif "in_cache_0" in k:
                interp.set_tensor(d["index"], caches[0])
            elif "in_cache_1" in k:
                interp.set_tensor(d["index"], caches[1])
            elif "in_cache_2" in k:
                interp.set_tensor(d["index"], caches[2])
            elif "in_cache_3" in k:
                interp.set_tensor(d["index"], caches[3])
            elif "right_context" in k:
                if use_conv2d_format:
                    # 旧格式：需要 permute 到 (B, D, T_rc, 1)
                    rc_conv2d = rc.transpose(0, 2, 1)[:, :, :, np.newaxis]  # (B, 2, D) -> (B, D, 2, 1)
                    interp.set_tensor(d["index"], rc_conv2d)
                else:
                    # 新格式：保持 (B, T_rc, D)
                    interp.set_tensor(d["index"], rc)
        if measure_latency:
            t0 = time.perf_counter()
        interp.invoke()
        if measure_latency:
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
        # out_details_list 已按 :0,:1,:2,:3,:4 排序，0=logits, 1..4=out_cache_0..3
        logits = interp.get_tensor(out_details_list[0]["index"]).copy()
        # 如果是量化模型，需要反量化输出
        if output_quant_params is not None and len(output_quant_params) > 0:
            scales = output_quant_params.get('scales', None)
            if scales is not None and len(scales) > 0:
                logits = dequantize(logits, output_quant_params)
        if logits.ndim == 3 and logits.shape[-1] == 2599:
            out_list.append(logits)
        if len(out_details_list) >= 5:
            cache_out = [
                interp.get_tensor(out_details_list[1]["index"]).copy(),
                interp.get_tensor(out_details_list[2]["index"]).copy(),
                interp.get_tensor(out_details_list[3]["index"]).copy(),
                interp.get_tensor(out_details_list[4]["index"]).copy(),
            ]
            # 如果模型输出是旧格式 (B, D, 9, 1)，需要 squeeze 最后一维
            if use_conv2d_format and cache_out[0].ndim == 4:
                caches = [c.squeeze(-1) for c in cache_out]  # (B, D, 9, 1) -> (B, D, 9)
            else:
                caches = cache_out
    out = np.concatenate(out_list, axis=1)
    if measure_latency:
        return out[:, :T, :], latencies
    return out[:, :T, :]


def compare_logits(name_a, logits_a, name_b, logits_b):
    """logits 形状需一致，返回 max/mean abs diff。"""
    logits_a = np.asarray(logits_a, dtype=np.float64)
    logits_b = np.asarray(logits_b, dtype=np.float64)
    min_len = min(logits_a.shape[1], logits_b.shape[1])
    la, lb = logits_a[:, :min_len, :], logits_b[:, :min_len, :]
    diff = np.abs(la - lb)
    return np.max(diff), np.mean(diff)


def _wav_list_from_merge_trans(trans_path):
    """
    从 merge_trans.txt 格式（每行 utt_id\\t 或 utt_id 空格 文本）解析 utt_id，
    wav 路径为 trans 同目录下的 wav/{utt_id}.wav（与 a2/example_kws 一致）。
    """
    trans_abs = os.path.abspath(trans_path)
    if not os.path.isfile(trans_abs):
        return []
    trans_dir = os.path.dirname(trans_abs)
    wav_dir = os.path.join(trans_dir, "wav")
    wavs = []
    missing = []
    with open(trans_abs, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 第一列为 utt_id（tab 或空格分隔）
            if "\t" in line:
                utt_id = line.split("\t", 1)[0].strip()
            else:
                utt_id = line.split(None, 1)[0].strip()
            if not utt_id:
                continue
            wav_path = os.path.join(wav_dir, utt_id + ".wav")
            if os.path.isfile(wav_path):
                wavs.append(wav_path)
            else:
                missing.append(utt_id)
    if missing:
        print("  [merge_trans] 未找到 wav 的 utt_id 数量: %d（如 %s）" % (len(missing), missing[:3]))
    return wavs


def main():
    parser = argparse.ArgumentParser(description="Encoder 后端对比 / 导出量化用代表性特征")
    parser.add_argument("--save-repr-npy", type=str, default=None, help="仅导出 fbank 特征到 .npy（形状 (T_total, 400)），供 export_tflite.py --repr-npy 使用后退出")
    parser.add_argument("--save-repr-stateful", type=str, default=None, help="用 PT 流式跑一遍，保存每步 (input, right_context, c0..c3) 到 .npz，供 export_tflite.py --repr-stateful 做真实流式校准")
    parser.add_argument("--repr-trans", type=str, default=None, help="与 --save-repr-npy/--save-repr-stateful 同用：从 merge_trans.txt 读 utt_id，到同目录 wav/{utt_id}.wav 取音频（与 a2 一致流程）")
    args = parser.parse_args()

    device = "cpu"
    model = AutoModel(
        model="/root/volume/ctc/speech_charctc_kws_phone-xiaoyun",
        keywords="小云小云",
        output_dir=os.path.join(SCRIPT_DIR, "outputs", "compare"),
        device=device,
        disable_update=True,
    )
    kwargs = model.kwargs
    frontend = kwargs["frontend"]
    tokenizer = kwargs["tokenizer"]
    keywords = kwargs["keywords"]

    # 与 a2.py 一致：example_kws 在项目上层目录
    test_wavs = [
        os.path.join(SCRIPT_DIR, "..", "res", "example_kws", "wav", "20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_21.wav"),
        os.path.join(SCRIPT_DIR, "..", "res", "test_xiaoyun.wav"),
    ]
    test_wavs = [w for w in test_wavs if os.path.isfile(w)]
    if not test_wavs:
        print("未找到任何测试 wav，请将 test_wavs 指向有效 wav 路径")
        sys.exit(1)

    if args.save_repr_stateful is not None:
        # 用 PT encoder 流式跑一遍，保存每步 (input, right_context, c0..c3) 到 .npz，校准用真实流式 cache
        if args.repr_trans is not None:
            wav_list = _wav_list_from_merge_trans(args.repr_trans)
            if not wav_list:
                print("--repr-trans 未解析到任何有效 wav，请检查路径与 wav/ 目录")
                sys.exit(1)
            print("从 %s 解析到 %d 条 wav，用 PT 流式收集 stateful 代表性数据" % (args.repr_trans, len(wav_list)))
        else:
            wav_list = test_wavs
        print("加载 PT encoder ...")
        pt_encoder = load_pt_encoder(device)
        inputs, rcs, c0s, c1s, c2s, c3s = [], [], [], [], [], []
        for test_wav in wav_list:
            audio_list = load_audio_text_image_video(
                test_wav,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type="sound",
                tokenizer=tokenizer,
            )
            speech, _ = extract_fbank(
                audio_list,
                data_type="fbank",
                frontend=frontend,
            )
            speech = speech.to(device=device, dtype=torch.float32)
            for chunk_np, rc_np, c0_np, c1_np, c2_np, c3_np in collect_stateful_repr_chunks(pt_encoder, speech, device):
                inputs.append(chunk_np)
                rcs.append(rc_np)
                c0s.append(c0_np)
                c1s.append(c1_np)
                c2s.append(c2_np)
                c3s.append(c3_np)
        # 确保输出目录存在
        output_dir = os.path.dirname(args.save_repr_stateful)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print("已创建目录: %s" % output_dir)
        np.savez(
            args.save_repr_stateful,
            input=np.stack(inputs, axis=0),
            right_context=np.stack(rcs, axis=0),
            c0=np.stack(c0s, axis=0),
            c1=np.stack(c1s, axis=0),
            c2=np.stack(c2s, axis=0),
            c3=np.stack(c3s, axis=0),
        )
        print("已保存流式代表性数据到 %s，共 %d 块（真实 cache）" % (args.save_repr_stateful, len(inputs)))
        print("用法: python export_tflite.py --repr-stateful %s" % args.save_repr_stateful)
        sys.exit(0)

    if args.save_repr_npy is not None:
        # 仅导出代表性 fbank 到 .npy，供 export_tflite.py --repr-npy 使用（与 a2 相同 load_audio + extract_fbank）
        if args.repr_trans is not None:
            wav_list = _wav_list_from_merge_trans(args.repr_trans)
            if not wav_list:
                print("--repr-trans 未解析到任何有效 wav，请检查路径与 wav/ 目录")
                sys.exit(1)
            print("从 %s 解析到 %d 条 wav，使用 a2 同流程提取 fbank" % (args.repr_trans, len(wav_list)))
        else:
            wav_list = test_wavs
        parts = []
        for test_wav in wav_list:
            audio_list = load_audio_text_image_video(
                test_wav,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type="sound",
                tokenizer=tokenizer,
            )
            speech, _ = extract_fbank(
                audio_list,
                data_type="fbank",
                frontend=frontend,
            )
            parts.append(speech.cpu().numpy().squeeze(0))
        repr_fbank = np.concatenate(parts, axis=0).astype(np.float32)
        # 确保输出目录存在
        output_dir = os.path.dirname(args.save_repr_npy)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print("已创建目录: %s" % output_dir)
        np.save(args.save_repr_npy, repr_fbank)
        print("已保存代表性 fbank 到 %s，形状 %s" % (args.save_repr_npy, repr_fbank.shape))
        print("用法: python export_tflite.py --repr-npy %s" % args.save_repr_npy)
        sys.exit(0)

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

    print("加载 PT encoder ...")
    pt_encoder = load_pt_encoder(device)

    use_onnx_stateful = os.path.isfile(ONNX_STATEFUL_PATH)
    onnx_stateful_session = None
    if use_onnx_stateful:
        print("加载 ONNX encoder (带 cache) ...")
        onnx_stateful_session = load_onnx_encoder_stateful()
    else:
        print("未找到 ONNX 带 cache 文件，跳过:", ONNX_STATEFUL_PATH, "(先运行 export_tflite.py 生成)")

    import tensorflow as tf
    tflite_stateful_interps = {}
    for name, path in TFLITE_STATEFUL_PATHS.items():
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print("加载 TFLite encoder (带 cache) %s ..." % name)
            interp = tf.lite.Interpreter(model_path=path)
            interp.allocate_tensors()
            tflite_stateful_interps[name] = interp
    if not tflite_stateful_interps:
        print("未找到 TFLite 带 cache 文件，跳过 (先运行 export_tflite.py 生成)")

    for test_wav in test_wavs:
        print("\n" + "=" * 60)
        print("测试音频:", test_wav)
        print("=" * 60)

        audio_list = load_audio_text_image_video(
            test_wav,
            fs=frontend.fs,
            audio_fs=kwargs.get("fs", 16000),
            data_type="sound",
            tokenizer=tokenizer,
        )
        speech, speech_lengths = extract_fbank(
            audio_list,
            data_type="fbank",
            frontend=frontend,
        )
        speech = speech.to(device=device, dtype=torch.float32)
        B, T, D_in = speech.shape[0], speech.shape[1], speech.shape[2]
        speech_np = speech.cpu().numpy()
        T_valid = speech_lengths.item() if speech_lengths.dim() > 0 else T

        # ----- PT 整段（用于解码参考） -----
        ref_logits_full = run_pt_encoder(pt_encoder, speech, device)
        print("speech.shape:", speech.shape)

        # ----- PT 带 cache 分块（与 a2 流式一致：不 pad、末块 rc=None） -----
        ref_logits_pt_stateful, pt_latencies = run_pt_encoder_stateful_chunked(pt_encoder, speech, device, measure_latency=True)
        max_d_pt, mean_d_pt = compare_logits("PT-整段", ref_logits_full, "PT-流式(raw)", ref_logits_pt_stateful)
        print("PT(整段) vs PT(流式raw, %d帧/块): max_diff = %.4e, mean_diff = %.4e" % (CHUNK_SIZE, max_d_pt, mean_d_pt))
        if pt_latencies:
            audio_duration_ms = T_valid * FRAME_RATE_MS
            total_latency_ms = sum(pt_latencies)
            avg_chunk_latency_ms = np.mean(pt_latencies)
            rtf = total_latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0
            print("  PT 延迟: 单块平均=%.2fms, 总=%.2fms, 音频时长=%.2fms, RTF=%.3f" % (
                avg_chunk_latency_ms, total_latency_ms, audio_duration_ms, rtf))

        # ----- ONNX 带 cache 分块，仅输出 decode（延迟 2 帧） -----
        if onnx_stateful_session is not None:
            onnx_stateful_logits, onnx_latencies = run_onnx_encoder_stateful_chunked(onnx_stateful_session, speech_np, measure_latency=True)
            onnx_st_delayed = apply_delay_frames(onnx_stateful_logits, DELAY_FRAMES)
            x_onnx_st = torch.from_numpy(onnx_st_delayed[0, :T_valid, :].astype(np.float32))
            print("decode(ONNX-带cache, 延迟%d帧):" % DELAY_FRAMES, kws_decoder.decode(x_onnx_st))
            if onnx_latencies:
                audio_duration_ms = T_valid * FRAME_RATE_MS
                total_latency_ms = sum(onnx_latencies)
                avg_chunk_latency_ms = np.mean(onnx_latencies)
                rtf = total_latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0
                print("  ONNX 延迟: 单块平均=%.2fms, 总=%.2fms, 音频时长=%.2fms, RTF=%.3f" % (
                    avg_chunk_latency_ms, total_latency_ms, audio_duration_ms, rtf))

        # ----- TFLite 带 cache 各版本，仅输出 decode（延迟 2 帧） -----
        for name, interp in tflite_stateful_interps.items():
            tflite_stateful_logits, tflite_latencies = run_tflite_encoder_stateful_chunked(interp, speech_np, measure_latency=True)
            tflite_st_delayed = apply_delay_frames(tflite_stateful_logits, DELAY_FRAMES)
            x_tf_st = torch.from_numpy(tflite_st_delayed[0, :T_valid, :].astype(np.float32))
            print("decode(TFLite-带cache-%s, 延迟%d帧):" % (name, DELAY_FRAMES), kws_decoder.decode(x_tf_st))
            if tflite_latencies:
                audio_duration_ms = T_valid * FRAME_RATE_MS
                total_latency_ms = sum(tflite_latencies)
                avg_chunk_latency_ms = np.mean(tflite_latencies)
                rtf = total_latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0
                print("  TFLite-%s 延迟: 单块平均=%.2fms, 总=%.2fms, 音频时长=%.2fms, RTF=%.3f" % (
                    name, avg_chunk_latency_ms, total_latency_ms, audio_duration_ms, rtf))

        # ----- PT 整段 / 流式 解码（流式用延迟后输出，与 a2 一致） -----
        x_pt_full = torch.from_numpy(ref_logits_full[0, :T_valid, :])
        pt_stateful_delayed = apply_delay_frames(ref_logits_pt_stateful, DELAY_FRAMES)
        x_pt_stream = torch.from_numpy(pt_stateful_delayed[0, :T_valid, :].astype(np.float32))
        print("decode(PT-整段):", kws_decoder.decode(x_pt_full))
        print("decode(PT-流式, 延迟%d帧):" % DELAY_FRAMES, kws_decoder.decode(x_pt_stream))

    print("\n对比测试完成。")


if __name__ == "__main__":
    main()
