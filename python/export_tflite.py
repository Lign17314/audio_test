# -*- coding: utf-8 -*-
"""
TFLite 导出脚本：仅 encoder 部分，导出 float32 / float16 / 16x8 / int8 四个版本，
并打印每个 tflite 模型包含的算子。

依赖: torch, onnx, onnx2tf, tensorflow
导出时输入形状固定为 (1, 10, 400)，与 encoder 输入 (B, T, D) 一致。

量化校准（改善 16x8/int8 精度）：
  - 流式模型建议用「真实流式 cache」校准，避免静音帧主导：
    1. python compare_encoder_backends.py --save-repr-stateful tflite_models/repr_stateful.npz --repr-trans example_kws/merge_trans.txt
    2. python export_tflite.py --repr-stateful tflite_models/repr_stateful.npz
  - 仅用 fbank 时可用 --repr-energy-percentile 50~70 丢弃最静音一半块：
    python export_tflite.py --repr-npy tflite_models/repr_fbank.npy [--repr-energy-percentile 50]

TFLite Micro 兼容性：
  - TRANSPOSE/STRIDED_SLICE/DEQUANTIZE 算子已在 tflite-micro 中支持（2021-2025年实现）
  - 建议使用 MicroMutableOpResolver 仅注册模型需要的算子以节省内存
  - 详见: https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/kernels
"""

from __future__ import print_function

import argparse
import os
import sys

# 可选：抑制 TF 部分日志
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# 禁用 XNNPACK delegate（为了 TFLite Micro 兼容性）
# 注意：必须在导入 tensorflow 之前设置
# 如果需要在导出时启用 XNNPACK，可以通过命令行参数控制
if 'TF_DISABLE_XNNPACK' not in os.environ:
    # 默认禁用 XNNPACK，以确保 TFLite Micro 兼容性
    # 如果需要启用，设置环境变量: export TF_ENABLE_XNNPACK=1
    if os.environ.get('TF_ENABLE_XNNPACK', '0') != '1':
        os.environ['TF_DISABLE_XNNPACK'] = '1'

import numpy as np
import torch

# 从 a2 导入模型与加载逻辑（不执行 a2 的 __main__）
from a2 import FsmnKWS, FSMNExport, load_pretrained_model


# ---------------------------------------------------------------------------
# 1) Encoder 仅、单输入单输出包装（内部用零 cache，整段推理）
# ---------------------------------------------------------------------------
class FSMNExportWrapper(torch.nn.Module):
    """仅 encoder：单输入 (B, T, D)、单输出 (B, T, output_dim)，内部用零 cache。"""

    def __init__(self, encoder_export: FSMNExport):
        super().__init__()
        self.encoder = encoder_export
        self.num_layers = 4
        self.proj_dim = 128
        self.left_ctx = 9   # (lorder-1)*lstride
        self.right_ctx = 2  # rorder*rstride

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, T, D)
        B = input.shape[0]
        device = input.device
        dtype = input.dtype
        # Conv1d 版本：cache 形状为 (B, proj_dim, left_ctx)，无最后一维
        zero_caches = [
            torch.zeros(B, self.proj_dim, self.left_ctx, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        out, _ = self.encoder(input, *zero_caches, right_context=None)
        return out


# ---------------------------------------------------------------------------
# 1b) 带 cache 的包装：分块时由调用方传入/接收每层 cache，与 PT 流式一致
# ---------------------------------------------------------------------------
class FSMNExportWrapperStateful(torch.nn.Module):
    """Encoder：input + 4×in_cache + right_context，输出 logits + 4×out_cache。right_context 为下一块前 2 帧 (B,2,D)，无则传零。"""

    def __init__(self, encoder_export: FSMNExport):
        super().__init__()
        self.encoder = encoder_export
        self.num_layers = 4
        self.proj_dim = 128
        self.left_ctx = 9
        self.right_ctx = 2

    def forward(
        self,
        input: torch.Tensor,
        in_cache_0: torch.Tensor,
        in_cache_1: torch.Tensor,
        in_cache_2: torch.Tensor,
        in_cache_3: torch.Tensor,
        right_context: torch.Tensor,
    ):
        # input: (B, T, D), in_cache_*: (B, proj_dim, left_ctx) - Conv1d版本，无最后一维, right_context: (B, 2, D)，无下一块时传零
        out, out_caches = self.encoder(
            input,
            in_cache_0, in_cache_1, in_cache_2, in_cache_3,
            right_context=right_context,
        )
        return out, out_caches[0], out_caches[1], out_caches[2], out_caches[3]


# ---------------------------------------------------------------------------
# 2) 打印 TFLite 模型中的算子（名称 + 数量）
# ---------------------------------------------------------------------------
def _build_builtin_op_names():
    from tensorflow.lite.python import schema_py_generated as schema_fb
    return {
        v: k for k, v in vars(schema_fb.BuiltinOperator).items()
        if isinstance(v, int)
    }


# TFLite Micro 算子支持状态（2025年更新）
# 参考: https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/kernels
# 
# ✅ 已支持的算子（已验证实现）：
# - TRANSPOSE: 已支持（2025年实现，支持 float32/int8/int16）
#   - 来源：FSMNBlock 中的 permute 操作（(B,T,D) <-> (B,D,T)）
#   - 文件：tensorflow/lite/micro/kernels/transpose.cc
# - STRIDED_SLICE: 已支持（2023年实现，支持 float32/int8/int16/int32/bool）
#   - 来源：FSMNBlock 中的切片操作（cache 和 y_right）
#   - 文件：tensorflow/lite/micro/kernels/strided_slice.cc
# - DEQUANTIZE: 已支持（2021年实现，支持 int8/int16/uint8 -> float32）
#   - 已通过全整数量化移除（int8/16x8 模型的输入输出也是整数），但算子本身已支持
#   - 文件：tensorflow/lite/micro/kernels/dequantize.cc
#
# 注意：虽然这些算子已实现，但某些平台/版本可能未包含在 AllOpsResolver 中，
# 建议使用 MicroMutableOpResolver 仅注册模型需要的算子以节省内存。
TFLITE_MICRO_TYPICALLY_UNSUPPORTED = frozenset({
    # 这些算子现在都已支持，但保留此集合以便未来添加其他不支持的算子
    # "TRANSPOSE",      # ✅ 已支持（2025）
    # "STRIDED_SLICE",  # ✅ 已支持（2023）
    # "DEQUANTIZE",     # ✅ 已支持（2021）
})


def print_tflite_ops(model_path: str, title: str = None) -> None:
    """
    解析 tflite 模型并打印其中包含的算子类型及数量。
    """
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as schema_fb
    from tensorflow.lite.python import schema_util

    if title is None:
        title = os.path.basename(model_path)

    with open(model_path, "rb") as f:
        buf = f.read()

    model = schema_fb.Model.GetRootAs(buf, 0)
    op_names = _build_builtin_op_names()

    op_counts = {}
    for sg_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(sg_idx)
        for op_idx in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(op_idx)
            code_idx = op.OpcodeIndex()
            if model.OperatorCodesLength() <= code_idx:
                continue
            opcode = model.OperatorCodes(code_idx)
            code = schema_util.get_builtin_code_from_operator_code(opcode)
            name = op_names.get(code, opcode.CustomCode() if opcode.CustomCode() else "CUSTOM_%d" % code)
            op_counts[name] = op_counts.get(name, 0) + 1

    print("\n" + "=" * 60)
    print("TFLite 算子列表: %s" % title)
    print("=" * 60)
    if not op_counts:
        print("  (无算子)")
    else:
        for name in sorted(op_counts.keys()):
            print("  %s: %d" % (name, op_counts[name]))
    print("  合计: %d 个算子" % sum(op_counts.values()))
    # TFLite Micro 兼容性：标出可能不被 micro 支持的算子
    unsupported_in_model = [n for n in op_counts if n in TFLITE_MICRO_TYPICALLY_UNSUPPORTED]
    if unsupported_in_model:
        print("  [TFLite Micro] ⚠️ 以下算子可能不被 tflite-micro 支持，需确认或自行 port:")
        for n in unsupported_in_model:
            print("    - %s (%d 个)" % (n, op_counts[n]))
    else:
        # 检查常见算子是否都在模型中（用于信息提示）
        common_ops = ["TRANSPOSE", "STRIDED_SLICE", "DEQUANTIZE"]
        present_ops = [op for op in common_ops if op in op_counts]
        if present_ops:
            print("  [TFLite Micro] ✅ 模型中的关键算子（TRANSPOSE/STRIDED_SLICE/DEQUANTIZE）已在 tflite-micro 中支持")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 2b) 从真实特征 npy 构建量化用代表性数据（改善 16x8/int8 精度）
# ---------------------------------------------------------------------------
def load_representative_chunks(repr_npy_path, batch, time_steps, feat_dim, num_samples=200, energy_percentile=0):
    """
    从 .npy 加载 fbank 特征，切分为 (1, 10, 400) 的块，用于 representative_dataset。
    npy 形状可为 (T_total, 400) 或 (N, T, 400)，内部会展平为 (T_total, 400)。
    energy_percentile: 只保留能量 >= 该百分位的块（0=不筛，50=丢弃最静音一半），避免静音帧主导校准。
    返回 (default_chunks, stateful_chunks)。
    """
    data = np.load(repr_npy_path).astype(np.float32)
    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
    T_total, D = data.shape
    if D != feat_dim:
        raise ValueError("repr npy 最后一维应为 %d (feat_dim)，当前为 %d" % (feat_dim, D))
    if T_total < time_steps + 2:
        raise ValueError("repr npy 帧数 %d 不足 (需要 >= %d)" % (T_total, time_steps + 2))

    # Conv1d 版本：cache 形状为 (batch, 128, 9)，无最后一维
    cache_shape_np = (batch, 128, 9)
    step = max(1, time_steps // 2)
    # 先收集所有候选块并计算能量（mean abs）
    candidates = []
    for start in range(0, T_total - time_steps - 2, step):
        input_ = np.asarray(data[start : start + time_steps], dtype=np.float32).reshape(batch, time_steps, feat_dim)
        right_context = np.asarray(data[start + time_steps : start + time_steps + 2], dtype=np.float32).reshape(batch, 2, feat_dim)
        energy = float(np.mean(np.abs(input_)))
        candidates.append((energy, input_, right_context))
    if energy_percentile > 0 and candidates:
        thresh = np.percentile([c[0] for c in candidates], energy_percentile)
        candidates = [c for c in candidates if c[0] >= thresh]
        if not candidates:
            raise ValueError("energy_percentile=%d 过滤后无样本，请降低或设为 0" % energy_percentile)
    # 取前 num_samples 个
    default_chunks = []
    stateful_chunks = []
    c0_z = np.zeros(cache_shape_np, dtype=np.float32)
    c1_z = np.zeros(cache_shape_np, dtype=np.float32)
    c2_z = np.zeros(cache_shape_np, dtype=np.float32)
    c3_z = np.zeros(cache_shape_np, dtype=np.float32)
    for _, input_, right_context in candidates[:num_samples]:
        default_chunks.append(input_)
        stateful_chunks.append([c3_z.copy(), input_, right_context, c2_z.copy(), c0_z.copy(), c1_z.copy()])
    return default_chunks, stateful_chunks


def load_stateful_representative_from_npz(npz_path, batch, time_steps, feat_dim):
    """
    从 .npz 加载「流式真实 cache」代表性数据（由 compare_encoder_backends --save-repr-stateful 生成）。
    npz 含: input (N,1,10,400), right_context (N,1,2,400), c0..c3 (N,1,128,9) - Conv1d 版本，cache 无最后一维。
    返回 list of [c3, input_, right_context, c2, c0, c1]（TFLite 输入顺序）。
    """
    z = np.load(npz_path)
    inp = z["input"]  # (N, 1, 10, 400)
    rc = z["right_context"]
    c0, c1, c2, c3 = z["c0"], z["c1"], z["c2"], z["c3"]
    N = inp.shape[0]
    out = []
    for i in range(N):
        out.append([
            c3[i].astype(np.float32),
            inp[i].astype(np.float32),
            rc[i].astype(np.float32),
            c2[i].astype(np.float32),
            c0[i].astype(np.float32),
            c1[i].astype(np.float32),
        ])
    return out


# ---------------------------------------------------------------------------
# 3) 导出流程：ONNX → SavedModel → TFLite (float32/float16/16x8/int8)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="导出 FSMN encoder 为 ONNX / TFLite")
    parser.add_argument("--repr-npy", type=str, default=None, help="用于量化校准的真实 fbank 特征 .npy 路径（形状 (T,400) 或 (N,T,400)）")
    parser.add_argument("--repr-num-samples", type=int, default=200, help="代表性样本数量（默认 200）")
    parser.add_argument("--repr-energy-percentile", type=float, default=0, help="只保留能量>=该百分位的块，0=不筛，建议 50~70 减少静音帧主导（默认 0）")
    parser.add_argument("--repr-stateful", type=str, default=None, help="流式真实 cache 的 .npz（由 compare_encoder_backends --save-repr-stateful 生成），优先于 --repr-npy 的 stateful 校准")
    parser.add_argument("--enable-xnnpack", action="store_true", help="启用 XNNPACK delegate（默认禁用，以确保 TFLite Micro 兼容性）")
    args = parser.parse_args()
    
    # 根据命令行参数决定是否禁用 XNNPACK
    if args.enable_xnnpack:
        # 用户明确要求启用 XNNPACK
        if 'TF_DISABLE_XNNPACK' in os.environ:
            del os.environ['TF_DISABLE_XNNPACK']
        os.environ['TF_ENABLE_XNNPACK'] = '1'
        print("⚠️  XNNPACK delegate 已启用（模型将不兼容 TFLite Micro）")
    else:
        # 默认禁用 XNNPACK（确保 TFLite Micro 兼容性）
        if 'TF_DISABLE_XNNPACK' not in os.environ:
            os.environ['TF_DISABLE_XNNPACK'] = '1'
        print("ℹ️  XNNPACK delegate 已禁用（确保 TFLite Micro 兼容性）")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, "finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt")
    out_dir = os.path.join(script_dir, "tflite_models")
    os.makedirs(out_dir, exist_ok=True)

    # Encoder 输入形状 (B, T, D)：B=batch, T=时间帧数, D=特征维度
    # - 1：batch=1，单条推理
    # - 10：T 与流式 chunk_size 一致（a2.py 里 chunk_size=10）
    # - 400：D 由模型固定，FSMN 里 self.input_dim=400（见 a2.py）
    batch, time_steps, feat_dim = 1, 10, 400

    default_repr_chunks = None
    stateful_repr_chunks = None
    if args.repr_stateful and os.path.isfile(args.repr_stateful):
        print("加载流式代表性数据 (真实 cache): %s" % args.repr_stateful)
        stateful_repr_chunks = load_stateful_representative_from_npz(
            args.repr_stateful, batch, time_steps, feat_dim
        )
        print("  得到 %d 个 stateful 块（真实流式 cache）" % len(stateful_repr_chunks))
    if args.repr_npy and os.path.isfile(args.repr_npy):
        print("加载代表性数据: %s (最多 %d 样本, energy_percentile=%.0f)" % (
            args.repr_npy, args.repr_num_samples, args.repr_energy_percentile))
        default_from_npy, stateful_from_npy = load_representative_chunks(
            args.repr_npy, batch, time_steps, feat_dim,
            num_samples=args.repr_num_samples,
            energy_percentile=args.repr_energy_percentile,
        )
        print("  得到 %d 个 default 块、%d 个 stateful 块（零 cache）" % (len(default_from_npy), len(stateful_from_npy)))
        if default_repr_chunks is None:
            default_repr_chunks = default_from_npy
        if stateful_repr_chunks is None:
            stateful_repr_chunks = stateful_from_npy
    if default_repr_chunks is None and stateful_repr_chunks is not None:
        default_repr_chunks = [chunk[1] for chunk in stateful_repr_chunks]

    print("加载 encoder 并包装为单入单出 ...")
    model_kws = FsmnKWS()
    model_kws.eval()
    load_pretrained_model(
        model=model_kws,
        path=ckpt_path,
        ignore_init_mismatch=True,  # 允许 Conv2d->Conv1d 的形状不匹配
        oss_bucket=None,
        scope_map=[],
        excludes=["ctc"],
    )
    # 自动转换 Conv2d 权重到 Conv1d（从 checkpoint 加载）
    from a2 import convert_conv2d_to_conv1d_weights
    convert_conv2d_to_conv1d_weights(model_kws, checkpoint_path=ckpt_path)
    model_kws.eval()
    encoder_export = FSMNExport(model=model_kws.encoder)
    wrapper = FSMNExportWrapper(encoder_export)
    wrapper.eval()

    dummy = torch.randn(batch, time_steps, feat_dim)
    with torch.no_grad():
        _ = wrapper(dummy)
    print("  wrapper 测试通过")

    # ONNX（无状态，单入单出）
    onnx_path = os.path.join(out_dir, "fsmn_encoder.onnx")
    print("导出 ONNX: %s" % onnx_path)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "time"}, "output": {0: "batch", 1: "time"}},
        opset_version=14,
    )

    # ONNX 带 cache + right_context（分块时传入/接收 cache，无下一块时 right_context 传零）
    # Conv1d 版本：cache 形状为 (batch, 128, 9)，无最后一维
    cache_shape = (batch, 128, 9)  # proj_dim=128, left_ctx=9
    dummy_caches = [torch.zeros(cache_shape) for _ in range(4)]
    dummy_right = torch.zeros(batch, 2, feat_dim)  # (B, 2, D)，无下一块时用零
    wrapper_stateful = FSMNExportWrapperStateful(encoder_export)
    wrapper_stateful.eval()
    onnx_stateful_path = os.path.join(out_dir, "fsmn_encoder_stateful.onnx")
    print("导出 ONNX (带 cache + right_context): %s" % onnx_stateful_path)
    torch.onnx.export(
        wrapper_stateful,
        (dummy, *dummy_caches, dummy_right),
        onnx_stateful_path,
        input_names=["input", "in_cache_0", "in_cache_1", "in_cache_2", "in_cache_3", "right_context"],
        output_names=["output", "out_cache_0", "out_cache_1", "out_cache_2", "out_cache_3"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "output": {0: "batch", 1: "time"},
        },
        opset_version=14,
    )

    # ONNX → SavedModel (onnx2tf)
    try:
        import onnx2tf
    except ImportError:
        print("请安装 onnx2tf: pip install onnx2tf")
        sys.exit(1)

    saved_model_dir = os.path.join(out_dir, "fsmn_encoder_saved_model")
    print("ONNX → SavedModel (onnx2tf): %s" % saved_model_dir)
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        output_signaturedefs=True,  # 避免 OP 名含前导 / 导致 SavedModel 命名校验失败
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        overwrite_input_shape=[f"input:{batch},{time_steps},{feat_dim}"],
        keep_shape_absolutely_input_names=["input"],
    )

    # ONNX 带 cache → SavedModel（多输入多输出，用于 TFLite 带 cache）
    saved_model_stateful_dir = os.path.join(out_dir, "fsmn_encoder_stateful_saved_model")
    # Conv1d 版本：cache 形状为 (batch, 128, 9)，无最后一维
    cache_shape_str = f"{batch},128,9"
    right_shape_str = f"{batch},2,{feat_dim}"
    overwrite_stateful = [
        f"input:{batch},{time_steps},{feat_dim}",
        "in_cache_0:" + cache_shape_str,
        "in_cache_1:" + cache_shape_str,
        "in_cache_2:" + cache_shape_str,
        "in_cache_3:" + cache_shape_str,
        "right_context:" + right_shape_str,
    ]
    print("ONNX (带 cache) → SavedModel: %s" % saved_model_stateful_dir)
    onnx2tf.convert(
        input_onnx_file_path=onnx_stateful_path,
        output_folder_path=saved_model_stateful_dir,
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        overwrite_input_shape=overwrite_stateful,
        keep_shape_absolutely_input_names=["input", "in_cache_0", "in_cache_1", "in_cache_2", "in_cache_3", "right_context"],
    )

    # TFLite 四个版本（无状态）
    import tensorflow as tf

    def _default_representative_gen():
        if default_repr_chunks is not None:
            for inp in default_repr_chunks:
                yield [inp]
        else:
            for _ in range(5):
                yield [np.random.randn(batch, time_steps, feat_dim).astype(np.float32)]

    # float32
    tflite_fp32_path = os.path.join(out_dir, "fsmn_encoder_float32.tflite")
    print("导出 TFLite float32: %s" % tflite_fp32_path)
    conv_fp32 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    conv_fp32.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_fp32 = conv_fp32.convert()
    with open(tflite_fp32_path, "wb") as f:
        f.write(tflite_fp32)
    print_tflite_ops(tflite_fp32_path, "float32")

    # float16
    tflite_fp16_path = os.path.join(out_dir, "fsmn_encoder_float16.tflite")
    print("导出 TFLite float16: %s" % tflite_fp16_path)
    conv_fp16 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_fp16.target_spec.supported_types = [tf.float16]
    conv_fp16.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_fp16 = conv_fp16.convert()
    with open(tflite_fp16_path, "wb") as f:
        f.write(tflite_fp16)
    print_tflite_ops(tflite_fp16_path, "float16")

    # 16x8 (int16 activations, int8 weights)
    tflite_16x8_path = os.path.join(out_dir, "fsmn_encoder_16x8.tflite")
    print("导出 TFLite 16x8: %s" % tflite_16x8_path)
    conv_16x8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    conv_16x8.optimizations = [tf.lite.Optimize.DEFAULT]
    # 16x8 量化：输入输出使用 int16（激活类型），避免 DEQUANTIZE（TFLite Micro 不支持）
    conv_16x8.inference_input_type = tf.int16
    conv_16x8.inference_output_type = tf.int16
    conv_16x8.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    conv_16x8.representative_dataset = _default_representative_gen
    # 16x8 量化：输入输出使用 int16（激活类型），避免 DEQUANTIZE（TFLite Micro 不支持）
    conv_16x8.inference_input_type = tf.int16
    conv_16x8.inference_output_type = tf.int16
    tflite_16x8 = conv_16x8.convert()
    with open(tflite_16x8_path, "wb") as f:
        f.write(tflite_16x8)
    print_tflite_ops(tflite_16x8_path, "16x8")

    # int8 (full integer quantization) - 两个版本：校准和不校准
    # 版本1：使用校准数据（如果有）
    if default_repr_chunks is not None:
        tflite_int8_calib_path = os.path.join(out_dir, "fsmn_encoder_int8.tflite")
        print("导出 TFLite int8 (使用校准): %s" % tflite_int8_calib_path)
        conv_int8_calib = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        conv_int8_calib.optimizations = [tf.lite.Optimize.DEFAULT]
        conv_int8_calib.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        conv_int8_calib.representative_dataset = _default_representative_gen
        # 全整数量化：输入输出也是 int8，避免 DEQUANTIZE（TFLite Micro 不支持）
        conv_int8_calib.inference_input_type = tf.int8
        conv_int8_calib.inference_output_type = tf.int8
        print("  使用代表性数据集进行全整数量化")
        tflite_int8_calib = conv_int8_calib.convert()
        with open(tflite_int8_calib_path, "wb") as f:
            f.write(tflite_int8_calib)
        print_tflite_ops(tflite_int8_calib_path, "int8 (校准)")
    
    # 版本2：不使用校准数据（仅权重量化）
    tflite_int8_no_calib_path = os.path.join(out_dir, "fsmn_encoder_int8_no_calib.tflite")
    print("导出 TFLite int8 (不使用校准): %s" % tflite_int8_no_calib_path)
    conv_int8_no_calib = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    conv_int8_no_calib.optimizations = [tf.lite.Optimize.DEFAULT]
    # 不使用 TFLITE_BUILTINS_INT8（它需要代表性数据集），只使用 TFLITE_BUILTINS 进行权重量化
    conv_int8_no_calib.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # 不使用代表性数据集：只做权重量化，激活保持 float32
    print("  仅量化权重（激活保持 float32），未使用代表性数据集")
    # 不设置 inference_input_type 和 inference_output_type，让 TFLite 使用默认行为
    tflite_int8_no_calib = conv_int8_no_calib.convert()
    with open(tflite_int8_no_calib_path, "wb") as f:
        f.write(tflite_int8_no_calib)
    print_tflite_ops(tflite_int8_no_calib_path, "int8 (不校准)")

    # TFLite 带 cache：float32 / float16 / 16x8 / int8（多输入多输出）
    # representative_dataset 的 yield 顺序必须与 TFLite 转换后的输入顺序一致（非 SavedModel 顺序）
    # 实测 stateful 模型 TFLite 输入顺序：0=in_cache_3, 1=input, 2=right_context, 3=in_cache_2, 4=in_cache_0, 5=in_cache_1
    def _stateful_representative_gen():
        if stateful_repr_chunks is not None:
            for inputs in stateful_repr_chunks:
                yield list(inputs)
        else:
            # Conv1d 版本：cache 形状为 (batch, 128, 9)，无最后一维
            cache_shape_np = (batch, 128, 9)
            right_shape_np = (batch, 2, feat_dim)
            num_random = max(200, args.repr_num_samples)  # int8 需要更多样本
            print("  警告: 未提供真实代表性数据，使用 %d 个随机样本（int8 精度可能较差，建议用 --repr-stateful）" % num_random)
            for _ in range(num_random):
                input_ = np.random.randn(batch, time_steps, feat_dim).astype(np.float32)
                c0 = np.random.randn(*cache_shape_np).astype(np.float32)
                c1 = np.random.randn(*cache_shape_np).astype(np.float32)
                c2 = np.random.randn(*cache_shape_np).astype(np.float32)
                c3 = np.random.randn(*cache_shape_np).astype(np.float32)
                right_context = np.random.randn(*right_shape_np).astype(np.float32)
                yield [c3, input_, right_context, c2, c0, c1]

    tflite_stateful_fp32_path = os.path.join(out_dir, "fsmn_encoder_stateful_float32.tflite")
    print("导出 TFLite (带 cache) float32: %s" % tflite_stateful_fp32_path)
    conv_s = tf.lite.TFLiteConverter.from_saved_model(saved_model_stateful_dir)
    conv_s.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    with open(tflite_stateful_fp32_path, "wb") as f:
        f.write(conv_s.convert())
    print_tflite_ops(tflite_stateful_fp32_path, "stateful float32")

    tflite_stateful_fp16_path = os.path.join(out_dir, "fsmn_encoder_stateful_float16.tflite")
    print("导出 TFLite (带 cache) float16: %s" % tflite_stateful_fp16_path)
    conv_s_fp16 = tf.lite.TFLiteConverter.from_saved_model(saved_model_stateful_dir)
    conv_s_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_s_fp16.target_spec.supported_types = [tf.float16]
    conv_s_fp16.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    with open(tflite_stateful_fp16_path, "wb") as f:
        f.write(conv_s_fp16.convert())
    print_tflite_ops(tflite_stateful_fp16_path, "stateful float16")

    tflite_stateful_16x8_path = os.path.join(out_dir, "fsmn_encoder_stateful_16x8.tflite")
    print("导出 TFLite (带 cache) 16x8: %s" % tflite_stateful_16x8_path)
    try:
        conv_s_16x8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_stateful_dir)
        conv_s_16x8.optimizations = [tf.lite.Optimize.DEFAULT]
        conv_s_16x8.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        conv_s_16x8.representative_dataset = _stateful_representative_gen
        # 16x8 量化：输入输出使用 int16（激活类型），避免 DEQUANTIZE（TFLite Micro 不支持）
        conv_s_16x8.inference_input_type = tf.int16
        conv_s_16x8.inference_output_type = tf.int16
        tflite_bytes = conv_s_16x8.convert()
        with open(tflite_stateful_16x8_path, "wb") as f:
            f.write(tflite_bytes)
        print_tflite_ops(tflite_stateful_16x8_path, "stateful 16x8")
    except Exception as e:
        print("  跳过 16x8 (多输入 stateful 可能不支持): %s" % e)
        if os.path.isfile(tflite_stateful_16x8_path) and os.path.getsize(tflite_stateful_16x8_path) == 0:
            os.remove(tflite_stateful_16x8_path)

    tflite_stateful_int8_path = os.path.join(out_dir, "fsmn_encoder_stateful_int8.tflite")
    # int8 (带 cache) - 两个版本：校准和不校准
    tflite_stateful_int8_no_calib_path = os.path.join(out_dir, "fsmn_encoder_stateful_int8_no_calib.tflite")
    
    # 版本1：使用校准数据（如果有）
    if stateful_repr_chunks is not None:
        print("导出 TFLite (带 cache) int8 (使用校准): %s" % tflite_stateful_int8_path)
        print("  使用 %d 个真实流式代表性样本" % len(stateful_repr_chunks))
        print("  注意: int8 全整数量化（int8 激活 + int8 权重）精度损失较大，在复杂/噪声场景可能失败")
        print("  若测试发现唤醒率不足，强烈建议使用 16x8（int16 激活 + int8 权重，精度更好）")
        try:
            conv_s_int8_calib = tf.lite.TFLiteConverter.from_saved_model(saved_model_stateful_dir)
            conv_s_int8_calib.optimizations = [tf.lite.Optimize.DEFAULT]
            conv_s_int8_calib.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            conv_s_int8_calib.representative_dataset = _stateful_representative_gen
            # 全整数量化：输入输出也是 int8，避免 DEQUANTIZE（TFLite Micro 不支持）
            conv_s_int8_calib.inference_input_type = tf.int8
            conv_s_int8_calib.inference_output_type = tf.int8
            tflite_bytes_calib = conv_s_int8_calib.convert()
            with open(tflite_stateful_int8_path, "wb") as f:
                f.write(tflite_bytes_calib)
            print_tflite_ops(tflite_stateful_int8_path, "stateful int8 (校准)")
        except Exception as e:
            print("  跳过 int8 校准版本 (多输入 stateful 可能不支持): %s" % e)
            import traceback
            traceback.print_exc()
            if os.path.isfile(tflite_stateful_int8_path) and os.path.getsize(tflite_stateful_int8_path) == 0:
                os.remove(tflite_stateful_int8_path)
    else:
        print("  跳过 int8 校准版本（未提供代表性数据集）")
    
    # 版本2：不使用校准数据（仅权重量化）
    print("导出 TFLite (带 cache) int8 (不使用校准): %s" % tflite_stateful_int8_no_calib_path)
    print("  仅量化权重（激活保持 float32），未使用代表性数据集")
    print("  注意: 这样无法实现全整数量化，但可以导出模型")
    try:
        conv_s_int8_no_calib = tf.lite.TFLiteConverter.from_saved_model(saved_model_stateful_dir)
        conv_s_int8_no_calib.optimizations = [tf.lite.Optimize.DEFAULT]
        # 不使用 TFLITE_BUILTINS_INT8（它需要代表性数据集），只使用 TFLITE_BUILTINS 进行权重量化
        conv_s_int8_no_calib.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        # 不使用代表性数据集：只做权重量化，激活保持 float32
        # 不设置 inference_input_type 和 inference_output_type，让 TFLite 使用默认行为
        tflite_bytes_no_calib = conv_s_int8_no_calib.convert()
        with open(tflite_stateful_int8_no_calib_path, "wb") as f:
            f.write(tflite_bytes_no_calib)
        print_tflite_ops(tflite_stateful_int8_no_calib_path, "stateful int8 (不校准)")
    except Exception as e:
        print("  跳过 int8 不校准版本 (多输入 stateful 可能不支持): %s" % e)
        import traceback
        traceback.print_exc()
        if os.path.isfile(tflite_stateful_int8_no_calib_path) and os.path.getsize(tflite_stateful_int8_no_calib_path) == 0:
            os.remove(tflite_stateful_int8_no_calib_path)

    print("\n全部完成。TFLite 文件目录: %s" % out_dir)


if __name__ == "__main__":
    main()
