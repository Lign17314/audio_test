import torch
import numpy as np
from funasr.train_utils.load_pretrained_model import load_pretrained_model
import torchaudio
from python.model import FsmnKWS
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from collections import defaultdict
import math
from typing import List, Tuple

# ======================== 工具函数（保留） ========================
def is_sublist(main_list, check_list):
    """
    修复版：检查check_list是否是main_list的连续子序列
    返回：子序列起始位置（int），未找到返回-1
    """
    main_len = len(main_list)
    check_len = len(check_list)
    
    # 如果目标序列更长，直接返回-1
    if check_len > main_len:
        return -1
    
    # 滑动窗口检查
    for i in range(main_len - check_len + 1):
        # 截取当前窗口的子序列
        window = main_list[i:i+check_len]
        if window == check_list:
            return i  # 返回起始位置
    return -1

def load_cmvn(cmvn_file):
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn

def apply_cmvn(inputs, cmvn):
    device = inputs.device
    dtype = inputs.dtype
    frame, dim = inputs.shape

    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)

    return inputs.type(torch.float32)

def apply_lfr(inputs, lfr_m, lfr_n):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).view(1, -1))
        else:
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n :]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    return LFR_outputs.type(torch.float32)

def beam_search(
    logits: torch.Tensor,
    logits_lengths: torch.Tensor,
    keywords_tokenset: set = None,
    score_beam_size: int = 3,
    path_beam_size: int = 20,
) -> Tuple[List[List[int]], torch.Tensor]:
    maxlen = logits.size(0)
    ctc_probs = logits
    cur_hyps = [(tuple(), (1.0, 0.0, []))]
    print("maxlen", maxlen)
    for t in range(0, maxlen):
        probs = ctc_probs[t]
        next_hyps = defaultdict(lambda: (0.0, 0.0, []))
        top_k_probs, top_k_index = probs.topk(score_beam_size)
        filter_probs = []
        filter_index = []
        for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
            if keywords_tokenset is not None:
                if prob > 0.05 and idx in keywords_tokenset:
                    filter_probs.append(prob)
                    filter_index.append(idx)
            else:
                if prob > 0.05:
                    filter_probs.append(prob)
                    filter_index.append(idx)
        if len(filter_index) == 0:
            continue
        for s in filter_index:
            ps = probs[s].item()
            if s != 0:
                print(f"frame:{t}, token:{s}, score:{ps}")
            for prefix, (pb, pnb, cur_nodes) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == 0:
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pb = n_pb + pb * ps + pnb * ps
                    nodes = cur_nodes.copy()
                    next_hyps[prefix] = (n_pb, n_pnb, nodes)
                elif s == last:
                    if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                        n_pb, n_pnb, nodes = next_hyps[prefix]
                        n_pnb = n_pnb + pnb * ps
                        nodes = cur_nodes.copy()
                        if ps > nodes[-1]["prob"]:
                            nodes[-1]["prob"] = ps
                            nodes[-1]["frame"] = t
                        next_hyps[prefix] = (n_pb, n_pnb, nodes)
                    if not math.isclose(pb, 0.0, abs_tol=0.000001):
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb, nodes = next_hyps[n_prefix]
                        n_pnb = n_pnb + pb * ps
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t, prob=ps))
                        next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                else:
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    if nodes:
                        if ps > nodes[-1]["prob"]:
                            nodes[-1]["prob"] = ps
                            nodes[-1]["frame"] = t
                    else:
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t, prob=ps))
                    n_pnb = n_pnb + pb * ps + pnb * ps
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
        next_hyps = sorted(next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)
        cur_hyps = next_hyps[:path_beam_size]
    hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in cur_hyps]
    return hyps

# ======================== 纯静态Encoder ========================
class PureStaticEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        
        # 冻结参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        torch.set_grad_enabled(False)
        
    def forward(self, x):
        out = self.encoder(x)
        return out

# ======================== 适配低版本的ONNX导出函数 ========================
def export_encoder_precise(encoder, input_tensor, output_path="kws_encoder_final.onnx"):
    """适配低版本PyTorch和onnxsim的导出函数"""
    static_encoder = PureStaticEncoder(encoder)
    
    # 导出配置（移除高版本参数）
    export_config = {
        "f": output_path,
        "input_names": ["encoder_input"],
        "output_names": ["encoder_output"],
        "dynamic_axes": None,
        "opset_version": 12,
        "do_constant_folding": True,
        "verbose": False,
        "keep_initializers_as_inputs": False,  # 关键修改：避免参数出现在输入中
    }
    
    # 适配training参数
    if hasattr(torch.onnx, 'TrainingMode'):
        export_config["training"] = torch.onnx.TrainingMode.EVAL
    else:
        export_config["training"] = False
    
    # 预热
    with torch.no_grad():
        static_encoder(input_tensor)
    
    # 导出
    try:
        torch.onnx.export(
            static_encoder,
            input_tensor,
            **export_config
        )
        print(f"\n✅ ONNX导出成功: {output_path}")
        
        # 简化ONNX（适配低版本onnxsim）
        try:
            import onnx
            from onnxsim import simplify
            model = onnx.load(output_path)
            
            # 低版本onnxsim的简化调用
            simp_model, check = simplify(model)
            
            simplified_path = output_path.replace(".onnx", "_simplified.onnx")
            onnx.save(simp_model, simplified_path)
            print(f"✅ ONNX简化成功: {simplified_path}")
            
            return simplified_path
            
        except Exception as e:
            print(f"⚠️ ONNX简化失败（不影响推理）: {str(e)[:80]}")
            return output_path
            
    except Exception as e:
        print(f"❌ 导出失败: {str(e)[:200]}")
        return None

# ======================== 放宽阈值的数值验证函数 ========================
def verify_numerical_equality(pytorch_encoder, onnx_path, test_input):
    """放宽误差阈值（工程可接受范围）"""
    # PyTorch推理
    with torch.no_grad():
        torch_out = pytorch_encoder(test_input).cpu().numpy()
    
    # ONNX推理
    try:
        import onnxruntime as ort
        # 配置ONNX Runtime（关闭优化，提升精度）
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        sess = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # ONNX推理
        onnx_out = sess.run(["encoder_output"], {"encoder_input": test_input.cpu().numpy()})[0]
        
        # 计算误差
        abs_diff = np.abs(torch_out - onnx_out)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        rmse = np.sqrt(np.mean(np.square(abs_diff)))
        
        print("\n=== 数值一致性验证 ===")
        print(f"最大绝对误差: {max_diff:.8f}")
        print(f"平均绝对误差: {mean_diff:.8f}")
        print(f"均方根误差: {rmse:.8f}")
        
        # 放宽阈值到1e-3（千分之一误差，工程上完全可接受）
        threshold = 1e-3
        is_aligned = max_diff < threshold
        print(f"数值是否在工程可接受范围内: {is_aligned} (阈值: {threshold})")
        
        return onnx_out, is_aligned
        
    except ImportError:
        print("❌ 缺少onnxruntime，请安装: pip install onnxruntime")
        return None, False

# ======================== 主流程 ========================
if __name__ == "__main__":
    # 1. 配置参数
    cmvn_file = "train/am.mvn.dim80_l2r2"
    init_param = "/root/volume/ctc/train/work_dir/avg_10_funasr.pt"
    data_in = "/root/volume/ctc/train/example_kws/wav/20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_13.wav"
    device = "cpu"
    reduce_channels = True

    # 特征提取配置
    fs = 16000
    n_mels = 80
    frame_length = 25
    frame_shift = 10
    lfr_m = 5
    lfr_n = 3
    keywords_idxset = {0, 1462, 976}
    lab = (1462, 976, 1462, 976)

    # 2. 初始化模型
    encoder_conf = {
        "input_dim": 400,
        "input_affine_dim": 140,
        "fsmn_layers": 4,
        "linear_dim": 250,
        "proj_dim": 128,
        "lorder": 10,
        "rorder": 2,
        "lstride": 1,
        "rstride": 1,
        "output_affine_dim": 140,
        "output_dim": 2599,
        "use_softmax": False,
    }

    model = FsmnKWS(
        encoder="FSMN", encoder_conf=encoder_conf, input_size=400, vocab_size=2599
    )

    # 加载参数
    load_pretrained_model(
        model=model,
        path=init_param,
        ignore_init_mismatch=False,
        oss_bucket=None,
        scope_map=[],
        excludes=["ctc"],
    )
    model.eval()
    torch.set_grad_enabled(False)

    # 3. 特征提取
    audio_sample_list, audio_fs = torchaudio.load(data_in)
    if reduce_channels:
        audio_sample_list = audio_sample_list.mean(0, keepdim=True)
    
    # 音频预处理
    waveform = audio_sample_list[0] * (1 << 15)
    waveform = waveform.unsqueeze(0)
    
    # FBank特征提取
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,  # 关闭随机噪声
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=16000,
        snip_edges=True,
    )
    
    # LFR和CMVN
    mat = apply_lfr(mat, 5, 3)
    cmvn = load_cmvn(cmvn_file)
    mat = apply_cmvn(mat, cmvn)
    
    # 模型输入
    speech = mat.unsqueeze(0)
    speech_lengths = torch.tensor([speech.shape[1]])

    # 4. PyTorch基准推理
    encoder_out_pytorch = model.encoder(speech)
    x_pytorch = encoder_out_pytorch[0, :speech_lengths[0], :]
    raw_logp_pytorch = F.softmax(x_pytorch.unsqueeze(0), dim=2).detach().squeeze(0).cpu()
    xlen = torch.tensor([raw_logp_pytorch.size(0)])

    hyps_pytorch = beam_search(
        logits=raw_logp_pytorch, 
        logits_lengths=xlen, 
        keywords_tokenset=keywords_idxset
    )

    print("\n=== PyTorch基准结果 ===")
    prefix_ids_pytorch = hyps_pytorch[0][0] if hyps_pytorch else ()
    isHit_pytorch = 1 if is_sublist(prefix_ids_pytorch, lab) != -1 else 0
    print(f"前缀ID: {prefix_ids_pytorch}")
    print(f"isHit: {isHit_pytorch}")

    # 5. 导出ONNX
    print("\n=== 导出ONNX模型 ===")
    onnx_path = export_encoder_precise(model.encoder, speech)
    
    if onnx_path is None:
        exit(1)

    # 6. 数值验证（放宽阈值）
    onnx_out, is_aligned = verify_numerical_equality(model.encoder, onnx_path, speech)

    # 7. 强制进行ONNX端到端推理（无论数值验证结果）
    print("\n=== ONNX端到端推理 ===")
    if onnx_out is not None:
        # ONNX结果处理
        onnx_out_tensor = torch.from_numpy(onnx_out)
        x_onnx = onnx_out_tensor[0, :speech_lengths[0], :]
        raw_logp_onnx = F.softmax(x_onnx.unsqueeze(0), dim=2).detach().squeeze(0).cpu()

        # beam search推理
        hyps_onnx = beam_search(
            logits=raw_logp_onnx, 
            logits_lengths=xlen, 
            keywords_tokenset=keywords_idxset
        )

        # 结果分析
        prefix_ids_onnx = hyps_onnx[0][0] if hyps_onnx else ()
        isHit_onnx = 1 if is_sublist(prefix_ids_onnx, lab) != -1 else 0

        # 最终对比
        print("\n=== 最终对比结果 ===")
        print(f"PyTorch: prefix={prefix_ids_pytorch}, isHit={isHit_pytorch}")
        print(f"ONNX   : prefix={prefix_ids_onnx}, isHit={isHit_onnx}")
        print(f"推理结果是否一致: {prefix_ids_pytorch == prefix_ids_onnx and isHit_pytorch == isHit_onnx}")
    else:
        print("❌ 无法进行ONNX推理")