from funasr import AutoModel
import torch
import os
import torch.nn as nn
import onnx
import warnings
import numpy as np
import librosa
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ===================== 全局配置 =====================
# 模型/关键词配置
MODEL_NAME = "iic/speech_charctc_kws_phone-xiaoyun"
KEYWORDS = "小云小云"
DEVICE = "cpu"
# 固定输入输出维度（微控制器专用）
INPUT_DIM = 400
FIXED_TIME_STEPS = 100
VOCAB_SIZE = 2599
# 输出路径
ONNX_EXPORT_DIR = "./fsmn_kws_onnx_final_no_if"
ONNX_PATH = os.path.join(ONNX_EXPORT_DIR, "fsmn_kws_final_no_if.onnx")
# 真实音频路径
LOCAL_WAV_PATH = "./test_xiaoyun.wav"  # 替换为你的真实音频路径
# 可视化保存路径
FRAME_COMPARE_PLOT = "./frame_level_comparison.png"
# 小云小云对应的核心token ID（关键：用于重点标注）
CORE_KEYWORD_TOKENS = {85: "小", 90: "云"}

# ===================== 工具函数：提取真实音频逐帧特征 =====================
def extract_real_audio_features(wav_path, target_len=100, target_dim=400, sample_rate=16000):
    """
    提取真实音频的逐帧特征（适配模型输入）
    :return: 模型输入特征 + 真实有效帧数 + 音频时长
    """
    # 1. 加载音频并去静音（保留有效语音段）
    wav_data, sr = librosa.load(wav_path, sr=sample_rate)
    wav_data, _ = librosa.effects.trim(wav_data, top_db=20)  # 20dB以下为静音
    audio_duration = len(wav_data) / sample_rate
    
    # 2. 提取Fbank特征（逐帧）
    n_fft = int(sample_rate * 0.025)  # 25ms帧长
    hop_length = int(sample_rate * 0.01)  # 10ms帧移
    fbank = librosa.feature.melspectrogram(
        y=wav_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=40
    )
    fbank = librosa.power_to_db(fbank, ref=np.max)
    raw_feat = np.transpose(fbank)  # (T, 40) → T为真实帧数
    
    # 3. 适配模型输入：40→400维 + 补零到固定长度
    feat_400 = np.repeat(raw_feat, 10, axis=1)  # (T, 400)
    raw_feat_len = len(feat_400)  # 真实有效帧数（非补零）
    
    # 4. 补零到固定长度（100帧）
    if len(feat_400) < target_len:
        pad_len = target_len - len(feat_400)
        feat_fixed = np.pad(feat_400, ((0, pad_len), (0, 0)), mode="constant")
    else:
        feat_fixed = feat_400[:target_len]
    
    # 5. 归一化 + 转为torch张量（适配模型）
    feat_fixed = (feat_fixed - np.mean(feat_fixed)) / (np.std(feat_fixed) + 1e-8)
    model_input = torch.from_numpy(feat_fixed).unsqueeze(0).float()  # (1, 100, 400)
    model_input.requires_grad = False
    
    return {
        "model_input": model_input,
        "raw_feat_len": raw_feat_len,
        "audio_duration": audio_duration,
        "hop_length": hop_length
    }

# ===================== 工具函数：逐帧计算误差（新增token ID标注） =====================
def calculate_frame_level_error(pytorch_logits, onnx_logits, raw_feat_len):
    """
    逐帧计算两个模型输出的误差（修复索引越界 + 新增token ID标注）
    :param raw_feat_len: 真实有效帧数（非补零）
    :return: 逐帧误差字典（含最高概率token ID）
    """
    # 模型输出只有100帧，截断真实帧数到100帧（避免索引越界）
    valid_frame_len = min(raw_feat_len, FIXED_TIME_STEPS)  # 取较小值：100帧
    print(f"\n⚠️ 模型输出仅支持{FIXED_TIME_STEPS}帧，截断真实帧数到{valid_frame_len}帧（剩余帧无模型输出）")
    
    # 转为numpy并截断到有效帧（模型输出只有100帧）
    pytorch_np = pytorch_logits.detach().cpu().numpy()[0][:valid_frame_len]  # (100, 2599)
    onnx_np = onnx_logits[0][:valid_frame_len]  # (100, 2599)
    
    # 逐帧计算核心指标（新增token ID相关字段）
    frame_metrics = {
        "frame_idx": [],          # 帧索引
        "score_pytorch": [],      # 原模型每帧最大概率
        "score_onnx": [],         # ONNX模型每帧最大概率
        "score_diff": [],         # 分数差异（绝对值）
        "idx_pytorch": [],        # 原模型每帧预测索引（最高概率token ID）
        "idx_onnx": [],           # ONNX模型每帧预测索引（最高概率token ID）
        "token_name_pytorch": [], # 原模型最高概率token名称（如"小/云/未知"）
        "token_name_onnx": [],    # ONNX模型最高概率token名称
        "idx_match": [],          # 索引是否匹配（True/False）
        "mae_per_frame": [],      # 每帧MAE（logits层面）
        # 新增：核心关键词token占比
        "core_token_pct_pytorch": [],  # 原模型核心token（85/90）总占比
        "core_token_pct_onnx": []      # ONNX模型核心token总占比
    }
    
    for frame_idx in range(valid_frame_len):  # 仅遍历100帧
        # 1. 单帧logits
        frame_pytorch = pytorch_np[frame_idx]
        frame_onnx = onnx_np[frame_idx]
        
        # 2. 分数计算（LogSoftmax转概率）
        prob_pytorch = np.exp(frame_pytorch) / np.sum(np.exp(frame_pytorch))
        prob_onnx = np.exp(frame_onnx) / np.sum(np.exp(frame_onnx))
        
        # 3. 最高概率token（核心修改：记录ID+名称）
        max_prob_pytorch = np.max(prob_pytorch)
        max_prob_onnx = np.max(prob_onnx)
        idx_pytorch = np.argmax(prob_pytorch)
        idx_onnx = np.argmax(prob_onnx)
        
        # 4. token名称映射（重点标注小云小云）
        token_name_pytorch = CORE_KEYWORD_TOKENS.get(idx_pytorch, f"未知({idx_pytorch})")
        token_name_onnx = CORE_KEYWORD_TOKENS.get(idx_onnx, f"未知({idx_onnx})")
        
        # 5. 核心关键词token总占比（85+90的概率和）
        core_pct_pytorch = prob_pytorch[85] + prob_pytorch[90] if 85 < len(prob_pytorch) and 90 < len(prob_pytorch) else 0.0
        core_pct_onnx = prob_onnx[85] + prob_onnx[90] if 85 < len(prob_onnx) and 90 < len(prob_onnx) else 0.0
        
        # 6. 误差计算
        score_diff = abs(max_prob_pytorch - max_prob_onnx)
        mae = np.mean(np.abs(frame_pytorch - frame_onnx))
        idx_match = (idx_pytorch == idx_onnx)
        
        # 存入结果
        frame_metrics["frame_idx"].append(frame_idx)
        frame_metrics["score_pytorch"].append(max_prob_pytorch)
        frame_metrics["score_onnx"].append(max_prob_onnx)
        frame_metrics["score_diff"].append(score_diff)
        frame_metrics["idx_pytorch"].append(idx_pytorch)
        frame_metrics["idx_onnx"].append(idx_onnx)
        frame_metrics["token_name_pytorch"].append(token_name_pytorch)
        frame_metrics["token_name_onnx"].append(token_name_onnx)
        frame_metrics["idx_match"].append(idx_match)
        frame_metrics["mae_per_frame"].append(mae)
        frame_metrics["core_token_pct_pytorch"].append(core_pct_pytorch)
        frame_metrics["core_token_pct_onnx"].append(core_pct_onnx)
    
    # 计算全局统计
    frame_metrics["avg_score_diff"] = np.mean(frame_metrics["score_diff"])
    frame_metrics["avg_mae"] = np.mean(frame_metrics["mae_per_frame"])
    frame_metrics["idx_match_rate"] = np.sum(frame_metrics["idx_match"]) / len(frame_metrics["idx_match"])
    frame_metrics["valid_frame_len"] = valid_frame_len
    # 新增：核心token平均占比
    frame_metrics["avg_core_pct_pytorch"] = np.mean(frame_metrics["core_token_pct_pytorch"])
    frame_metrics["avg_core_pct_onnx"] = np.mean(frame_metrics["core_token_pct_onnx"])
    
    return frame_metrics

# ===================== 工具函数：提取真实音频逐帧特征（补充说明） =====================
def extract_real_audio_features(wav_path, target_len=100, target_dim=400, sample_rate=16000):
    """
    提取真实音频的逐帧特征（适配模型输入）
    :return: 模型输入特征 + 真实有效帧数 + 音频时长
    """
    # 1. 加载音频并去静音（保留有效语音段）
    wav_data, sr = librosa.load(wav_path, sr=sample_rate)
    wav_data, _ = librosa.effects.trim(wav_data, top_db=20)  # 20dB以下为静音
    audio_duration = len(wav_data) / sample_rate
    
    # 2. 提取Fbank特征（逐帧）
    n_fft = int(sample_rate * 0.025)  # 25ms帧长
    hop_length = int(sample_rate * 0.01)  # 10ms帧移
    fbank = librosa.feature.melspectrogram(
        y=wav_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=40
    )
    fbank = librosa.power_to_db(fbank, ref=np.max)
    raw_feat = np.transpose(fbank)  # (T, 40) → T为真实帧数
    
    # 3. 适配模型输入：40→400维 + 截断/补零到固定长度（100帧）
    feat_400 = np.repeat(raw_feat, 10, axis=1)  # (T, 400)
    raw_feat_len = len(feat_400)  # 真实有效帧数（非补零）
    
    # 4. 截断/补零到固定长度（100帧）：超过100帧则截断，不足则补零
    if len(feat_400) > target_len:
        feat_fixed = feat_400[:target_len]  # 截断到100帧（取前100帧）
        print(f"⚠️ 音频特征帧数({len(feat_400)})超过模型固定输入({target_len})，已截断到前{target_len}帧")
    else:
        pad_len = target_len - len(feat_400)
        feat_fixed = np.pad(feat_400, ((0, pad_len), (0, 0)), mode="constant")
    
    # 5. 归一化 + 转为torch张量（适配模型）
    feat_fixed = (feat_fixed - np.mean(feat_fixed)) / (np.std(feat_fixed) + 1e-8)
    model_input = torch.from_numpy(feat_fixed).unsqueeze(0).float()  # (1, 100, 400)
    model_input.requires_grad = False
    
    return {
        "model_input": model_input,
        "raw_feat_len": raw_feat_len,
        "audio_duration": audio_duration,
        "hop_length": hop_length,
        "actual_input_frames": min(raw_feat_len, target_len)  # 实际输入模型的帧数
    }

# ===================== 工具函数：可视化逐帧对比（新增token标注） =====================
def plot_frame_level_comparison(frame_metrics, save_path):
    """可视化逐帧对比结果（新增token ID/核心占比）"""
    # 配置中文显示
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 创建3行1列子图（新增核心token占比）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 子图1：逐帧分数+最高token标注
    ax1.plot(frame_metrics["frame_idx"], frame_metrics["score_pytorch"], 
             label="原PyTorch模型（最高概率）", color="#2E86AB", linewidth=2)
    ax1.plot(frame_metrics["frame_idx"], frame_metrics["score_onnx"], 
             label="ONNX模型（最高概率）", color="#E63946", linestyle="--", linewidth=2)
    ax1.plot(frame_metrics["frame_idx"], frame_metrics["score_diff"], 
             label="分数差异", color="#F1A208", linewidth=1)
    ax1.axhline(y=0.01, color="#707070", linestyle=":", label="差异阈值(0.01)")
    ax1.set_ylabel("最高概率值/分数差异")
    ax1.set_title("逐帧最高概率对比 + 核心token（小/云）标注")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 子图2：逐帧MAE + 索引匹配
    ax2.plot(frame_metrics["frame_idx"], frame_metrics["mae_per_frame"], 
             label="逐帧MAE（Logits）", color="#3A98B9", linewidth=2)
    ax2.axhline(y=0.01, color="#707070", linestyle=":", label="MAE阈值(0.01)")
    # 标记索引不匹配的帧
    mismatch_frames = [i for i, match in enumerate(frame_metrics["idx_match"]) if not match]
    if mismatch_frames:
        ax2.scatter(mismatch_frames, [frame_metrics["mae_per_frame"][i] for i in mismatch_frames], 
                    color="red", s=50, label="索引不匹配帧", zorder=5)
    ax2.set_ylabel("逐帧MAE")
    ax2.set_title("逐帧MAE及索引匹配情况")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 子图3：核心token（小+云）总占比
    ax3.plot(frame_metrics["frame_idx"], frame_metrics["core_token_pct_pytorch"], 
             label="原PyTorch模型（小+云占比）", color="#2E86AB", linewidth=2)
    ax3.plot(frame_metrics["frame_idx"], frame_metrics["core_token_pct_onnx"], 
             label="ONNX模型（小+云占比）", color="#E63946", linestyle="--", linewidth=2)
    ax3.axhline(y=0.5, color="#707070", linestyle=":", label="核心token占比阈值(0.5)")
    ax3.set_xlabel("帧索引（1帧=10ms）")
    ax3.set_ylabel("核心token总占比（小+云）")
    ax3.set_title("逐帧核心关键词（小云小云）token占比")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ 逐帧对比可视化图已保存：{save_path}")

# ===================== 1. 核心配置（仅推理） =====================
# 加载模型（修复writer错误）
model = AutoModel(
    model=MODEL_NAME,
    keywords=KEYWORDS,
    device=DEVICE,
    disable_log=True,
)
if not hasattr(model.model, "writer"):
    model.model.writer = {"detect": {}, "score": {}}
fsmn_kws_model = model.model.eval()

# 全局禁用梯度
for param in fsmn_kws_model.parameters():
    param.requires_grad = False

# ===================== 2. 极简静态封装（仅推理） =====================
class StaticFSMNInferWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.cache = {}  # 空cache消除所有动态分支
        
    def forward(self, speech):
        """纯推理前向：无梯度、无动态、固定形状"""
        assert speech.shape == (1, FIXED_TIME_STEPS, INPUT_DIM), \
            f"输入必须是 (1,{FIXED_TIME_STEPS},{INPUT_DIM})，当前: {speech.shape}"
        with torch.no_grad():
            encoder_result = self.model.encode(speech, self.cache)
            encoder_out = encoder_result[0] if isinstance(encoder_result, tuple) else encoder_result
            ctc_logits = nn.functional.log_softmax(encoder_out, dim=2)
        return ctc_logits

# ===================== 3. 构造输入并预热模型 =====================
dummy_speech = torch.randn(1, FIXED_TIME_STEPS, INPUT_DIM, dtype=torch.float32)
dummy_speech.requires_grad = False
static_wrapper = StaticFSMNInferWrapper(fsmn_kws_model)
static_wrapper.eval()

# 预热模型
with torch.no_grad():
    for _ in range(2):
        warmup_output = static_wrapper(dummy_speech)
print(f"\n✅ 模型预热成功！输出形状: {warmup_output.shape}")

# ===================== 4. Trace冻结计算图（纯静态） =====================
with torch.no_grad():
    traced_model = torch.jit.trace(static_wrapper, dummy_speech, strict=False)
traced_model.eval()
print("✅ Trace模型冻结成功（纯静态计算图）")

# ===================== 5. 导出无If/无梯度的ONNX（微控制器专用） =====================
os.makedirs(ONNX_EXPORT_DIR, exist_ok=True)
torch.onnx.export(
    traced_model,
    dummy_speech,
    ONNX_PATH,
    input_names=["speech"],
    output_names=["ctc_logits"],
    dynamic_axes=None,          
    opset_version=10,           
    do_constant_folding=True,   
    verbose=False,
    keep_initializers_as_inputs=False,
    training=torch.onnx.TrainingMode.EVAL,
)

# ===================== 6. 最终验证（微控制器兼容性） =====================
onnx_model = onnx.load(ONNX_PATH)
all_op_types = [node.op_type for node in onnx_model.graph.node]
has_if = "If" in all_op_types
has_dynamic = any(op in all_op_types for op in ["DynamicSlice", "Where", "Loop"])

# ONNX Runtime推理验证
import onnxruntime as ort
ort_sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
ort_inputs = {"speech": dummy_speech.numpy()}
ort_outputs = ort_sess.run(["ctc_logits"], ort_inputs)

print(f"\n{'='*70}")
print(f"✅ 纯推理ONNX模型导出完成（微控制器专用）")
print(f"  - 模型路径: {ONNX_PATH}")
print(f"  - 含If算子: {'❌ 无（彻底移除）' if not has_if else '✅ 仍存在'}")
print(f"  - 含动态算子: {'❌ 无（纯静态）' if not has_dynamic else '✅ 有'}")
print(f"  - 输入形状: (1, {FIXED_TIME_STEPS}, {INPUT_DIM})（固定）")
print(f"  - 输出形状: {ort_outputs[0].shape}（预期: (1,{FIXED_TIME_STEPS},{VOCAB_SIZE})）")
print(f"  - 模型体积: {os.path.getsize(ONNX_PATH) / 1024:.2f} KB")
print(f"{'='*70}")

# ===================== 7. 误差评估：ONNX vs 原PyTorch模型（保留原有逻辑） =====================
def calculate_error_metrics(pytorch_output, onnx_output):
    pytorch_np = pytorch_output.detach().cpu().numpy() if isinstance(pytorch_output, torch.Tensor) else pytorch_output
    onnx_np = onnx_output if isinstance(onnx_output, np.ndarray) else onnx_output.detach().cpu().numpy()
    mae = np.mean(np.abs(pytorch_np - onnx_np))
    mse = np.mean((pytorch_np - onnx_np) **2)
    pytorch_flat = pytorch_np.flatten()
    onnx_flat = onnx_np.flatten()
    cos_sim = np.dot(pytorch_flat, onnx_flat) / (
        np.linalg.norm(pytorch_flat) * np.linalg.norm(onnx_flat) + 1e-8
    )
    max_abs_err = np.max(np.abs(pytorch_np - onnx_np))
    return {
        "MAE（平均绝对误差）": mae,
        "MSE（均方误差）": mse,
        "余弦相似度": cos_sim,
        "最大绝对误差": max_abs_err
    }

# 原有多样化输入测试（保留）
test_inputs = {
    "随机输入": torch.randn(1, FIXED_TIME_STEPS, INPUT_DIM, dtype=torch.float32).to(DEVICE),
    "全零输入": torch.zeros(1, FIXED_TIME_STEPS, INPUT_DIM, dtype=torch.float32).to(DEVICE),
    "单位输入": torch.ones(1, FIXED_TIME_STEPS, INPUT_DIM, dtype=torch.float32).to(DEVICE) * 0.1
}
print(f"\n{'='*80}")
print(f"📊 ONNX模型 vs 原PyTorch模型 误差评估（模拟输入）")
print(f"{'='*80}")
for input_name, test_input in test_inputs.items():
    test_input.requires_grad = False
    with torch.no_grad():
        pytorch_output = static_wrapper(test_input)
    ort_inputs = {"speech": test_input.cpu().numpy()}
    onnx_output = ort_sess.run(["ctc_logits"], ort_inputs)[0]
    metrics = calculate_error_metrics(pytorch_output, onnx_output)
    print(f"\n🔹 测试输入：{input_name}")
    for metric_name, value in metrics.items():
        if "余弦相似度" in metric_name:
            print(f"  {metric_name}: {value:.6f} (✅ 合格)" if value > 0.999 else f"  {metric_name}: {value:.6f} (⚠️ 需关注)")
        elif "MAE" in metric_name:
            print(f"  {metric_name}: {value:.6f} (✅ 合格)" if value < 0.01 else f"  {metric_name}: {value:.6f} (⚠️ 需关注)")
        else:
            print(f"  {metric_name}: {value:.6f}")

# ===================== 8. 核心改造：真实音频逐帧对比（新增token ID输出） =====================
print(f"\n{'='*80}")
print(f"🎯 关键词检测结果一致性验证（真实音频+逐帧token ID对比）")
print(f"{'='*80}")

# 步骤1：提取真实音频特征
if not os.path.exists(LOCAL_WAV_PATH):
    # 下载测试音频（若本地无）
    import requests
    TEST_WAV_URL = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"
    response = requests.get(TEST_WAV_URL)
    with open(LOCAL_WAV_PATH, "wb") as f:
        f.write(response.content)
    print(f"✅ 测试音频已下载：{LOCAL_WAV_PATH}")

# 提取特征（调用修复后的函数）
feat_info = extract_real_audio_features(LOCAL_WAV_PATH, FIXED_TIME_STEPS, INPUT_DIM)
real_input = feat_info["model_input"]
raw_feat_len = feat_info["raw_feat_len"]
actual_input_frames = feat_info["actual_input_frames"]  # 实际输入模型的帧数（100帧）
print(f"\n✅ 真实音频特征提取完成：")
print(f"  - 音频总时长：{feat_info['audio_duration']:.3f} 秒")
print(f"  - 真实有效帧数：{raw_feat_len} 帧（1帧=10ms）")
print(f"  - 模型实际输入帧数：{actual_input_frames} 帧（截断/补零后）")
print(f"  - 模型输入形状：{real_input.shape}")

# 步骤2：原模型逐帧推理
with torch.no_grad():
    pytorch_logits = static_wrapper(real_input)  # (1, 100, 2599)

# 步骤3：ONNX模型逐帧推理
onnx_logits = ort_sess.run(["ctc_logits"], {"speech": real_input.cpu().numpy()})[0]

# 步骤4：逐帧计算误差（传入实际输入帧数）
frame_metrics = calculate_frame_level_error(pytorch_logits, onnx_logits, actual_input_frames)

# 步骤5：打印逐帧统计结果（新增token相关）
print(f"\n📈 逐帧对比统计结果（真实音频）：")
print(f"  - 有效对比帧数：{frame_metrics['valid_frame_len']} 帧")
print(f"  - 平均分数差异：{frame_metrics['avg_score_diff']:.6f} (✅ 合格)" if frame_metrics['avg_score_diff'] < 0.01 else f"  - 平均分数差异：{frame_metrics['avg_score_diff']:.6f} (⚠️ 需关注)")
print(f"  - 平均MAE（Logits）：{frame_metrics['avg_mae']:.6f} (✅ 合格)" if frame_metrics['avg_mae'] < 0.01 else f"  - 平均MAE（Logits）：{frame_metrics['avg_mae']:.6f} (⚠️ 需关注)")
print(f"  - 索引匹配率：{frame_metrics['idx_match_rate']:.6f} (✅ 合格)" if frame_metrics['idx_match_rate'] > 0.99 else f"  - 索引匹配率：{frame_metrics['idx_match_rate']:.6f} (⚠️ 需关注)")
# 新增：核心token平均占比
print(f"  - 原模型核心token（小+云）平均占比：{frame_metrics['avg_core_pct_pytorch']:.6f}")
print(f"  - ONNX模型核心token（小+云）平均占比：{frame_metrics['avg_core_pct_onnx']:.6f}")

# 步骤6：打印前20帧明细（新增token ID/名称）
print(f"\n🔍 前20帧明细（真实音频）：")
print(f"{'帧索引':<6} {'原模型最高token':<15} {'原模型分数':<12} {'ONNX最高token':<15} {'ONNX分数':<12} {'分数差异':<10} {'索引匹配':<8} {'核心token占比(原)':<15} {'核心token占比(ONNX)':<15}")
print("-" * 120)
for i in range(min(183, frame_metrics['valid_frame_len'])):
    print(f"{i:<6} {frame_metrics['token_name_pytorch'][i]:<15} {frame_metrics['score_pytorch'][i]:<12.6f} "
          f"{frame_metrics['token_name_onnx'][i]:<15} {frame_metrics['score_onnx'][i]:<12.6f} "
          f"{frame_metrics['score_diff'][i]:<10.6f} {frame_metrics['idx_match'][i]} "
          f"{frame_metrics['core_token_pct_pytorch'][i]:<15.6f} {frame_metrics['core_token_pct_onnx'][i]:<15.6f}")

# 步骤7：可视化逐帧对比
plot_frame_level_comparison(frame_metrics, FRAME_COMPARE_PLOT)

# ===================== 9. 最终结论 =====================
print(f"\n{'='*80}")
print(f"📝 最终评估结论（真实音频验证）")
print(f"{'='*80}")
final_mae = frame_metrics["avg_mae"]
final_match_rate = frame_metrics["idx_match_rate"]
final_core_pct_diff = abs(frame_metrics["avg_core_pct_pytorch"] - frame_metrics["avg_core_pct_onnx"])

if final_mae < 0.01 and final_match_rate > 0.999 and final_core_pct_diff < 0.01:
    print(f"✅ ONNX模型与原模型在真实音频上逐帧一致性极高，核心token（小/云）占比匹配，满足部署要求！")
elif final_mae < 0.05 and final_match_rate > 0.99 and final_core_pct_diff < 0.05:
    print(f"✅ ONNX模型与原模型在真实音频上逐帧一致性良好，核心token占比基本匹配，可部署（建议监控实际效果）！")
else:
    print(f"⚠️ ONNX模型与原模型在真实音频上逐帧误差较大，或核心token占比差异明显，需检查导出配置或重新导出！")