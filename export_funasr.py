from funasr import AutoModel
import torch
import os
import torch.nn as nn
model = AutoModel(
    model="iic/speech_charctc_kws_phone-xiaoyun",
    keywords="小云小云",
    output_dir="./outputs/debug",
    device='cpu'
)

test_wav = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"

res = model.generate(input=test_wav, cache={},)
print(res)


# 获取底层FsmnKWS模型实例
fsmn_kws_model = model.model.eval()  # 强制推理模式


print(fsmn_kws_model)

# 从模型结构提取核心参数（精准匹配）
INPUT_DIM = 400  # 从in_linear1的in_features=400获取
VOCAB_SIZE = 2599  # 从out_linear2的out_features=2599获取
DUMMY_TIME_STEPS = 100  # 任意合理值（如1秒音频的特征步数）

# ===================== 2. 定义导出封装类（精准匹配模型前向） =====================
class FsmnKWSExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, speech, speech_lengths):
        """
        严格匹配模型的encode + ctc前向逻辑
        input:
            speech: (1, T, 400) 语音特征
            speech_lengths: (1,) 特征长度
        output:
            ctc_logits: (1, T, 2599) CTC输出logits
            encoder_out_lens: (1,) 编码后长度
        """
        with torch.no_grad():
            # 1. 特征编码（完全复用模型的encode方法）
            encoder_out, encoder_out_lens = self.model.encode(speech, speech_lengths)
            # 2. CTC log_softmax（匹配模型的ctc层）
            ctc_logits = self.model.ctc.log_softmax(encoder_out)
        return ctc_logits, encoder_out_lens

# ===================== 3. 构造精准匹配的虚拟输入 =====================
# 关键：输入维度设为400（不是360）
dummy_speech = torch.randn(1, DUMMY_TIME_STEPS, INPUT_DIM, dtype=torch.float32)
dummy_lengths = torch.tensor([DUMMY_TIME_STEPS], dtype=torch.int32)

# ===================== 4. 封装模型并导出 =====================
wrapper = FsmnKWSExportWrapper(fsmn_kws_model)

# 导出目录
export_dir = "./fsmn_kws_onnx_400dim"
os.makedirs(export_dir, exist_ok=True)
onnx_path = os.path.join(export_dir, "fsmn_kws_400dim.onnx")

# 执行导出（精准匹配模型维度）
torch.onnx.export(
    wrapper,
    (dummy_speech, dummy_lengths),
    onnx_path,
    input_names=["speech", "speech_lengths"],
    output_names=["ctc_logits", "encoder_out_lens"],
    dynamic_axes={
        "speech": {1: "time"},          # 支持任意时间步长
        "speech_lengths": {0: "batch"}, # 支持任意batch大小
        "ctc_logits": {1: "time"}       # 输出时间步和输入一致
    },
    opset_version=12,          # 兼容多数部署框架
    do_constant_folding=True,  # 优化模型
    verbose=False,
    keep_initializers_as_inputs=False  # 减少冗余节点
)

# ===================== 5. 验证导出的模型 =====================
import onnxruntime as ort

# 加载ONNX模型
ort_sess = ort.InferenceSession(onnx_path)

# 验证输入输出维度
print("="*50)
print("✅ 模型导出成功！核心信息验证：")
print(f"  - 导出路径: {onnx_path}")
print(f"  - 输入特征维度: {INPUT_DIM} (匹配in_linear1的400维)")
print(f"  - 输出词汇表大小: {VOCAB_SIZE} (匹配out_linear2的2599维)")

# 执行ONNX推理验证
ort_inputs = {
    "speech": dummy_speech.numpy(),
    "speech_lengths": dummy_lengths.numpy()
}
ort_outputs = ort_sess.run(["ctc_logits", "encoder_out_lens"], ort_inputs)

print(f"  - ONNX输出形状: ctc_logits={ort_outputs[0].shape} (应输出 (1, 100, 2599))")
print(f"  - 维度验证: {'✅ 匹配' if ort_outputs[0].shape == (1, 100, 2599) else '❌ 不匹配'}")
print("="*50)