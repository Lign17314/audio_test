from funasr import AutoModel
import torch
import os
import torch.nn as nn
import onnx
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 加载模型 =====================
model = AutoModel(
    model="iic/speech_charctc_kws_phone-xiaoyun",
    keywords="小云小云",
    output_dir="./outputs/debug",
    device='cpu'
)

# 测试原模型
test_wav = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"
res = model.generate(input=test_wav, cache={},)
print("原模型推理结果:", res)

fsmn_kws_model = model.model.eval()
INPUT_DIM = 400
FIXED_TIME_STEPS = 100
VOCAB_SIZE = 2599

# ===================== 2. 极简静态封装（无任何层属性依赖） =====================
class StaticFSMNWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.cache = {}  # 空cache，让动态分支无意义
        
    def forward(self, speech):
        """
        极简前向：仅固定输入形状，直接调用模型，不修改内部结构
        """
        # 强制固定输入形状
        assert speech.shape == (1, FIXED_TIME_STEPS, INPUT_DIM), \
            f"输入必须是 (1,{FIXED_TIME_STEPS},{INPUT_DIM})，当前: {speech.shape}"
        
        # 直接调用模型的encode（黑盒式，不关心内部结构）
        # 兼容返回值为张量/元组的情况
        encoder_result = self.model.encode(speech, self.cache)
        encoder_out = encoder_result[0] if isinstance(encoder_result, tuple) else encoder_result
        
        # 静态CTC计算
        ctc_logits = nn.functional.log_softmax(encoder_out, dim=2)
        return ctc_logits

# ===================== 3. 构造固定输入并预热模型 =====================
# 1. 构造固定形状输入
dummy_speech = torch.randn(1, FIXED_TIME_STEPS, INPUT_DIM, dtype=torch.float32)

# 2. 实例化封装模型
static_wrapper = StaticFSMNWrapper(fsmn_kws_model)
static_wrapper.eval()

# 3. 预热模型（关键：先运行一次前向，让模型内部所有层初始化）
with torch.no_grad():
    # 多次运行，确保动态分支被「固定」为静态路径
    for _ in range(2):
        warmup_output = static_wrapper(dummy_speech)
print(f"\n✅ 模型预热成功！输出形状: {warmup_output.shape}")

# ===================== 4. 黑盒式Trace（彻底冻结计算图） =====================
# 关键：strict=False，完全不检查层结构，只记录计算流程
with torch.no_grad():
    traced_model = torch.jit.trace(static_wrapper, dummy_speech, strict=False)
traced_model.eval()
print("\n✅ Trace模型冻结成功！")

# ===================== 5. 导出无If算子的ONNX =====================
export_dir = "./fsmn_kws_onnx_final_no_if"
os.makedirs(export_dir, exist_ok=True)
onnx_path = os.path.join(export_dir, "fsmn_kws_final_no_if.onnx")

# 导出配置（极致静态，禁用所有动态）
torch.onnx.export(
    traced_model,
    dummy_speech,
    onnx_path,
    input_names=["speech"],
    output_names=["ctc_logits"],
    dynamic_axes=None,          # 完全静态
    opset_version=10,           # 更低版本，更兼容tflite-micro
    do_constant_folding=True,   # 折叠所有常量
    verbose=False,
    keep_initializers_as_inputs=False,
    training=torch.onnx.TrainingMode.EVAL,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    # grad_enabled=False,         # 禁用梯度
    # 额外配置：禁用动态形状推断
    # enable_onnx_checker=False,  # 跳过严格检查，适配Trace后的图
)

# ===================== 6. 终极验证：检查If算子 + 推理 =====================
# 1. 检查If算子
onnx_model = onnx.load(onnx_path)
all_op_types = [node.op_type for node in onnx_model.graph.node]
has_if = "If" in all_op_types
has_dynamic = any(op in all_op_types for op in ["Slice", "DynamicSlice", "Where"])

# 2. ONNX Runtime推理验证
import onnxruntime as ort
ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_inputs = {"speech": dummy_speech.numpy()}
ort_outputs = ort_sess.run(["ctc_logits"], ort_inputs)

# 打印验证结果
print(f"\n{'='*60}")
print(f"✅ 最终验证结果：")
print(f"  - ONNX模型路径: {onnx_path}")
print(f"  - 含If算子: {'❌ 无（彻底移除）' if not has_if else '✅ 仍存在'}")
print(f"  - 含动态算子: {'❌ 无' if not has_dynamic else '✅ 有'}")
print(f"  - ONNX输出形状: {ort_outputs[0].shape} (预期: (1,{FIXED_TIME_STEPS},{VOCAB_SIZE}))")
print(f"  - 维度验证: {'✅ 匹配' if ort_outputs[0].shape == (1,FIXED_TIME_STEPS,VOCAB_SIZE) else '❌ 不匹配'}")
print(f"{'='*60}")
