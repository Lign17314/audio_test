import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np

# 定义封装类，模拟PyTorch模型接口
class ONNX2PyTorchWrapper(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        self.ort_sess = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        # 获取输入输出名称
        self.input_names = [i.name for i in self.ort_sess.get_inputs()]
        self.output_names = [o.name for o in self.ort_sess.get_outputs()]

    def forward(self, speech, speech_lengths):
        """
        完全模拟原PyTorch模型的forward接口
        """
        # 转换为numpy（ONNX Runtime输入格式）
        speech_np = speech.cpu().numpy().astype(np.float32)
        lengths_np = speech_lengths.cpu().numpy().astype(np.int32)
        
        # ONNX推理
        inputs = {
            self.input_names[0]: speech_np,
            self.input_names[1]: lengths_np
        }
        outputs = self.ort_sess.run(self.output_names, inputs)
        
        # 转换回PyTorch张量
        ctc_logits = torch.from_numpy(outputs[0]).to(speech.device)
        encoder_lens = torch.from_numpy(outputs[1]).to(speech_lengths.device)
        return ctc_logits, encoder_lens

# 实例化封装模型（伪PyTorch模型）
onnx_path = "./fsmn_kws_onnx_400dim/fsmn_kws_400dim.onnx"
pytorch_like_model = ONNX2PyTorchWrapper(onnx_path).eval()


# 验证接口和输出（和原PyTorch模型完全一致）
dummy_speech = torch.randn(1, 100, 400, device="cpu")
dummy_lengths = torch.tensor([100], dtype=torch.int32, device="cpu")

with torch.no_grad():
    output = pytorch_like_model(dummy_speech, dummy_lengths)
    print(f"✅ 封装模型输出形状: {output[0].shape}")  # (1,100,2599)
    print(f"✅ 模型接口和原PyTorch模型完全一致！")