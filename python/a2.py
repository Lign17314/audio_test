from funasr import AutoModel
import torch
from torch.cuda.amp import autocast
from funasr.utils.load_utils import load_audio_text_image_video
from ctc import CTC, KwsCtcPrefixDecoder
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from typing import Optional, Dict

from typing import Tuple, Dict
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from funasr.register import tables


class LinearTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, input):
        output = self.linear(input)

        return output


class AffineTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        output = self.linear(input)

        return output


class RectifiedLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out = self.relu(input)
        return out


class FSMNBlock(nn.Module):
    """固定: dim=128, lorder=10, rorder=2, lstride=1, rstride=1
    使用 Conv1d 替代 Conv2d，减少 TFLite 中的 TRANSPOSE 算子
    """

    def __init__(self):
        super(FSMNBlock, self).__init__()
        self.dim = 128
        self.lorder = 10
        self.rorder = 2
        self.lstride = 1
        self.rstride = 1
        # 使用 Conv1d 替代 Conv2d，减少 TRANSPOSE
        self.conv_left = nn.Conv1d(
            self.dim, self.dim, self.lorder,
            dilation=self.lstride, groups=self.dim, bias=False,
        )
        self.conv_right = nn.Conv1d(
            self.dim, self.dim, self.rorder,
            dilation=self.rstride, groups=self.dim, bias=False,
        )

    def forward(
        self,
        input: torch.Tensor,  # (B, T, D)
        cache: torch.Tensor = None,  # (B, D, left_ctx) 而不是 (B, D, left_ctx, 1)
        right_context: torch.Tensor = None,  # (B, T_rc, D) 而不是 (B, D, T_rc, 1)
    ):
        # 直接 permute 到 (B, D, T)，无需 unsqueeze
        x_per = input.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)

        # 左上下文（前向帧）：有 cache 时用 cache，无则零 pad
        if cache is not None:
            cache = cache.to(device=x_per.device, dtype=x_per.dtype)
            y_left = torch.cat((cache, x_per), dim=2).contiguous()
            cache = y_left[:, :, -(self.lorder - 1) * self.lstride :].contiguous()
        else:
            y_left = F.pad(x_per, [(self.lorder - 1) * self.lstride, 0])

        # Conv1d 处理
        T_out = x_per.shape[2]
        if self.lstride == 1:
            y_left = self.conv_left(y_left)  # (B, D, T_out)
        else:
            left_out_list = []
            for t in range(T_out):
                w = y_left[
                    :, :, t : t + (self.lorder - 1) * self.lstride + 1 : self.lstride
                ]
                left_out_list.append(self.conv_left(w))
            y_left = torch.cat(left_out_list, dim=2)
        out = x_per + y_left

        # 右上下文（后向帧）：有 right_context 时用外部提供的几帧，无则零 pad
        if self.conv_right is not None:
            if right_context is not None:
                right_context = right_context.to(device=x_per.device, dtype=x_per.dtype)
                rc_per = right_context.permute(0, 2, 1)  # (B, T_rc, D) -> (B, D, T_rc)
                y_right = torch.cat((x_per, rc_per), dim=2)
            else:
                y_right = F.pad(x_per, [0, self.rorder * self.rstride])
            y_right = y_right[:, :, self.rstride :]
            if self.rstride == 1:
                y_right = self.conv_right(y_right)  # (B, D, T_out)
            else:
                right_out_list = []
                for t in range(T_out):
                    w = y_right[
                        :, :, t : t + (self.rorder - 1) * self.rstride + 1 : self.rstride
                    ]
                    right_out_list.append(self.conv_right(w))
                y_right = torch.cat(right_out_list, dim=2)
            out += y_right

        # permute 回 (B, T, D)
        output = out.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)

        return output, cache


# FSMN 栈中会使用 cache 的层：BasicBlock 用 dict cache（原地更新），BasicBlock_export 用 per-layer tensor（显式 in_cache/out_cache）


class BasicBlock(nn.Module):
    """固定: linear_dim=250, proj_dim=128, lorder=10, rorder=2；cache 用 Dict 按层 key 原地更新。"""

    def __init__(self, stack_layer: int):
        super(BasicBlock, self).__init__()
        self.lorder = 10
        self.rorder = 2
        self.lstride = 1
        self.rstride = 1
        self.stack_layer = stack_layer
        self.linear = LinearTransform(250, 128)
        self.fsmn_block = FSMNBlock()
        self.affine = AffineTransform(128, 250)
        self.relu = RectifiedLinear(250, 250)

    def forward(self, input: torch.Tensor, cache: Dict[str, torch.Tensor] = None):
        x1 = self.linear(input)  # B T D
        if cache is not None:
            cache_layer_name = "cache_layer_{}".format(self.stack_layer)
            if cache_layer_name not in cache:
                cache[cache_layer_name] = torch.zeros(
                    x1.shape[0], x1.shape[-1], (self.lorder - 1) * self.lstride
                )
            x2, cache[cache_layer_name] = self.fsmn_block(x1, cache[cache_layer_name])
        else:
            x2, _ = self.fsmn_block(x1, None)
        x3 = self.affine(x2)
        x4 = self.relu(x3)
        return x4


class BasicBlock_export(nn.Module):
    """使用 cache：每层一个 in_cache (Tensor)，返回 out_cache，供下一块作为 in_cache。"""

    def __init__(
        self,
        model,
    ):
        super(BasicBlock_export, self).__init__()
        self.linear = model.linear
        self.fsmn_block = model.fsmn_block
        self.affine = model.affine
        self.relu = model.relu

    def forward(
        self,
        input: torch.Tensor,
        in_cache: torch.Tensor,
        right_context: torch.Tensor = None,
    ):
        x = self.linear(input)  # B T D
        x, out_cache = self.fsmn_block(x, in_cache, right_context)
        x = self.affine(x)
        x = self.relu(x)
        return x, out_cache


class FsmnStack(nn.Sequential):
    def __init__(self, *args):
        super(FsmnStack, self).__init__(*args)

    def forward(self, input: torch.Tensor, cache: Dict[str, torch.Tensor]):
        x = input
        for module in self._modules.values():
            x = module(x, cache)
        return x


"""
FSMN net for keyword spotting
input_dim:              input dimension
linear_dim:             fsmn input dimensionll
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
"""


class FSMN(nn.Module):
    """固定: 400→140→250, proj 128, lorder=10 rorder=2, 4 层, 输出 2599"""

    def __init__(self):
        super().__init__()
        self.input_dim = 400
        self.output_dim = 2599
        self.in_linear1 = AffineTransform(400, 140)
        self.in_linear2 = AffineTransform(140, 250)
        self.relu = RectifiedLinear(250, 250)
        self.fsmn = FsmnStack(*[BasicBlock(i) for i in range(4)])
        self.out_linear1 = AffineTransform(250, 140)
        self.out_linear2 = AffineTransform(140, 2599)
        self.use_softmax = False
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=-1)

    def fuse_modules(self):
        pass

    def output_size(self) -> int:
        return self.output_dim

    def forward(
        self, input: torch.Tensor, cache: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): Input tensor (B, T, D)
            cache: when cache is not None, the forward is in streaming. The type of cache is a dict, egs,
            {'cache_layer_1': torch.Tensor(B, T1, D)}, T1 is equal to self.lorder. It is {} for the 1st frame
        """

        x1 = self.in_linear1(input)
        x2 = self.in_linear2(x1)
        x3 = self.relu(x2)
        x4 = self.fsmn(x3, cache)  # self.cache will update automatically in self.fsmn
        x5 = self.out_linear1(x4)
        x6 = self.out_linear2(x5)

        if self.use_softmax:
            x7 = self.softmax(x6)
            return x7

        return x6


class FSMNExport(nn.Module):
    def __init__(
        self,
        model,
        **kwargs,
    ):
        super().__init__()

        # self.input_dim = input_dim
        # self.input_affine_dim = input_affine_dim
        # self.fsmn_layers = fsmn_layers
        # self.linear_dim = linear_dim
        # self.proj_dim = proj_dim
        # self.output_affine_dim = output_affine_dim
        # self.output_dim = output_dim
        #
        # self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        # self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        # self.relu = RectifiedLinear(linear_dim, linear_dim)
        # self.fsmn = FsmnStack(*[BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i) for i in
        #                         range(fsmn_layers)])
        # self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        # self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        # self.softmax = nn.Softmax(dim=-1)

        self.in_linear1 = model.in_linear1
        self.in_linear2 = model.in_linear2
        self.relu = model.relu
        # self.fsmn = model.fsmn
        self.out_linear1 = model.out_linear1
        self.out_linear2 = model.out_linear2
        # self.softmax = model.softmax
        self.fsmn = model.fsmn
        for i, d in enumerate(model.fsmn):
            if isinstance(d, BasicBlock):
                self.fsmn[i] = BasicBlock_export(d)

    def fuse_modules(self):
        pass

    def forward(
        self,
        input: torch.Tensor,
        *args,
        right_context: torch.Tensor = None,
    ):
        """
        Args:
            input (B, T, D)
            *args: per-layer in_cache (B, proj_dim, left_ctx) - Conv1d 版本，无最后一维
            right_context: 可选 (B, 2, D)，流式时由调用方传入下一段前 2 帧
        """
        x = self.in_linear1(input)
        x = self.in_linear2(x)
        x = self.relu(x)
        rc_proj_list = None
        if right_context is not None:
            rc = self.in_linear1(right_context)
            rc = self.in_linear2(rc)
            rc = self.relu(rc)
            # Conv1d 版本：right_context 保持 (B, T_rc, D)，在 FSMNBlock 内部 permute
            rc_proj_list = [d.linear(rc) for d in self.fsmn]  # (B, T_rc, D)
        out_caches = list()
        for i, d in enumerate(self.fsmn):
            rc_i = rc_proj_list[i] if rc_proj_list is not None else None
            x, out_cache = d(x, args[i], rc_i)
            out_caches.append(out_cache)
        x = self.out_linear1(x)
        x = self.out_linear2(x)
        return x, out_caches


def convert_conv2d_to_conv1d_weights(model, checkpoint_path=None):
    """
    将模型中的 Conv2d 权重转换为 Conv1d（用于从旧 checkpoint 加载）。
    如果提供了 checkpoint_path，会从 checkpoint 加载并转换权重后加载到模型。
    Conv2d weight: (out_channels, in_channels/groups, kernel_h, kernel_w) = (128, 1, 10, 1)
    Conv1d weight: (out_channels, in_channels/groups, kernel_w) = (128, 1, 10)
    转换方法：squeeze(-1) 去掉最后一维
    """
    if checkpoint_path and os.path.isfile(checkpoint_path):
        # 从 checkpoint 加载并转换
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        # 从 state_dict 中转换权重
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'fsmn_block.conv_left.weight' in key or 'fsmn_block.conv_right.weight' in key:
                if value.dim() == 4:  # Conv2d weight
                    new_state_dict[key] = value.squeeze(-1)  # 转换为 Conv1d
                else:
                    new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict, strict=False)
    else:
        # 直接转换模型中的权重（如果已经是 Conv2d）
        for name, module in model.named_modules():
            if isinstance(module, FSMNBlock):
                if isinstance(module.conv_left, nn.Conv2d):
                    conv2d_weight_left = module.conv_left.weight.data  # (128, 1, 10, 1)
                    conv1d_weight_left = conv2d_weight_left.squeeze(-1)  # (128, 1, 10)
                    conv1d_left = nn.Conv1d(
                        module.dim, module.dim, module.lorder,
                        dilation=module.lstride, groups=module.dim, bias=False,
                    )
                    conv1d_left.weight.data = conv1d_weight_left
                    module.conv_left = conv1d_left
                    
                if isinstance(module.conv_right, nn.Conv2d):
                    conv2d_weight_right = module.conv_right.weight.data  # (128, 1, 2, 1)
                    conv1d_weight_right = conv2d_weight_right.squeeze(-1)  # (128, 1, 2)
                    conv1d_right = nn.Conv1d(
                        module.dim, module.dim, module.rorder,
                        dilation=module.rstride, groups=module.dim, bias=False,
                    )
                    conv1d_right.weight.data = conv1d_weight_right
                    module.conv_right = conv1d_right
    return model


class FsmnKWS(torch.nn.Module):
    """FSMN KWS，参数在 FSMN/ BasicBlock/ FSMNBlock 内写死"""

    def __init__(self):
        super().__init__()
        self.encoder = FSMN()


WavFrontend_config = {
    "fs": 16000,
    "window": "hamming",
    "n_mels": 80,
    "frame_length": 25,
    "frame_shift": 10,
    "lfr_m": 5,
    "lfr_n": 3,
    "cmvn_file": "/root/volume/ctc/ctc_tflite_micro/res/am.mvn.dim80_l2r2",
}
from torch.nn.utils.rnn import pad_sequence
from wav_frontend import WavFrontend
frontend = WavFrontend(**WavFrontend_config)


def extract_fbank(data, data_len=None, data_type: str = "sound", frontend=None, **kwargs):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        data_len = [data.shape[1]] if data_len is None else data_len
    elif isinstance(data, torch.Tensor):
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        data_len = [data.shape[1]] if data_len is None else data_len
    elif isinstance(data, (list, tuple)):
        data_list, data_len = [], []
        for data_i in data:
            if isinstance(data_i, np.ndarray):
                data_i = torch.from_numpy(data_i)
            data_list.append(data_i)
            data_len.append(data_i.shape[0])
        data = pad_sequence(data_list, batch_first=True)  # data: [batch, N]

    data, data_len = frontend(data, data_len, **kwargs)

    if isinstance(data_len, (list, tuple)):
        data_len = torch.tensor([data_len])
    return data.to(torch.float32), data_len.to(torch.int32)

if __name__ == "__main__":
    model = AutoModel(
        model="/root/volume/ctc/speech_charctc_kws_phone-xiaoyun",
        keywords="小云小云",
        output_dir="./outputs/debug",
        device="cpu",
        disable_update=True,
    )
    kwargs = model.kwargs
    model_kws = FsmnKWS()
    model_kws.eval()
    # print("kwargs", kwargs)

    load_pretrained_model(
        model=model_kws,
        path="/root/volume/ctc/ctc_tflite_micro/python/finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt",
        ignore_init_mismatch=True,  # 允许 Conv2d->Conv1d 的形状不匹配
        oss_bucket=None,
        scope_map=[],
        excludes=["ctc"],
    )
    # 自动转换 Conv2d 权重到 Conv1d（从 checkpoint 加载）
    convert_conv2d_to_conv1d_weights(model_kws, checkpoint_path="/root/volume/ctc/ctc_tflite_micro/python/finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt")

    model_kws_export = FsmnKWS()
    model_kws_export.eval()
    test_wavs = [
        "/root/volume/ctc/ctc_tflite_micro/python/example_kws/wav/20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_21.wav",
        "/root/volume/ctc/ctc_tflite_micro/python/test_xiaoyun.wav",
    ]

    load_pretrained_model(
        model=model_kws_export,
        path="/root/volume/ctc/ctc_tflite_micro/python/finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt",
        ignore_init_mismatch=True,
        oss_bucket=None,
        scope_map=[],
        excludes=["ctc"],
    )
    convert_conv2d_to_conv1d_weights(model_kws_export, checkpoint_path="/root/volume/ctc/ctc_tflite_micro/python/finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt")

    with torch.no_grad():
        # 保证误差可复现与一致：固定随机种子；统一 float32、避免混合精度带来额外差异
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        model_kws.float()
        model_kws_export.float()
        # 可选：强制确定性（部分 op 可能变慢或不可用）
        # torch.use_deterministic_algorithms(True, warn_only=True)
        tokenizer = kwargs["tokenizer"]
        keywords = kwargs["keywords"]
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
        model_export = FSMNExport(model=model_kws_export.encoder)
        num_layers = 4
        proj_dim = 128
        left_ctx = 9   # (lorder-1)*lstride
        right_ctx = 2  # rorder*rstride
        delay_frames = 2
        chunk_size = 10

        for test_wav in test_wavs:
            print("\n========== 测试音频:", test_wav, "==========")
            audio_sample_list = load_audio_text_image_video(
                test_wav,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type="sound",
                tokenizer=tokenizer,
            )
            speech, speech_lengths = extract_fbank(
                audio_sample_list,
                data_type="fbank",
                frontend=frontend,
            )
            # 统一 dtype=float32，避免混合精度带来额外差异（与 torch.no_grad() 一起保证误差可复现）
            speech = speech.to(device=kwargs["device"], dtype=torch.float32)
            speech_lengths = speech_lengths.to(device=kwargs["device"])
            B, T, D_in = speech.shape[0], speech.shape[1], speech.shape[2]
            print("speech.shape", speech.shape)

            # --- 流式参数与整段基准（与 FsmnKWS 固定参数一致：lorder=10→left_ctx=9, rorder=2→right_ctx=2）---
            # Conv1d 版本：cache 形状为 (B, proj_dim, left_ctx)，无最后一维
            zero_caches = [
                torch.zeros(B, proj_dim, left_ctx, device=speech.device, dtype=torch.float32)
                for _ in range(num_layers)
            ]

            ref_full_export, _ = model_export(speech, *zero_caches)
            ref_full = model_kws.encoder(speech)
            ref_full = ref_full[0] if isinstance(ref_full, tuple) else ref_full

            # --- 流式流程：手动传入 right_context（非末块传 speech[:, end:end+2, :]）；输出延迟两帧 ---
            ref_chunks = []
            caches = zero_caches
            start = 0
            while start < T:
                end = min(start + chunk_size, T)
                chunk = speech[:, start:end, :]
                if end < T:
                    rc = speech[:, end : end + right_ctx, :]
                    if rc.shape[1] < right_ctx:
                        rc = F.pad(rc, [0, 0, 0, right_ctx - rc.shape[1]])  # 末段不足 2 帧时零填充
                else:
                    rc = None
                logits, caches = model_export(chunk, *caches, right_context=rc)
                ref_chunks.append(logits)
                start = end
            raw_stream = torch.cat(ref_chunks, dim=1)[:, :T, :]
            ref_stream = torch.cat(
                [raw_stream[:, :1, :].expand(-1, delay_frames, -1), raw_stream[:, : T - delay_frames, :]],
                dim=1,
            )  # 延迟 2 帧：ref_stream[t] = raw_stream[t-2]（t>=2），前 2 帧用首帧填充

            # --- 对比与解码 ---
            diff_max = (ref_full_export - raw_stream).abs().max().item()
            print("整段 vs 流式(%d帧/块, %d块) max diff = %.4e" % (chunk_size, len(ref_chunks), diff_max))
            x_full = ref_full[0, : speech_lengths[0], :]
            x_stream = ref_stream[0, : speech_lengths[0], :]  # 流式解码用延迟后的输出（历史第二帧）
            print("decode(整段):", kws_decoder.decode(x_full))
            print("decode(流式):", kws_decoder.decode(x_stream))
