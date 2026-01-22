from funasr import AutoModel
import torch
import os
import torch.nn as nn
import time

import json
import time
import copy
import torch
import random
import string
import logging
import os.path
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict
import torch.nn.functional as F

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

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lorder=None,
        rorder=None,
        lstride=1,
        rstride=1,
    ):
        super(FSMNBlock, self).__init__()

        self.dim = input_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.conv_left = nn.Conv2d(
            self.dim, self.dim, [lorder, 1], dilation=[lstride, 1], groups=self.dim, bias=False
        )

        if self.rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim, self.dim, [rorder, 1], dilation=[rstride, 1], groups=self.dim, bias=False
            )
        else:
            self.conv_right = None

    def forward(self, input: torch.Tensor, cache: torch.Tensor = None):
        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)  # B D T C

        if cache is not None:
            cache = cache.to(x_per.device)
            y_left = torch.cat((cache, x_per), dim=2)
            cache = y_left[:, :, -(self.lorder - 1) * self.lstride :, :]
        else:
            y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])

        y_left = self.conv_left(y_left)
        out = x_per + y_left

        if self.conv_right is not None:
            # maybe need to check
            y_right = F.pad(x_per, [0, 0, 0, self.rorder * self.rstride])
            y_right = y_right[:, :, self.rstride :, :]
            y_right = self.conv_right(y_right)
            out += y_right

        out_per = out.permute(0, 3, 2, 1)
        output = out_per.squeeze(1)

        return output, cache

class BasicBlock(nn.Module):
    def __init__(
        self,
        linear_dim: int,
        proj_dim: int,
        lorder: int,
        rorder: int,
        lstride: int,
        rstride: int,
        stack_layer: int,
    ):
        super(BasicBlock, self).__init__()
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        self.stack_layer = stack_layer
        self.linear = LinearTransform(linear_dim, proj_dim)
        self.fsmn_block = FSMNBlock(proj_dim, proj_dim, lorder, rorder, lstride, rstride)
        self.affine = AffineTransform(proj_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)

    def forward(self, input: torch.Tensor, cache: Dict[str, torch.Tensor] = None):
        x1 = self.linear(input)  # B T D

        if cache is not None:
            cache_layer_name = 'cache_layer_{}'.format(self.stack_layer)
            if cache_layer_name not in cache:
                cache[cache_layer_name] = torch.zeros(
                    x1.shape[0], x1.shape[-1], (self.lorder - 1) * self.lstride, 1
                )
            x2, cache[cache_layer_name] = self.fsmn_block(x1, cache[cache_layer_name])
        else:
            x2, _ = self.fsmn_block(x1, None)
        x3 = self.affine(x2)
        x4 = self.relu(x3)
        return x4

class FsmnStack(nn.Sequential):
    def __init__(self, *args):
        super(FsmnStack, self).__init__(*args)

    def forward(self, input: torch.Tensor, cache: Dict[str, torch.Tensor]):
        x = input
        for module in self._modules.values():
            x = module(x, cache)
        return x

class FSMN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_affine_dim: int,
        fsmn_layers: int,
        linear_dim: int,
        proj_dim: int,
        lorder: int,
        rorder: int,
        lstride: int,
        rstride: int,
        output_affine_dim: int,
        output_dim: int,
        use_softmax: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_affine_dim = input_affine_dim
        self.fsmn_layers = fsmn_layers
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.output_affine_dim = output_affine_dim
        self.output_dim = output_dim

        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)
        self.fsmn = FsmnStack(
            *[
                BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i)
                for i in range(fsmn_layers)
            ]
        )
        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)

        self.use_softmax = use_softmax
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=-1)

    def fuse_modules(self):
        pass

    def output_size(self) -> int:
        return self.output_dim

    def forward(
        self,
        input: torch.Tensor,
        cache: Dict[str, torch.Tensor] = None
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



import time
import torch
import logging
from torch.cuda.amp import autocast
from typing import Union, Dict, List, Tuple, Optional

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

class FsmnKWS(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
        self,
        specaug: Optional[str] = None,
        specaug_conf: Optional[Dict] = None,
        normalize: str = None,
        normalize_conf: Optional[Dict] = None,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        ctc: str = None,
        ctc_conf: Optional[Dict] = None,
        ctc_weight: float = 1.0,
        input_size: int = 360,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        **kwargs,
    ):
        super().__init__()

        encoder = FSMN(**encoder_conf)
        # print("encoder_conf",encoder_conf)
        encoder_output_size = encoder.output_size()
        if ctc_conf is None:
            ctc_conf = {}
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf
        )

        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight

        # self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.ctc = ctc

        self.error_calculator = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        batch_size = speech.shape[0]

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        # Collect CTC branch stats
        stats = dict()
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
        stats["cer_ctc"] = cer_ctc

        loss = self.ctc_weight * loss_ctc

        stats["cer"] = cer_ctc
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """

        with autocast(False):
            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # Forward encoder
        encoder_out = self.encoder(speech)
        encoder_out_lens = speech_lengths

        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)

        return loss_ctc, cer_ctc

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list=None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        keywords = kwargs.get("keywords")
        from funasr.utils.kws_utils import KwsCtcPrefixDecoder
        self.kws_decoder = KwsCtcPrefixDecoder(
            ctc=self.ctc,
            keywords=keywords,
            token_list=tokenizer.token_list,
            seg_dict=tokenizer.seg_dict,
        )

        meta_data = {}
        if isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank": # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is not None:
                speech_lengths = speech_lengths.squeeze(-1)
            else:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(data_in, fs=frontend.fs, audio_fs=kwargs.get("fs", 16000), data_type=kwargs.get("data_type", "sound"), tokenizer=tokenizer)
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend)
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        results = []
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))

        for i in range(encoder_out.size(0)):
            x = encoder_out[i, :encoder_out_lens[i], :]
            detect_result = self.kws_decoder.decode(x)
            is_deted, det_keyword, det_score = detect_result[0], detect_result[1], detect_result[2]

            if is_deted:
                self.writer["detect"][key[i]] = "detected " + det_keyword + " " + str(det_score)
                det_info = "detected " + det_keyword + " " + str(det_score)
            else:
                self.writer["detect"][key[i]] = "rejected"
                det_info = "rejected"

            result_i = {"key": key[i], "text": det_info}
            results.append(result_i)

        return results, meta_data
