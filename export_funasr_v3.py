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

from omegaconf import DictConfig, ListConfig
from funasr.utils.misc import deep_update
from funasr.register import tables
from funasr.utils.load_utils import load_bytes
from funasr.download.file import download_from_url
from funasr.utils.timestamp_tools import timestamp_sentence
from funasr.utils.timestamp_tools import timestamp_sentence_en
from funasr.download.download_model_from_hub import download_model
from funasr.utils.vad_utils import slice_padding_audio_samples
from funasr.utils.vad_utils import merge_vad
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr.utils import export_utils
from funasr.utils import misc
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.datadir_writer import DatadirWriter
from python.ctc_decoder import KwsCtcPrefixDecoder
import re
import logging
from python.ctc_e import CTC
import torch
import torchaudio
from python.model import FSMN, WavFrontend
model = AutoModel(
    model="iic/speech_charctc_kws_phone-xiaoyun",
    keywords="小云小云",
    output_dir="./outputs/debug",
    device='cpu'
)

test_wav = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"


def deep_update(original, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            if len(value) == 0:
                original[key] = value
            deep_update(original[key], value)
        else:
            original[key] = value



def prepare_data_iterator(data_in, input_len=None, data_type=None, key=None):
    """ """
    data_list = []
    key_list = []
    filelist = [".scp", ".txt", ".json", ".jsonl", ".text"]

    chars = string.ascii_letters + string.digits
    if isinstance(data_in, str):
        if data_in.startswith("http://") or data_in.startswith("https://"):  # url
            data_in = download_from_url(data_in)

    if isinstance(data_in, str) and os.path.exists(
        data_in
    ):  # wav_path; filelist: wav.scp, file.jsonl;text.txt;
        _, file_extension = os.path.splitext(data_in)
        file_extension = file_extension.lower()
        if file_extension in filelist:  # filelist: wav.scp, file.jsonl;text.txt;
            with open(data_in, encoding="utf-8") as fin:
                for line in fin:
                    key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                    if data_in.endswith(".jsonl"):  # file.jsonl: json.dumps({"source": data})
                        lines = json.loads(line.strip())
                        data = lines["source"]
                        key = data["key"] if "key" in data else key
                    else:  # filelist, wav.scp, text.txt: id \t data or data
                        lines = line.strip().split(maxsplit=1)
                        data = lines[1] if len(lines) > 1 else lines[0]
                        key = lines[0] if len(lines) > 1 else key

                    data_list.append(data)
                    key_list.append(key)
        else:
            if key is None:
                # key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                key = misc.extract_filename_without_extension(data_in)
            data_list = [data_in]
            key_list = [key]
    elif isinstance(data_in, (list, tuple)):
        if data_type is not None and isinstance(data_type, (list, tuple)):  # mutiple inputs
            data_list_tmp = []
            for data_in_i, data_type_i in zip(data_in, data_type):
                key_list, data_list_i = prepare_data_iterator(
                    data_in=data_in_i, data_type=data_type_i
                )
                data_list_tmp.append(data_list_i)
            data_list = []
            for item in zip(*data_list_tmp):
                data_list.append(item)
        else:
            # [audio sample point, fbank, text]
            data_list = data_in
            key_list = []
            for data_i in data_in:
                if isinstance(data_i, str) and os.path.exists(data_i):
                    key = misc.extract_filename_without_extension(data_i)
                else:
                    if key is None:
                        key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                key_list.append(key)

    else:  # raw text; audio sample point, fbank; bytes
        if isinstance(data_in, bytes):  # audio bytes
            data_in = load_bytes(data_in)
        if key is None:
            key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
        data_list = [data_in]
        key_list = [key]

    return key_list, data_list

from torch.cuda.amp import autocast
def fsmn_kws_inference(
    self,
    data_in,
    key: list=None,
    tokenizer=None,
    # frontend=None,
    **kwargs,
):

    frontend_conf={'fs': 16000, 'window': 'hamming', 'n_mels': 80, 'frame_length': 25, 'frame_shift': 10, 'lfr_m': 5, 'lfr_n': 3, 'cmvn_file': '/mnt/workspace/.cache/modelscope/iic/speech_charctc_kws_phone-xiaoyun/funasr/am.mvn.dim80_l2r2'}
    frontend=WavFrontend(**frontend_conf)
    ctc_conf={'dropout_rate': 0.0, 'ctc_type': 'builtin', 'reduce': True, 'ignore_nan_grad': True, 'extra_linear': False}
    vocab_size=2599
    ctc = CTC(
        odim=vocab_size, encoder_output_size=2599, **ctc_conf
    )
    encoder_conf={'input_dim': 400, 'input_affine_dim': 140, 'fsmn_layers': 4, 'linear_dim': 250, 'proj_dim': 128, 'lorder': 10, 'rorder': 2, 'lstride': 1, 'rstride': 1, 'output_affine_dim': 140, 'output_dim': 2599, 'use_softmax': False}

    print("frontend", frontend.fs)
    keywords = kwargs.get("keywords")
    kws_decoder = KwsCtcPrefixDecoder(
        ctc=ctc,
        keywords=keywords,
        token_list=tokenizer.token_list,
        seg_dict=tokenizer.seg_dict,
    )
    data_type="sound"
    audio_fs=16000
    device="cpu"

    meta_data = {}
    # extract fbank feats
    audio_sample_list = load_audio_text_image_video(data_in, fs=frontend.fs, audio_fs=audio_fs, data_type=data_type, tokenizer=tokenizer)
    speech, speech_lengths = extract_fbank(audio_sample_list, data_type=data_type, frontend=frontend)
    meta_data["batch_data_time"] = speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
    speech = speech.to(device=device)
    speech_lengths = speech_lengths.to(device=device)
    # Encoder
    # encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)


    # encoder_out = self.encoder(speech)
    # encoder_out_lens = speech_lengths

    model1 = FsmnKWS(encoder="FSMN",encoder_conf=encoder_conf,ctc_conf=ctc_conf,input_size=400,vocab_size=2599)
    print(model1)

    # 加载参数
    init_param = "/mnt/workspace/.cache/modelscope/iic/speech_charctc_kws_phone-xiaoyun/funasr/finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt"
    if init_param is not None:
        if os.path.exists(init_param):
            logging.info(f"Loading pretrained params from {init_param}")
            load_pretrained_model(
                model=model1,
                path=init_param,
                ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
                oss_bucket=kwargs.get("oss_bucket", None),
                scope_map=kwargs.get("scope_map", []),
                excludes=kwargs.get("excludes", None),
            )
        else:
            print(f"error, init_param does not exist!: {init_param}")
    model1.eval()
    encoder_out = model1.encoder(speech) 


    encoder_out_lens = speech_lengths

    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]
    results = []
    if kwargs.get("output_dir") is not None:
        if not hasattr(self, "writer"):
            self.writer = DatadirWriter(kwargs.get("output_dir"))
    for i in range(encoder_out.size(0)):
        x = encoder_out[i, :encoder_out_lens[i], :]
        detect_result = kws_decoder.decode(x)
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


def inference(self,model=None):
    kwargs = self.kwargs
    deep_update(kwargs, {})
    # print(kwargs)
    model.eval()
    key_list = ['kws_xiaoyunxiaoyun'] 
    data_list = ['/root/volume/ctc/speech_charctc_kws_phone-xiaoyun/unittest/example_kws/wav/20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_47.wav']
    speed_stats = {}
    asr_result_list = []
    time_speech_total = 0.0
    time_escape_total = 0.0
    batch = {"data_in": data_list[0], "key": key_list[0]}
    print(batch)
    time1 = time.perf_counter()
    with torch.no_grad():
        # res = model.inference(**batch, **kwargs)
        res = fsmn_kws_inference(self = model, **batch, **kwargs)
        print(res)
        if isinstance(res, (list, tuple)):
            results = res[0] if len(res) > 0 else [{"text": ""}]
            meta_data = res[1] if len(res) > 1 else {}
    time2 = time.perf_counter()
    asr_result_list.extend(results)
    # batch_data_time = time_per_frame_s * data_batch_i["speech_lengths"].sum().item()
    batch_data_time = meta_data.get("batch_data_time", -1)
    time_escape = time2 - time1
    speed_stats["load_data"] = meta_data.get("load_data", 0.0)
    speed_stats["extract_feat"] = meta_data.get("extract_feat", 0.0)
    speed_stats["forward"] = f"{time_escape:0.3f}"
    speed_stats["batch_size"] = f"{len(results)}"
    speed_stats["rtf"] = f"{(time_escape) / batch_data_time:0.3f}"
    time_speech_total += batch_data_time
    time_escape_total += time_escape
    torch.cuda.empty_cache()
    return asr_result_list

# print(model.model)



# print(model.model)
# res = inference(model, model=model.model)

# # res = model.generate(input=test_wav, cache={},)
# print(res)

from python.model import FsmnKWS


kwargs = model.kwargs

deep_update(kwargs, {})
tokenizer = kwargs.get("tokenizer")

encoder_conf={'input_dim': 400, 'input_affine_dim': 140, 'fsmn_layers': 4, 'linear_dim': 250, 'proj_dim': 128, 'lorder': 10, 'rorder': 2, 'lstride': 1, 'rstride': 1, 'output_affine_dim': 140, 'output_dim': 2599, 'use_softmax': False}
ctc_conf={'dropout_rate': 0.0, 'ctc_type': 'builtin', 'reduce': True, 'ignore_nan_grad': True, 'extra_linear': False}
vocab_size=2599

ctc = CTC(
    odim=vocab_size, encoder_output_size=2599, **ctc_conf
)


model1 = FsmnKWS(encoder="FSMN",encoder_conf=encoder_conf,ctc_conf=ctc_conf,input_size=400,vocab_size=2599)
# print(model1)
# 加载参数
init_param = "/root/volume/ctc/speech_charctc_kws_phone-xiaoyun/funasr/basetrain_fsmn_4e_l10r2_250_128_fdim80_t2599.pt"
# init_param = "/mnt/workspace/.cache/modelscope/iic/speech_charctc_kws_phone-xiaoyun/funasr/finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt"
load_pretrained_model(
    model=model1,
    path=init_param,
    ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
    oss_bucket=kwargs.get("oss_bucket", None),
    scope_map=kwargs.get("scope_map", []),
    excludes=kwargs.get("excludes", None),
)
model1.eval()

from torch.nn.utils.rnn import pad_sequence


def extract_fbank1(data, data_len=None, data_type: str = "sound", frontend=None, **kwargs):

    print("ssssss 1", len(data.shape))
    if len(data.shape) < 2:
        data = data[None, :]  # data: [batch, N]
    data_len = [data.shape[1]] if data_len is None else data_len
    data, data_len = frontend(data, data_len, **kwargs)
    return data.to(torch.float32), data_len.to(torch.int32)

data_in="test_xiaoyun.wav"
data_type="sound"
audio_fs=16000
device="cpu"
meta_data = {}
# extract fbank feats
audio_sample_list, audio_fs = torchaudio.load(data_in)
if kwargs.get("reduce_channels", True):
    audio_sample_list = audio_sample_list.mean(0)
print(audio_sample_list)

frontend_conf={'fs': 16000, 'window': 'hamming', 'n_mels': 80, 'frame_length': 25, 'frame_shift': 10, 'lfr_m': 5, 'lfr_n': 3, 'cmvn_file': '/mnt/workspace/.cache/modelscope/iic/speech_charctc_kws_phone-xiaoyun/funasr/am.mvn.dim80_l2r2'}
frontend=WavFrontend(**frontend_conf)

speech, speech_lengths = extract_fbank1(audio_sample_list, data_type=data_type, frontend=frontend)
meta_data["batch_data_time"] = speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
speech = speech.to(device=device)
speech_lengths = speech_lengths.to(device=device)

# print(speech,speech_lengths)

# Encoder
encoder_out = model1.encoder(speech) 


encoder_out_lens = speech_lengths
# print(encoder_out)

keywords="小云小云"
key='kws_xiaoyunxiaoyun'
kws_decoder = KwsCtcPrefixDecoder(
        ctc=ctc,
        keywords=keywords,
        token_list=tokenizer.token_list,
        seg_dict=tokenizer.seg_dict,
    )

results = []
print("encoder_out.size(0)",encoder_out.size(0))
for i in range(encoder_out.size(0)):
    x = encoder_out[i, :encoder_out_lens[i], :]
    detect_result = kws_decoder.decode(x)
    print(detect_result)
    is_deted, det_keyword, det_score = detect_result[0], detect_result[1], detect_result[2]
    print("is_deted", is_deted)
    if is_deted:
        # self.writer["detect"][key[i]] = "detected " + det_keyword + " " + str(det_score)
        det_info = "detected " + det_keyword + " " + str(det_score)
    else:
        # self.writer["detect"][key[i]] = "rejected"
        det_info = "rejected"
    result_i = {"key": key[i], "text": det_info}
    results.append(result_i)
print("results", results)
# res = inference(model, model=model.model)


