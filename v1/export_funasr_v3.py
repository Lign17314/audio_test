import torch
import torch
import numpy as np
from funasr.train_utils.load_pretrained_model import load_pretrained_model
import torch
import torchaudio
from python.model import FsmnKWS
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from collections import defaultdict
import math
from typing import List, Tuple


def is_sublist(main_list, check_list):
    if len(main_list) < len(check_list):
        return -1
    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1
    for i in range(len(main_list) - len(check_list)):
        if main_list[i] == check_list[0]:
            for j in range(len(check_list)):
                if main_list[i + j] != check_list[j]:
                    break
            else:
                return i
    else:
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


def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """

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
        else:  # process last LFR frame
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
    """CTC prefix beam search inner implementation
    Args:
        logits (torch.Tensor): (1, max_len, vocab_size)
        logits_lengths (torch.Tensor): (1, )
        keywords_tokenset (set): token set for filtering score
        score_beam_size (int): beam size for score
        path_beam_size (int): beam size for path
    Returns:
        List[List[int]]: nbest results
    """
    maxlen = logits.size(0)
    ctc_probs = logits
    cur_hyps = [(tuple(), (1.0, 0.0, []))]
    print("maxlen", maxlen)
    # CTC beam search step by step
    for t in range(0, maxlen):
        probs = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (0.0, 0.0, []))
        # 2.1 First beam prune: select topk best
        top_k_probs, top_k_index = probs.topk(score_beam_size)  # (score_beam_size,)
        # filter prob score that is too small
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
                if s == 0:  # blank
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pb = n_pb + pb * ps + pnb * ps
                    nodes = cur_nodes.copy()
                    next_hyps[prefix] = (n_pb, n_pnb, nodes)
                elif s == last:
                    if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                        # Update *ss -> *s;
                        n_pb, n_pnb, nodes = next_hyps[prefix]
                        n_pnb = n_pnb + pnb * ps
                        nodes = cur_nodes.copy()
                        if ps > nodes[-1]["prob"]:  # update frame and prob
                            nodes[-1]["prob"] = ps
                            nodes[-1]["frame"] = t
                        next_hyps[prefix] = (n_pb, n_pnb, nodes)
                    if not math.isclose(pb, 0.0, abs_tol=0.000001):
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb, nodes = next_hyps[n_prefix]
                        n_pnb = n_pnb + pb * ps
                        nodes = cur_nodes.copy()
                        nodes.append(
                            dict(token=s, frame=t, prob=ps)
                        )  # to record token prob
                        next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                else:
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    if nodes:
                        if ps > nodes[-1]["prob"]:  # update frame and prob
                            nodes[-1]["prob"] = ps
                            nodes[-1]["frame"] = t
                    else:
                        nodes = cur_nodes.copy()
                        nodes.append(
                            dict(token=s, frame=t, prob=ps)
                        )  # to record token prob
                    n_pnb = n_pnb + pb * ps + pnb * ps
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
        # 2.2 Second beam prune
        next_hyps = sorted(
            next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True
        )
        cur_hyps = next_hyps[:path_beam_size]
    hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in cur_hyps]
    return hyps


cmvn_file = "/root/volume/ctc/train/am.mvn.dim80_l2r2"
init_param = "/root/volume/ctc/train/work_dir/avg_10_funasr.pt"
data_in = "/root/volume/ctc/train/example_kws/wav/20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_13.wav"
data_type = "sound"
audio_fs = 16000
device = "cpu"
meta_data = {}
reduce_channels = True

# 前端特征提取配置（适配小云小云KWS场景）
fs = 16000  # 音频采样率
window = "hamming"  # 窗函数类型（汉明窗）
n_mels = 80  # FBank梅尔滤波器数量（80维）
frame_length = 25  # 帧长（ms）
frame_shift = 10  # 帧移（ms）
lfr_m = 5  # LFR帧合并数（5帧合并）
lfr_n = 3  # LFR帧移（每3帧取一次）
dither = 1
snip_edges = True
keywords_idxset = {0, 1462, 976}
lab = (1462, 976, 1462, 976)


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
# print(model)
# 加载参数

load_pretrained_model(
    model=model,
    path=init_param,
    ignore_init_mismatch=True,
    oss_bucket=None,
    scope_map=[],
    excludes=None,
)
model.eval()

# extract fbank feats
audio_sample_list, audio_fs = torchaudio.load(data_in)
if reduce_channels == True:
    audio_sample_list = audio_sample_list.mean(0)
print(
    "audio_sample_list",
    audio_sample_list,
    "audio_sample_list.shape",
    audio_sample_list.shape,
)
if len(audio_sample_list.shape) < 2:  # 里面只有一个情况下扩充成两个
    print("len(audio_sample_list.shape)", len(audio_sample_list.shape))
    audio_sample_list = audio_sample_list[None, :]  # data: [batch, N]
data_len = [audio_sample_list.shape[1]]


feats = []
feats_lens = []
cmvn = load_cmvn(cmvn_file)
waveform_length = data_len[0]
waveform = audio_sample_list[0][:waveform_length]
# upsacle_samples
print("waveform", waveform)
# 转换成16 位整型 PCM
waveform = waveform * (1 << 15)
print("waveform_length", waveform_length)
waveform = waveform.unsqueeze(0)
# 提取fbank
mat = kaldi.fbank(
    waveform,
    num_mel_bins=n_mels,
    frame_length=min(frame_length, waveform_length / fs * 1000),
    frame_shift=frame_shift,
    dither=dither,
    energy_floor=0.0,
    window_type=window,
    sample_frequency=fs,
    snip_edges=snip_edges,
)
mat = apply_lfr(mat, lfr_m, lfr_n)
mat = apply_cmvn(mat, cmvn)
feat_length = mat.size(0)
feats.append(mat)
feats_lens.append(feat_length)
feats_lens = torch.as_tensor(feats_lens)
feats_pad = feats[0][None, :, :]

audio_sample_list = feats_pad
data_len = feats_lens

speech = audio_sample_list.to(torch.float32)
speech_lengths = data_len.to(torch.int32)

meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000


# Encoder
encoder_out = model.encoder(speech)


encoder_out_lens = speech_lengths
# print(encoder_out)

print("encoder_out.size(0)", encoder_out.size(0))
x = encoder_out[0, : encoder_out_lens[0], :]

raw_logp = F.softmax(x.unsqueeze(0), dim=2).detach().squeeze(0).cpu()
xlen = torch.tensor([raw_logp.size(1)])



hyps = beam_search(
    logits=raw_logp, logits_lengths=xlen, keywords_tokenset=keywords_idxset
)

print(hyps)
prefix_ids = hyps[0][0]
# path_score = one_hyp[1]
prefix_nodes = hyps[0][2]

isHit = 0

offset = is_sublist(prefix_ids, lab)
hit_score = 1.0
if offset != -1:
    for idx in range(offset, offset + len(lab)):
        hit_score *= prefix_nodes[idx]["prob"]
        isHit = 1
hit_score = math.sqrt(hit_score)
print("hit_score", hit_score, isHit)
