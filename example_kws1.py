# coding = utf-8

import os
import sys
import copy
import datetime
import os
import math
from typing import Dict
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader
from collections import defaultdict
from modelscope.metainfo import Trainers
from modelscope.msdatasets.dataset_cls.custom_datasets.audio.kws_nearfield_dataset import \
    kws_nearfield_dataset
from modelscope.utils.config import Config

from modelscope.utils.hub import read_config
from modelscope.utils.hub import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
import torch.utils.data as data  # 新增：导入DataLoader
from modelscope.utils.config import Config

from kws_utils.det_utils import compute_det
from kws_utils.file_utils import query_token_set, read_lexicon, read_token


def ctc_prefix_beam_search(
    logits: torch.Tensor,
    logits_lengths: torch.Tensor,
    keywords_tokenset: set = None,
    score_beam_size: int = 3,
    path_beam_size: int = 20,
) -> Tuple[List[List[int]], torch.Tensor]:
    """ CTC prefix beam search inner implementation

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
    # ctc_probs = logits.softmax(1)  # (1, maxlen, vocab_size)
    ctc_probs = logits

    cur_hyps = [(tuple(), (1.0, 0.0, []))]

    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        probs = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (0.0, 0.0, []))

        # 2.1 First beam prune: select topk best
        top_k_probs, top_k_index = probs.topk(
            score_beam_size)  # (score_beam_size,)

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
                        if ps > nodes[-1]['prob']:  # update frame and prob
                            nodes[-1]['prob'] = ps
                            nodes[-1]['frame'] = t
                        next_hyps[prefix] = (n_pb, n_pnb, nodes)

                    if not math.isclose(pb, 0.0, abs_tol=0.000001):
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb, nodes = next_hyps[n_prefix]
                        n_pnb = n_pnb + pb * ps
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t,
                                          prob=ps))  # to record token prob
                        next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                else:
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    if nodes:
                        if ps > nodes[-1]['prob']:  # update frame and prob
                            nodes[-1]['prob'] = ps
                            nodes[-1]['frame'] = t
                    else:
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t,
                                          prob=ps))  # to record token prob
                    n_pnb = n_pnb + pb * ps + pnb * ps
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

        # 2.2 Second beam prune
        next_hyps = sorted(
            next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)

        cur_hyps = next_hyps[:path_beam_size]

    hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in cur_hyps]
    return hyps


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


def executor_test(model, data_loader, device, keywords_token, keywords_idxset,
                  args):
    ''' Test model with decoder
    '''
    assert args.get('test_dir', None) is not None, \
        'Please config param: test_dir, to store score file'
    score_abs_path = os.path.join(args['test_dir'], 'score.txt')
    log_interval = args.get('log_interval', 10)

    model.eval()
    infer_seconds = 0.0
    decode_seconds = 0.0
    with torch.no_grad(), open(score_abs_path, 'w', encoding='utf8') as fout:
        for batch_idx, batch in enumerate(data_loader):
            batch_start_time = datetime.datetime.now()

            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            feats_lengths = feats_lengths.to(device)
            if target_lengths is not None:
                target_lengths = target_lengths.to(device)
            num_utts = feats_lengths.size(0)
            if num_utts == 0:
                continue

            logits, _ = model(feats)
            logits = logits.softmax(2)  # (1, maxlen, vocab_size)
            logits = logits.cpu()
            
            infer_end_time = datetime.datetime.now()
            for i in range(len(keys)):
                key = keys[i]
                print(f'key: {key}', f'feats_lengths: {feats_lengths[i]}')
                score = logits[i][:feats_lengths[i]]
                hyps = ctc_prefix_beam_search(score, feats_lengths[i],
                                              keywords_idxset)
                hit_keyword = None
                hit_score = 1.0
                # start = 0; end = 0
                for one_hyp in hyps:
                    prefix_ids = one_hyp[0]
                    # path_score = one_hyp[1]
                    prefix_nodes = one_hyp[2]
                    assert len(prefix_ids) == len(prefix_nodes)
                    for word in keywords_token.keys():
                        lab = keywords_token[word]['token_id']
                        offset = is_sublist(prefix_ids, lab)
                        if offset != -1:
                            hit_keyword = word
                            # start = prefix_nodes[offset]['frame']
                            # end = prefix_nodes[offset+len(lab)-1]['frame']
                            for idx in range(offset, offset + len(lab)):
                                hit_score *= prefix_nodes[idx]['prob']
                            break
                    if hit_keyword is not None:
                        hit_score = math.sqrt(hit_score)
                        break

                if hit_keyword is not None:
                    # fout.write('{} detected [{:.2f} {:.2f}] {} {:.3f}\n'\
                    #          .format(key, start*0.03, end*0.03, hit_keyword, hit_score))
                    fout.write('{} detected {} {:.3f}\n'.format(
                        key, hit_keyword, hit_score))
                else:
                    fout.write('{} rejected\n'.format(key))

            decode_end_time = datetime.datetime.now()
            infer_seconds += (infer_end_time
                              - batch_start_time).total_seconds()
            decode_seconds += (decode_end_time
                               - infer_end_time).total_seconds()

            if batch_idx % log_interval == 0:
                print('Progress batch {}'.format(batch_idx))
                sys.stdout.flush()
        print(
            'Total infer cost {:.2f} mins, decode cost {:.2f} mins'.format(
                infer_seconds / 60.0,
                decode_seconds / 60.0,
            ))

    return score_abs_path

def evaluate(self) -> Dict[str, float]:
        testing_config = {}
        work_dir = "/root/volume/ctc/train/work_dir"
        keywords = '小云小云'
        test_dir = os.path.join(work_dir, 'test_dir')
        # test_scp = './example_kws/test_wav.scp'
        # trans_file = './example_kws/merge_trans.txt'

        # 1. get checkpoint
        model_dir = "/root/volume/ctc/speech_charctc_kws_phone-xiaoyun"
        eval_checkpoint = os.path.join(work_dir,'avg_10.pt')
        testing_config['test_dir'] = test_dir
        if not os.path.exists(testing_config['test_dir']):
            os.makedirs(testing_config['test_dir'])

        kwargs = dict(
            test_dir=test_dir,
            average_num=10,
            gpu=0,  # 注意：你设置了device='cpu'，这里gpu=0会被忽略，不影响
            keywords=keywords,
            batch_size=32,  # 修复：改为和训练一致的32，避免内存溢出
            )
        # 2. get test data and trans
        test_data = './example_kws/test_wav.scp'
        trans_data = './example_kws/merge_trans.txt'
        print(f'test data: {test_data}')
        print(f'trans data: {trans_data}')

        # 3. prepare dataset and dataloader
        test_conf = copy.deepcopy(self.configs['preprocessor'])
        print("test conf", test_conf)
        test_conf['filter_conf']['max_length'] = 102400
        test_conf['filter_conf']['min_length'] = 0
        test_conf['speed_perturb'] = False
        test_conf['spec_aug'] = False
        test_conf['shuffle'] = False
        if kwargs.get('batch_size', None) is not None:
            test_conf['batch_conf']['batch_size'] = kwargs['batch_size']


        # 2. prepare preset files
        token_file = os.path.join(model_dir, 'train/tokens.txt')
        token_table = read_token(token_file)
        lexicon_file = os.path.join(model_dir, 'train/lexicon.txt')
        lexicon_table = read_lexicon(lexicon_file)

        test_dataset = kws_nearfield_dataset(test_data, trans_data, test_conf,
                                             token_table,
                                             lexicon_table, False, '',
                                             False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=None,
            pin_memory=kwargs.get('pin_memory', False),
            persistent_workers=True,
            num_workers=0,
            prefetch_factor=2)

        # 4. parse keywords tokens
        keywords_str = kwargs['keywords']
        keywords_list = keywords_str.strip().replace(' ', '').split(',')
        keywords_token = {}
        keywords_idxset = {0}
        keywords_strset = {'<blk>'}
        keywords_tokenmap = {'<blk>': 0}
        for keyword in keywords_list:
            strs, indexes = query_token_set(keyword, token_table,
                                            lexicon_table)
            keywords_token[keyword] = {}
            keywords_token[keyword]['token_id'] = indexes
            keywords_token[keyword]['token_str'] = ''.join('%s ' % str(i)
                                                           for i in indexes)
            [keywords_strset.add(i) for i in strs]
            [keywords_idxset.add(i) for i in indexes]
            for txt, idx in zip(strs, indexes):
                if keywords_tokenmap.get(txt, None) is None:
                    keywords_tokenmap[txt] = idx

        token_print = ''
        for txt, idx in keywords_tokenmap.items():
            token_print += f'{txt}({idx}) '
        print(f'Token set is: {token_print}', keywords_token)

        # 5. build model and load checkpoint
        # support assign specific gpu device
        # Init kws model from configs

        test_model = self.build_model(self.configs)
        print(test_model)
        checkpoint = torch.load(eval_checkpoint, map_location='cpu')
        state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint[
            'state_dict']
        test_model.load_state_dict(state_dict)

        # 6. executing evaluation and get score file
        print('Start evaluating...')
        totaltime = datetime.datetime.now()
        score_file = executor_test(test_model, test_dataloader,
                                   torch.device('cpu'), keywords_token,
                                   keywords_idxset, testing_config)
        totaltime = datetime.datetime.now() - totaltime
        print('Total time spent: {:.2f} hours'.format(
            totaltime.total_seconds() / 3600.0))

        # 7. compute det statistic file with score file
        det_kwargs = dict(
            keywords=keywords_str,
            test_data=test_data,
            trans_data=trans_data,
            score_file=score_file,
        )
        det_results = compute_det(**det_kwargs)
        print(det_results)
    
    
def print_current_line(msg=""):
    # 关键：用 1 表示「调用这个函数的代码」的栈帧
    caller_frame = sys._getframe(1)
    # 获取调用者的行号、文件名、函数名
    line_no = caller_frame.f_lineno
    file_name = caller_frame.f_code.co_filename
    func_name = caller_frame.f_code.co_name  # 调用者所在的函数名
    # 格式化输出
    print(f"[{file_name}] 函数[{func_name}] 行[{line_no}] {msg}")


from modelscope.utils.config import Config, ConfigDict
from modelscope.models.builder import build_model

def from_pretrained_my(model_name_or_path: str,
                    cfg_dict: Config = None,
                    **kwargs):
    print_current_line()
    local_model_dir = model_name_or_path
    print(f'initialize model from {local_model_dir}')
    cfg = cfg_dict
    task_name = "keyword-spotting"
    model_cfg = getattr(cfg, 'model', ConfigDict())
    # use ms
    print("use ms 11", model_cfg)
    model_cfg.model_dir = local_model_dir
    for k, v in kwargs.items():
        model_cfg[k] = v
    model = build_model(model_cfg, task_name=task_name)
    if hasattr(cfg, 'pipeline'):
        model.pipeline = cfg.pipeline
    if not hasattr(model, 'cfg'):
        model.cfg = cfg
    model_cfg.pop('model_dir', None)
    model.name = model_name_or_path
    model.model_dir = local_model_dir
    return model


def test_fun(trainer, work_dir, test_scp, trans_file, config_file):
    keywords = '小云小云'
    test_dir = os.path.join(work_dir, 'test_dir')

    kwargs = dict(
        test_dir=test_dir,
        test_data=test_scp,
        trans_data=trans_file,
        average_num=10,
        gpu=0,  # 注意：你设置了device='cpu'，这里gpu=0会被忽略，不影响
        keywords=keywords,
        batch_size=32,  # 修复：改为和训练一致的32，避免内存溢出
        )
    # trainer.evaluate(test_checkpoint, None, **kwargs)
    configs = Config.from_file(config_file)
    model_dir = "/root/volume/ctc/speech_charctc_kws_phone-xiaoyun"
    model = from_pretrained_my(
        model_name_or_path=model_dir, cfg_dict=configs, training=True)
    print(model.model)
    pass
def main():
    enable_training = False
    enable_testing = True
    work_dir = './work_dir'

    model_id = 'damo/speech_charctc_kws_phone-xiaoyun'
    model_dir = snapshot_download(model_id)

    configs = read_config(model_id)

    # update some configs
    configs.train.max_epochs = 10
    configs.preprocessor.batch_conf.batch_size = 32
    configs.train.dataloader.workers_per_gpu = 0
    configs.evaluation.dataloader.workers_per_gpu = 0

    config_file = os.path.join(work_dir, 'config.json')
    configs.dump(config_file)

    kwargs = dict(
        model=model_id,
        work_dir=work_dir,
        cfg_file=config_file,
        seed=666,
        local_rank=-1,
        device='cpu',
    )
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '-1'

    trainer = build_trainer(
        Trainers.speech_kws_fsmn_char_ctc_nearfield, default_args=kwargs)

    # ====================== 完善的核心修复代码 ======================
    # 临时重写DataLoader的初始化函数，解决所有多进程参数冲突
    original_dataloader_init = data.DataLoader.__init__
    def custom_dataloader_init(self, *args, **kwargs):
        # 当num_workers=0时，强制禁用所有仅支持多进程的参数
        if kwargs.get('num_workers', 0) == 0:
            kwargs['prefetch_factor'] = None       # 解决prefetch_factor冲突
            kwargs['persistent_workers'] = False   # 解决persistent_workers冲突
        # 执行原始的初始化逻辑
        original_dataloader_init(self, *args, **kwargs)

    # 替换原始的DataLoader初始化函数
    data.DataLoader.__init__ = custom_dataloader_init
    # ====================== 核心修复代码结束 ======================
    
    train_scp = './example_kws/train_wav.scp'
    cv_scp = './example_kws/cv_wav.scp'
    test_scp = './example_kws/test_wav.scp'
    trans_file = './example_kws/merge_trans.txt'

    train_checkpoint = ''
    test_checkpoint = ''

    if enable_training:
        kwargs = dict(
            train_data=train_scp,
            cv_data=cv_scp,
            trans_data=trans_file,
            checkpoint=train_checkpoint,
            tensorboard_dir='tb_test',
            need_dump=True
        )
        trainer.train(**kwargs)

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size > 1 and rank != 0:
        enable_testing = False

    if enable_testing:
        evaluate(trainer)

    enable_testing = False
    if enable_testing:
        keywords = '小云小云'
        test_dir = os.path.join(work_dir, 'test_dir')

        kwargs = dict(
            test_dir=test_dir,
            test_data=test_scp,
            trans_data=trans_file,
            average_num=10,
            gpu=0,  # 注意：你设置了device='cpu'，这里gpu=0会被忽略，不影响
            keywords=keywords,
            batch_size=32,  # 修复：改为和训练一致的32，避免内存溢出
            )
        trainer.evaluate(test_checkpoint, None, **kwargs)
    # 【可选】训练+评估完成后恢复原始DataLoader，避免影响后续代码
    data.DataLoader.__init__ = original_dataloader_init

if __name__ == '__main__':
    main()