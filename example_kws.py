# coding = utf-8

import os

from modelscope.utils.hub import read_config
from modelscope.utils.hub import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
import torch.utils.data as data  # 新增：导入DataLoader

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