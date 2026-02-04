# 1. 提取特征
python prepare_fbank_features.py ../res/test_xiaoyun.wav ./features
python prepare_fbank_features.py ../res/example_kws/wav/20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_21.wav ./features


# tflite模型的导出以及校准
python compare_encoder_backends.py --save-repr-stateful tflite_models/repr_stateful.npz --repr-trans ../res/example_kws/merge_trans.txt
python export_tflite.py --repr-stateful tflite_models/repr_stateful.npz
python compare_encoder_backends.py


# stream tflite 16x8 测试
cd tflite_micro_cpp/streaming_pipe
./run_30ms_threaded_16x8_int16.sh









# 步骤 1: 提取 FBank 特征
cd tflite_micro_cpp/streaming_fbank_only
./run.sh

# 步骤 2: Encoder 推理
cd ../encoder_runner
./run.sh ../tflite_models/fsmn_encoder_stateful_float32.tflite \
         ../streaming_fbank_only/build/streaming_fbank_output.npy \
         streaming_output.npy --skip-compare

# 步骤 3: CTC 解码验证
cd ../..
python3 verify_encoder_output.py \
        tflite_micro_cpp/streaming_fbank_only/build/streaming_fbank_output.npy \
        tflite_micro_cpp/build/streaming_output.npy \
        /root/volume/ctc/train/funasr_test/test_xiaoyun.wav
