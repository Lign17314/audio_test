#!/bin/bash

# Run streaming FBank + Encoder (16x8 quantized) + KWS Decoder
# This version includes real-time keyword spotting for "小云小云"

echo "========================================="
echo "Streaming FBank + Encoder + KWS Decoder"
echo "========================================="
echo ""
echo "Model: 16x8 quantized (INT16 activations, INT8 weights)"
echo "Keyword: 小云小云"
echo "Detection method: State machine (SimpleKwsDecoder)"
echo ""

# Build if needed
# if [ ! -f "build/streaming_fbank_30ms_threaded_16x8_int16_kws" ]; then
    echo "Building..."
    cmake -S . -B build
    cmake --build build --target streaming_fbank_30ms_threaded_16x8_int16_kws
    echo ""
# fi

# Run
./build/streaming_fbank_30ms_threaded_16x8_int16_kws \
    ../res/output_16k_mono.wav \
    ../res/tflite_models/fsmn_encoder_stateful_16x8.tflite

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
