/* Copyright 2025
 * Test program for TFLite Micro library
 * 
 * This program tests if the compiled TFLite Micro library can:
 * 1. Load a TFLite model from file
 * 2. Create an interpreter
 * 3. Allocate tensors
 * 4. Run inference
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 读取文件到内存
bool ReadFile(const char* filename, std::vector<uint8_t>& buffer) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    MicroPrintf("Failed to open file: %s\n", filename);
    return false;
  }
  
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  buffer.resize(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    MicroPrintf("Failed to read file: %s\n", filename);
    return false;
  }
  
  MicroPrintf("Successfully loaded model file: %s (%zu bytes)\n", filename, size);
  return true;
}

// 打印张量信息
void PrintTensorInfo(const char* name, const TfLiteTensor* tensor) {
  if (!tensor) {
    MicroPrintf("%s tensor: (null)\n", name);
    return;
  }
  
  MicroPrintf("%s tensor info:\n", name);
  MicroPrintf("  name: %s\n", tensor->name ? tensor->name : "(null)");
  MicroPrintf("  type: %d\n", tensor->type);
  MicroPrintf("  bytes: %zu\n", tensor->bytes);
  
  if (tensor->dims) {
    MicroPrintf("  dims: [");
    for (int i = 0; i < tensor->dims->size; i++) {
      MicroPrintf("%d", tensor->dims->data[i]);
      if (i < tensor->dims->size - 1) MicroPrintf(", ");
    }
    MicroPrintf("]\n");
  } else {
    MicroPrintf("  dims: (null)\n");
  }
  
  // 打印量化参数（如果有）
  if (tensor->quantization.type == kTfLiteAffineQuantization && tensor->quantization.params) {
    const TfLiteAffineQuantization* quant_params =
        static_cast<const TfLiteAffineQuantization*>(tensor->quantization.params);
    if (quant_params && quant_params->scale && quant_params->scale->size > 0) {
      MicroPrintf("  quantization scale: %f\n", quant_params->scale->data[0]);
      if (quant_params->zero_point && quant_params->zero_point->size > 0) {
        MicroPrintf("  quantization zero_point: %d\n", quant_params->zero_point->data[0]);
      }
    }
  }
}

int main(int argc, char** argv) {
  const char* model_filename = nullptr;
  
  // 解析命令行参数
  if (argc < 2) {
    MicroPrintf("Usage: %s <model.tflite>\n", argv[0]);
    MicroPrintf("Example: %s tflite_models/fsmn_encoder_float32.tflite\n", argv[0]);
    return 1;
  }
  model_filename = argv[1];
  
  MicroPrintf("========================================\n");
  MicroPrintf("TFLite Micro Library Test\n");
  MicroPrintf("========================================\n");
  MicroPrintf("Model file: %s\n", model_filename);
  MicroPrintf("\n");
  
  // 1. 读取模型文件
  std::vector<uint8_t> model_data;
  if (!ReadFile(model_filename, model_data)) {
    MicroPrintf("ERROR: Failed to read model file\n");
    return 1;
  }
  
  // 2. 验证模型格式
  const tflite::Model* model = tflite::GetModel(model_data.data());
  if (!model) {
    MicroPrintf("ERROR: Failed to get model from data\n");
    return 1;
  }
  MicroPrintf("Model schema version: %d\n", model->version());
  
  // 3. 创建操作解析器（添加常用的操作）
  // 注意：根据模型需要添加相应的操作
  tflite::MicroMutableOpResolver<50> resolver;
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddTranspose();
  resolver.AddStridedSlice();
  resolver.AddDequantize();
  resolver.AddQuantize();
  resolver.AddPad();
  resolver.AddConcatenation();
  resolver.AddSoftmax();
  resolver.AddMean();
  resolver.AddSum();
  resolver.AddMaximum();
  resolver.AddMinimum();
  resolver.AddSub();
  resolver.AddDiv();
  resolver.AddSqrt();
  resolver.AddRsqrt();
  resolver.AddSquare();
  resolver.AddLogistic();
  resolver.AddTanh();
  resolver.AddExp();
  resolver.AddLog();
  resolver.AddAbs();
  resolver.AddNeg();
  resolver.AddSin();
  resolver.AddCos();
  resolver.AddRound();
  resolver.AddFloor();
  resolver.AddCeil();
  resolver.AddSlice();
  resolver.AddSplit();
  resolver.AddUnpack();
  resolver.AddPack();
  resolver.AddGather();
  resolver.AddSelectV2();
  resolver.AddSqueeze();
  resolver.AddExpandDims();
  resolver.AddCast();
  resolver.AddShape();
  resolver.AddFill();
  resolver.AddZerosLike();
  MicroPrintf("Created MicroMutableOpResolver with common ops\n");
  
  // 4. 分配内存区域（tensor arena）
  // 注意：实际使用中需要根据模型大小调整
  constexpr int kTensorArenaSize = 1024 * 1024;  // 1MB
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
  MicroPrintf("Allocated tensor arena: %d bytes\n", kTensorArenaSize);
  
  // 5. 创建解释器
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize);
  MicroPrintf("Created MicroInterpreter\n");
  
  // 6. 分配张量
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("ERROR: AllocateTensors() failed with status %d\n", allocate_status);
    return 1;
  }
  MicroPrintf("Allocated tensors successfully\n");
  
  // 7. 打印输入输出信息
  MicroPrintf("\n");
  MicroPrintf("Input tensors: %zu\n", interpreter.inputs_size());
  for (size_t i = 0; i < interpreter.inputs_size(); i++) {
    PrintTensorInfo("Input", interpreter.input(i));
  }
  
  MicroPrintf("\n");
  MicroPrintf("Output tensors: %zu\n", interpreter.outputs_size());
  for (size_t i = 0; i < interpreter.outputs_size(); i++) {
    PrintTensorInfo("Output", interpreter.output(i));
  }
  
  // 8. 设置输入数据（使用零初始化作为测试）
  MicroPrintf("\n");
  MicroPrintf("Setting input data (zeros for testing)...\n");
  for (size_t i = 0; i < interpreter.inputs_size(); i++) {
    TfLiteTensor* input_tensor = interpreter.input(i);
    if (input_tensor && input_tensor->data.raw) {
      memset(input_tensor->data.raw, 0, input_tensor->bytes);
      MicroPrintf("  Input %zu: initialized with zeros (%zu bytes)\n", 
                  i, input_tensor->bytes);
    } else {
      MicroPrintf("  Input %zu: (null tensor or data)\n", i);
    }
  }
  
  // 9. 运行推理
  MicroPrintf("\n");
  MicroPrintf("Running inference...\n");
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("ERROR: Invoke() failed with status %d\n", invoke_status);
    return 1;
  }
  MicroPrintf("Inference completed successfully!\n");
  
  // 10. 打印输出数据（前几个值）
  MicroPrintf("\n");
  MicroPrintf("Output data (first 10 values):\n");
  for (size_t i = 0; i < interpreter.outputs_size(); i++) {
    TfLiteTensor* output_tensor = interpreter.output(i);
    if (!output_tensor || !output_tensor->data.raw) {
      MicroPrintf("  Output %zu: (null tensor or data)\n", i);
      continue;
    }
    
    MicroPrintf("  Output %zu:\n", i);
    
    // 根据数据类型打印
    int num_values = std::min(10, static_cast<int>(output_tensor->bytes / 
                          (output_tensor->type == kTfLiteFloat32 ? sizeof(float) :
                           output_tensor->type == kTfLiteInt8 ? sizeof(int8_t) :
                           output_tensor->type == kTfLiteInt16 ? sizeof(int16_t) : 1)));
    
    if (output_tensor->type == kTfLiteFloat32 && output_tensor->data.f) {
      float* data = output_tensor->data.f;
      for (int j = 0; j < num_values; j++) {
        MicroPrintf("    [%d] = %f\n", j, data[j]);
      }
    } else if (output_tensor->type == kTfLiteInt8 && output_tensor->data.int8) {
      int8_t* data = output_tensor->data.int8;
      for (int j = 0; j < num_values; j++) {
        MicroPrintf("    [%d] = %d\n", j, static_cast<int>(data[j]));
      }
    } else if (output_tensor->type == kTfLiteInt16 && output_tensor->data.i16) {
      int16_t* data = output_tensor->data.i16;
      for (int j = 0; j < num_values; j++) {
        MicroPrintf("    [%d] = %d\n", j, static_cast<int>(data[j]));
      }
    } else {
      MicroPrintf("    (type %d, unsupported for printing)\n", output_tensor->type);
    }
  }
  
  // 11. 打印内存使用情况
  MicroPrintf("\n");
  MicroPrintf("Memory usage:\n");
  MicroPrintf("  Tensor arena size: %d bytes\n", kTensorArenaSize);
  MicroPrintf("  Used: %zu bytes\n", interpreter.arena_used_bytes());
  MicroPrintf("  Available: %zu bytes\n", 
              kTensorArenaSize - interpreter.arena_used_bytes());
  
  MicroPrintf("\n");
  MicroPrintf("========================================\n");
  MicroPrintf("Test completed successfully!\n");
  MicroPrintf("========================================\n");
  
  return 0;
}
