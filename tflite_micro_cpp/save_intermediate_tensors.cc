// 修改 run_encoder.cc 来保存中间层输出
// 这个文件展示了如何修改代码来保存中间层 tensor

// 在 run_encoder.cc 的 Invoke() 调用之前和之后，添加以下代码：

/*
// 保存中间层输出（在 Invoke() 之前）
const auto* subgraph = model->subgraphs()->Get(0);
size_t num_tensors = subgraph->tensors()->size();

// 在 Invoke() 之前保存所有 tensor 的值
std::vector<std::vector<float>> tensors_before(num_tensors);
for (size_t i = 0; i < num_tensors; i++) {
    TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(i, 0);
    if (eval_tensor && eval_tensor->data.raw && eval_tensor->type == kTfLiteFloat32) {
        float* data = eval_tensor->data.f;
        size_t num_elements = 1;
        if (eval_tensor->dims) {
            for (int j = 0; j < eval_tensor->dims->size; j++) {
                num_elements *= eval_tensor->dims->data[j];
            }
        }
        tensors_before[i].resize(num_elements);
        memcpy(tensors_before[i].data(), data, num_elements * sizeof(float));
    }
}

// 运行推理
status = interpreter.Invoke();

// 在 Invoke() 之后保存所有 tensor 的值
std::vector<std::vector<float>> tensors_after(num_tensors);
for (size_t i = 0; i < num_tensors; i++) {
    TfLiteEvalTensor* eval_tensor = interpreter.GetTensor(i, 0);
    if (eval_tensor && eval_tensor->data.raw && eval_tensor->type == kTfLiteFloat32) {
        float* data = eval_tensor->data.f;
        size_t num_elements = 1;
        if (eval_tensor->dims) {
            for (int j = 0; j < eval_tensor->dims->size; j++) {
                num_elements *= eval_tensor->dims->data[j];
            }
        }
        tensors_after[i].resize(num_elements);
        memcpy(tensors_after[i].data(), data, num_elements * sizeof(float));
    }
}

// 找出哪些 tensor 在 Invoke() 后发生了变化（即算子的输出）
// 这些就是中间层的输出
*/
