import onnx
from collections import Counter

def analyze_onnx_operators(onnx_path):
    """
    分析ONNX模型的算子信息
    参数：onnx_path - ONNX模型文件路径
    """
    # 加载ONNX模型
    try:
        model = onnx.load(onnx_path)
        print(f"✅ 成功加载模型: {onnx_path}")
        print(f"📊 模型基本信息:")
        print(f"   - IR版本: {model.ir_version}")
        print(f"   - OPSET版本: {model.opset_import[0].version}")
        print(f"   - 输入数量: {len(model.graph.input)}")
        print(f"   - 输出数量: {len(model.graph.output)}")
        
        # 1. 统计所有算子类型和数量
        op_types = [node.op_type for node in model.graph.node]
        op_counter = Counter(op_types)
        
        print(f"\n🔍 算子类型及数量:")
        for op_type, count in sorted(op_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {op_type}: {count}个")
        
        # 2. 检查动态算子（If/Loop/Scan）
        dynamic_ops = [op for op in op_types if op in ["If", "Loop", "Scan"]]
        if dynamic_ops:
            print(f"\n⚠️  检测到动态算子: {Counter(dynamic_ops)}")
        else:
            print(f"\n✅ 未检测到动态算子（If/Loop/Scan）")
        
        # 3. 列出所有输入输出节点
        print(f"\n📥 输入节点:")
        for input_node in model.graph.input:
            print(f"   - 名称: {input_node.name}")
            if input_node.type.tensor_type.shape.dim:
                shape = [dim.dim_value if dim.dim_value != 0 else "dynamic" 
                         for dim in input_node.type.tensor_type.shape.dim]
                print(f"     形状: {shape}")
            print(f"     数据类型: {onnx.TensorProto.DataType.Name(input_node.type.tensor_type.elem_type)}")
        
        print(f"\n📤 输出节点:")
        for output_node in model.graph.output:
            print(f"   - 名称: {output_node.name}")
            if output_node.type.tensor_type.shape.dim:
                shape = [dim.dim_value if dim.dim_value != 0 else "dynamic" 
                         for dim in output_node.type.tensor_type.shape.dim]
                print(f"     形状: {shape}")
            print(f"     数据类型: {onnx.TensorProto.DataType.Name(output_node.type.tensor_type.elem_type)}")
        
        # 4. 可选：列出所有算子的详细信息（适合调试）
        # print(f"\n📋 所有算子详细信息:")
        # for i, node in enumerate(model.graph.node):
        #     print(f"   算子{i+1}: {node.op_type} (名称: {node.name})")
        #     print(f"      输入: {node.input}")
        #     print(f"      输出: {node.output}")
        
        return op_counter
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        return None

# 执行分析
if __name__ == "__main__":
    # 替换为你的ONNX模型路径
    onnx_file = "/root/volume/ctc/train/v1/kws_encoder_final.onnx"
    # onnx_file = "kws_encoder_final.onnx"  # 也可以分析原始模型
    analyze_onnx_operators(onnx_file)
    onnx_file = "/root/volume/ctc/train/v1/kws_encoder_final_simplified.onnx"
    analyze_onnx_operators(onnx_file)