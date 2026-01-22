import torch
import re

def rename_ckpt_params(ckpt_path, save_path):
    """
    定制化重命名你的FSMN-CTC权重参数（backbone→encoder），适配FunASR加载
    :param ckpt_path: 原始权重路径（finetune_avg_10.pt）
    :param save_path: 转换后权重保存路径
    """
    # 1. 加载原始权重
    ckpt = torch.load(ckpt_path, map_location="cpu")
    new_ckpt = {}
    
    # 2. 定制化参数名映射规则（完全匹配你的权重结构）
    rename_rules = [
        # 规则1：in_linear1/in_linear2（backbone→encoder）
        (r"^backbone\.in_linear1\.linear\.weight$", "encoder.in_linear1.linear.weight"),
        (r"^backbone\.in_linear1\.linear\.bias$", "encoder.in_linear1.linear.bias"),
        (r"^backbone\.in_linear2\.linear\.weight$", "encoder.in_linear2.linear.weight"),
        (r"^backbone\.in_linear2\.linear\.bias$", "encoder.in_linear2.linear.bias"),
        
        # 规则2：fsmn.N.0.linear.weight → encoder.fsmn.N.linear.linear.weight
        (r"^backbone\.fsmn\.(\d+)\.0\.linear\.weight$", r"encoder.fsmn.\1.linear.linear.weight"),
        
        # 规则3：fsmn.N.1.conv_left/right → encoder.fsmn.N.fsmn_block.conv_left/right
        (r"^backbone\.fsmn\.(\d+)\.1\.conv_left\.weight$", r"encoder.fsmn.\1.fsmn_block.conv_left.weight"),
        (r"^backbone\.fsmn\.(\d+)\.1\.conv_right\.weight$", r"encoder.fsmn.\1.fsmn_block.conv_right.weight"),
        
        # 规则4：fsmn.N.2.linear.weight/bias → encoder.fsmn.N.affine.linear.weight/bias
        (r"^backbone\.fsmn\.(\d+)\.2\.linear\.weight$", r"encoder.fsmn.\1.affine.linear.weight"),
        (r"^backbone\.fsmn\.(\d+)\.2\.linear\.bias$", r"encoder.fsmn.\1.affine.linear.bias"),
        
        # 规则5：out_linear1/out_linear2（若有，补充此规则）
        (r"^backbone\.out_linear1\.linear\.weight$", "encoder.out_linear1.linear.weight"),
        (r"^backbone\.out_linear1\.linear\.bias$", "encoder.out_linear1.linear.bias"),
        (r"^backbone\.out_linear2\.linear\.weight$", "encoder.out_linear2.linear.weight"),
        (r"^backbone\.out_linear2\.linear\.bias$", "encoder.out_linear2.linear.bias"),
        
        # 规则6：ctc层（若有，补充此规则，需根据你的完整参数名调整）
        (r"^ctc_head\.weight$", "ctc.ctc_lo.weight"),
        (r"^ctc_head\.bias$", "ctc.ctc_lo.bias"),
    ]
    
    # 3. 批量重命名参数（保留未匹配的参数，如global_cmvn）
    for old_k, v in ckpt.items():
        new_k = old_k  # 默认保留原名称（如global_cmvn）
        # 遍历所有规则，匹配并替换
        for pattern, repl in rename_rules:
            if re.match(pattern, old_k):
                new_k = re.sub(pattern, repl, old_k)
                break
        new_ckpt[new_k] = v
        # 打印重命名日志，便于验证
        if old_k != new_k:
            print(f"✅ 重命名：{old_k} → {new_k}")
        else:
            print(f"ℹ️ 保留原名称：{old_k}")
    
    # 4. 保存转换后的权重
    torch.save(new_ckpt, save_path)
    print(f"\n📌 转换后的权重已保存到：{save_path}")
    print(f"📌 共处理 {len(new_ckpt)} 个参数")

# ---------------------- 调用脚本（替换为你的路径） ----------------------
if __name__ == "__main__":
    # 你的原始权重路径
    CKPT_PATH = "/root/volume/ctc/train/work_dir/avg_10.pt"
    # 转换后权重保存路径
    SAVE_PATH = "/root/volume/ctc/train/work_dir/avg_10_funasr.pt"
    
    rename_ckpt_params(CKPT_PATH, SAVE_PATH)