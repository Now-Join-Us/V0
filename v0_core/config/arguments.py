import torch
import argparse

# =============================================================================
# 参数解析配置
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generalist Value Model")

    # --- 路径相关 ---
    parser.add_argument("--time_str", type=str, required=True)
    parser.add_argument("--qwen_path", type=str, required=True, help="Qwen 模型路径")
    parser.add_argument("--tabpfn_checkpoint", type=str, required=True, help="TabPFN Checkpoint 路径")
    
    # 数据路径配置
    parser.add_argument("--context_data_paths", type=str, required=True, help="Context Pool Jsonl路径 (支持多个,逗号分隔)")
    parser.add_argument("--train_data_paths", type=str, default=None, help="Train Query Pool Jsonl路径 (支持多个,逗号分隔)")
    parser.add_argument("--eval_data_paths", type=str, default=None, help="Test Query Pool Jsonl路径 (支持多个,逗号分隔)")
    parser.add_argument("--validity_data_paths", type=str, default=None, help="Validity Test Pool Jsonl路径 (支持多个,逗号分隔)")
    
    parser.add_argument("--prompt_dict_path", type=str, required=True, help="Prompt 字典 JSON 路径")
    
    # --- Checkpoint 保存相关 ---
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="模型保存目录")
    parser.add_argument("--save_interval", type=int, default=1, help="每隔多少个 Epoch 保存一次模型")
    parser.add_argument("--max_keep_checkpoints", type=int, default=2, help="最多保留多少个最新的 Checkpoint")
    parser.add_argument("--resume", action="store_true", help="是否尝试从 checkpoint_dir 恢复训练")
    parser.add_argument("--resume_from_specific_epoch", type=int, default=None, help="指定要 resume 的 epoch")

    # --- 日志相关 ---
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10, help="保存间隔")
    parser.add_argument("--metric_path", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="context-v", help="Wandb 项目名称")
    parser.add_argument("--wandb_interval", type=int, default=1, help="Wandb 记录间隔")
    parser.add_argument("--wandb_id", type=str, default=None)

    # --- 运行模式与策略 ---
    parser.add_argument("--run_mode", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--pooling_strategy", type=str, default="dynamic_query", 
                        choices=["last_token", "fixed_query", "dynamic_query"],
                        help="Embedding 提取策略")


    parser.add_argument("--label_strategy", type=str, default="binary", 
                        choices=["binary", "minmax_norm"],
                        help="Label 处理策略")
    parser.add_argument("--loss_type", type=str, default="ce_hard", 
                        choices=["ce_hard", "ce_soft", "kl_div", "pairwise", "combined"],
                        help="Loss 函数类型: combined = pairwise + ce_soft")
    parser.add_argument("--loss_alpha", type=float, default=0.5, 
                        help="Combined Loss 中 Pairwise 的权重 (0.0-1.0)。Total = alpha * Pair + (1-alpha) * CE")
    parser.add_argument("--loss_balance", action="store_true", help="是否对正负样本加权")

    parser.add_argument("--kl_temperature", type=float, default=1.0, 
                        help="KL 散度或 Softmax 的温度系数 T")
    # --- 降维参数 ---
    parser.add_argument("--reduce_method", type=str, default="none", 
                        choices=["none", "avg_pool", "max_pool"])
    parser.add_argument("--target_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument("--context_clustering", action="store_true", help="是否启用 Support Set 聚类筛选")
    parser.add_argument("--context_num_clusters", type=int, default=128, help="聚类保留的原型数量 (k值)")

    # --- 模型超参数 ---
    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--tabpfn_estimators", type=int, default=4)
    parser.add_argument("--dynamic_query_generator_bottleneck_dim", type=int, default=128)
    parser.add_argument("--dynamic_query_generator_dropout_rate", type=float, default=0.2)

    # --- 训练超参数 ---
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--meta_batch_size", type=int, default=1, help="每次forward处理多少个Task(一个Task包含Support+Query)")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    
    parser.add_argument("--train_query_batch_size", type=int, default=8, help="每个Task包含多少个Query样本 (必须来自同一个Step)")
    parser.add_argument("--eval_query_batch_size", type=int, default=8, help="每个Task包含多少个Query样本 (必须来自同一个Step)")
    parser.add_argument("--support_size", type=int, default=256, help="每个Task采样的Context样本数量")

    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--lr_tabpfn", type=float, default=1e-5)
    
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--train_embed_bs", type=int, default=4)
    parser.add_argument("--eval_embed_bs", type=int, default=4)
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    
    def split_paths(path_str):
        if not path_str: return []
        return [p.strip() for p in path_str.split(',') if p.strip()]

    args.context_data_paths = split_paths(args.context_data_paths)
    args.train_data_paths = split_paths(args.train_data_paths)
    args.eval_data_paths = split_paths(args.eval_data_paths)
    args.validity_data_paths = split_paths(args.validity_data_paths)
    args.prompt_dict_path = split_paths(args.prompt_dict_path)

    return args

def print_elegant_args(args):
    """
    打印参数列表
    """
    args_dict = vars(args)
    keys = sorted(args_dict.keys())
    # 计算最长键名以便对齐
    max_k = max([len(k) for k in keys]) if keys else 10
    
    # 定义颜色
    C_KEY = "\033[36m"    # 青色用于键
    C_VALUE = "\033[33m"  # 黄色用于值（如果不想要颜色，设为 "" 即可）
    C_RESET = "\033[0m"   # 重置
    
    print(f"\n{C_VALUE}Arguments:{C_RESET}")
    
    for k in keys:
        val = str(args_dict[k])
        # 格式说明：
        # {k:<{max_k}} : 让键名左对齐并填充空格
        # val          : 完整打印值，不截断
        print(f"  {C_KEY}{k:<{max_k}}{C_RESET} : {val}")
        
    print() # 打印末尾空行
