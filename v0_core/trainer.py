import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import wandb

# =============================================================================
# 聚类: Feature保留梯度，Label进行平滑
# =============================================================================
def cluster_and_select_prototypes(X_sup, y_sup, n_clusters, seed=42):
    """
    对 Support Set 进行聚类。
    - X (Feature): 选择距离聚类中心最近的真实样本 (Prototype)，以保留梯度流。
    - y (Label): 计算该聚类簇内所有样本 Label 的平均值 (Label Smoothing / Denoising)。
    """
    # 边界情况：如果样本数少于聚类数，无法聚类，直接返回原数据
    if len(X_sup) <= n_clusters:
        return X_sup, y_sup
    
    # 1. 转移到 CPU 进行 KMeans 计算 (sklearn 仅支持 CPU)
    # X_sup: [N, D]
    X_np = X_sup.detach().cpu().numpy()
    device = X_sup.device
    
    # 2. 运行 MiniBatchKMeans
    # batch_size=256 保证速度，random_state 保证在同一 Step 下结果可复现
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, 
        random_state=seed, 
        batch_size=256,
        n_init='auto'
    )
    kmeans.fit(X_np)
    
    # 获取聚类结果
    cluster_labels = kmeans.labels_  # 每个样本所属的簇 ID
    centers = kmeans.cluster_centers_ # 簇中心坐标
    
    selected_indices = []
    new_y_values = []
    
    # 3. 遍历每个簇，提取代表性 X 和 平均 y
    unique_labels = np.unique(cluster_labels)
    
    for k in unique_labels:
        # 找到属于簇 k 的所有样本的索引
        indices_in_cluster = np.where(cluster_labels == k)[0]
        
        if len(indices_in_cluster) == 0:
            continue

        # --- 处理 y (Label): 计算该簇所有样本的平均值 ---
        # 这一步实现了降噪：簇内 8个正例 2个负例 -> Label 变成 0.8
        y_in_cluster = y_sup[indices_in_cluster]
        avg_y = y_in_cluster.float().mean()
        new_y_values.append(avg_y)
        
        # --- 处理 X (Feature): 找到距离该簇中心最近的样本索引 ---
        # 取出该簇所有样本的 Feature
        X_in_cluster = X_np[indices_in_cluster]
        center_k = centers[k].reshape(1, -1)
        
        # 计算距离中心最近的样本
        local_closest_idx, _ = pairwise_distances_argmin_min(center_k, X_in_cluster)
        local_idx = local_closest_idx[0]
        
        # 映射回全局索引，以便从原始 Tensor 中切片
        global_idx = indices_in_cluster[local_idx]
        selected_indices.append(global_idx)
    
    # 4. 组装结果
    # X 使用索引切片，保留了原始计算图，确保梯度可以反向传播到 Qwen Backbone
    X_selected = X_sup[selected_indices]
    
    # y 是新计算的数值，stack 回 Tensor
    y_selected = torch.stack(new_y_values).to(device)
    
    return X_selected, y_selected

# =============================================================================
# 训练与评估 Loop
# =============================================================================
def run_epoch(phase, dataloader, model, tabpfn, optimizer=None, scheduler=None, criterion=None,
              train_infer_bs=16, eval_infer_bs=32,
              reduce_method='avg_pool', target_dim=1024, grad_accum_steps=1, is_master=True,
              epoch_idx=0, wandb_interval=1, max_grad_norm=1.0, 
              loss_type='ce_hard', kl_temperature=1.0, loss_alpha=0.5, loss_balance=False,
              context_clustering=False, context_num_clusters=32):
    """
    Revised run_epoch:
    1. Removes batch-level AUC calculation.
    2. Collects all predictions, labels, and metadata for post-epoch global evaluation.
    """
    is_train = (phase == 'train')
    infer_bs = train_infer_bs if is_train else eval_infer_bs

    if is_train: 
        model.train()
        if optimizer: optimizer.zero_grad()
    else: 
        model.eval()
    
    total_loss = 0.0
    steps = 0
    device = model.device
    
    local_results = []
    
    if is_master: pbar = tqdm(dataloader, desc=phase, leave=False)
    else: pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        flat_prompts = batch['flat_prompts']
        flat_labels = batch['flat_labels'].to(device)
        metadata = batch['metadata']
        
        with torch.set_grad_enabled(is_train):
            all_features = model(flat_prompts, batch_size=infer_bs) 
        
        physical_batch_loss_tensor = 0.0 
        valid_tasks_in_batch = 0
        for i, task_info in enumerate(metadata):
            start = task_info['start']
            end = start + task_info['len']
            split = task_info['split']
            
            # Extract metadata
            q_ids = task_info['q_ids']
            pair_ids = task_info['pair_ids']
            dataset_name, model_name, step_val = task_info['key']
            stats = task_info['stats']

            task_feats = all_features[start:end]
            task_y = flat_labels[start:end]
            
            X_sup, y_sup = task_feats[:split], task_y[:split]
            X_que, y_que = task_feats[split:], task_y[split:]
            
            if len(torch.unique(y_sup)) < 2 or len(y_que) == 0: continue

            if not is_train:
                X_sup = X_sup.detach()
                y_sup = y_sup.detach()
            
            if context_clustering and len(X_sup) > context_num_clusters:
                # 使用 Step 值作为随机种子，保证同一 Step 的聚类结果稳定                
                X_sup, y_sup = cluster_and_select_prototypes(
                    X_sup, y_sup, 
                    n_clusters=context_num_clusters, 
                    seed=int(step_val)
                )

            # 使用离散标签 (Long, 0/1)，适用于标准 Classification
            y_sup_hard = (y_sup >= 0.5).long()
            tabpfn.fit(X_sup, y_sup_hard)

            if is_train:
                logits = tabpfn.forward(X_que, use_inference_mode=False, return_logits=True)
            else:
                with torch.no_grad():
                    logits = tabpfn.forward(X_que, use_inference_mode=True, return_logits=True)
            
            if logits.ndim == 3: logits = logits.squeeze(0)

            is_batch_pairwise = task_info.get('is_pairwise', False)
            actual_loss_type = 'ce_hard' if (loss_type in ['pairwise', 'combined'] and not is_batch_pairwise) else loss_type
            # Pairwise Loss
            if actual_loss_type == 'combined':
                # --- Part A: Pairwise Loss (学习相对排序) ---
                # 1. 计算 Score (Log-Odds Difference)
                scores = logits[:, 1] - logits[:, 0]
                
                # 2. Reshape to pairs (Batch Size 必定为偶数)
                if scores.shape[0] % 2 != 0:
                    scores = scores[:-1]
                
                pos_scores = scores[0::2]
                neg_scores = scores[1::2]
                
                diff = pos_scores - neg_scores
                # Pairwise Loss Component
                loss_pairwise = -F.logsigmoid(diff).mean()

                # --- Part B: CE Soft Loss (学习绝对数值) ---
                # 重新计算 Context 权重 (为了 CE 部分的类别平衡)
                n_pos = stats['n_pos']
                n_neg = stats['n_neg']
                if loss_balance and (n_pos > 0 and n_neg > 0):
                    n_total = n_pos + n_neg
                    w_pos = n_total / (2.0 * n_pos)
                    w_neg = n_total / (2.0 * n_neg)
                else:
                    w_pos, w_neg = 1.0, 1.0
                
                # y_que 已经是通过 label_strategy 处理过的
                sample_weights = torch.where(
                    y_que >= 0.5, 
                    torch.tensor(w_pos, device=device, dtype=logits.dtype), 
                    torch.tensor(w_neg, device=device, dtype=logits.dtype)
                )

                # 构造 Soft Targets [1-y, y]
                target_probs = torch.stack([1.0 - y_que, y_que], dim=1).to(device)
                
                # CE Soft Loss Component
                per_sample_ce = F.cross_entropy(logits, target_probs, reduction='none')
                loss_ce = (per_sample_ce * sample_weights).mean()

                # --- Part C: Combine ---                
                task_loss = loss_alpha * loss_pairwise + (1.0 - loss_alpha) * loss_ce

            elif actual_loss_type == 'pairwise':
                # 检查数据是否支持 Pairwise (由 Dataset 保证 Train 是 Pairwise 的)
                # Dataset 构造保证了 batch 顺序是 [Pos, Neg, Pos, Neg...]
                # 且 Batch Size 必定是偶数
                
                # 1. 计算 Score (Log-Odds Difference)
                # s = log(P(1)) - log(P(0)) = log(e^z1 / sum) - log(e^z0 / sum) = z1 - z0
                scores = logits[:, 1] - logits[:, 0]
                
                # 2. Reshape to pairs
                # 假设 batch_size = 2N, 结果为 [N, 2] -> (Pos_Score, Neg_Score)
                # 偶数索引是 Pos, 奇数索引是 Neg
                # 安全检查：确保 tensor 长度是偶数
                if scores.shape[0] % 2 != 0:
                    scores = scores[:-1]
                
                pos_scores = scores[0::2]
                neg_scores = scores[1::2]
                
                # 3. Bradley-Terry Loss: -log(sigmoid(s_pos - s_neg))
                # 使用 log_sigmoid 更加数值稳定: log(1 / (1 + exp(-x))) = -softplus(-x)
                # s_pos > s_neg, 即 diff > 0
                diff = pos_scores - neg_scores
                
                # Average over batch
                task_loss = -F.logsigmoid(diff).mean()
                
            # Pointwise Losses
            else:
                # --- Context-Aware Re-weighting Loss Calculation ---
                # 计算加权系数
                n_pos = stats['n_pos']
                n_neg = stats['n_neg']
                if loss_balance and (n_pos > 0 and n_neg > 0):
                    n_total = n_pos + n_neg
                    w_pos = n_total / (2.0 * n_pos)
                    w_neg = n_total / (2.0 * n_neg)
                else:
                    w_pos, w_neg = 1.0, 1.0
                
                # 构造样本级权重张量 (Shape: [Batch_Query])
                # y_que >= 0.5 视为正样本，反之为负
                sample_weights = torch.where(
                    y_que >= 0.5, 
                    torch.tensor(w_pos, device=device, dtype=logits.dtype), 
                    torch.tensor(w_neg, device=device, dtype=logits.dtype)
                )

                # 计算 Loss (应用权重)
                if actual_loss_type == 'ce_soft':
                    # Soft Label: Manual BCE with weighting
                    target_probs = torch.stack([1.0 - y_que, y_que], dim=1).to(device)
                    # F.cross_entropy with soft targets requires PyTorch >= 1.10
                    # 如果传入的 criterion 是默认 reduction='mean'，这里需要用 functional 强制为 none
                    # 为了安全，这里统一使用 Functional 接口
                    per_sample_loss = F.cross_entropy(logits, target_probs, reduction='none')
                    task_loss = (per_sample_loss * sample_weights).mean()

                elif actual_loss_type == 'ce_hard':
                    y_que_hard = (y_que >= 0.5).long()
                    per_sample_loss = F.cross_entropy(logits, y_que_hard, reduction='none')
                    task_loss = (per_sample_loss * sample_weights).mean()

                elif actual_loss_type == 'kl_div':
                    # KL 散度加权相对直接：给 KL 项乘权重
                    target_probs = torch.stack([1.0 - y_que, y_que], dim=1).to(device)
                    log_probs = F.log_softmax(logits / kl_temperature, dim=1)
                    # F.kl_div 默认是 mean，需要 none
                    # KL output shape [Batch, 2] -> sum over classes -> [Batch]
                    per_sample_kl = F.kl_div(log_probs, target_probs, reduction='none').sum(dim=1) * (kl_temperature ** 2)
                    task_loss = (per_sample_kl * sample_weights).mean()

            # -----------------------------------------------------------

            if is_train: physical_batch_loss_tensor += task_loss
            valid_tasks_in_batch += 1

            loss_val = task_loss.item()
            total_loss += loss_val
            steps += 1
            
            # --- Collection for Global Metrics ---
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                probs_np = probs.cpu().numpy()[:, 1]
                preds_np = preds.cpu().numpy()
                y_que_np = y_que.cpu().numpy()
                
                for idx_q in range(len(y_que)):
                    entry = {
                        "id": q_ids[idx_q],
                        "dataset": dataset_name,
                        "model": model_name,
                        "step": step_val,
                        "pred": int(preds_np[idx_q]),
                        "prob": float(probs_np[idx_q]),
                        "label": float(y_que_np[idx_q]),
                        "epoch": epoch_idx
                    }
                    entry["pair_id"] = pair_ids[idx_q] if idx_q < len(pair_ids) else None
                    local_results.append(entry)

            # WandB logging
            if is_master and steps > 0 and (steps % wandb_interval == 0):
                 cur_lr = scheduler.get_last_lr()[0] if scheduler else 0.0
                 wandb.log({
                    f"{phase}/batch_loss": loss_val,
                    f"{phase}/lr": cur_lr,
                    "epoch": epoch_idx
                 })
        
        # --- Backprop ---
        if is_train and valid_tasks_in_batch > 0:
            final_loss_to_backward = physical_batch_loss_tensor / (valid_tasks_in_batch * grad_accum_steps)
            final_loss_to_backward.backward()
            if (batch_idx + 1) % grad_accum_steps == 0:
                if max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler: scheduler.step()
                optimizer.zero_grad()
        
        del all_features, logits

        if is_master and steps > 0:
            pbar.set_description(f"{phase} | Loss:{total_loss/steps:.4f}")

    if is_train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    if hasattr(tabpfn, 'X_'): tabpfn.X_ = None
    if hasattr(tabpfn, 'y_'): tabpfn.y_ = None

    final_avg_loss = total_loss / steps if steps > 0 else 0.0
    
    return final_avg_loss, local_results
