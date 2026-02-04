import os
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.distributed as dist

def append_jsonl(path, data):
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error appending to jsonl: {e}")

# =============================================================================
# Global Metrics Calculation & Aggregation
# =============================================================================
def calculate_metrics_by_group(all_results, phase, epoch, is_master=True, output_dir=None, dataset_name_tag="", avg_loss=None):

    # 1. Gather from all ranks
    world_size = dist.get_world_size()
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, all_results)
    
    if not is_master:
        return {}

    # Flatten list of lists
    flat_results = []
    for rank_res in gathered_results:
        flat_results.extend(rank_res)
    
    print(f"[{phase}] Collected {len(flat_results)} samples for evaluation.")
    
    if len(flat_results) == 0:
        return {}

    metrics_summary = {"epoch": epoch}
    # =========================================================================
    # Part A: Pair-wise Metrics Calculation (Global)
    # =========================================================================
    
    pair_grouping = defaultdict(lambda: {'pos': [], 'neg': []})
    
    for r in flat_results:
        pid = r.get('pair_id')
        if pid is not None:
            if r['label'] >= 0.5:
                pair_grouping[pid]['pos'].append(r)
            else:
                pair_grouping[pid]['neg'].append(r)

    valid_pairs = [] 
    
    for pid, group in pair_grouping.items():
        if len(group['pos']) == 1 and len(group['neg']) == 1:
            valid_pairs.append((group['pos'][0], group['neg'][0]))
    
    total_valid_pairs = len(valid_pairs)
    strict_pair_correct_count = 0
    rlhf_pair_correct_count = 0
    
    # 3. Calculate Global Pair Metrics
    for pos_item, neg_item in valid_pairs:
        # Strict
        if (pos_item['pred'] == 1) and (neg_item['pred'] == 0):
            strict_pair_correct_count += 1
        # RLHF
        if pos_item['prob'] > neg_item['prob']:
            rlhf_pair_correct_count += 1

    metrics_summary[f"{phase}/global_strict_pair_acc"] = strict_pair_correct_count / total_valid_pairs if total_valid_pairs > 0 else -1
    metrics_summary[f"{phase}/global_rlhf_pair_acc"] = rlhf_pair_correct_count / total_valid_pairs if total_valid_pairs > 0 else -1
    metrics_summary[f"{phase}/num_valid_pairs"] = total_valid_pairs

    # =========================================================================
    # Part B: Standard Global Metrics (Acc / AUC)
    # =========================================================================
    y_true_binary = [1 if r['label'] >= 0.5 else 0 for r in flat_results]
    y_scores = [r['prob'] for r in flat_results]
    y_preds = [r['pred'] for r in flat_results]

    def get_auc_strict(y_t, y_s):
        try:
            return roc_auc_score(y_t, y_s) if len(set(y_t)) > 1 else -1
        except:
            return -1
    
    g_auc = get_auc_strict(y_true_binary, y_scores)
    metrics_summary[f"{phase}/global_acc"] = accuracy_score(y_true_binary, y_preds)
    metrics_summary[f"{phase}/global_auc"] = g_auc

    if avg_loss is not None:
        metrics_summary[f"{phase}/loss"] = avg_loss

    # =========================================================================
    # Part C: Step-wise Metrics
    # =========================================================================
    step_groups = defaultdict(list)
    for r in flat_results:
        step_groups[r['step']].append(r)
    
    step_valid_pairs = defaultdict(list)
    for pos_item, neg_item in valid_pairs:
        if pos_item['step'] == neg_item['step']:
            step_valid_pairs[pos_item['step']].append((pos_item, neg_item))

    print(f"[{phase}] Calculating metrics for {len(step_groups)} distinct steps...")

    gauc_weighted_sum = 0.0
    gauc_total_weight = 0.0
    valid_gauc_steps = 0

    step_details_list = []

    for step_val, items in step_groups.items():
        s_true = [1 if x['label'] >= 0.5 else 0 for x in items]
        s_scores = [x['prob'] for x in items]
        s_preds = [x['pred'] for x in items]
        
        # 1. Basic Step Metrics
        step_acc = accuracy_score(s_true, s_preds)
        step_auc = get_auc_strict(s_true, s_scores) # Returns None if only 1 class
        
        step_record = {
            "step": step_val,
            "count": len(items),
            "acc": step_acc,
            "auc": step_auc
        }
        
        if step_auc != -1:
            weight = len(items) 
            gauc_weighted_sum += step_auc * weight
            gauc_total_weight += weight
            valid_gauc_steps += 1
        
        # 3. Step Pair Metrics
        pairs_in_step = step_valid_pairs.get(step_val, [])
        n_pairs = len(pairs_in_step)
        
        if n_pairs > 0:
            s_strict_corr = sum(1 for p, n in pairs_in_step if (p['pred'] == 1 and n['pred'] == 0))
            s_rlhf_corr = sum(1 for p, n in pairs_in_step if p['prob'] > n['prob'])
            
            step_record["pair_count"] = n_pairs
            step_record["strict_pair_acc"] = s_strict_corr / n_pairs
            step_record["rlhf_pair_acc"] = s_rlhf_corr / n_pairs
        else:
            step_record["pair_count"] = 0
            step_record["strict_pair_acc"] = -1
            step_record["rlhf_pair_acc"] = -1

        step_details_list.append(step_record)

    # Calculate Weighted gAUC
    final_gauc = gauc_weighted_sum / gauc_total_weight if gauc_total_weight > 0 else -1

    metrics_summary[f"{phase}/gAUC"] = final_gauc
    metrics_summary[f"{phase}/gAUC_valid_steps"] = valid_gauc_steps
    
    print(f"[{phase}] gAUC: {final_gauc:.4f} (Computed over {valid_gauc_steps} valid steps out of {len(step_groups)})")

    # =========================================================================
    # Part D: Save Logs
    # =========================================================================
    if output_dir:
        # 1. Save Raw Predictions (Keep as is)
        log_filename = f"{phase}_predictions_epoch_{epoch}{dataset_name_tag}.jsonl"
        log_path = os.path.join(output_dir, log_filename)
        valid_pair_ids = set(p[0]['pair_id'] for p in valid_pairs)
        
        print(f"Saving raw predictions to {log_path}...")
        with open(log_path, 'w', encoding='utf-8') as f:
            for item in flat_results:
                item['is_valid_pair_part'] = item.get('pair_id') in valid_pair_ids
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        # 2. Save Global Metrics (Only summary)
        metric_filename = "all_metrics.jsonl"
        metric_path = os.path.join(output_dir, metric_filename)
        append_jsonl(metric_path, metrics_summary)

        # 3. [NEW] Save Step-wise Details to a separate file
        step_log_filename = f"{phase}_step_metrics_epoch_{epoch}{dataset_name_tag}.jsonl"
        step_log_path = os.path.join(output_dir, step_log_filename)
        print(f"Saving step-wise metrics to {step_log_path}...")
        
        # Sort by step for readability
        step_details_list.sort(key=lambda x: x['step'] if isinstance(x['step'], int) else -1)
        
        with open(step_log_path, 'w', encoding='utf-8') as f:
            for item in step_details_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return metrics_summary
