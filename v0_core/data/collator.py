import torch

def meta_collate_fn(batch):
    all_prompts = []
    all_labels = []
    metadata = [] 
    current_start = 0
    for item in batch:
        t_len = len(item['prompts'])
        all_prompts.extend(item['prompts'])
        all_labels.append(item['labels'])
        metadata.append({
            'start': current_start,
            'len': t_len,
            'split': item['split_idx'],
            'q_ids': item['q_ids'],
            'pair_ids': item['pair_ids'],
            'pair_types': item['pair_types'],
            'key': item['key'],
            'stats': item['stats']
        })
        current_start += t_len
    return {
        'flat_prompts': all_prompts,
        'flat_labels': torch.cat(all_labels),
        'metadata': metadata
    }
