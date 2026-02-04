import json
import torch
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import Dataset
from v0_core.data.utils import load_jsonl_lines

# =============================================================================
# 数据与日志工具
# =============================================================================
class ValueModelDataset(Dataset):
    def __init__(self, 
                 context_paths, 
                 query_paths, 
                 prompt_dict_path, 
                 label_strategy='binary', 
                 query_batch_size=8,
                 support_size=256,
                 mode='train'):
        """
        args:
            context_paths: List of paths to context_pool jsonl files
            query_paths: List of paths to query_pool jsonl files (train/test/validity)
            prompt_dict_path: List of paths to prompt dictionaries
            query_batch_size: Number of queries in one forward pass (all from same step)
            support_size: Number of context samples to sample
            mode: 'train' (shuffle queries before chunking) or 'eval' (sequential)
        """
        self.label_strategy = label_strategy
        self.query_batch_size = query_batch_size
        self.support_size = support_size
        self.mode = mode

        # 1. Load Prompt Dictionary
        print(f"Loading prompts from {prompt_dict_path}...")
        self.prompt_map = {}
        for path in prompt_dict_path:
            with open(path, 'r', encoding='utf-8') as f:
                self.prompt_map.update(json.load(f))

        # 2. Load Context Pool and Index it
        # Structure: {(dataset, model, step): [list of sample dicts]}
        print("Loading Context Pool...")
        self.context_pool = defaultdict(list)
        self.context_pool_fallback = defaultdict(list)
        raw_context = load_jsonl_lines(context_paths)
        for item in raw_context:
            key = (item['dataset'], item['model'], item['step'])
            self.context_pool[key].append(item)
            fallback_key = (item['model'], item['step'])
            self.context_pool_fallback[fallback_key].append(item)
        print(f"Loaded Context Pool with {len(self.context_pool)} unique (dataset, model, step) keys.")

        # 3. Load Query Pool
        print(f"Loading Query Pool from {query_paths}...")
        raw_queries = load_jsonl_lines(query_paths)
        
        # 4. Group Queries by Key
        self.queries_by_key = defaultdict(list)
        print("Grouping Queries...")
        for item in raw_queries:
            key = (item['dataset'], item['model'], item['step'])
            # Pre-fetch prompt text to save time later, if ID exists
            s_id_str = f"{item['dataset']}_{item['id']}"
            item['text'] = self.prompt_map[s_id_str]
            self.queries_by_key[key].append(item)
        
        # 5. Pre-calculate Class Statistics for Context-Aware Re-weighting
        # 统计每个Context Key下，Query Pool中的正负样本总数，用于计算加权Loss
        print("Calculating Global Context Statistics for Re-weighting...")
        self.context_stats = {}
        if mode == 'train': 
            for key, items in self.queries_by_key.items():
                # 定义正样本: score >= 0
                n_pos = sum(1 for x in items if float(x.get('score', -1)) >= 0)
                n_neg = len(items) - n_pos
                self.context_stats[key] = {'n_pos': n_pos, 'n_neg': n_neg}
            
            print("\n" + "="*60)
            print(f"Top 10 Steps Statistics ({mode} mode)")
            print(f"{'Dataset':<15} | {'Model':<15} | {'Step':<6} | {'n_pos':<6} | {'n_neg':<6} | {'Total':<6}")
            print("-" * 60)
            
            sorted_keys = sorted(list(self.context_stats.keys()))
            
            for i, key in enumerate(sorted_keys[:10]):

                dataset_name, model_name, step_val = key
                stats = self.context_stats[key]
                total = stats['n_pos'] + stats['n_neg']
                print(f"{dataset_name:<15} | {model_name:<15} | {str(step_val):<6} | "
                        f"{stats['n_pos']:<6} | {stats['n_neg']:<6} | {total:<6}")
            
            print(f"... (Total {len(sorted_keys)} steps loaded)")
            print("="*60 + "\n")
        
        # 6. Create Tasks (Chunks of Queries)
        self.tasks = []
        self.generate_tasks(shuffle=(self.mode == 'train'))
        
        print(f"Dataset Initialized. Total Tasks: {len(self.tasks)}")

    def generate_tasks(self, shuffle=True):
        """
        Pairwise Task Generation with Cyclic Oversampling.
        目标：保留所有样本，不进行丢弃。对于数量较少的一方，循环重复使用以匹配数量较多的一方。
        """
        new_tasks = []
        keys = sorted(list(self.queries_by_key.keys()))
        
        if shuffle:
            random.shuffle(keys)

        dropped_steps = 0
        total_pairs = 0

        for key in keys:
            samples = list(self.queries_by_key[key])
            
            if self.mode == 'train':
                # 1. 分离正负样本
                pos_list = [x for x in samples if self._process_label(x['score']) >= 0.5]
                neg_list = [x for x in samples if self._process_label(x['score']) < 0.5]
                
                n_pos = len(pos_list)
                n_neg = len(neg_list)

                # 2. 如果某一方完全缺失，不得不跳过 (无法构建 Pair)
                if n_pos == 0 or n_neg == 0:
                    dropped_steps += 1
                    continue
                
                # 3. Shuffle (保证每次 Epoch 重复使用的样本是随机顺序的)
                if shuffle:
                    random.shuffle(pos_list)
                    random.shuffle(neg_list)
                
                # 4. Maximize Pairs via Cyclic Oversampling
                # 取最大长度，保证所有样本至少被用到一次
                n_pairs = max(n_pos, n_neg)
                
                paired_samples = []
                for i in range(n_pairs):
                    p = pos_list[i % n_pos]
                    n = neg_list[i % n_neg]
                    
                    paired_samples.append(p)
                    paired_samples.append(n)
                
                total_pairs += n_pairs
                
                # 5. Chunking
                # query_batch_size 必须是偶数
                bs = self.query_batch_size
                if bs % 2 != 0:
                    bs -= 1
                if bs < 2: bs = 2

                for i in range(0, len(paired_samples), bs):
                    chunk = paired_samples[i : i + bs]
                    
                    # 丢弃末尾不完整的 Pair (极少发生，仅当 chunk 长度为奇数时)
                    if len(chunk) % 2 != 0:
                        chunk = chunk[:-1]
                    
                    context_key_to_use = None
                    if key in self.context_pool and len(self.context_pool[key]) > 0:
                        context_key_to_use = key
                    else:
                        fallback_key = (key[1], key[2]) # (model, step)
                        if fallback_key in self.context_pool_fallback and len(self.context_pool_fallback[fallback_key]) > 0:
                            context_key_to_use = fallback_key

                    if len(chunk) > 0 and context_key_to_use is not None:
                        new_tasks.append({
                            'key': key,
                            'context_key': context_key_to_use,
                            'queries': chunk,
                            'is_pairwise': True
                        })

            else:
                if shuffle: random.shuffle(samples)
                for i in range(0, len(samples), self.query_batch_size):
                    chunk = samples[i : i + self.query_batch_size]
                    context_key_to_use = None
                    if key in self.context_pool and len(self.context_pool[key]) > 0:
                        context_key_to_use = key
                    else:
                        fallback_key = (key[1], key[2]) # (model, step)
                        if fallback_key in self.context_pool_fallback and len(self.context_pool_fallback[fallback_key]) > 0:
                            context_key_to_use = fallback_key

                    if context_key_to_use is not None:
                        new_tasks.append({
                            'key': key,
                            'context_key': context_key_to_use,
                            'queries': chunk,
                            'is_pairwise': False
                        })

        self.tasks = new_tasks
        if self.mode == 'train':
            print(f"  >>> [Dataset] Generated {len(self.tasks)} tasks from {len(keys)} contexts.")
            print(f"  >>> [Pairwise Stats] Total Pairs: {total_pairs} (Using Oversampling). Dropped Steps (0 pos or 0 neg): {dropped_steps}")

    def _process_label(self, reward):
        val = float(reward)
        if self.label_strategy == "binary":
            return 1.0 if val >= 0 else 0.0
        elif self.label_strategy == "minmax_norm":
            return (np.clip(val, -1.0, 1.0) + 1.0) / 2.0
        return val

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        key = task['key'] # (dataset, model, step)
        query_samples = task['queries']

        # 1. Sample Context
        context_key = task.get('context_key', key)
        available_context = self.context_pool[key] if context_key == key else self.context_pool_fallback[context_key]
        
        if len(available_context) >= self.support_size:
            support_samples = random.sample(available_context, self.support_size)
        else:
            support_samples = available_context

        # 2. Format Data
        prompts = []
        labels = []
        
        # Process Support
        for item in support_samples:
            s_id_str = f"{item['dataset']}_{item['id']}"
            text = self.prompt_map[s_id_str]
            if text:
                prompts.append(text)
                labels.append(self._process_label(item['score']))
        
        split_idx = len(prompts) # Boundary
        
        # Process Query
        q_ids = []
        pair_ids = [] 
        pair_types = [] 

        for item in query_samples:
            prompts.append(item['text'])
            labels.append(self._process_label(item['score']))
            q_ids.append(item['id'])
            if 'pair_id' in item:
                pair_ids.append(item['pair_id'])
            if 'pair_type' in item:
                pair_types.append(item['pair_type'])

        # 获取该Context的全局正负样本统计量
        stats = self.context_stats.get(key, {'n_pos': 0, 'n_neg': 0})

        return {
            "prompts": prompts,
            "labels": torch.tensor(labels, dtype=torch.float),
            "split_idx": split_idx,
            "q_ids": q_ids,
            "pair_ids": pair_ids, 
            "pair_types": pair_types,
            "key": key,
            "stats": stats # Pass stats to collate
        }
