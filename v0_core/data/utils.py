import os
import json

def load_jsonl_lines(paths):
    """读取多个文件路径并将所有行合并为一个列表"""
    all_lines = []
    if not isinstance(paths, list): paths = [paths]
    for p in paths:
        if not p or not os.path.exists(p): 
            print(f"Warning: Path not found {p}")
            continue
        print(f"Loading {p}...")
        try:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_lines.append(json.loads(line.strip()))
        except Exception as e:
            print(f"Error reading {p}: {e}")
    return all_lines
