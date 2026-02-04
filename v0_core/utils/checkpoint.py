import os
import re
import glob
import torch

# =============================================================================
# Checkpoint 管理器
# =============================================================================
class CheckpointManager:
    def __init__(self, checkpoint_dir, max_keep=2, is_master=False):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        self.is_master = is_master
        if self.is_master and self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch, args, wandb_run_id=None):
        if not self.is_master or not self.checkpoint_dir: return
        raw_model = model.module if hasattr(model, 'module') else model
        state = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'args': vars(args),
            'wandb_run_id': wandb_run_id
        }
        filename = f"checkpoint_epoch_{epoch:04d}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        tmp_filepath = filepath + ".tmp"
        print(f">> Saving Checkpoint to {filepath} (Atomic)...")
        try:
            # 1. 先写入临时文件
            torch.save(state, tmp_filepath)
            # 2. 强制刷盘，确保数据落盘
            if os.path.exists(tmp_filepath):
                with open(tmp_filepath, 'rb') as f:
                    os.fsync(f.fileno())
            # 3. 原子重命名 (如果掉电发生在这里之前，旧文件还在；之后，新文件生效)
            os.replace(tmp_filepath, filepath)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            return
        self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        # 保持原逻辑不变，但增加健壮性检查
        files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt"))
        # 过滤掉 .tmp 文件
        files = [f for f in files if not f.endswith('.tmp')]
        
        def extract_epoch(f):
            try:
                match = re.search(r"epoch_(\d+)", f)
                return int(match.group(1)) if match else -1
            except: return -1
            
        files.sort(key=extract_epoch)
        if len(files) > self.max_keep:
            to_delete = files[: -self.max_keep]
            for f in to_delete:
                try:
                    print(f"Removing old checkpoint: {f}")
                    os.remove(f)
                except OSError as e:
                    print(f"Error removing {f}: {e}")

    def find_latest_epoch_num(self):
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir): return 0
        files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt"))
        files = [f for f in files if not f.endswith('.tmp')]
        if not files: return 0
        def extract_epoch(f):
            match = re.search(r"epoch_(\d+)", f)
            return int(match.group(1)) if match else -1
        files.sort(key=extract_epoch)
        return extract_epoch(files[-1])

    def load_specific_epoch(self, target_epoch, model, optimizer, scheduler, device):
        if target_epoch <= 0: return 1 
        filename = f"checkpoint_epoch_{target_epoch:04d}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            import time
            print(f">> [Warning] Checkpoint {filepath} not found immediately. Waiting for FS sync...")
            time.sleep(5) 
            if not os.path.exists(filepath): raise FileNotFoundError(f"Checkpoint {filepath} does not exist.")
        print(f">> Resuming from checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        
        state_dict = checkpoint['model_state_dict']
        raw_model = model.module if hasattr(model, 'module') else model
        
        # 检查是否 key 不匹配 (例如保存时有 module. 读取时没有，或者反之)
        model_keys = set(raw_model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # 简单的 key 修正逻辑
        if list(model_keys)[0].startswith('module.') and not list(ckpt_keys)[0].startswith('module.'):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        elif not list(model_keys)[0].startswith('module.') and list(ckpt_keys)[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        raw_model.load_state_dict(state_dict)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        wandb_id = checkpoint.get('wandb_run_id', None)
        
        print(f"✅ Successfully resumed. Next epoch: {start_epoch}. WandB ID: {wandb_id}")
        return start_epoch, wandb_id
