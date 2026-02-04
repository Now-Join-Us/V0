import os
import gc
import sys
import torch
import torch.nn as nn
import numpy as np
import socket
import random
import wandb
from transformers import get_scheduler
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from v0_core.config.arguments import parse_args, print_elegant_args
from v0_core.utils.checkpoint import CheckpointManager
from v0_core.models.v0 import QwenEmbeddingModel, TabPFNClassifier
from v0_core.data.dataset import ValueModelDataset
from v0_core.data.collator import meta_collate_fn
from v0_core.utils.metrics import calculate_metrics_by_group

from v0_core.trainer import run_epoch


# =============================================================================
# Training Main
# =============================================================================
def main():
    base_seed = 42
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_master = (global_rank == 0)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.device = device 

    if is_master:
        print_elegant_args(args)

    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print(f"[Node Info] Host: {hostname} | IP: {ip} | Global Rank: {global_rank} | World Size: {world_size}")

    dist.barrier()
    if is_master:
        print("✅ All nodes initialized and connected.")

    if is_master and args.log_path:
        done_file_path = os.path.join(os.path.dirname(args.log_path), "done")
        if os.path.exists(done_file_path):
            print(f"✅ Found DONE file. Exiting.")
            dist.destroy_process_group()
            sys.exit(0)
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        if args.metric_path: os.makedirs(os.path.dirname(args.metric_path), exist_ok=True)
        if args.checkpoint_dir: os.makedirs(args.checkpoint_dir, exist_ok=True)
    else:
        args.log_path, args.metric_path = None, None

    ckpt_manager = CheckpointManager(args.checkpoint_dir, args.max_keep_checkpoints, is_master)

    start_epoch = 1
    resumed_wandb_id = None

    found_epoch = 0
    if args.run_mode == 'train' and args.resume:
        if is_master: 
            found_epoch = ckpt_manager.find_latest_epoch_num()
        
        epoch_tensor = torch.tensor(found_epoch, dtype=torch.long, device=device)
        dist.broadcast(epoch_tensor, src=0)
        found_epoch = epoch_tensor.item()

    model = QwenEmbeddingModel(
        model_path=args.qwen_path,
        pooling_type=args.pooling_strategy,
        num_queries=args.num_queries,
        embed_dim=args.embed_dim,
        reduce_method=args.reduce_method,
        target_dim=args.target_dim,
        num_heads=args.num_heads,
        generator_bottleneck_dim=args.dynamic_query_generator_bottleneck_dim,
        generator_dropout_rate=args.dynamic_query_generator_dropout_rate,
        device=device
    )

    if args.run_mode == 'train':
        print(">> Freezing Qwen Backbone parameters...")
        model.backbone.eval()
        for param in model.backbone.parameters(): param.requires_grad = False
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        print(">> Freezing All Parameters for Evaluation...")
        model.eval()
        for p in model.parameters(): p.requires_grad = False
    
    print(f"Loading TabPFN from {args.tabpfn_checkpoint}...")
    tabpfn = TabPFNClassifier(model_path=args.tabpfn_checkpoint, device=device, n_estimators=args.tabpfn_estimators, inference_precision=torch.float32, differentiable_input=True)
    tabpfn._initialize_model_variables()
    if hasattr(tabpfn, 'models_'):
        for m in tabpfn.models_: 
            m.eval()
            for p in m.parameters(): p.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    def create_loader(query_paths, is_train=False, batch_size=1):
        if not query_paths: return None
        ds = ValueModelDataset(
            context_paths=args.context_data_paths,
            query_paths=query_paths,
            prompt_dict_path=args.prompt_dict_path,
            label_strategy=args.label_strategy,
            query_batch_size=args.train_query_batch_size if is_train else args.eval_query_batch_size,
            support_size=args.support_size,
            mode='train' if is_train else 'eval'
        )
        sampler = DistributedSampler(ds, shuffle=is_train)
        return DataLoader(
            ds, batch_size=batch_size, collate_fn=meta_collate_fn, 
            sampler=sampler, num_workers=4, pin_memory=True
        )

    validity_loaders = {}
    if args.validity_data_paths:
        for val_path in args.validity_data_paths:
            d_name = os.path.basename(val_path).split('.')[0]
            if is_master: print(f"Preparing Validity Loader for {d_name}...")
            validity_loaders[d_name] = create_loader([val_path], is_train=False, batch_size=args.meta_batch_size)

    # === Eval Mode ===
    if args.run_mode == 'eval':
        if args.resume: _ = ckpt_manager.load_specific_epoch(args.resume_from_specific_epoch, model, None, None, device)
        
        if args.eval_data_paths:
            for eval_path in args.eval_data_paths:
                dataset_name = os.path.basename(eval_path).split('.')[0]
                test_loader = create_loader([eval_path], is_train=False, batch_size=args.meta_batch_size)
                
                print(f">>> Starting Evaluation on {dataset_name}...")
                v_loss, v_results = run_epoch(
                    'val', test_loader, model, tabpfn, criterion=criterion,
                    train_infer_bs=args.train_embed_bs, eval_infer_bs=args.eval_embed_bs,
                    log_interval=args.log_interval, reduce_method=args.reduce_method, target_dim=args.target_dim,
                    grad_accum_steps=1, is_master=is_master, wandb_interval=args.wandb_interval, 
                    loss_type=args.loss_type, kl_temperature=args.kl_temperature, loss_alpha=args.loss_alpha, loss_balance=args.loss_balance,
                    context_clustering=args.context_clustering, context_num_clusters=args.context_num_clusters
                )
                
                metrics = calculate_metrics_by_group(
                    v_results, 'val', 0, is_master=is_master, 
                    output_dir=os.path.dirname(args.log_path) if args.log_path else None,
                    dataset_name_tag=f"_{dataset_name}", avg_loss=v_loss
                )
                
                if is_master:
                    print(f"KEval Result [{dataset_name}]: {metrics}")
                    wandb.log(metrics)

    # === Train Mode ===
    elif args.run_mode == 'train':
        print(f"Preparing Train Loader...")
        
        test_loaders = {}
        if args.eval_data_paths:
            for eval_path in args.eval_data_paths:
                d_name = os.path.basename(eval_path).split('.')[0]
                test_loaders[d_name] = create_loader([eval_path], is_train=False, batch_size=args.meta_batch_size)

        raw_model = model.module if isinstance(model, DDP) else model
        trainable_params = [p for n, p in raw_model.named_parameters() if p.requires_grad]
        no_decay = ["bias", "LayerNorm.weight", "ln_q.weight", "ln_kv.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in raw_model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, "lr": args.lr_adapter},
            {"params": [p for n, p in raw_model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": args.lr_adapter},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        train_loader = create_loader(args.train_data_paths, is_train=True, batch_size=args.meta_batch_size)
        num_update_steps_per_epoch = len(train_loader) // args.grad_accum_steps
        max_train_steps = args.epochs * num_update_steps_per_epoch
        scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=int(max_train_steps * args.warmup_ratio), num_training_steps=max_train_steps)
        
        start_epoch = 1
        if args.resume and found_epoch > 0:
            start_epoch, resumed_wandb_id = ckpt_manager.load_specific_epoch(
                found_epoch, model, optimizer, scheduler, device
            )
            if is_master:
                print(f"🚀 Resuming training from Epoch {start_epoch}. WandB ID: {resumed_wandb_id}")

    if is_master and args.log_path:
        final_wandb_id = resumed_wandb_id if resumed_wandb_id else args.wandb_id
        
        wandb.init(
            project=args.wandb_project, 
            name=f"{args.time_str}", 
            config=vars(args), 
            mode="offline", 
            id=final_wandb_id,
            resume="allow"
        )

    if args.run_mode == 'train':
        dist.barrier()
        
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_seed = base_seed + epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)

            train_loader = create_loader(args.train_data_paths, is_train=True, batch_size=args.meta_batch_size)

            dist.barrier()
            train_loader.sampler.set_epoch(epoch)
            
            print(f"\n[Epoch {epoch}/{args.epochs}] Training started...")
            
            t_loss, t_results = run_epoch(
                'train', train_loader, model, tabpfn, optimizer, scheduler, criterion,
                train_infer_bs=args.train_embed_bs, eval_infer_bs=args.eval_embed_bs,
                reduce_method=args.reduce_method, target_dim=args.target_dim,
                grad_accum_steps=args.grad_accum_steps, is_master=is_master,
                epoch_idx=epoch, wandb_interval=args.wandb_interval,
                max_grad_norm=args.max_grad_norm, loss_type=args.loss_type, kl_temperature=args.kl_temperature, loss_alpha=args.loss_alpha, loss_balance=args.loss_balance,
                context_clustering=args.context_clustering, context_num_clusters=args.context_num_clusters
            )
            
            train_metrics = calculate_metrics_by_group(
                t_results, 'train', epoch, is_master=is_master,
                output_dir=os.path.dirname(args.log_path) if args.log_path else None, avg_loss=t_loss
            )
            if is_master:
                print(f"Train Global Metrics: {train_metrics}")
                wandb.log(train_metrics)

            gc.collect()
            torch.cuda.empty_cache()

            if epoch % args.save_interval == 0:
                current_run_id = wandb.run.id if is_master and wandb.run else None
                ckpt_manager.save(model, optimizer, scheduler, epoch, args, wandb_run_id=current_run_id)
            
            for d_name, t_loader in test_loaders.items():
                v_loss, v_results = run_epoch(
                    'val', t_loader, model, tabpfn, criterion=criterion,
                    train_infer_bs=args.train_embed_bs, eval_infer_bs=args.eval_embed_bs,
                    reduce_method=args.reduce_method, target_dim=args.target_dim,
                    grad_accum_steps=1, is_master=is_master, epoch_idx=epoch, wandb_interval=args.wandb_interval,
                    loss_type=args.loss_type, kl_temperature=args.kl_temperature, loss_alpha=args.loss_alpha, loss_balance=args.loss_balance,
                    context_clustering=args.context_clustering, context_num_clusters=args.context_num_clusters
                )
                
                val_metrics = calculate_metrics_by_group(
                    v_results, f'val_{d_name}', epoch, is_master=is_master,
                    output_dir=os.path.dirname(args.log_path) if args.log_path else None,
                    dataset_name_tag=f"_{d_name}", avg_loss=v_loss
                )
                
                if is_master:
                    val_metrics[f'val_{d_name}/loss'] = v_loss
                    print(f"Valid [{d_name}] Metrics: {val_metrics}")
                    wandb.log(val_metrics)
            
            if validity_loaders:
                for d_name, v_loader in validity_loaders.items():
                    if hasattr(v_loader.sampler, 'set_epoch'): v_loader.sampler.set_epoch(epoch)
                    print(f">> Validating Validity on {d_name}...")
                    
                    val_loss, val_results = run_epoch(
                        'val_validity', v_loader, model, tabpfn, criterion=criterion,
                        train_infer_bs=args.train_embed_bs, eval_infer_bs=args.eval_embed_bs,
                        reduce_method=args.reduce_method, target_dim=args.target_dim,
                        grad_accum_steps=1, is_master=is_master, epoch_idx=epoch, wandb_interval=args.wandb_interval,
                        loss_type=args.loss_type, kl_temperature=args.kl_temperature, loss_alpha=args.loss_alpha, loss_balance=args.loss_balance,
                        context_clustering=args.context_clustering, context_num_clusters=args.context_num_clusters
                    )
                    
                    validity_metrics = calculate_metrics_by_group(
                        val_results, f'val_validity_{d_name}', epoch, is_master=is_master,
                        output_dir=os.path.dirname(args.log_path) if args.log_path else None,
                        dataset_name_tag=f"_{d_name}_validity", avg_loss=val_loss
                    )
                    
                    if is_master:
                        print(f"KValidity [{d_name}] Metrics: {validity_metrics}")
                        wandb.log(validity_metrics)

            gc.collect()
            torch.cuda.empty_cache()

    if is_master:
        wandb.finish()
        if args.log_path:
             with open(done_file_path, 'w') as f: f.write("finished")
    dist.destroy_process_group()
    print("\nProcess Complete.")

if __name__ == "__main__":
    main()