import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# =============================================================================
# TabPFN 修复补丁
# =============================================================================
try:
    from tabpfn import TabPFNClassifier
except ImportError as e:
    print(f"导入 TabPFN 模块失败: {e}")
    print("请确保已安装 tabpfn，并且处于包含 tabpfn 源代码的环境中。")
    exit(1)

from v0_core.utils.tabpfn_patches import fixed_fit, fixed_forward

# Apply Patches
TabPFNClassifier.fit = fixed_fit
TabPFNClassifier.forward = fixed_forward
# print("已应用 TabPFNClassifier 的 fit 和 forward 最终修复补丁。")

# =============================================================================
# Qwen Official Pooling
# =============================================================================
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# =============================================================================
# Adapter 策略模块
# =============================================================================
class FixedQueryAdapter(nn.Module):
    def __init__(self, input_dim, num_queries=10, embed_dim=32, num_heads=4):
        super().__init__()
        self.proj_kv = nn.Linear(input_dim, embed_dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size = hidden_states.size(0)
        kv = self.proj_kv(hidden_states) 
        q = self.queries.repeat(batch_size, 1, 1) 
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        attn_out, _ = self.mha(query=self.ln_q(q), key=self.ln_kv(kv), value=kv, key_padding_mask=key_padding_mask) 
        return attn_out.reshape(batch_size, -1)

class DynamicQueryAdapter(nn.Module):
    def __init__(self, input_dim, num_queries=10, embed_dim=32, num_heads=4, generator_bottleneck_dim=128, generator_dropout_rate=0.2):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.static_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.generator = nn.Sequential(
            nn.Linear(input_dim, generator_bottleneck_dim),
            nn.LayerNorm(generator_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(generator_dropout_rate),
            nn.Linear(generator_bottleneck_dim, num_queries * embed_dim)
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)
        self.proj_kv = nn.Linear(input_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states, attention_mask):
        batch_size = hidden_states.size(0)
        v_global = last_token_pool(hidden_states, attention_mask)
        delta_q = self.generator(v_global).view(batch_size, self.num_queries, self.embed_dim)
        q_final = self.static_queries.repeat(batch_size, 1, 1) + delta_q
        kv = self.proj_kv(hidden_states)
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        attn_out, _ = self.mha(query=self.ln_q(q_final), key=self.ln_kv(kv), value=kv, key_padding_mask=key_padding_mask)
        return attn_out.reshape(batch_size, -1)

# =============================================================================
# Qwen Embedding 模型封装
# =============================================================================
class QwenEmbeddingModel(nn.Module):
    def __init__(self, model_path, pooling_type='last_token', num_queries=10, embed_dim=32,
                 reduce_method='avg_pool', target_dim=1024, num_heads=4, generator_bottleneck_dim=128, generator_dropout_rate=0.2, device='cuda'):
        super().__init__()
        self.device = device
        self.pooling_type = pooling_type
        self.reduce_method = reduce_method
        self.target_dim = target_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        self.backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.backbone.train() 

        with torch.no_grad(): hidden_size = self.backbone.config.hidden_size
        
        if self.pooling_type == 'fixed_query':
            self.adapter_layer = FixedQueryAdapter(input_dim=hidden_size, num_queries=num_queries, embed_dim=embed_dim, num_heads=num_heads).to(device)
        elif self.pooling_type == 'dynamic_query':
            self.adapter_layer = DynamicQueryAdapter(input_dim=hidden_size, num_queries=num_queries, embed_dim=embed_dim, num_heads=num_heads, generator_bottleneck_dim=generator_bottleneck_dim, generator_dropout_rate=generator_dropout_rate).to(device)
        elif self.pooling_type == 'last_token':
            self.adapter_layer = last_token_pool

    def forward(self, prompts, batch_size=32):
        embeddings = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_dict = self.tokenizer(batch_prompts, max_length=2048, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.backbone(**batch_dict)
                last_hidden_state = outputs.last_hidden_state
            emb = self.adapter_layer(last_hidden_state, batch_dict['attention_mask'])
            
            if self.reduce_method == 'avg_pool' and emb.shape[1] > self.target_dim:
                emb = F.adaptive_avg_pool1d(emb.unsqueeze(1), self.target_dim).squeeze(1)
            elif self.reduce_method == 'max_pool' and emb.shape[1] > self.target_dim:
                emb = F.adaptive_max_pool1d(emb.unsqueeze(1), self.target_dim).squeeze(1)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)


class V0:
    def __init__(self, embedding_model, tabpfn_model, device):
        self.embedding_model = embedding_model
        self.tabpfn = tabpfn_model
        self.device = device

    @classmethod
    def from_pretrained(cls, 
                        checkpoint_path, 
                        embedding_model_path, 
                        tabpfn_head_path, 
                        device="cuda",
                        num_queries=168,
                        embed_dim=6,
                        num_heads=3,
                        bottleneck_dim=128,
                        tabpfn_estimators=4):
                
        # 1. Initialize Embedding Model (Qwen + Adapter)
        embedding_model = QwenEmbeddingModel(
            model_path=embedding_model_path,
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=num_heads,
            generator_bottleneck_dim=bottleneck_dim,
            generator_dropout_rate=0.0, # Dropout not needed for inference
            device=device
        )

        # 2. Load Trained Weights (Adapter + potentially Backbone)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict']
        
        # Clean DDP 'module.' prefix if present
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load weights
        msg = embedding_model.load_state_dict(state_dict, strict=False)

        # 3. Initialize TabPFN
        tabpfn = TabPFNClassifier(
            model_path=tabpfn_head_path, 
            device=device, 
            n_estimators=tabpfn_estimators, 
            inference_precision=torch.float32,
            differentiable_input=True # As per training script
        )
        # Manual init to ensure weights are loaded
        tabpfn._initialize_model_variables() 
        
        return cls(embedding_model, tabpfn, device)

    def predict(self, context_prompts, context_labels, target_prompts, batch_size=32):
        """
        Args:
            context_prompts: List[str] - Support Set Texts
            context_labels: List[float] - Support Set Scores (0.0 to 1.0)
            target_prompts: List[str] - Query Set Texts to be scored
        Returns:
            scores: List[float] - Predicted scores (probability of class 1)
        """
        # 1. Encode Context (Support Set)
        X_sup = self.embedding_model(context_prompts, batch_size=batch_size)
        
        # 2. Process Labels (Training script logic: >= 0.5 is Positive)
        y_sup = torch.tensor(context_labels, device=self.device)
        y_sup_hard = (y_sup >= 0.5).long() # Convert to class indices 0 or 1
        
        # 3. Fit TabPFN (In-Context Learning)
        # TabPFN learns from this specific batch of context
        self.tabpfn.fit(X_sup, y_sup_hard)
        
        # 4. Encode Targets (Query Set)
        X_que = self.embedding_model(target_prompts)
        
        # 5. Predict
        # use_inference_mode=True as per eval logic in run_epoch
        with torch.no_grad():
            logits = self.tabpfn.forward(X_que, use_inference_mode=True, return_logits=True)
            probs = torch.softmax(logits, dim=1)
            
        # Return probability of the positive class (class 1)
        # If batch size is 1, output might be squeezed, handling that:
        if probs.dim() == 1:
            return [probs[1].item()]
        else:
            return probs[:, 1].tolist()