###
# monitoring.py
#
# Classes for monitoring experiments.
# Dylan Everingham
# 30.04.2026
###

# Dependencies
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from transformers import AutoConfig
from moe_hooks import *


# MoE monitoring probe: attaches a callback to routing modules which computes routing metrics
class MoEProbe(MoEHook):
    
    def __init__(self, model, n_experts=64, k=2):
        
        super(MoEProbe, self).__init__(model, n_experts=n_experts, k=k)
        
        self.logs = {}  # Metrics are logged here after each router activation
                        # indexed by layer name
        self.most_recent = {} # Holds per-router metrics for most recent activation
    
    # Clears monitoring logs
    def clear(self):
        self.logs = {}
        self.most_recent = {}

    def print_count(self):
        print(f"MoEProbe: Captured {len(self.logs)} routing events from {self.n_routers} router modules.")
    
    # get router probabilities
    def get_probs(self, batch_size=1):
        
        all_routers_probs = []
        for n in self.router_names_sorted:
            
            # Reshape each step to [batch, seq, n_experts] and cat along seq dimension
            unflattened_logs = []
            for l in self.logs[n]:
                
                step_probs = l['probs'] # [batch * seq_len, n_experts] or [n_experts] if batch == seq_len == 1
                # n_experts replaced here with k for deepseek models
                if step_probs.dim() == 1: # Add batch dim if missing
                    step_probs = step_probs.unsqueeze(0)
                seq_len = step_probs.shape[0] // batch_size
                unflattened_logs.append(step_probs.view(batch_size, seq_len, step_probs.size(-1)))
                
            router_seq_probs = torch.cat(unflattened_logs, dim=1) # [batch, total_seq_len, n_experts]
            all_routers_probs.append(router_seq_probs)
            
        # [batch, padded_seq_len, n_experts, n_routers]
        # stacking here works for different seq_len due to padding from HF
        return torch.stack(all_routers_probs, dim=-1).cpu()
    
    # get active experts
    def get_active_experts(self, batch_size=1):
        all_routers_experts = []
        for n in self.router_names_sorted:
            unflattened_logs = []
            for l in self.logs[n]:
                step_experts = l['active_experts'] # [batch * seq_len, k] or [k] if batch == seq_len == 1
                if step_experts.dim() == 1: # Add batch dim if missing
                    step_experts = step_experts.unsqueeze(0)
                seq_len = step_experts.shape[0] // batch_size
                unflattened_logs.append(step_experts.view(batch_size, seq_len, step_experts.size(-1)))
                
            router_seq_experts = torch.cat(unflattened_logs, dim=1)
            all_routers_experts.append(router_seq_experts)
        
        # [batch, padded_seq_len, k, n_routers]
        return torch.stack(all_routers_experts, dim=-1).cpu()
    
    def plot_loadbalance(self, router_id=0):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Collate data
        active_experts = self.get_active_experts() # [batch, tokens, k, n_routers]
        active_experts = active_experts[:, :, :, router_id].flatten().cpu().tolist()
            
        counts = Counter(active_experts)
        #avg_entropy = np.mean(all_entropies)
        expert_ids = np.array(range(self.n_experts))
        count_per_expert = np.array([counts[i] for i in expert_ids])
        tokens = counts.total()
        freqs = count_per_expert / tokens
        expected_freq = 1/self.n_experts
        ax.bar(expert_ids, freqs, \
                label=f"tokens: {tokens}", \
                alpha=0.7)

        ax.set_title("Expert Activation Frequency (Load)")
        ax.set_xlabel(f"Expert Index (0-{self.n_experts-1})")
        ax.set_ylabel("Activation Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.axhline(y=expected_freq, color='red', linestyle='--', linewidth=2, label="expected frequency (1/n_experts)")    
            
# Implentation for Qwen MoE models ex. Qwen1.5-MoE-A2.7B-Chat
class MoEProbeQwen(MoEProbe, MoEHookQwen):
    
    def __init__(self, model):

        # Query expert information
        n_experts = model.config.num_experts
        k = model.config.num_experts_per_tok
        self.n_routers = model.config.num_hidden_layers

        super(MoEProbeQwen, self).__init__(model, n_experts=n_experts, k=k)
        
        if hasattr(model.config, "shared_expert_intermediate_size") and hasattr(model.config, "moe_intermediate_size"):
            shared_size = model.config.shared_expert_intermediate_size
            routed_size = model.config.moe_intermediate_size
            print(f"MoEProbe: Qwen1.5-MoE-A2.7B model also has shared expert with intermediate size: {shared_size} (Equivalent to ~{shared_size // routed_size} routed experts)")
    
    # Function used for extracting router metrics
    def hook_fn(self, module, inputs, outputs):
        
        # outputs are the raw logits [batch * seq_len, n_experts]
        router_logits = outputs
        
        # Calculate probabilities
        probs = torch.softmax(router_logits, dim=-1)
        
        # Metric: router entropy (uncertainty)
        # High entropy = Router is unsure (or load balancing is forcing uniformity)
        # Low entropy = Strong specialization
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        
        # Metric: expert activation (Load)
        # Manually recalculate top-k
        topk_values, topk_indices = torch.topk(probs, self.k, dim=-1)
        
        # Store lightweight statistics (move to cpu to save vram)
        name = self.routers[module]["name"]
        log = {
            "entropy": entropy.item(),
            "active_experts": topk_indices.squeeze().cpu(),
            "probs": probs.squeeze().detach().cpu(),
        }
        
        if name in self.logs:
            self.logs[name].append(log)
        else:
            self.logs[name] = [log]
        self.most_recent[name] = log


# Implentation for DeepSeek MoE models ex. DeepSeek-V2-Lite-Chat
class MoEProbeDeepSeek(MoEProbe, MoEHookDeepSeek):
    
    def __init__(self, model):

        # Query expert information
        n_experts = model.config.n_routed_experts
        k = model.config.num_experts_per_tok
        self.n_routers = model.config.num_hidden_layers - model.config.first_k_dense_replace

        super(MoEProbeDeepSeek, self).__init__(model, n_experts=n_experts, k=k)
        
        if hasattr(model.config, "shared_expert_intermediate_size") and hasattr(model.config, "moe_intermediate_size"):
            shared_size = model.config.shared_expert_intermediate_size
            n_shared_experts = model.config.n_share_experts
            routed_size = model.config.moe_intermediate_size
            print(f"MoEProbe: DeepSeek-V2-Lite-Chat model also has {n_shared_experts} shared experts with the same intermediate size: {shared_size} (Compared to {routed_size} for routed experts)")
    
    # Function used for extracting router metrics
    # Mostly the same as Qwen, but DeepSeek routers return the topk expert indices directly as a second return value
    def hook_fn(self, module, inputs, outputs):
        
        # DeepSeek router outputs: tuple of tokp indices, logits, and None
        # each of size [batch * seq_len, k]
        topk_indices, router_logits, _ = outputs
        
        # Calculate probabilities
        probs = torch.softmax(router_logits, dim=-1)
        
        # Metric: router entropy (uncertainty)
        # High entropy = Router is unsure (or load balancing is forcing uniformity)
        # Low entropy = Strong specialization
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        
        # Store lightweight statistics (move to cpu to save vram)
        name = self.routers[module]["name"]
        log = {
            "entropy": entropy.item(),
            "active_experts": topk_indices.cpu(),
            "probs": probs.detach().cpu(),
        }
        if name in self.logs:
            self.logs[name].append(log)
        else:
            self.logs[name] = [log]
        self.most_recent[name] = log
        
        
# Implentation for Mistral MoE models ex. Mixtral-8x7B-Instruct-v0.1
class MoEProbeMistral(MoEProbe, MoEHookMistral):
    
    def __init__(self, model):

        # Query expert information
        n_experts = 8 # Mixtral experts set at 8 (see model name)
        k = model.config.num_experts_per_tok
        self.n_routers = model.config.num_hidden_layers

        super(MoEProbeMistral, self).__init__(model, n_experts=n_experts, k=k)
        
        # No shared experts in Mixtral
    
    # Function used for extracting router metrics
    # Mostly the same as Qwen, but DeepSeek routers return the topk expert indices directly as a second return value
    def hook_fn(self, module, inputs, outputs):
        
        # outputs are the raw logits [batch * seq_len, n_experts]
        router_logits = outputs
        
        # Calculate probabilities
        probs = torch.softmax(router_logits, dim=-1)
        
        # Metric: router entropy (uncertainty)
        # High entropy = Router is unsure (or load balancing is forcing uniformity)
        # Low entropy = Strong specialization
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        
        # Metric: expert activation (Load)
        # Manually recalculate top-k
        topk_values, topk_indices = torch.topk(probs, self.k, dim=-1)
        
        # Store lightweight statistics (move to cpu to save vram)
        name = self.routers[module]["name"]
        log = {
            "entropy": entropy.item(),
            "active_experts": topk_indices.cpu(),
            "probs": probs.detach().cpu(),
        }
        if name in self.logs:
            self.logs[name].append(log)
        else:
            self.logs[name] = [log]
        self.most_recent[name] = log