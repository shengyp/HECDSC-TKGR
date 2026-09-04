from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import autocast_if_available, get_total_rank
from aggregator import HistoricalEmbeddingAggregator
from rgcn_model import RGCN_Clean




class SafeBatchNorm1d(nn.BatchNorm1d):

    def forward(self, x):
        if self.training and x.dim() == 2 and x.size(0) < 2:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, training=False,
                momentum=self.momentum, eps=self.eps
            )
        return super().forward(x)

class ConvTransE(nn.Module):

    def __init__(self, num_entities: int, embedding_dim: int, dropout: float = 0.2):
        super().__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim

        self.bn0 = nn.BatchNorm2d(1, momentum=0.05, eps=1e-5)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.05, eps=1e-5)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.05, eps=1e-5)
        self.bn3 = SafeBatchNorm1d(embedding_dim, momentum=0.01, eps=1e-5)

        self.inp_drop = nn.Dropout(dropout)
        self.feature_map_drop = nn.Dropout2d(dropout)
        self.hidden_drop = nn.Dropout(dropout * 1.5)

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), 1, 1, bias=True)

        self.fc1 = nn.Linear(64 * embedding_dim, 2 * embedding_dim)
        self.fc2 = nn.Linear(2 * embedding_dim, embedding_dim)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu', a=0.1)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.3)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.3)

        if self.conv1.bias is not None:
            nn.init.uniform_(self.conv1.bias, -0.01, 0.01)
        if self.conv2.bias is not None:
            nn.init.uniform_(self.conv2.bias, -0.01, 0.01)
        if self.fc1.bias is not None:
            nn.init.uniform_(self.fc1.bias, -0.01, 0.01)
        if self.fc2.bias is not None:
            nn.init.uniform_(self.fc2.bias, -0.01, 0.01)

    def forward(self, e1_embedded, rel_embedded, ent_emb):
        x = torch.cat([e1_embedded, rel_embedded], 1)
        x = x.view(-1, 1, 2, self.embedding_dim)

        x = self.bn0(x); x = self.inp_drop(x)
        x = self.conv1(x); x = self.bn1(x); x = F.relu(x)
        x = self.feature_map_drop(x)

        x = self.conv2(x); x = self.bn2(x); x = F.relu(x)
        x = self.feature_map_drop(x)

        x = x.mean(dim=2)
        x = x.view(x.size(0), 64 * self.embedding_dim)

        x = self.fc1(x); x = F.relu(x); x = self.hidden_drop(x)
        x = self.fc2(x); x = self.hidden_drop(x)
        x = self.bn3(x)

        eps = 1e-8
        temperature = 0.8
        x_scaled = x / temperature

        x_scaled = torch.clamp(x_scaled, min=-20, max=20)

        scores = torch.mm(x_scaled, ent_emb.transpose(1, 0))

        if torch.any(torch.isnan(scores)) or torch.any(torch.isinf(scores)):
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=eps)
            x_normalized = x / x_norm
            ent_emb_norm = torch.norm(ent_emb, p=2, dim=-1, keepdim=True).clamp(min=eps)
            ent_emb_normalized = ent_emb / ent_emb_norm
            scores = torch.mm(x_normalized, ent_emb_normalized.transpose(1, 0)) * 5.0

        return scores

class DualEncoderModel(nn.Module):

    def __init__(self, dataset_name, num_nodes, num_rels, embd_dim, hidden_dim,
                 n_head, num_hidden_layers, d_k, d_v, d_inner, dropout, self_loop,
                 skip_connect, entity_prediction, relation_prediction,
                 use_cuda, gpu, dataset_path, eval_bz,
                 enable_cache, cache_size, num_workers,
                 train_last_time_idx: int = 1,
                 max_seq_len: int = 10, agg_nhead: int = 4, agg_num_layers: int = 2,
                 rgcn_layers: int = 1):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.embd_dim = embd_dim
        self.eval_bz = eval_bz
        self.train_last_time_idx = int(train_last_time_idx)

        self.shared_ent_emb = nn.Parameter(torch.zeros(num_nodes, embd_dim))
        nn.init.xavier_uniform_(self.shared_ent_emb, gain=0.3)

        self.shared_rel_emb = nn.Parameter(torch.zeros(num_rels * 2, embd_dim))

        with torch.no_grad():
            nn.init.xavier_uniform_(self.shared_rel_emb[:num_rels], gain=0.3)
            nn.init.xavier_uniform_(self.shared_rel_emb[num_rels:], gain=0.3)

        print("relation embeddings initialized:")
        print(f"   - Forward relations: [0:{num_rels}]")
        print(f"   - Inverse relations: [{num_rels}:{num_rels*2}]")
        print(f"   - Total relation embeddings: {num_rels * 2}")

        print("Initializing shared components...")

        max_embeddings = max(8192, train_last_time_idx * 2)

        self.shared_aggregator = HistoricalEmbeddingAggregator(
            d_model=embd_dim,
            nhead=agg_nhead,
            num_layers=agg_num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

        from aggregator import TimeHistoryProcessor
        time_processor = TimeHistoryProcessor(
            dataset_path=dataset_path, num_rels=num_rels,
            dataset_name=dataset_name, split_name="train"
        )
        self.shared_aggregator.set_data_processor(time_processor)
        print(f"   Using HistoricalEmbeddingAggregator")

        self.shared_aggregator_test = self.shared_aggregator

        self.shared_rgcn = RGCN_Clean(
            num_ents=num_nodes, num_rels=num_rels, h_dim=embd_dim,
            dataset_name=dataset_name, num_layers_rgcn=rgcn_layers, dropout=dropout, self_loop=self_loop,
            data_root=dataset_path.replace(f'/{dataset_name}', '') if dataset_path else './data',
            use_attention_rgcn=True, rgcn_attention_heads=n_head
        )

        self.shared_rgcn.set_relation_embedding(self.shared_rel_emb)

        self.subject_aggregator = self.shared_aggregator
        self.object_aggregator = self.shared_aggregator
        self.subject_rgcn = self.shared_rgcn
        self.object_rgcn = self.shared_rgcn

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embd_dim, 
            num_heads=n_head, 
            dropout=dropout, 
            batch_first=True
        )

        self.output_ln = nn.LayerNorm(embd_dim)

        print(f"   Shared aggregator: {type(self.shared_aggregator).__name__}")
        print(f"   Shared RGCN: {type(self.shared_rgcn).__name__}")
        print(f"   Fusion: Gated fusion with residual connection")
        print(f"   Parameter reduction: ~66% (3 modules → 1 module each)")

        self.subject_head = ConvTransE(num_nodes, embd_dim, dropout=dropout)
        self.object_head = ConvTransE(num_nodes, embd_dim, dropout=dropout)

        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, triples: torch.Tensor, all_ans_list, agg_time_idx: int, rgcn_time_idx: int = None, mode_lk: str = None, prediction_type='subject'):
        if rgcn_time_idx is None:
            rgcn_time_idx = agg_time_idx

        if mode_lk is None:
            mode_lk = rgcn_time_idx if isinstance(rgcn_time_idx, str) else "Test"

        device = self.shared_ent_emb.device
        triples = triples.to(device).long()

        if mode_lk == 'Training':
            history_time_idx = int(agg_time_idx)
            graph_time_idx = int(agg_time_idx)
        else:
            history_time_idx = int(agg_time_idx)
            graph_time_idx = int(rgcn_time_idx)

        if prediction_type == 'subject':
            return self._subject_centric_forward(
                triples, all_ans_list, history_time_idx, graph_time_idx, mode_lk, agg_time_idx
            )
        elif prediction_type == 'object':
            return self._object_centric_forward(
                triples, all_ans_list, history_time_idx, graph_time_idx, mode_lk, agg_time_idx
            )
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}. Use 'subject' or 'object'.")

    def _subject_centric_forward(self, triples, all_ans_list, history_time_idx, graph_time_idx, mode_lk, time_idx):
        self.shared_rgcn.set_entity_embedding(self.shared_ent_emb)

        fused = self._encode_subject_centric(
            triples, history_time_idx, graph_time_idx, (mode_lk != 'Training'), mode_lk
        )

        r_ids = triples[:, 1].long()
        r_emb = self.shared_rel_emb[r_ids]

        score = self.subject_head(fused, r_emb, self.shared_ent_emb)

        if mode_lk == 'Training':
            target = triples[:, 2].long()
            loss = self.cross_entropy_loss(score, target)
            return loss

        return self._evaluate_predictions(triples, score, all_ans_list, graph_time_idx, mode_lk, local_time_idx=history_time_idx)

    def _object_centric_forward(self, triples, all_ans_list, history_time_idx, graph_time_idx, mode_lk, time_idx):
        self.shared_rgcn.set_entity_embedding(self.shared_ent_emb)

        fused = self._encode_object_centric(
            triples, history_time_idx, graph_time_idx, (mode_lk != 'Training'), mode_lk
        )

        r_ids = triples[:, 1].long() + self.num_rels
        r_emb = self.shared_rel_emb[r_ids]

        score = self.object_head(fused, r_emb, self.shared_ent_emb)

        if mode_lk == 'Training':
            target = triples[:, 0].long()
            loss = self.cross_entropy_loss(score, target)
            return loss

        obj_triples_for_eval = torch.stack(
            [triples[:, 2], r_ids, triples[:, 0]], dim=1
        )
        return self._evaluate_predictions(obj_triples_for_eval, score, all_ans_list, graph_time_idx, mode_lk, local_time_idx=history_time_idx)

    def _gated_fusion(self, agg_out, rgcn_out, static_emb):
        concat = torch.cat([agg_out, rgcn_out], dim=-1)
        g = self.fusion_gate(concat)
        fused = g * agg_out + (1 - g) * rgcn_out

        fused = fused + static_emb

        fused = self.output_ln(fused)

        return fused

    def _cross_attention_fusion(self, agg_out, rgcn_out, static_emb):
        q = rgcn_out.unsqueeze(1)
        k = agg_out.unsqueeze(1)
        v = agg_out.unsqueeze(1)

        attn_out, _ = self.cross_attn(query=q, key=k, value=v)

        attn_out = attn_out.squeeze(1)

        fused = attn_out + rgcn_out

        fused = fused + static_emb

        fused = self.output_ln(fused)

        return fused

    def _encode_subject_centric(self, triples, history_time_idx, graph_time_idx, eval_mode, mode_lk):
        agg_module = self.shared_aggregator
        split_name = 'train' if mode_lk == 'Training' else 'test'

        if hasattr(agg_module, '_update_split_name'):
            agg_module._update_split_name(split_name)
        else:
            agg_module.split_name = split_name

        full_rel_emb = self.shared_rel_emb

        agg_out = agg_module(
            triples=triples, time_idx=int(history_time_idx),
            ent_embeds=self.shared_ent_emb, rel_embeds=full_rel_emb, direction='forward', split_name=split_name
        )

        entity_ids = triples[:, 0].long()
        rel_indices = triples[:, 1].long()

        rgcn_out = self.shared_rgcn(
            query_entity_ids=entity_ids, rel_ids=rel_indices, current_time_idx=int(graph_time_idx),
            current_time_triples=triples, eval_mode=eval_mode, direction='forward', split_name=split_name
        )

        s_ids = triples[:, 0].long()
        static_emb = self.shared_ent_emb[s_ids]
        fused = self._cross_attention_fusion(agg_out, rgcn_out, static_emb)

        return fused

    def _encode_object_centric(self, triples, history_time_idx, graph_time_idx, eval_mode, mode_lk):
        agg_module = self.shared_aggregator
        split_name = 'train' if mode_lk == 'Training' else 'test'

        if hasattr(agg_module, '_update_split_name'):
            agg_module._update_split_name(split_name)
        else:
            agg_module.split_name = split_name

        full_rel_emb = self.shared_rel_emb

        agg_out = agg_module(
            triples=triples, time_idx=int(history_time_idx),
            ent_embeds=self.shared_ent_emb, rel_embeds=full_rel_emb, direction='inverse', split_name=split_name
        )

        entity_ids = triples[:, 2].long()
        rel_indices = triples[:, 1].long() + (full_rel_emb.size(0) // 2)

        rgcn_out = self.shared_rgcn(
            query_entity_ids=entity_ids, rel_ids=rel_indices, current_time_idx=int(graph_time_idx),
            current_time_triples=triples, eval_mode=eval_mode, direction='inverse', split_name=split_name
        )

        o_ids = triples[:, 2].long()
        static_emb = self.shared_ent_emb[o_ids]
        fused = self._cross_attention_fusion(agg_out, rgcn_out, static_emb)

        return fused

    def _evaluate_predictions(self, triples, score, all_ans_list, global_time_idx, mode_lk, local_time_idx=None):
        rel_predict = 0

        eval_time_idx = local_time_idx if local_time_idx is not None else global_time_idx

        if all_ans_list is None:
            all_mrr_filter, all_mrr, ranks_f, ranks = get_total_rank(
                triples, score, None, self.eval_bz, rel_predict,
                None, None, self.num_rels
            )
            return 0.0, all_mrr_filter, all_mrr, ranks, ranks_f

        all_mrr_filter, all_mrr, ranks_f, ranks = get_total_rank(
            triples, score, None, self.eval_bz, rel_predict,
            all_ans_list, eval_time_idx, self.num_rels
        )
        return 0.0, all_mrr_filter, all_mrr, ranks, ranks_f

Model = DualEncoderModel
