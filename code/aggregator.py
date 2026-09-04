from typing import Tuple, Optional, Iterable, Union, Dict, Set
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agg_get_history import AGGDataManager


def get_device(emb) -> torch.device:
    if isinstance(emb, nn.Module):
        for p in emb.parameters():
            return p.device
        return torch.device("cpu")
    if isinstance(emb, torch.Tensor):
        return emb.device
    return torch.device("cpu")


def get_dim(emb) -> int:
    if isinstance(emb, nn.Embedding):
        return emb.embedding_dim
    if isinstance(emb, nn.Parameter) or isinstance(emb, torch.Tensor):
        return emb.size(-1)
    raise ValueError("Unsupported embedding holder")


def embed_lookup(emb, idx: torch.Tensor) -> torch.Tensor:
    if isinstance(emb, nn.Embedding):
        return emb(idx)
    if isinstance(emb, nn.Parameter):
        return emb.index_select(0, idx)
    if isinstance(emb, torch.Tensor):
        return torch.index_select(emb, 0, idx)
    raise ValueError("Unsupported embedding type")


class TimeHistoryProcessor:

    def __init__(self, dataset_path=None, num_rels=None, dataset_name=None, data_root="./data", split_name="train"):
        self.dataset_path = dataset_path
        self.num_rels = num_rels
        self.dataset_name = dataset_name or os.path.basename(dataset_path or "")
        self.data_root = data_root
        self.split_name = split_name
        self._hist_cache: Dict[int, Dict[Tuple[int,int], Set[int]]] = {}
        self._time_cache: Dict[int, Dict[int, int]] = {}
        self._mgr_cache = {}
        self._combined_mgr = None
    
    def clear_cache(self):
        self._hist_cache.clear()
        self._time_cache.clear()

    def clear_old_cache(self, keep_recent=5):
        if len(self._hist_cache) > keep_recent:
            oldest_keys = sorted(self._hist_cache.keys())[:-keep_recent]
            for key in oldest_keys:
                self._hist_cache.pop(key, None)
        if len(self._time_cache) > keep_recent:
            oldest_keys = sorted(self._time_cache.keys())[:-keep_recent]
            for key in oldest_keys:
                self._time_cache.pop(key, None)

    def _load_hist_dict(self, time_idx: int, split_name: str) -> Dict[Tuple[int,int], Set[int]]:
        cache_key = (time_idx, split_name)
        if cache_key in self._hist_cache:
            return self._hist_cache[cache_key]

        if split_name not in self._mgr_cache:
            self._mgr_cache[split_name] = AGGDataManager(
                self.dataset_name,
                self.data_root,
                split_name
            )

        mgr = self._mgr_cache[split_name]
        data = mgr.load_time_slice_data(time_idx) or {}

        hist = {k: (set(v) if not isinstance(v, set) else v) for k, v in data.items()}
        self._hist_cache[cache_key] = hist
        return hist

    def _load_hist_time_dict(self, time_idx: int, split_name: str) -> Dict[int, int]:
        cache_key = (time_idx, split_name)
        if cache_key in self._time_cache:
            return self._time_cache[cache_key]

        if split_name not in self._mgr_cache:
            self._mgr_cache[split_name] = AGGDataManager(
                self.dataset_name,
                self.data_root,
                split_name
            )

        mgr = self._mgr_cache[split_name]

        tpath = os.path.join(mgr.cache_dir, f"agg_{split_name}_sro_time_{time_idx}.npy")
        if os.path.exists(tpath):
            loaded = np.load(tpath, allow_pickle=True)
            time_map = loaded.item() if hasattr(loaded, 'item') else dict(loaded)
        else:
            legacy_dir = os.path.join(mgr.data_root, mgr.dataset_name, f"his_time_dict_snap_{split_name}")
            legacy = os.path.join(legacy_dir, f"global_sro_time_{time_idx}.npy")
            if os.path.exists(legacy):
                loaded = np.load(legacy, allow_pickle=True)
                time_map = loaded.item() if hasattr(loaded, 'item') else dict(loaded)
            else:
                time_map = {}

        self._time_cache[cache_key] = time_map
        return time_map

    def convert_triples_to_pairs(self, triples: Union[torch.Tensor, np.ndarray, Iterable[Tuple[int,int,int]]], direction='forward'):
        if isinstance(triples, torch.Tensor):
            triples = triples.tolist()
        pairs = []
        if direction == 'forward':
            for (s, r, o) in triples:
                pairs.append((int(s), int(r)))
        elif direction == 'inverse':
            for (s, r, o) in triples:
                offset = self.num_rels if self.num_rels is not None else 0
                pairs.append((int(o), int(r) + offset))
        else:
            raise ValueError("direction must be 'forward' or 'inverse'")
        return pairs

    def time_decay_weighted_mean(self, sr_pair, hist_dict, time_map, ent_embeds, tau: int, lam: float = 0.005, topk: int = 16):
        device = get_device(ent_embeds)
        d_model = get_dim(ent_embeds)
        obj_set = hist_dict.get(sr_pair, set())
        if len(obj_set) == 0:
            return torch.zeros(1, d_model, device=device)

        objs = list(obj_set)
        times = [time_map.get(int(o), tau) for o in objs]

        obj_ids = torch.tensor(objs, dtype=torch.long, device=device)
        o_emb = embed_lookup(ent_embeds, obj_ids)

        dt = torch.clamp(torch.tensor(float(tau), device=device) - torch.tensor(times, device=device, dtype=torch.float32), min=0.0)
        w_time_raw = torch.exp(-lam * dt)

        s_id, r_id = sr_pair
        q = embed_lookup(ent_embeds, torch.tensor([s_id], device=device))
        sim = F.cosine_similarity(q, o_emb, dim=-1).clamp_min(0)

        combined_weights = w_time_raw * sim

        if topk > 0 and len(combined_weights) > topk:
            _, topk_indices = torch.topk(combined_weights, k=topk, largest=True)
            objs = [objs[i] for i in topk_indices.tolist()]
            times = [times[i] for i in topk_indices.tolist()]
            o_emb = o_emb[topk_indices]
            combined_weights = combined_weights[topk_indices]

        w = combined_weights / (combined_weights.sum() + 1e-9)
        mean = torch.sum(o_emb * w.unsqueeze(-1), dim=0, keepdim=True)
        return mean

    def load_histories(self, time_idx: int, sr_pairs: Iterable[Tuple[int,int]], split_name: str = None):
        if split_name is None:
            split_name = self.split_name

        hist_dict = self._load_hist_dict(time_idx, split_name)
        time_map = self._load_hist_time_dict(time_idx, split_name)
        filtered_pairs = [sr for sr in sr_pairs if (sr in hist_dict and len(hist_dict[sr]) > 0)]
        return hist_dict, time_map, filtered_pairs


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = output + residual
        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output = enc_output * non_pad_mask
        return enc_output, enc_slf_attn

class Encoder(nn.Module):

    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src_seq, event_time, non_pad_mask=None, slf_attn_mask=None):
        enc_output = src_seq
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return enc_output


class HistoricalEmbeddingAggregator(nn.Module):

    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, max_seq_len: int = 10):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.agg_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.time_w = nn.Parameter(torch.Tensor(1, d_model))
        self.time_b = nn.Parameter(torch.Tensor(1, d_model))
        nn.init.xavier_uniform_(self.time_w)
        nn.init.zeros_(self.time_b)

        self.pos_emb = nn.Embedding(max_seq_len + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.time_processor = None
        self._mgr_cache = {}
        self.split_name = "train"

    def set_data_processor(self, time_processor):
        self.time_processor = time_processor
        self.split_name = time_processor.split_name

    def _update_split_name(self, split_name):
        if self.time_processor is not None:
            self.time_processor.split_name = split_name
        self.split_name = split_name

    def _get_history_sequence_fast(self, preprocessed_history, current_time, entity_embs_all, device, time_map=None):
        seq_pooled_objs = []
        seq_rel_times = []

        num_entities = entity_embs_all.shape[0] if hasattr(entity_embs_all, 'shape') else entity_embs_all.size(0)

        for obj_id in preprocessed_history:
            obj_id = int(obj_id)

            if obj_id < 0 or obj_id >= num_entities:
                continue

            ts = time_map.get(obj_id, current_time) if time_map else current_time
            if float(ts) < float(current_time):
                obj_tensor = torch.tensor([obj_id], dtype=torch.long, device=device)

                obj_embs = entity_embs_all[obj_tensor]
                pooled_obj = obj_embs.mean(dim=0)

                seq_pooled_objs.append(pooled_obj)
                seq_rel_times.append(float(current_time) - float(ts))

                if len(seq_pooled_objs) >= self.max_seq_len:
                    break

        return seq_pooled_objs, seq_rel_times

    def forward(self, triples: torch.Tensor, time_idx: int,
                ent_embeds, rel_embeds, direction: str, split_name: str = None):
        device = get_device(ent_embeds)
        d_model = get_dim(ent_embeds)

        if triples.dim() == 1:
            triples = triples.unsqueeze(0)
        B = triples.size(0)

        if direction == 'inverse':
            entity_ids = triples[:, 2].long()
            r_ids = triples[:, 1].long() + (rel_embeds.size(0) // 2)
        else:
            entity_ids = triples[:, 0].long()
            r_ids = triples[:, 1].long()

        entity_ids = entity_ids.to(device)
        r_ids = r_ids.to(device)

        num_entities = ent_embeds.shape[0] if hasattr(ent_embeds, 'shape') else ent_embeds.size(0)
        num_relations = rel_embeds.size(0)

        if (entity_ids < 0).any() or (entity_ids >= num_entities).any():
            print(f"[WARNING] 发现越界的实体ID，已裁剪到 [0, {num_entities-1}]")
            entity_ids = torch.clamp(entity_ids, min=0, max=num_entities-1)

        if (r_ids < 0).any() or (r_ids >= num_relations).any():
            print(f"[WARNING] 发现越界的关系ID，已裁剪到 [0, {num_relations-1}]")
            r_ids = torch.clamp(r_ids, min=0, max=num_relations-1)

        if isinstance(ent_embeds, nn.Parameter):
            entity_embs_all = ent_embeds
        else:
            entity_embs_all = ent_embeds

        r_embs = embed_lookup(rel_embeds, r_ids)

        entity_embs = entity_embs_all[entity_ids]

        seq_tensor = torch.zeros(B, self.max_seq_len, self.d_model, device=device)
        rel_time_tensor = torch.zeros(B, self.max_seq_len, device=device)
        padding_mask = torch.ones(B, self.max_seq_len, dtype=torch.bool, device=device)

        if self.time_processor is not None:
            target_pairs = self.time_processor.convert_triples_to_pairs(triples, direction=direction)
            hist_dict, time_map, _ = self.time_processor.load_histories(
                int(time_idx), target_pairs, split_name=split_name
            )

            for i in range(B):
                e_id = int(entity_ids[i].item())
                c_time = float(time_idx)

                key = (e_id, int(r_ids[i].item()))

                hist_data = hist_dict.get(key, set())

                pooled_objs, rel_times = self._get_history_sequence_fast(hist_data, c_time, entity_embs_all, device, time_map)

                seq_len = min(len(pooled_objs), self.max_seq_len)

                if seq_len > 0:
                    pooled_stack = torch.stack(pooled_objs[:seq_len])
                    time_stack = torch.tensor(rel_times[:seq_len], dtype=torch.float, device=device)

                    sq_emb = entity_embs[i].unsqueeze(0).expand(seq_len, -1)
                    rq_emb = r_embs[i].unsqueeze(0).expand(seq_len, -1)

                    step_inputs = torch.cat([sq_emb, rq_emb, pooled_stack], dim=-1)
                    step_features = self.agg_mlp(step_inputs)

                    seq_tensor[i, :seq_len, :] = step_features
                    rel_time_tensor[i, :seq_len] = time_stack
                    padding_mask[i, :seq_len] = False

        time_enc = torch.cos(rel_time_tensor.unsqueeze(-1) * self.time_w + self.time_b)

        positions = torch.arange(1, self.max_seq_len + 1, device=device).unsqueeze(0).expand(B, -1)
        pos_enc = self.pos_emb(positions)

        transformer_input = seq_tensor + time_enc + pos_enc
        transformer_input = transformer_input.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        all_masked = padding_mask.all(dim=1)
        padding_mask = padding_mask.clone()
        padding_mask[all_masked, 0] = False

        out = self.transformer(transformer_input, src_key_padding_mask=padding_mask)

        out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        valid_lens = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        final_history_emb = out.sum(dim=1) / valid_lens

        return final_history_emb
