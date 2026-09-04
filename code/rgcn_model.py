import os
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as dglfn

class SimpleDataManager:
    def __init__(self, dataset_name: str, data_root: str = "./data", train_test_split_idx: int = None):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.train_test_split_idx = train_test_split_idx

        self.train_cache_t = None
        self.train_cache = None
        self.test_cache_t = None
        self.test_cache = None

        self.train_root_dir = os.path.join(data_root, dataset_name, 'graph_snapshots_train')
        self.test_root_dir = os.path.join(data_root, dataset_name, 'graph_snapshots_test')

        if self.train_test_split_idx is None:
            self.train_test_split_idx = self._detect_split_idx()

        print(f"DataManager initialized for {dataset_name}")
        print(f"  Train graphs: {self.train_root_dir}")
        print(f"  Test graphs: {self.test_root_dir}")
        print(f"  Train/Test split index: {self.train_test_split_idx}")

    def _detect_split_idx(self) -> int:
        try:
            import sys
            import os
            utils_path = os.path.join(os.path.dirname(__file__), 'utils.py')
            if os.path.exists(utils_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("utils", utils_path)
                utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(utils)

                print("  Loading raw training data to detect split index...")
                train_data = utils.load_from_local(self.data_root, self.dataset_name, load_time=True)
                if train_data.train is not None and len(train_data.train) > 0:
                    train_snapshots = utils.split_by_time(train_data.train)
                    split_idx = len(train_snapshots)
                    print(f"  Detected {split_idx} time snapshots from raw training data")
                    return split_idx
                else:
                    print("  Warning: No training data found in raw data")
            else:
                print("  Warning: utils.py not found")
        except Exception as e:
            print(f"  Warning: Could not load raw training data: {e}")

        print("  Using default split_idx = 334")
        return 334

    def _snap_path(self, t_idx: int, split_name: str = None) -> str:
        if split_name == 'train':
            root_dir = self.train_root_dir
        elif split_name == 'test':
            root_dir = self.test_root_dir
        else:
            root_dir = self.train_root_dir if t_idx < self.train_test_split_idx else self.test_root_dir
        return os.path.join(root_dir, f'graph_{t_idx}.bin')

    def _load_t(self, t_idx: int, split_name: str = None):
        path = self._snap_path(t_idx, split_name)

        cache_key = (t_idx, split_name)

        if not os.path.exists(path):
            empty_cache = (torch.empty(0, dtype=torch.long), [])
            if split_name == 'train':
                self.train_cache_t, self.train_cache = t_idx, empty_cache
            elif split_name == 'test':
                self.test_cache_t, self.test_cache = t_idx, empty_cache
            return

        try:
            graphs, aux = dgl.load_graphs(path)
            subject_ids = aux.get('subject_ids', torch.empty(0, dtype=torch.long))
            cache_data = (subject_ids, graphs)

            if split_name == 'train':
                self.train_cache_t, self.train_cache = t_idx, cache_data
            elif split_name == 'test':
                self.test_cache_t, self.test_cache = t_idx, cache_data

        except Exception as e:
            print(f"Error loading graphs from {path}: {e}")
            empty_cache = (torch.empty(0, dtype=torch.long), [])
            if split_name == 'train':
                self.train_cache_t, self.train_cache = t_idx, empty_cache
            elif split_name == 'test':
                self.test_cache_t, self.test_cache = t_idx, empty_cache

    def get_graphs_for_subjects(self, t_idx: int, subj_ids: torch.Tensor,
                               split_name: str = None) -> List[dgl.DGLGraph]:
        if split_name is None:
            split_name = 'train' if t_idx < self.train_test_split_idx else 'test'

        if split_name == 'train':
            if self.train_cache_t != t_idx or self.train_cache is None:
                self._load_t(t_idx, split_name)
            subj_of_t, graphs_of_t = self.train_cache
        elif split_name == 'test':
            if self.test_cache_t != t_idx or self.test_cache is None:
                self._load_t(t_idx, split_name)
            subj_of_t, graphs_of_t = self.test_cache
        else:
            raise ValueError(f"Invalid split_name: {split_name}. Must be 'train' or 'test'.")

        if subj_of_t.numel() == 0:
            return [None for _ in range(subj_ids.shape[0])]

        id2idx = {int(s.item()): i for i, s in enumerate(subj_of_t)}
        out = []
        for s in subj_ids.tolist():
            gi = id2idx.get(int(s), None)
            out.append(graphs_of_t[gi] if gi is not None else None)
        return out
    
    def clear_cache(self):
        self.train_cache_t = None
        self.train_cache = None
        self.test_cache_t = None
        self.test_cache = None

class BasicRGCNLayer(nn.Module):
    def __init__(self, h_dim: int, num_rels: int,
                 activation=F.rrelu, self_loop: bool = True,
                 dropout: float = 0.1, skip_connect: bool = True):
        super().__init__()
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.rel_emb = None

        self.weight_neighbor = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(h_dim))
            nn.init.zeros_(self.skip_connect_bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, g: dgl.DGLGraph, h0: torch.Tensor, prev_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.rel_emb is None:
            raise ValueError("rel_emb is not set!")

        if h0.is_cuda and not g.device.type == 'cuda':
            g = g.to(h0.device)

        with g.local_scope():
            g.ndata['h'] = h0

            if self.self_loop:
                in_degrees = g.in_degrees()
                masked_index = torch.masked_select(
                    torch.arange(0, g.num_nodes(), dtype=torch.long, device=h0.device),
                    (in_degrees > 0)
                )

                loop_message = torch.mm(h0, self.evolve_loop_weight)

                if masked_index.numel() > 0:
                    loop_message[masked_index, :] = torch.mm(h0, self.loop_weight)[masked_index, :]

            if prev_h is not None and self.skip_connect:
                skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)

            if g.num_edges() > 0:
                rel = g.edata['type']
                if isinstance(self.rel_emb, nn.Parameter):
                    rel_vec = self.rel_emb.index_select(0, rel)
                elif isinstance(self.rel_emb, nn.Embedding):
                    rel_vec = self.rel_emb(rel)
                else:
                    rel_vec = torch.index_select(self.rel_emb, 0, rel)

                g.edata['r'] = rel_vec

                def msg_func(edges):
                    m = edges.src['h'] + edges.data['r']
                    m = torch.mm(m, self.weight_neighbor)
                    return {'m': m}

                def apply_func(nodes):
                    if 'norm' in nodes.data:
                        return {'h': nodes.data['h_new'] * nodes.data['norm']}
                    else:
                        return {'h': nodes.data['h_new']}

                g.update_all(msg_func, dglfn.sum('m', 'h_new'), apply_func)
                node_repr = g.ndata['h']
            else:
                node_repr = torch.zeros_like(h0)

            if prev_h is not None and self.skip_connect:
                if self.self_loop:
                    node_repr = node_repr + loop_message
                node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
            else:
                if self.self_loop:
                    node_repr = node_repr + loop_message

            if self.activation:
                node_repr = self.activation(node_repr)

            if self.dropout is not None:
                node_repr = self.dropout(node_repr)

            return node_repr

class RGCN_Clean(nn.Module):

    def __init__(self, num_ents: int, num_rels: int, h_dim: int, dataset_name: str,
                 num_layers_rgcn: int = 2, dropout: float = 0.1, self_loop: bool = True,
                 att_heads_num: int = 2, device: Optional[torch.device] = None, data_root: str = "./data",
                 use_attention_rgcn: bool = True, rgcn_attention_heads: int = 4, train_test_split_idx: int = None):
        super().__init__()

        self.num_ents = num_ents
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.device = device or torch.device('cpu')

        self.data_mgr = SimpleDataManager(dataset_name, data_root, train_test_split_idx)
        self.ent_emb = None

        self.use_attention_rgcn = use_attention_rgcn

        self.rgcn_layers = nn.ModuleList([
            BasicRGCNLayer(
                h_dim=h_dim,
                num_rels=num_rels,
                dropout=dropout,
                self_loop=self_loop,
                skip_connect=True
            )
            for _ in range(num_layers_rgcn)
        ])

        self.feature_adapter = nn.Sequential(
            nn.Linear(h_dim * 3, h_dim),
            nn.LayerNorm(h_dim),        
            nn.GELU(),                  
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim)     
        )

    def set_entity_embedding(self, ent_emb):
        self.ent_emb = ent_emb
        if ent_emb is not None:
            self.device = ent_emb.device

    def set_relation_embedding(self, rel_emb):
        if rel_emb is None:
            raise ValueError("rel_emb cannot be None")

        if isinstance(rel_emb, (nn.Parameter, torch.Tensor)):
            expected_size = 2 * self.num_rels
            actual_size = rel_emb.size(0)
            if actual_size != expected_size:
                raise ValueError(
                    f"rel_emb size mismatch! "
                    f"expected: {expected_size} (2 * num_rels), got: {actual_size}"
                )
            if rel_emb.size(1) != self.h_dim:
                raise ValueError(
                    f"rel_emb dimension mismatch! "
                    f"expected: {self.h_dim}, got: {rel_emb.size(1)}"
                )

        self.global_rel_emb = rel_emb

        for layer in self.rgcn_layers:
            layer.rel_emb = rel_emb

    def _gather_subject_repr(self, g: dgl.DGLGraph, h: torch.Tensor, subj_id: int) -> torch.Tensor:
        ids = g.ndata['id']

        mask = (ids == int(subj_id))
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False)[0].item()
            return h[idx].to(self.device)

        return torch.zeros_like(h[0]).to(self.device)
    
    def forward(self, query_entity_ids: torch.Tensor, rel_ids: torch.Tensor, current_time_idx: int,
                current_time_triples: Optional[torch.Tensor] = None, eval_mode: bool = False,
                direction: str = 'forward', split_name: str = 'train',
                pooled_rel_embs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.ent_emb is None:
            raise ValueError("Entity embedding not set. Call set_entity_embedding() first.")

        is_training = not eval_mode

        graphs = self.data_mgr.get_graphs_for_subjects(int(current_time_idx), query_entity_ids.detach().cpu(),
                                                      split_name=split_name)
        batch_graphs = []
        batch_owner = []

        for i, (sid, g) in enumerate(zip(query_entity_ids.tolist(), graphs)):
            if g is None or g.num_nodes() == 0:
                batch_graphs.append(None)
            else:
                batch_graphs.append(g.to(self.device))
                batch_owner.append(i)

        B = query_entity_ids.shape[0]
        outputs = [None] * B

        if any(g is not None for g in batch_graphs):
            valid_pairs = [(i, g) for i, g in enumerate(batch_graphs) if g is not None]
            valid_idx = [i for i, _ in valid_pairs]
            g_list = [g for _, g in valid_pairs]

            g_list = [g.to(self.device) for g in g_list]
            bg = dgl.batch(g_list)
            bg = bg.to(self.device)

            node_global_ids = bg.ndata['id'].to(self.device)

            max_valid_id = self.num_ents - 1
            valid_mask = (node_global_ids >= 0) & (node_global_ids <= max_valid_id)

            if not valid_mask.all():
                invalid_count = (~valid_mask).sum().item()
                if invalid_count > 0:
                    node_global_ids = torch.where(valid_mask, node_global_ids, torch.zeros_like(node_global_ids))

            h0 = self.ent_emb[node_global_ids]
            h = h0

            prev_h = None
            for layer in self.rgcn_layers:
                h_new = layer(bg, h, prev_h)
                prev_h = h
                h = h_new

            sg_list = dgl.unbatch(bg)
            sg_list = [sg.to(self.device) for sg in sg_list]

            cursor = 0
            for out_pos, sg in zip(valid_idx, sg_list):
                n = sg.num_nodes()
                h_sg = h[cursor: cursor + n]
                cursor += n

                s_id = int(query_entity_ids[out_pos].item())

                outputs[out_pos] = self._gather_subject_repr(sg, h_sg, s_id)

        for i in range(B):
            if outputs[i] is None:
                sid = int(query_entity_ids[i].item())
                outputs[i] = self.ent_emb[sid]

        if any(o is None for o in outputs):
             raise ValueError("Error: static embedding fallback failed")

        outputs = [out.to(self.device) for out in outputs]
        X = torch.stack(outputs, dim=0).to(self.device)

        if not hasattr(self, 'global_rel_emb') or self.global_rel_emb is None:
            raise ValueError("global_rel_emb not set, call set_relation_embedding first")

        rel_ids = rel_ids.to(self.device)

        max_valid_rel_id = self.global_rel_emb.size(0) - 1
        if (rel_ids < 0).any() or (rel_ids > max_valid_rel_id).any():
            invalid_count = ((rel_ids < 0) | (rel_ids > max_valid_rel_id)).sum().item()
            print(f"[WARNING] {invalid_count} out-of-range rel_ids, clipped to [0, {max_valid_rel_id}]")
            rel_ids = torch.clamp(rel_ids, min=0, max=max_valid_rel_id)

        query_rel_emb = self.global_rel_emb[rel_ids]

        pooled_rel = torch.zeros_like(X)

        if pooled_rel_embs is not None:
            pooled_rel = pooled_rel_embs.to(self.device)

        elif current_time_triples is not None and len(current_time_triples) > 0:
            current_time_triples = current_time_triples.to(self.device)

            src = current_time_triples[:, 0]
            rel = current_time_triples[:, 1]
            dst = current_time_triples[:, 2]

            valid_src_mask = (src >= 0) & (src < self.num_ents)
            valid_dst_mask = (dst >= 0) & (dst < self.num_ents)
            valid_rel_mask = (rel >= 0) & (rel < self.num_rels * 2)

            all_ent_rel_sum = torch.zeros(self.num_ents, self.h_dim, device=self.device)
            all_ent_rel_cnt = torch.zeros(self.num_ents, 1, device=self.device)

            if direction == 'forward':
                valid_mask = valid_src_mask & valid_rel_mask
                if valid_mask.any():
                    valid_src = torch.masked_select(src, valid_mask)
                    valid_rel = torch.masked_select(rel, valid_mask)
                    rel_embs = self.global_rel_emb[valid_rel]
                    all_ent_rel_sum.index_add_(0, valid_src, rel_embs)
                    all_ent_rel_cnt.index_add_(0, valid_src, torch.ones_like(valid_src, dtype=torch.float).unsqueeze(1))

            else:
                valid_mask = valid_dst_mask & valid_rel_mask
                if valid_mask.any():
                    valid_dst = torch.masked_select(dst, valid_mask)
                    valid_rel = torch.masked_select(rel, valid_mask)

                    num_rels_half = self.global_rel_emb.size(0) // 2
                    valid_rel_inv = valid_rel + num_rels_half
                    rel_embs_inv = self.global_rel_emb[valid_rel_inv]

                    all_ent_rel_sum.index_add_(0, valid_dst, rel_embs_inv)
                    all_ent_rel_cnt.index_add_(0, valid_dst, torch.ones_like(valid_dst, dtype=torch.float).unsqueeze(1))

            all_ent_rel_cnt = all_ent_rel_cnt.clamp(min=1.0)
            all_ent_rel_mean = all_ent_rel_sum / all_ent_rel_cnt

            valid_query_mask = (query_entity_ids >= 0) & (query_entity_ids < self.num_ents)
            if valid_query_mask.all():
                pooled_rel = all_ent_rel_mean[query_entity_ids.to(self.device)]
            else:
                pooled_rel = torch.zeros_like(X)

        combined_X = torch.cat([X, query_rel_emb, pooled_rel], dim=-1)

        out = self.feature_adapter(combined_X)

        return out
