import numpy as np
import torch
import dgl
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
from typing import Optional


def get_dataset_stat(data_root, dataset_name):
    stat_file = os.path.join(data_root, dataset_name, 'stat.txt')
    if not os.path.exists(stat_file):
        raise FileNotFoundError(f"找不到统计文件: {stat_file}")

    with open(stat_file, 'r', encoding='utf-8') as f:
        stats = f.read().strip().split()
        if len(stats) < 2:
            raise ValueError(f"{stat_file} 格式错误，应包含实体和关系数量")

        num_nodes = int(stats[0])
        num_rels = int(stats[1])

    return num_nodes, num_rels

try:
    from torch.cuda.amp import autocast as _amp_autocast
except Exception:
    _amp_autocast = None

class _NullContext:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): return False

class autocast_if_available:
    def __init__(self, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = torch.cuda.is_available()
        self._enabled = bool(enabled) and (_amp_autocast is not None)
        self._ctx = _amp_autocast(enabled=True) if self._enabled else _NullContext()

    def __enter__(self):
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)

def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
    if not x.is_floating_point():
        x = x.to(torch.float32)

    dtype = x.dtype
    if dtype == torch.float16:
        return torch.tensor(-3e4, dtype=dtype, device=x.device)
    else:
        finfo = torch.finfo(dtype)
        return torch.tensor(finfo.min / 2, dtype=dtype, device=x.device)

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]

        score[i][b_multi] = ground.new_tensor(_neg_inf_like(ground))
        score[i][ans] = ground

    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def filter_score_rhe_online(test_triples, score, all_data_by_time, time_idx, num_rels):
    if all_data_by_time is None or time_idx >= len(all_data_by_time):
        return score

    if not score.is_floating_point():
        score = score.float()

    neg_inf = _neg_inf_like(score)
    tt_cpu = test_triples.detach().cpu()
    B = tt_cpu.shape[0]

    current_slice = all_data_by_time[time_idx]
    if current_slice is None or len(current_slice) == 0:
        return score

    hr_to_objects = {}
    for triple in current_slice:
        h, r, t = int(triple[0]), int(triple[1]), int(triple[2])

        key = (h, r)
        if key not in hr_to_objects:
            hr_to_objects[key] = set()
        hr_to_objects[key].add(t)

        if r < num_rels:
            inv_key = (t, r + num_rels)
            if inv_key not in hr_to_objects:
                hr_to_objects[inv_key] = set()
            hr_to_objects[inv_key].add(h)

    for i in range(B):
        h, r, t = tt_cpu[i]
        h_item, r_item, t_item = int(h), int(r), int(t)

        key = (h_item, r_item)
        if key in hr_to_objects:
            filter_objects = hr_to_objects[key].copy()
            filter_objects.discard(t_item)

            if filter_objects:
                filter_ids = list(filter_objects)
                idx = torch.as_tensor(filter_ids, device=score.device, dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < score.shape[1])]
                if idx.numel() > 0:
                    score[i, idx] = neg_inf

    return score

def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score

    if not score.is_floating_point():
        score = score.float()

    neg_inf = _neg_inf_like(score)
    tt_cpu = test_triples.detach().cpu()
    B = tt_cpu.shape[0]

    for i in range(B):
        h, r, t = tt_cpu[i]
        h_item, r_item, t_item = int(h), int(r), int(t)

        if h_item in all_ans and r_item in all_ans[h_item]:
            cand = all_ans[h_item][r_item]

            if t_item in cand and len(cand) == 1:
                continue

            ids = [x for x in cand if x != t_item]
            if not ids:
                continue

            idx = torch.as_tensor(ids, device=score.device, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < score.shape[1])]
            if idx.numel() > 0:
                score[i, idx] = neg_inf

    return score


def filter_score_r_rhe_online(test_triples, score, all_data_by_time, time_idx, num_rels):
    if all_data_by_time is None or time_idx >= len(all_data_by_time):
        return score

    if not score.is_floating_point():
        score = score.float()

    neg_inf = _neg_inf_like(score)
    tt_cpu = test_triples.detach().cpu()
    B = tt_cpu.shape[0]

    current_slice = all_data_by_time[time_idx]
    if current_slice is None or len(current_slice) == 0:
        return score

    ht_to_relations = {}
    for triple in current_slice:
        h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
        key = (h, t)
        if key not in ht_to_relations:
            ht_to_relations[key] = set()
        ht_to_relations[key].add(r)

    for i in range(B):
        h, r, t = tt_cpu[i]
        h_item, r_item, t_item = int(h), int(r), int(t)

        key = (h_item, t_item)
        if key in ht_to_relations:
            filter_relations = ht_to_relations[key].copy()
            filter_relations.discard(r_item)

            if filter_relations:
                filter_ids = list(filter_relations)
                idx = torch.as_tensor(filter_ids, device=score.device, dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < score.shape[1])]
                if idx.numel() > 0:
                    score[i, idx] = neg_inf

    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    if not score.is_floating_point():
        score = score.float()
    neg_inf = _neg_inf_like(score)
    tt_cpu = test_triples.detach().cpu()
    B = tt_cpu.shape[0]
    for i in range(B):
        h, r, t = tt_cpu[i]
        h_item, r_item, t_item = int(h), int(r), int(t)
        if h_item in all_ans and t_item in all_ans[h_item]:
            cand = all_ans[h_item][t_item]
            if r_item in cand and len(cand) == 1:
                continue
            ids = [x for x in cand if x != r_item]
            if not ids:
                continue
            idx = torch.as_tensor(ids, device=score.device, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < score.shape[1])]
            if idx.numel() == 0:
                continue
            score[i, idx] = neg_inf
    return score

def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0, all_data_by_time=None, time_idx=None, num_rels=None):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank_filter, rank = [], []

    use_rhe_filtering = (all_data_by_time is not None and time_idx is not None and num_rels is not None)
    use_global_filtering = (all_ans is not None)
    
    for idx in range(n_batch):
        s, e = idx * eval_bz, min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[s:e]
        score_batch = score[s:e]

        if not score_batch.is_floating_point():
            score_batch = score_batch.float()

        target = triples_batch[:, 2] if rel_predict == 0 else triples_batch[:, 1]

        rank_batch = sort_and_rank(score_batch, target)
        rank.extend(rank_batch.cpu().numpy())

        if use_rhe_filtering:
            if rel_predict == 0:
                score_batch_filter = filter_score_rhe_online(
                    triples_batch, score_batch.clone(), all_data_by_time, time_idx, num_rels
                )
            else:
                score_batch_filter = filter_score_r_rhe_online(
                    triples_batch, score_batch.clone(), all_data_by_time, time_idx, num_rels
                )
        elif use_global_filtering:
            if rel_predict == 0:
                score_batch_filter = filter_score(triples_batch, score_batch.clone(), all_ans)
            else:
                score_batch_filter = filter_score_r(triples_batch, score_batch.clone(), all_ans)
        else:
            score_batch_filter = score_batch.clone()

        rank_batch_filter = sort_and_rank(score_batch_filter, target)
        rank_filter.extend(rank_batch_filter.cpu().numpy())
    
    rank = np.array(rank); rank_filter = np.array(rank_filter)
    mrr = np.mean(1.0 / (rank + 1))
    mrr_filter = np.mean(1.0 / (rank_filter + 1))

    return mrr_filter, mrr, rank_filter.tolist(), rank.tolist()

def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    all_ranks = []

    for ranks in rank_list:
        if isinstance(ranks, torch.Tensor):
            all_ranks.extend(ranks.cpu().numpy().tolist())
        elif isinstance(ranks, np.ndarray):
            all_ranks.extend(ranks.tolist())
        elif isinstance(ranks, list):
            all_ranks.extend(ranks)
        else:
            all_ranks.append(ranks)

    total_rank = torch.tensor(all_ranks, dtype=torch.float32)
    mrr = torch.mean(1.0 / (total_rank + 1))

    hit_values = []
    for hit in hits:
        avg_count = torch.mean((total_rank < hit).float())
        hit_values.append(avg_count.item())

    return mrr, hit_values


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)

    src, rel, dst = triples.transpose()
    g.add_edges(src, dst)

    norm = comp_deg_norm(g)
    g.ndata.update({'norm': norm.view(-1, 1)})

    g.edata['type'] = torch.LongTensor(rel)

    g = dgl.add_self_loop(g)

    if use_cuda:
        g.ndata.update({k: v.cuda(gpu) for k, v in g.ndata.items()})
        g.edata.update({k: v.cuda(gpu) for k, v in g.edata.items()})

    return g

def build_graph(num_nodes, num_rels, triples, use_cuda=False, gpu=0):
    src, rel, dst = triples.transpose()

    max_rel_id = rel.max() if len(rel) > 0 else 0
    if max_rel_id >= num_rels * 2:
        print(f"[WARNING] 发现无效关系ID (max_id={max_rel_id} >= 2*num_rels={num_rels*2})，进行裁剪...")
        valid_mask = (rel < num_rels * 2) & (rel >= 0) & (src >= 0) & (dst >= 0)
        src = src[valid_mask]
        dst = dst[valid_mask]
        rel = rel[valid_mask]
        print(f"[WARNING] 裁剪后剩余边数: {len(src)}")

    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.graph((src, dst), num_nodes=num_nodes)

    g.edata['type'] = torch.LongTensor(rel)

    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    g.ndata['norm'] = norm.view(-1, 1)

    if use_cuda:
        g = g.to(torch.device(f'cuda:{gpu}'))

    return g

def build_graphs_for_snapshots(train_list, num_nodes, num_rels, use_cuda=False, gpu=0):
    graphs = []
    for snapshot in train_list:
        if len(snapshot) > 0:
            g = build_graph(num_nodes, num_rels, snapshot, use_cuda, gpu)
            graphs.append(g)
        else:
            graphs.append(None)
    return graphs


def append_object(e1, e2, r, d):
    if not e1 in d: d[e1] = {}
    if not r in d[e1]: d[e1][r] = set()
    d[e1][r].add(e2)

def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d: d[e2] = {}
    if not r+num_rel in d[e2]: d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)

def add_object(e1, e2, r, d, num_rel):
    if not e1 in d: d[e1] = {}
    if not r in d[e1]: d[e1][r] = set()
    d[e1][r].add(e2)

def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    def _append_object(e1, e2, r, d):
        if not e1 in d: d[e1] = {}
        if not r in d[e1]: d[e1][r] = set()
        d[e1][r].add(e2)

    all_ans = {}

    for triple in total_data:
        if rel_p:
            _append_object(triple[0], triple[1], triple[2], all_ans)
        else:
            _append_object(triple[0], triple[2], triple[1], all_ans)

    return all_ans

def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for t_idx in range(len(all_snap)):
        all_ans_t = {}
        for snap_idx in range(t_idx + 1):
            snap = all_snap[snap_idx]
            for triple in snap:
                if rel_p:
                    append_object(triple[0], triple[1], triple[2], all_ans_t)
                else:
                    append_object(triple[0], triple[2], triple[1], all_ans_t)
        all_ans_list.append(all_ans_t)
    return all_ans_list


def construct_snap(test_triples, num_nodes, num_rels, final_score, topk):
    sorted_score, sorted_idx = torch.sort(final_score, dim=1, descending=True)
    top_idx = sorted_idx[:, :topk]

    constructed_snap = []
    for i in range(len(test_triples)):
        h, r = test_triples[i][0], test_triples[i][1]
        for j in range(topk):
            t = top_idx[i][j].item()
            constructed_snap.append([h, r, t])

    return np.array(constructed_snap)

def construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, topk):
    sorted_score, sorted_idx = torch.sort(final_r_score, dim=1, descending=True)
    top_idx = sorted_idx[:, :topk]

    constructed_snap = []
    for i in range(len(test_triples)):
        h, t = test_triples[i][0], test_triples[i][2]
        for j in range(topk):
            r = top_idx[i][j].item()
            constructed_snap.append([h, r, t])

    return np.array(constructed_snap)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def _read_triplets_as_list(filename, load_time):
    if not os.path.exists(filename):
        return []

    l = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                if len(parts) >= 4:
                    h, r, t, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    l.append([h, r, t, ts])
                elif len(parts) == 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    if load_time:
                        l.append([h, r, t, 0])
                    else:
                        l.append([h, r, t])
                else:
                    continue
            except ValueError:
                continue

    return l

class Data:
    def __init__(self, train, valid, test, num_nodes, num_rels):
        self.train = train
        self.valid = valid
        self.test = test
        self.num_nodes = num_nodes
        self.num_rels = num_rels

def load_from_local(dir, dataset, load_time=True):
    train_path = os.path.join(dir, dataset, 'train.txt')
    valid_path = os.path.join(dir, dataset, 'valid.txt')
    test_path  = os.path.join(dir, dataset, 'test.txt')
    stat_path  = os.path.join(dir, dataset, 'stat.txt')

    with open(stat_path, 'r') as f:
        parts = f.readline().strip().split()
        if len(parts) < 2:
            raise ValueError(f"stat.txt 格式不正确：期望至少2列，实际: {parts}")
        num_nodes, num_rels = int(parts[0]), int(parts[1])

    train = _read_triplets_as_list(train_path, load_time)
    valid = _read_triplets_as_list(valid_path, load_time)
    test  = _read_triplets_as_list(test_path,  load_time)

    return Data(train, valid, test, num_nodes, num_rels)

def load_data(dataset, bfs_level=3, relabel=False, load_time=False):
    return load_from_local('./data', dataset, load_time)

def split_by_time(data):
    snapshot_list = []

    if data is None or len(data) == 0:
        return snapshot_list
    
    first_item = data[0]
    if hasattr(first_item, '__len__') and len(first_item) < 4:
        return [np.array([[triple[0], triple[1], triple[2]] for triple in data])]
    snapshots = defaultdict(list)
    for triple in data:
        time_stamp = triple[3]
        snapshots[time_stamp].append([triple[0], triple[1], triple[2]])
    sorted_times = sorted(snapshots.keys())
    for time_stamp in sorted_times:
        snapshot_list.append(np.array(snapshots[time_stamp]))
    if len(snapshot_list) > 0:
        try:
            nodes = [len(set(snap[:, 0]) | set(snap[:, 2])) for snap in snapshot_list]
            rels = [len(set(snap[:, 1])) for snap in snapshot_list]
            edges_per_snapshot = [len(snap) for snap in snapshot_list]
            

            entity_sets = [set(snap[:, 0]) | set(snap[:, 2]) for snap in snapshot_list]
            relation_sets = [set(snap[:, 1]) for snap in snapshot_list]
            
            total_entities = len(set().union(*entity_sets)) if entity_sets else 0
            total_relations = len(set().union(*relation_sets)) if relation_sets else 0
            
            avg_entity_coverage = np.mean([n / total_entities for n in nodes]) if total_entities > 0 else 0
            avg_relation_coverage = np.mean([r / (total_relations * 2) for r in rels]) if total_relations > 0 else 0
            
            print("# Sanity Check:  ave node num : {:.4f}, ave rel num : {:.4f}, snapshots num: {:04d}, "
                  "max edges num: {:04d}, min edges num: {:04d}, total entities: {:04d}, total relations: {:04d}, "
                  "avg entity coverage: {:.4f}, avg relation coverage: {:.4f}"
                  .format(np.average(nodes), np.average(rels), len(snapshot_list),
                          max(edges_per_snapshot), min(edges_per_snapshot),
                          total_entities, total_relations, avg_entity_coverage, avg_relation_coverage))
        except Exception as e:
            print(f"警告: 统计信息计算失败: {e}")
    else:
        print("警告: 没有时间片数据")
    return snapshot_list

def load_time_snapshot_graphs(dataset_name, time_idx, data_root="./data"):
    graph_file = os.path.join(data_root, dataset_name, 'graph_snapshots', f'graph_{time_idx}.pkl')
    if not os.path.exists(graph_file):
        return None
    try:
        with open(graph_file, 'rb') as f:
            entity_graphs = pickle.load(f)
        return entity_graphs
    except Exception as e:
        print(f"Warning: Failed to load graph snapshot {graph_file}: {e}")
        return None

if __name__ == '__main__':
    print("Unified utility module loaded successfully!")
