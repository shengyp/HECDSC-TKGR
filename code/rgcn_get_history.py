import os
import argparse
import time
import gc
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np
import torch
import dgl

def load_quadruples(path: str) -> np.ndarray:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            h, r, t, ts = parts[:4]
            data.append((int(h), int(r), int(t), int(ts)))
    arr = np.array(data, dtype=np.int64)
    return arr

def split_by_time(quads: np.ndarray) -> Tuple[np.ndarray, Dict[int, List[Tuple[int,int,int]]]]:
    times = np.unique(quads[:, 3])
    times = np.sort(times)
    time2triples: Dict[int, List[Tuple[int,int,int]]] = defaultdict(list)

    for h, r, t, ts in quads:
        time2triples[int(ts)].append((int(h), int(r), int(t)))
    return times, time2triples

def build_subject_event_lists(historical_triples: List[List[Tuple[int,int,int]]],
                              current_triples: List[Tuple[int,int,int]], num_rels: int,
                              max_events_per_subject: int) -> Dict[int, List[Tuple[int,int,int]]]:
    incident = defaultdict(list)
    direct_neighbors = defaultdict(set)

    for triples in historical_triples:
        for h, r, t in triples:
            incident[h].append((h, r, t, True))
            direct_neighbors[h].add(t)
            incident[t].append((t, r + num_rels, h, True))
            direct_neighbors[t].add(h)

    query_entities = set()
    for h, r, t in current_triples:
        query_entities.add(h)
        query_entities.add(t)

    all_entities = set(incident.keys()) | set(direct_neighbors.keys())
    max_neighbors_per_entity = min(10, max_events_per_subject // 2)
    max_events_per_neighbor = 5

    for e in all_entities:
        neighbors = direct_neighbors.get(e, set())

        limited_neighbors = list(neighbors)[:max_neighbors_per_entity]

        for n in limited_neighbors:
            neighbor_events = incident.get(n, [])

            limited_events = neighbor_events[:max_events_per_neighbor]
            for (src, r, dst, _) in limited_events:
                incident[e].append((src, r, dst, False))

    subj2events: Dict[int, List[Tuple[int,int,int]]] = {}
    for e in all_entities:
        if e not in query_entities:
            continue

        direct = [ (s,r,d) for (s,r,d,is_dir) in incident[e] if is_dir ]
        neighb = [ (s,r,d) for (s,r,d,is_dir) in incident[e] if not is_dir ]

        ordered = direct + neighb
        seen = set()
        deduped = []

        for s, r, d in ordered:
            key = (s, r, d)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)

        if len(deduped) > max_events_per_subject:
            deduped = deduped[:max_events_per_subject]

        if len(deduped) == 0:
            print(f"Warning: Subject {e} has no events after deduplication, creating minimal graph")
            subj2events[e] = [(e, 0, e)]
        else:
            subj2events[e] = deduped

    return subj2events

def build_graph_from_events(events: List[Tuple[int,int,int]]) -> dgl.DGLGraph:
    nodes = set()
    for s, r, o in events:
        nodes.add(s); nodes.add(o)
    nodes = sorted(list(nodes))
    global2local = {nid:i for i, nid in enumerate(nodes)}

    if len(events) == 0:
        g = dgl.graph(([], []), num_nodes=len(nodes))
        g.ndata['id'] = torch.tensor(nodes, dtype=torch.int64) if nodes else torch.zeros((0,), dtype=torch.int64)
        g.ndata['norm'] = torch.ones((len(nodes), 1))
        g.edata['type'] = torch.zeros((0,), dtype=torch.int64)
        return g

    src = [global2local[s] for (s,_,o) in events]
    dst = [global2local[o] for (s,_,o) in events]
    ety = [r for (_,r,_) in events]

    g = dgl.graph((torch.tensor(src, dtype=torch.int64),
                   torch.tensor(dst, dtype=torch.int64)),
                  num_nodes=len(nodes))
    g.edata['type'] = torch.tensor(ety, dtype=torch.int64)
    g.ndata['id'] = torch.tensor(nodes, dtype=torch.int64)

    indeg = g.in_degrees().float().clamp(min=1.0)
    g.ndata['norm'] = (1.0 / torch.sqrt(indeg)).unsqueeze(-1)
    return g

def save_time_slice_graphs(save_dir: str, t_idx: int, graphs: List[dgl.DGLGraph], subject_ids: List[int]):
    path = os.path.join(save_dir, f'graph_{t_idx}.bin')
    aux = {'subject_ids': torch.tensor(subject_ids, dtype=torch.int64)}

    if not graphs:
        empty_g = dgl.graph(([], []), num_nodes=0)
        empty_g.ndata['id'] = torch.zeros((0,), dtype=torch.int64)
        empty_g.ndata['norm'] = torch.zeros((0, 1))
        empty_g.edata['type'] = torch.zeros((0,), dtype=torch.int64)
        graphs = [empty_g]

    dgl.save_graphs(path, graphs, aux)

def process_dataset_graphs(dataset_name: str, data_root: str, save_root: str,
                          max_events_per_subject: int, history_window: int = 5,
                          num_relations: int = None,
                          process_train: bool = True, process_test: bool = True):
    ds_dir = os.path.join(data_root, dataset_name)

    if num_relations is None:
        raise ValueError("num_relations 不能为 None，必须从 stat.txt 中读取！")
    num_rels = num_relations

    print(f'[{dataset_name}] Processing with num_relations={num_rels}, global time indexing')

    latest_t = 0

    train_history_context = None

    if process_train:
        train_path = os.path.join(ds_dir, 'train.txt')
        if os.path.exists(train_path):
            print(f'Processing training set...')
            train_save_dir = os.path.join(save_root, dataset_name, 'graph_snapshots_train')
            os.makedirs(train_save_dir, exist_ok=True)

            train_quads = load_quadruples(train_path)
            latest_t, train_history_context = _process_data_split(
                train_quads, num_rels, max_events_per_subject, history_window,
                train_save_dir, dataset_name, 'train',
                initial_history=None, latest_t=latest_t
            )
            print(f'Training set processed, latest_t={latest_t}')

    if process_test:
        test_path = os.path.join(ds_dir, 'test.txt')
        if os.path.exists(test_path):
            print(f'Processing test set (continuing from latest_t={latest_t})...')
            test_save_dir = os.path.join(save_root, dataset_name, 'graph_snapshots_test')
            os.makedirs(test_save_dir, exist_ok=True)

            test_quads = load_quadruples(test_path)

            if train_history_context is not None:
                print(f"  Using training set history context for test set (history_window={history_window})")
                initial_history = train_history_context[-history_window:] if len(train_history_context) >= history_window else train_history_context
            elif not process_train and os.path.exists(os.path.join(ds_dir, 'train.txt')):
                print("Loading train data to build history context for test set...")
                temp_train = load_quadruples(os.path.join(ds_dir, 'train.txt'))
                _, temp_time2triples = split_by_time(temp_train)
                temp_times = sorted(temp_time2triples.keys())

                initial_history = []
                for ts in temp_times[-history_window:]:
                    initial_history.append(temp_time2triples[ts])
            else:
                initial_history = None

            _process_data_split(
                test_quads, num_rels, max_events_per_subject, history_window,
                test_save_dir, dataset_name, 'test',
                initial_history=initial_history, latest_t=latest_t
            )
            print(f'Test set processed with global time indexing')

def _process_data_split(quads: np.ndarray, num_rels: int, max_events_per_subject: int,
                       history_window: int, save_dir: str, dataset_name: str, split_name: str,
                       initial_history: List[List[Tuple[int,int,int]]] = None, latest_t: int = 0):
    times, time2triples = split_by_time(quads)

    print(f'[{dataset_name}-{split_name}] unique times: {len(times)}, history_window: {history_window}, global latest_t: {latest_t}')

    for i, ts in enumerate(times):
        global_time_idx = latest_t + i
        start_time = time.time()

        historical_triples = []

        for j in range(max(0, global_time_idx - history_window), global_time_idx):
            if j >= latest_t:
                local_idx = j - latest_t
                if local_idx < len(times):
                    hist_ts = times[local_idx]
                    historical_triples.append(time2triples[int(hist_ts)])
            elif initial_history and (j - latest_t + len(initial_history)) >= 0:
                hist_idx = j - latest_t + len(initial_history)
                if hist_idx < len(initial_history):
                    historical_triples.append(initial_history[hist_idx])

        historical_triples.reverse()

        current_triples = time2triples[int(ts)]
        print(f'Processing {split_name} t={i} (original_ts={ts}) with {len(current_triples)} current triples, {len(historical_triples)} historical slices...')

        if len(current_triples) > 10000:
            print(f'  Warning: t={i} has {len(current_triples)} triples, which is very large. Skipping...')
            save_time_slice_graphs(save_dir, i, [], [])
            continue

        try:
            subj2events = build_subject_event_lists(historical_triples, current_triples, num_rels, max_events_per_subject)
            print(f'  Built event lists for {len(subj2events)} subjects from historical data')

            graphs = []
            subj_ids = []
            for j, (subj, evs) in enumerate(subj2events.items()):
                if j % 50 == 0 and j > 0:
                    elapsed = time.time() - start_time
                    print(f'  Processing subject {j}/{len(subj2events)} (elapsed: {elapsed:.1f}s)...')

                    if elapsed > 300:
                        print(f'  Timeout reached for t={i}, saving partial results...')
                        break

                g = build_graph_from_events(evs)
                graphs.append(g)
                subj_ids.append(subj)

            save_time_slice_graphs(save_dir, global_time_idx, graphs, subj_ids)
            graph_file = os.path.join(save_dir, f"graph_{global_time_idx}.bin")
            elapsed = time.time() - start_time
            print(f'{split_name} t={global_time_idx} (original_ts={ts}) | subjects={len(graphs)} saved: {graph_file} (took {elapsed:.1f}s)')

            if i % 5 == 0:
                gc.collect()

        except Exception as e:
            print(f'Error processing {split_name} t={i}: {e}')
            save_time_slice_graphs(save_dir, i, [], [])
            continue

    print(f'Total {len(times)} {split_name} time slices processed and saved.')

    times, time2triples = split_by_time(quads)
    history_context = [time2triples[ts] for ts in sorted(time2triples.keys())]

    new_latest_t = latest_t + len(times)
    return new_latest_t, history_context

def main():
    ap = argparse.ArgumentParser(description='时序知识图谱历史数据预处理工具')
    ap.add_argument('--dataset', type=str, required=True, help='数据集名称，例如 YAGO 或 WIKI')
    ap.add_argument('--data-root', type=str, default='./data', help='数据根目录')
    ap.add_argument('--save-root', type=str, default='./data', help='保存根目录')
    ap.add_argument('--max-events-per-subject', type=int, default=40, help='每个主体的最大事件数')
    ap.add_argument('--history-window', type=int, default=8, help='历史时间片窗口大小（默认为5）')
    ap.add_argument('--process-train', action='store_true', default=True, help='处理训练集')
    ap.add_argument('--process-test', action='store_true', default=True, help='处理测试集')
    ap.add_argument('--train-only', action='store_true', help='仅处理训练集')
    ap.add_argument('--test-only', action='store_true', help='仅处理测试集')
    args = ap.parse_args()

    process_train = not args.test_only
    process_test = not args.train_only

    if args.train_only:
        print("仅处理训练集")
    elif args.test_only:
        print("仅处理测试集")
    else:
        print("处理训练集和测试集")

    stat_file = os.path.join(args.data_root, args.dataset, 'stat.txt')
    if os.path.exists(stat_file):
        with open(stat_file, 'r', encoding='utf-8') as f:
            stats = f.read().strip().split()
            num_nodes = int(stats[0])
            num_rels = int(stats[1])
        print(f"[{args.dataset}] 从 stat.txt 中读取统计信息 - 实体数: {num_nodes}, 关系数: {num_rels}")
    else:
        raise FileNotFoundError(f"预处理失败：未找到 {stat_file}，无法获取准确的节点和关系总数！")

    process_dataset_graphs(
        dataset_name=args.dataset,
        data_root=args.data_root,
        save_root=args.save_root,
        max_events_per_subject=args.max_events_per_subject,
        history_window=args.history_window,
        num_relations=num_rels,
        process_train=process_train,
        process_test=process_test
    )

if __name__ == '__main__':
    main()
