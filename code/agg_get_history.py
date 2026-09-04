import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
import argparse
import gc
import pickle


def format_history_for_transformer(raw_events, safe_max_seq_len=50, max_objs_per_step=10):
    if not raw_events:
        return []

    time_groups = defaultdict(list)

    for event in raw_events:
        tail_id = int(event[0])
        ts = float(event[2])
        time_groups[ts].append(tail_id)

    sorted_times = sorted(time_groups.keys(), reverse=True)

    selected_times = sorted_times[:safe_max_seq_len]

    processed_history = []
    for ts in selected_times:
        tails = time_groups[ts][:max_objs_per_step]
        processed_history.append((ts, tails))

    return processed_history


def load_quadruples(inPath, fileName, fileName2=None):
    quadrupleList = []
    times = set()

    file_path = os.path.join(inPath, fileName)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as fr:
        for line_num, line in enumerate(fr, 1):
            line = line.strip()
            if not line:
                continue

            line_split = line.split()
            if len(line_split) < 4:
                continue

            if not all(part.isdigit() for part in line_split[:4]):
                continue

            head = int(line_split[0])
            rel = int(line_split[1])
            tail = int(line_split[2])
            time = int(line_split[3])

            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    if fileName2 is not None:
        file_path2 = os.path.join(inPath, fileName2)
        if os.path.exists(file_path2):
            with open(file_path2, 'r', encoding='utf-8') as fr:
                for line_num, line in enumerate(fr, 1):
                    line = line.strip()
                    if not line:
                        continue

                    line_split = line.split()
                    if len(line_split) < 4:
                        continue

                    if not all(part.isdigit() for part in line_split[:4]):
                        continue

                    head = int(line_split[0])
                    rel = int(line_split[1])
                    tail = int(line_split[2])
                    time = int(line_split[3])
                    quadrupleList.append([head, rel, tail, time])
                    times.add(time)

    if not quadrupleList:
        raise ValueError("没有成功加载任何有效数据")

    times = sorted(list(times))
    return np.asarray(quadrupleList), np.asarray(times)

def update_dict(subg_arr, s_to_sro, sr_to_sro, num_rels, sr_time_map=None, t_idx=None):
    if len(subg_arr) == 0:
        return

    try:
        inverse_subg = subg_arr[:, [2, 1, 0]].copy()
        inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels

        subg_triples = np.concatenate([subg_arr, inverse_subg])

        for (src, rel, dst) in subg_triples:
            src, rel, dst = int(src), int(rel), int(dst)

            s_to_sro[src].add((src, rel, dst))

            sr_key = (src, rel)
            sr_to_sro[sr_key].add(dst)

            if sr_time_map is not None and t_idx is not None:
                sr_time_map[sr_key][dst] = int(t_idx)

    except Exception as e:
        raise

def split_by_time(data):
    if len(data) == 0:
        return []

    snapshot_list = []
    snapshot = []
    latest_t = None

    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t is None or latest_t != t:
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
            snapshot = []
        snapshot.append(train[:3])

    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())

    if not snapshot_list:
        return []

    nodes, rels = [], []
    edges_per_snapshot = []
    all_entities = set()
    all_relations = set()

    for snapshot in snapshot_list:
        if len(snapshot) == 0:
            continue
        uniq_v = np.unique(np.concatenate([snapshot[:,0], snapshot[:,2]]))
        uniq_r = np.unique(snapshot[:,1])

        nodes.append(len(uniq_v))
        rels.append(len(uniq_r) * 2)
        edges_per_snapshot.append(len(snapshot))

        all_entities.update(uniq_v)
        all_relations.update(uniq_r)

    total_entities = len(all_entities)
    total_relations = len(all_relations)

    return snapshot_list

def get_total_number(inPath, fileName):
    file_path = os.path.join(inPath, fileName)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"统计文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue

            line_split = line.split()
            if len(line_split) < 2:
                continue

            try:
                num_entities = int(line_split[0])
                num_relations = int(line_split[1])
                if num_entities <= 0 or num_relations <= 0:
                    continue
                return num_entities, num_relations
            except ValueError:
                continue

    raise ValueError(f"无法从统计文件 {file_path} 中读取有效的实体数和关系数")




def process_dataset(dataset_name, data_root="./data", save_root="./data",
                   safe_max_seq_len=50, max_objs_per_step=10,
                   process_train=True, process_test=True):

    dataset_path = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_path):
        return False

    num_nodes, num_rels = get_total_number(dataset_path, 'stat.txt')
    success = True

    sr_to_sro = defaultdict(set)
    s_to_sro = defaultdict(set)
    sr_time_map = defaultdict(dict)

    latest_t = 0

    if process_train:
        print(f"  处理训练集...")
        train_data, train_times = load_quadruples(dataset_path, 'train.txt')
        train_list = split_by_time(train_data)
        if train_list:
            latest_t, sr_to_sro, s_to_sro, sr_time_map = _process_data_split(
                dataset_name, train_list, save_root,
                'train', num_rels, latest_t,
                sr_to_sro, s_to_sro, sr_time_map,
                safe_max_seq_len, max_objs_per_step
            )
            print(f"  训练集处理完成，latest_t={latest_t}")
        else:
            print(f"  警告: 训练集为空")

    if process_test:
        print(f"  处理测试集（继承训练集历史状态，latest_t={latest_t}）...")
        test_data, test_times = load_quadruples(dataset_path, 'test.txt')
        test_list = split_by_time(test_data)
        if test_list:
            _process_data_split(dataset_name, test_list, save_root,
                              'test', num_rels, latest_t,
                              sr_to_sro, s_to_sro, sr_time_map,
                              safe_max_seq_len, max_objs_per_step)
            print(f"  测试集处理完成")
        else:
            print(f"  警告: 测试集为空")

    return success


def _process_data_split(dataset_name, data_list, save_root, split_name, global_num_rels, latest_t=0,
                       shared_sr_to_sro=None, shared_s_to_sro=None, shared_sr_time_map=None,
                       safe_max_seq_len=50, max_objs_per_step=10):
    idx = list(range(len(data_list)))

    if shared_sr_to_sro is not None:
        sr_to_sro = shared_sr_to_sro
        s_to_sro = shared_s_to_sro
        sr_time_map = shared_sr_time_map
    else:
        sr_to_sro = defaultdict(set)
        s_to_sro = defaultdict(set)
        sr_time_map = defaultdict(dict)

    save_dir_snap = os.path.join(save_root, dataset_name, f'his_dict_snap_{split_name}')
    save_dir_all = os.path.join(save_root, dataset_name, f'his_dict_all_{split_name}')
    save_dir_time_snap = os.path.join(save_root, dataset_name, f'his_time_dict_snap_{split_name}')
    save_dir_time_all = os.path.join(save_root, dataset_name, f'his_time_dict_all_{split_name}')

    for save_dir in [save_dir_snap, save_dir_all, save_dir_time_snap, save_dir_time_all]:
        os.makedirs(save_dir, exist_ok=True)

    for sample_num in tqdm(idx, desc=f"处理{dataset_name}-{split_name}"):
        global_time_idx = latest_t + sample_num

        snap_file = os.path.join(save_dir_snap, f"global_sro_{global_time_idx}.npy")
        time_snap_file = os.path.join(save_dir_time_snap, f"global_sro_time_{global_time_idx}.npy")

        min_valid_time = max(0, global_time_idx - safe_max_seq_len)

        preprocessed_history = {}
        empty_keys = []

        for sr_key, objs in sr_to_sro.items():
            raw_events = []
            objs_to_remove = []

            for obj in objs:
                ts = sr_time_map.get(sr_key, {}).get(obj, -1)
                if ts >= min_valid_time:
                    raw_events.append((obj, sr_key[1], ts))
                else:
                    objs_to_remove.append(obj)

            for obj in objs_to_remove:
                objs.remove(obj)
                if obj in sr_time_map.get(sr_key, {}):
                    del sr_time_map[sr_key][obj]

            if not objs:
                empty_keys.append(sr_key)
            elif raw_events:
                preprocessed_history[sr_key] = set(item[0] for item in raw_events)

        for k in empty_keys:
            del sr_to_sro[k]
            if k in sr_time_map:
                del sr_time_map[k]

        gc.disable()
        try:
            with open(snap_file, 'wb') as f:
                pickle.dump(preprocessed_history, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(time_snap_file, 'wb') as f:
                pickle.dump(sr_time_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            gc.enable()

        if sample_num < len(data_list):
            current_graph = data_list[sample_num]

            if len(current_graph) > 0:
                update_dict(current_graph,
                           s_to_sro, sr_to_sro, global_num_rels,
                           sr_time_map=sr_time_map, t_idx=global_time_idx)

        if sample_num % 10 == 0:
            gc.collect()

    all_file = os.path.join(save_dir_all, f"{split_name}_s_r.npy")
    time_all_file = os.path.join(save_dir_time_all, f"{split_name}_s_r_time.npy")

    print(f"\n  正在极速保存全局汇总字典...")
    gc.disable()
    try:
        with open(all_file, 'wb') as f:
            pickle.dump(dict(sr_to_sro), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(time_all_file, 'wb') as f:
            pickle.dump(dict(sr_time_map), f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        gc.enable()

    try:
        data_manager = AGGDataManager(dataset_name, save_root, split_name)
        data_manager.create_time_index(len(data_list), split_name, latest_t)
        data_manager.save_agg_info(sr_to_sro, sr_time_map, split_name)
    except Exception as e:
        print(f"创建AGG混合存储索引({split_name})时出错: {e}")

    new_latest_t = latest_t + len(data_list)
    return new_latest_t, sr_to_sro, s_to_sro, sr_time_map


class AGGDataManager:

    def __init__(self, dataset_name, data_root="./data", split_name="train"):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.split_name = split_name

        self.cache_dir = os.path.join(data_root, dataset_name, 'hybrid_cache')

        self.legacy_snap_dir = os.path.join(data_root, dataset_name, f'his_dict_snap_{split_name}')
        self.legacy_all_dir = os.path.join(data_root, dataset_name, f'his_dict_all_{split_name}')

        os.makedirs(self.cache_dir, exist_ok=True)
    
    def create_time_index(self, num_time_slices, split_name=None, latest_t=0):
        if split_name is None:
            split_name = self.split_name

        time_index = {}

        for local_idx in range(0, num_time_slices):
            global_time_idx = latest_t + local_idx
            snap_file = os.path.join(self.legacy_snap_dir, f"global_sro_{global_time_idx}.npy")
            if os.path.exists(snap_file):
                file_size = os.path.getsize(snap_file)
                data = np.load(snap_file, allow_pickle=True)

                time_index[global_time_idx] = {
                    'snap_file': f"global_sro_{global_time_idx}.npy",
                    'file_size_bytes': file_size,
                    'num_pairs': len(data),
                    'total_objects': sum(len(objects) for objects in data.values()) if data else 0
                }

        index_file = os.path.join(self.cache_dir, f"agg_time_index_{split_name}.pkl")
        with open(index_file, 'wb') as f:
            pickle.dump(time_index, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"AGG{split_name}时间片索引保存成功: {index_file}")
        print(f"包含 {len(time_index)} 个时间片的索引")
        return True
    
    def save_agg_info(self, sr_to_sro, sr_time_map, split_name=None):
        if split_name is None:
            split_name = self.split_name

        agg_info = {
            'dataset_name': self.dataset_name,
            'split_name': split_name,
            'storage_type': 'time_slice_based',
            'snap_dir': f'his_dict_snap_{split_name}',
            'all_dir': f'his_dict_all_{split_name}',
            'time_snap_dir': f'his_time_dict_snap_{split_name}',
            'time_all_dir': f'his_time_dict_all_{split_name}',
            'total_sr_pairs': len(sr_to_sro),
            'total_objects': sum(len(objects) for objects in sr_to_sro.values()),
            'has_time_info': len(sr_time_map) > 0
        }

        info_file = os.path.join(self.cache_dir, f"agg_storage_info_{split_name}.pkl")
        with open(info_file, 'wb') as f:
            pickle.dump(agg_info, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"AGG{split_name}存储信息保存成功: {info_file}")
        return True
    
    def load_time_slice_data(self, time_idx, split_name=None):
        if split_name is None:
            split_name = self.split_name

        hybrid_file = os.path.join(self.cache_dir, f"agg_{split_name}_sro_{time_idx}.npy")
        legacy_file = os.path.join(self.legacy_snap_dir, f"global_sro_{time_idx}.npy")

        if os.path.exists(hybrid_file):
            data = np.load(hybrid_file, allow_pickle=True)
            return data

        if os.path.exists(legacy_file):
            data = np.load(legacy_file, allow_pickle=True)
            return data

        print(f"{split_name}时间片 {time_idx} 的数据文件不存在 (已尝试: {hybrid_file} 和 {legacy_file})")
        return None
    
    def get_time_index(self, split_name=None):
        if split_name is None:
            split_name = self.split_name

        index_file = os.path.join(self.cache_dir, f"agg_time_index_{split_name}.pkl")

        if not os.path.exists(index_file):
            return None

        with open(index_file, 'rb') as f:
            return pickle.load(f)
    
    def get_cache_stats(self, split_name=None):
        if split_name is None:
            split_name = self.split_name

        time_index = self.get_time_index(split_name)
        if not time_index:
            return None

        total_size_bytes = sum(info['file_size_bytes'] for info in time_index.values())
        total_pairs = sum(info['num_pairs'] for info in time_index.values())
        total_objects = sum(info['total_objects'] for info in time_index.values())

        return {
            'dataset': self.dataset_name,
            'split_name': split_name,
            'num_time_slices': len(time_index),
            'total_size_mb': total_size_bytes / (1024 * 1024),
            'total_sr_pairs': total_pairs,
            'total_objects': total_objects,
            'avg_pairs_per_slice': total_pairs / len(time_index) if time_index else 0,
            'avg_objects_per_slice': total_objects / len(time_index) if time_index else 0
        }


def load_agg_history_data(dataset_name, time_idx, data_root="./data", split_name="train"):
    data_manager = AGGDataManager(dataset_name, data_root, split_name)
    return data_manager.load_time_slice_data(time_idx, split_name)


def main():

    parser = argparse.ArgumentParser(description='预处理时序知识图谱数据集，构建历史图')
    parser.add_argument('--datasets', nargs='+', default=["ICEWS14"],
                       help='要处理的数据集列表 (default: ["ICEWS14"])')
    parser.add_argument('--data-root', type=str, default="./data",
                       help='数据集根目录 (default: ./data)')
    parser.add_argument('--save-root', type=str, default="./data",
                       help='保存根目录 (default: ./data)')
    parser.add_argument('--safe-max-seq-len', type=int, default=50,
                       help='预处理阶段保存的时间片上限 (default: 50)')
    parser.add_argument('--max-objs-per-step', type=int, default=10,
                       help='超参数 K：防止过平滑，每个时间片内最多保留的历史客体数量')
    parser.add_argument('--train-only', action='store_true',
                       help='只处理训练集')
    parser.add_argument('--test-only', action='store_true',
                       help='只处理测试集')

    args = parser.parse_args()

    process_train = not args.test_only
    process_test = not args.train_only

    success_count = 0

    for dataset in args.datasets:
        print(f"\n正在处理数据集: {dataset}")

        if process_dataset(dataset, args.data_root, args.save_root,
                          args.safe_max_seq_len, args.max_objs_per_step,
                          process_train, process_test):
            success_count += 1
            print(f"[OK] {dataset} 处理成功")
        else:
            print(f"[FAIL] {dataset} 处理失败")

    print(f"\n处理完成: {success_count}/{len(args.datasets)} 个数据集成功")


if __name__ == "__main__":
    main()
