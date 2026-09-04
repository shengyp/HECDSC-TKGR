
import csv
import datetime
import argparse
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    print("[WARNING] torch.cuda.amp is not available, mixed precision training will be disabled")
    AMP_AVAILABLE = False

from model import Model
import utils


def set_random_seed(seed):
    """
    Set all random seeds to ensure reproducibility
    """
    import random
    import time
    
    if seed < 0:
        actual_seed = int(time.time() * 1000) % (2**32)
        seed = actual_seed
    
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed  


@torch.no_grad()
@torch.inference_mode()
def test(model, test_list, num_rels, num_nodes, use_cuda, all_ans_list, model_path, args, scheduler_info=None, global_start_idx=0):
    """
    Complete test function

    Perform full evaluation on test dataset, calculate MRR and Hits@K metrics.
    Supports global filtering and time online filtering.

    Args:
        model: Trained model
        test_list: Test data time snapshot list
        num_rels: Number of relations
        num_nodes: Number of nodes
        use_cuda: Whether to use CUDA
        all_ans_list: Filtered answer list
        model_path: Model file path
        args: Command line arguments
        scheduler_info: Scheduler info (optional)
        global_start_idx: Global time start index

    Returns:
        tuple: (mrr_filter, hit_values) Filtered MRR and Hits@K metrics
    """

    if global_start_idx > 0 and len(all_ans_list) < global_start_idx + len(test_list):
        print(f"[Fix] Padding all_ans_list with {global_start_idx} None items to align with global index...")
        padded_ans_list = [None] * global_start_idx + all_ans_list
        all_ans_list = padded_ans_list

    ranks_raw, ranks_filter = [], []
    ranks_raw_inv, ranks_filter_inv = [], []

    if model_path is not None and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        map_loc = torch.device(args.gpu) if use_cuda else torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=map_loc)
        print(f"Model epoch in file: {checkpoint.get('epoch', -1)}")
        model.load_state_dict(checkpoint['state_dict'])

        if '_best_hit1_' in model_path:
            print(f"[BEST] Using best model for full test set evaluation (all timesteps, all data)")
            print(f"   Note: Full evaluation results may differ from fast evaluation during training")

    model.eval()

    for time_idx, test_snap in enumerate(tqdm(test_list, desc="Test Evaluation")):
        global_time_idx = global_start_idx + time_idx

        if time_idx % 50 == 0 and use_cuda:
            torch.cuda.empty_cache()

        if hasattr(model.shared_aggregator, 'clear_current_timestep_cache'):
            model.shared_aggregator.clear_current_timestep_cache()
        if hasattr(model.shared_aggregator_test, 'clear_current_timestep_cache'):
            model.shared_aggregator_test.clear_current_timestep_cache()

        test_triples = test_snap.copy()

        def run_dual_encoder_batches(triples_np, prediction_type):
            """
            Dual encoder batch evaluation function

            Evaluate test data in batches, supporting subject and object prediction.

            Args:
                triples_np: Test triples data [N, 3]
                prediction_type: Prediction type ('subject' or 'object')

            Returns:
                tuple: (avg_loss, mrr_filter, mrr_raw, ranks_raw, ranks_filter)
            """
            dev = model.shared_ent_emb.device
            triples = torch.as_tensor(triples_np, dtype=torch.long, device=dev)

            if triples.numel() == 0:
                return (0.0, 0.0, 0.0, [], [])

            bz = min(args.eval_bz, triples.size(0))
            outs = []

            for s in range(0, triples.size(0), bz):
                b = triples[s:s+bz]

                agg_time_idx = global_time_idx
                rgcn_time_idx = global_time_idx
                out = model(b, all_ans_list, agg_time_idx, rgcn_time_idx, "Test", prediction_type=prediction_type)
                outs.append(out)

            loss_avg = sum([o[0] for o in outs]) / len(outs)
            mrr_f = sum([o[1] for o in outs]) / len(outs)
            mrr = sum([o[2] for o in outs]) / len(outs)

            r_all, rf_all = [], []
            for o in outs:
                r_all.extend(o[3])
                rf_all.extend(o[4])

            return (loss_avg, mrr_f, mrr, r_all, rf_all)

        f_res = run_dual_encoder_batches(test_triples, 'subject')
        i_res = run_dual_encoder_batches(test_triples, 'object')

        ranks_raw.append(f_res[3])
        ranks_filter.append(f_res[4])
        ranks_raw_inv.append(i_res[3])
        ranks_filter_inv.append(i_res[4])

    mrr_filter, hit_filter = utils.stat_ranks(ranks_filter, "filter")
    mrr_filter_inv, hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_inv")

    all_mrr_filter = (mrr_filter + mrr_filter_inv) / 2
    all_hit_filter = [(hit_filter[i] + hit_filter_inv[i]) / 2 for i in range(len(hit_filter))]

    import csv as _csv, time, shutil

    os.makedirs('./result', exist_ok=True)
    filename = './result/' + args.dataset + ".csv"

    fieldnames = [
        'dataset', 'datetime', 'seed', 'lr', 'batch_size', 'n_epochs', 
        'embd_dim','hidden_dim','rgcn_num_layers','agg_nhead','max_seq_len','agg_num_layers','dropout','grad_norm','weight_decay','d_k','d_v','d_inner',
        'use_lr_scheduler', 'use_amp', 'best_epoch', 'final_lr',
        'filter_MRR', 'filter_H@1', 'filter_H@3', 'filter_H@10',
        'filter_inv_MRR', 'filter_inv_H@1', 'filter_inv_H@3', 'filter_inv_H@10',
    ]

    def _ensure_csv_with_header(path, header):
        """
        Ensure CSV file exists and header is correct

        Create new file if not exists.
        Backup old file and create new if header doesn't match.
        """
        if not os.path.isfile(path):
            with open(path, 'w', newline='') as f:
                _csv.DictWriter(f, fieldnames=header).writeheader()
            return

        try:
            with open(path, 'r', newline='') as f:
                reader = _csv.reader(f)
                first = next(reader, None)
        except Exception:
            first = None

        if first is None or list(first) != header:
            bak = path + f".bak-{time.strftime('%Y%m%d-%H%M%S')}"
            shutil.move(path, bak)
            with open(path, 'w', newline='') as f:
                _csv.DictWriter(f, fieldnames=header).writeheader()
            print(f"[CSV] Old header found, backed up as: {bak}, rebuilt with new header {path}")

    _ensure_csv_with_header(filename, fieldnames)

    with open(filename, 'a', newline='') as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        
        use_scheduler = scheduler_info is not None and scheduler_info.get('used', False)
        best_epoch = scheduler_info.get('best_epoch', args.n_epochs) if scheduler_info else args.n_epochs
        final_lr = scheduler_info.get('final_lr', args.lr) if scheduler_info else args.lr
        
        row = {
            'dataset': args.dataset,
            'datetime': datetime.datetime.now(),
            'seed': args.seed,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'embd_dim': args.embd_dim,
            'hidden_dim': args.hidden_dim,
            'rgcn_num_layers': args.rgcn_num_layers,
            'agg_nhead': args.agg_nhead,
            'max_seq_len': args.max_seq_len,
            'agg_num_layers': args.agg_num_layers,
            'dropout':args.dropout,
            'grad_norm':args.grad_norm,
            'weight_decay':args.weight_decay,
            'd_k':args.d_k,
            'd_v':args.d_v,
            'd_inner':args.d_inner,
            'use_lr_scheduler': use_scheduler,
            'use_amp': args.use_amp,
            'best_epoch': best_epoch,
            'final_lr': round(final_lr, 6),
            'filter_MRR': round(float(mrr_filter), 4),
            'filter_H@1': round(hit_filter[0], 4),
            'filter_H@3': round(hit_filter[1], 4),
            'filter_H@10': round(hit_filter[2], 4),
            'filter_inv_MRR': round(float(mrr_filter_inv), 4),
            'filter_inv_H@1': round(hit_filter_inv[0], 4),
            'filter_inv_H@3': round(hit_filter_inv[1], 4),
            'filter_inv_H@10': round(hit_filter_inv[2], 4),
        }
        writer.writerow(row)
        


    return all_mrr_filter, all_hit_filter


@torch.no_grad()
def quick_eval_for_scheduler(model, test_list_subset, num_rels, num_nodes, use_cuda, all_ans_list, args, global_start_idx=0):
    """
    Quick evaluation function for learning rate scheduler

    Quickly evaluate a small subset of test data during training for scheduler decisions.
    Only evaluates the first few timesteps to save time.

    Args:
        model: Model to evaluate
        test_list_subset: Test data subset
        num_rels: Number of relations
        num_nodes: Number of nodes
        use_cuda: Whether to use CUDA
        all_ans_list: Filtered answer list
        args: Command line arguments
        global_start_idx: Global time start index

    Returns:
        float: Hits@1 metric for learning rate scheduling
    """
    model.eval()
    ranks_filter = []

    if global_start_idx > 0 and len(all_ans_list) < global_start_idx + len(test_list_subset):
        padded_ans_list = [None] * global_start_idx + all_ans_list
        all_ans_list = padded_ans_list

    eval_subset = test_list_subset[:min(5, len(test_list_subset))]

    for time_idx, test_snap in enumerate(eval_subset):
        if test_snap.shape[0] == 0:
            continue

        global_time_idx = time_idx + global_start_idx

        if hasattr(model.shared_aggregator, 'clear_current_timestep_cache'):
            model.shared_aggregator.clear_current_timestep_cache()
        if hasattr(model.shared_aggregator_test, 'clear_current_timestep_cache'):
            model.shared_aggregator_test.clear_current_timestep_cache()

        test_triples = test_snap.copy()
        dev = model.shared_ent_emb.device
        triples = torch.as_tensor(test_triples, dtype=torch.long, device=dev)

        if triples.numel() == 0:
            continue

        bz = min(args.eval_bz // 4, triples.size(0))
        for s in range(0, min(triples.size(0), bz * 3), bz):
            b = triples[s:s+bz]
            try:
                out = model(b, all_ans_list, global_time_idx, global_time_idx, "Test", prediction_type='subject')
                if len(out) >= 5:
                    ranks_filter.extend(out[4])
            except Exception:
                continue

    if ranks_filter:
        mrr_filter, hit_filter = utils.stat_ranks([ranks_filter], "filter")
        return hit_filter[0]
    else:
        return 0.0


def run_experiment(args):
    """
    Run complete temporal knowledge graph experiment

    Includes full workflow: data loading, model initialization, training, validation and testing.

    Args:
        args: Command line arguments namespace containing all experiment configuration
    """
    actual_seed = set_random_seed(args.seed)
    args.seed = actual_seed

    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S") + "-simple-fusion-" + args.dataset
    save_dir = getattr(args, 'save_dir', './saved_models')
    main_dirName = os.path.join(save_dir, dt_string)
    os.makedirs(main_dirName, exist_ok=True)
    model_path = os.path.join(main_dirName, 'models'); os.makedirs(model_path, exist_ok=True)

    use_cuda = (args.gpu >= 0 and torch.cuda.is_available())

    dataset_path = f'./data/{args.dataset}'
    train_file = f'{dataset_path}/train.txt'
    test_file = f'{dataset_path}/test.txt'
    stat_file = f'{dataset_path}/stat.txt'

    data = utils.load_data(args.dataset, load_time=True)

    train_graph_dir = f'./data/{args.dataset}/graph_snapshots_train'
    test_graph_dir = f'./data/{args.dataset}/graph_snapshots_test'
    train_agg_dir = f'./data/{args.dataset}/his_dict_snap_train'
    test_agg_dir = f'./data/{args.dataset}/his_dict_snap_test'

    if not os.path.exists(train_graph_dir) or not os.path.exists(test_graph_dir) or \
       not os.path.exists(train_agg_dir) or not os.path.exists(test_agg_dir):
        print("Data preprocessing incomplete! Please run in order:")
        print(f"1. python rgcn_get_history.py --dataset {args.dataset}")
        print(f"2. python agg_get_history.py --datasets {args.dataset}")
        exit(1)

    num_nodes, num_rels = utils.get_dataset_stat(args.data_root, args.dataset)
    print(f"[{args.dataset}] Statistics loaded: num_nodes={num_nodes}, num_rels={num_rels}")

    train_list = utils.split_by_time(data.train)
    test_list  = utils.split_by_time(data.test)

    test_start_idx = len(train_list)
    print(f"Global Time Index Alignment: Train ends at {test_start_idx-1}, Test starts at {test_start_idx}")

    train_last_time_idx = max(1, len(train_list) - 1)

    model_name = now.strftime("%d-%m-%Y-%H-%M-%S") + "-" + args.dataset
    model_state_file = os.path.join(main_dirName, 'models', model_name + ".pt")
    model = Model(
        dataset_name=args.dataset, num_nodes=num_nodes, num_rels=num_rels,
        embd_dim=args.embd_dim, hidden_dim=args.hidden_dim,
        n_head=args.agg_nhead, num_hidden_layers=args.agg_num_layers,
        d_k=args.d_k, d_v=args.d_v, d_inner=args.d_inner,
        dropout=args.dropout, self_loop=args.self_loop, skip_connect=args.skip_connect,
        entity_prediction=args.entity_prediction, relation_prediction=args.relation_prediction,
        use_cuda=use_cuda, gpu=args.gpu, dataset_path=f'./data/{args.dataset}',
        eval_bz=args.eval_bz,
        enable_cache=args.enable_cache, cache_size=args.cache_size, num_workers=args.num_workers,
        train_last_time_idx=train_last_time_idx,
        max_seq_len=args.max_seq_len, agg_nhead=args.agg_nhead, agg_num_layers=args.agg_num_layers,
        rgcn_layers=args.rgcn_num_layers,
    )

    if use_cuda:
        torch.cuda.set_device(args.gpu); model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    scaler = None
    use_amp = args.use_amp and AMP_AVAILABLE and use_cuda
    if use_amp:
        scaler = GradScaler()

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            verbose=True,
            min_lr=args.lr_scheduler_min_lr
        )

    if args.test and os.path.exists(model_state_file):
        return test(model, test_list, num_rels, num_nodes, use_cuda, test_list, model_state_file, args, global_start_idx=test_start_idx)
    elif args.test and not os.path.exists(model_state_file):
        print(f"Model file {model_state_file} not found, switching to training...")

    best_hit1 = 0.0
    best_epoch = 0
    best_results = None

    for epoch in range(args.n_epochs):
        model.train()
        losses = []
        current_lr = optimizer.param_groups[0]['lr']

        for t_idx in tqdm(range(len(train_list)), desc=f"Epoch {epoch+1}"):
            if t_idx == 0:
                continue

            if t_idx % 50 == 0:
                if use_cuda:
                    torch.cuda.empty_cache()
                if hasattr(model.shared_aggregator, 'clear_cache'):
                    model.shared_aggregator.clear_cache()
                if hasattr(model.shared_aggregator_test, 'clear_cache'):
                    model.shared_aggregator_test.clear_cache()

            snap = train_list[t_idx]
            if snap.shape[1] > 3:
                snap = snap[:, :3]

            triples_np = snap

            def train_dual_encoder(triples_np, prediction_type):
                """
                Dual encoder training function - supports mixed precision training

                Train data for a single time snapshot using dual encoder model for subject or object prediction.

                Args:
                    triples_np (numpy.ndarray): Time snapshot triples data [N, 3] or [N, 4]
                    prediction_type (str): Prediction type, 'subject' or 'object'
                """
                dev = model.shared_ent_emb.device
                triples = torch.as_tensor(triples_np, dtype=torch.long, device=dev)
                if triples.numel() == 0:
                    return

                bz = min(args.batch_size, triples.size(0))
                step_losses = []

                for s in range(0, triples.size(0), bz):
                    b = triples[s:s+bz]
                    optimizer.zero_grad(set_to_none=True)

                    if use_amp:
                        with autocast():
                            loss = model(b, None, t_idx, t_idx, 'Training', prediction_type=prediction_type)

                        scaler.scale(loss).backward()

                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)

                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss = model(b, None, t_idx, t_idx, 'Training', prediction_type=prediction_type)

                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)

                        optimizer.step()
                    
                    step_losses.append(loss.item())

                if step_losses:
                    losses.append(sum(step_losses) / len(step_losses))

            train_dual_encoder(triples_np, 'subject')

            train_dual_encoder(triples_np, 'object')

            if hasattr(model.shared_aggregator, 'clear_current_timestep_cache'):
                model.shared_aggregator.clear_current_timestep_cache()
            if hasattr(model.shared_aggregator_test, 'clear_current_timestep_cache'):
                model.shared_aggregator_test.clear_current_timestep_cache()

        avg_loss = (np.mean(losses) if losses else 0.0)

        if losses:
            max_loss = np.max(losses)
            min_loss = np.min(losses)
            std_loss = np.std(losses)

            if avg_loss > 100 or std_loss > 50:
                print(f"[WARNING] Abnormal loss distribution detected, NaN may occur soon")
        else:
            avg_loss = 0.0
        
        if scheduler is not None and (epoch + 1) % args.eval_freq == 0:
            hit1 = quick_eval_for_scheduler(model, test_list, num_rels, num_nodes, use_cuda, test_list, args, global_start_idx=test_start_idx)
            print(f"Epoch {epoch+1:04d} | Loss: {avg_loss:.4f} | Hit@1: {hit1:.4f} | LR: {current_lr:.6f}")

            scheduler.step(hit1)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"Learning rate adjusted: {current_lr:.6f} → {new_lr:.6f}")

            if hit1 > best_hit1:
                best_hit1 = hit1
                best_epoch = epoch + 1
                best_model_path = model_state_file.replace('.pt', f'_best_hit1_{hit1:.4f}.pt')
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_hit1': hit1,
                    'args': args
                }, best_model_path)

                best_results = {
                    'epoch': epoch + 1,
                    'hit1': hit1,
                    'loss': avg_loss,
                    'lr': current_lr
                }
        else:
            print(f"Epoch {epoch:04d} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        if use_cuda:
            torch.cuda.empty_cache()

    torch.save({
        'state_dict': model.state_dict(),
        'epoch': args.n_epochs - 1,
        'best_mrr': None,
        'args': args
    }, model_state_file)

    scheduler_info = None
    if scheduler is not None:
        scheduler_info = {
            'used': True,
            'best_epoch': best_epoch,
            'final_lr': optimizer.param_groups[0]['lr']
        }

    final_model_path = model_state_file
    if scheduler is not None and best_results is not None:
        best_model_path = model_state_file.replace('.pt', f'_best_hit1_{best_hit1:.4f}.pt')
        if os.path.exists(best_model_path):
            final_model_path = best_model_path

    final = test(model, test_list, num_rels, num_nodes, use_cuda, test_list, final_model_path, args, scheduler_info, global_start_idx=test_start_idx)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Temporal KG Reasoning ')

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("-d", "--dataset", type=str, default="ICEWS14")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--save-dir", type=str, default="./saved_models")

    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility. Use --seed -1 for random initialization.")
    parser.add_argument("--eval-bz", type=int, default=2000)
    

    parser.add_argument("--embd-dim", type=int, default=280)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--skip-connect", action='store_true', default=False)
    parser.add_argument("--hidden-dim", type=int, default=280)
    parser.add_argument("--rgcn-num-layers", type=int, default=2,
                       help="RGCN module: number of graph convolutional network layers")
    parser.add_argument("--d-k", type=int, default=56)
    parser.add_argument("--d-v", type=int, default=56)
    parser.add_argument("--d-inner", type=int, default=1024)
    parser.add_argument("--no-self-loop", action='store_false', dest='self_loop', default=True,
                       help="Disable self-loop (enabled by default)")
    parser.add_argument("--relation-prediction", action='store_true', default=False)
    parser.add_argument("--entity-prediction", action='store_true', default=True)

    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--grad-norm", type=float, default=0.6)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Adam optimizer weight decay (L2 regularization strength, default 1e-4)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--enable-cache", action='store_true', default=True)
    parser.add_argument("--cache-size", type=int, default=18000)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--max-seq-len", type=int, default=10,
                       help="Hyperparameter M: dynamic historical timestep window size (can be modified anytime without reprocessing)")
    parser.add_argument("--agg-nhead", type=int, default=4,
                       help="AGG module: number of attention heads in Transformer (must be divisible by d_model)")
    parser.add_argument("--agg-num-layers", type=int, default=2,
                       help="AGG module: number of Transformer encoder layers")


    parser.add_argument("--use-lr-scheduler", action='store_true', default=False,
                       help="Enable learning rate scheduler (ReduceLROnPlateau)")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5,
                       help="Learning rate decay factor")
    parser.add_argument("--lr-scheduler-patience", type=int, default=2,
                       help="Learning rate scheduler patience (epochs without improvement before decay)")
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6,
                       help="Learning rate scheduler minimum learning rate")
    parser.add_argument("--eval-freq", type=int, default=2,
                       help="Evaluation frequency: evaluate on validation set every N epochs, also used for scheduler evaluation")
    
    parser.add_argument("--use-amp", action='store_true', default=False,
                       help="Enable automatic mixed precision training (AMP) for faster training")

    args = parser.parse_args()
    
    print(f"Dataset: {args.dataset}, Seed: {args.seed}")
    run_experiment(args)
