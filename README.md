# # HECDSC-TKGR
The code of HECDSC-TKGR: Enhancing Temporal Knowledge Graph Reasoning through Jointly Modeling of Historical Event Context and Query-Oriented Dynamic Subgraph Context


## Brief Introduction

We present the HECDSC-TKGR framework, which enhances temporal knowledge graph extrapolation reasoning by jointly modeling the historical evolution context and the dynamic subgraph context.

<img src="E:\西大\毕设\自模型\HEDS\HEDS_git_hub\pic\overview.png" style="zoom:50%;" />

**Temporal Historical Context Encoder**. The core function of this module is to track the subject entity's own historical evolution trajectory. By utilizing a query-aware semantic fusion mechanism coupled with the Transformer self-attention mechanism, it effectively captures the entities' long-range temporal dependencies and evolutionary patterns.

**Relational Structural Context Encoder**. To provide crucial structural compensation in scenarios with sparse historical interactions, this module constructs a query-centric local relational subgraph. Subsequently, it introduces relation-aware structural encoding and employs a Relational Graph Convolutional Network (R-GCN) to deeply mine topological semantics.

**Joint Fusion Learning**. During the dual-path feature integration stage, we introduce a cross-attention fusion mechanism. This mechanism effectively achieves the complementary integration of temporal evolution patterns and spatial topological structures, thereby significantly boosting the model's reasoning performance in complex and sparse scenarios.

For more technical details, please refer to the HEDS paper.


## Getting Started

### 1. Environment Preparation
Please set up the required environment for the project first. You can run the following commands to install the underlying CUDA runtime libraries all at once and complete the dependency configurations for PyTorch and DGL:


```
conda create -n tkgr python=3.9 -y
conda activate tkgr

# 一次性补齐底层 CUDA 运行库
conda install -c conda-forge cudatoolkit=11.7 -y

# 安装 PyTorch (1.13.1 + cu117)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 安装对应版本的 DGL GPU 版
pip install dgl==1.1.0+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html

# 安装其他依赖
pip install numpy==1.26.4
# (其他依赖如 tqdm 等正常 pip install 即可)
```

### 2.Data Preparation

#### 1.Dataset Structure
Please extract your datasets into the `./data` directory. The directory structure should look as follows:
```
./data/
└── ICEWS14/
    ├── train.txt
    ├── valid.txt (optional)
    ├── test.txt
    └── stat.txt
```
- stat.txt should contain two numbers: num_entities and num_relations.
- Data files should contain quadruples: head relation tail timestamp.

#### 2.Preprocessing

The dual-path architecture of HEDS requires two types of preprocessed data: Graph Snapshots (for the structural encoder) and History Dictionaries (for the historical aggregator).

Run the following commands to generate them for the ICEWS14 dataset (or change the dataset name to YAGO, WIKI, etc.):

**Step 1:  Construct Graph Snapshots. **This will generate DGL graph structure files for the local relational subgraph encoding.

```
python rgcn_get_history.py --dataset ICEWS14 
```
Output: ./data/ICEWS14/graph_snapshots_train/ and ./data/ICEWS14/graph_snapshots_test/

**Step 2: Construct History Dictionaries.** This will extract the multi-hop neighborhood and generate hybrid cache dictionaries for the history-aware aggregator.

```
python agg_get_history.py --dataset ICEWS14
```
Output: ./data/ICEWS14/hybrid_cache/ and various his_dict_* folders.



### 3.Training HEDS
Use `main.py` to train the model. The model natively supports Automatic Mixed Precision (AMP) training and learning rate scheduler optimization.

Example Training Command:

```
python main.py --dataset ICEWS14 --n-epochs 50 --use-lr-scheduler --lr-scheduler-patience 2  --max-seq-len 40

```

## Cite
If you find this code useful for your research, please cite our paper:
~~~
@article{His-GraR,
  title={Temporal Knowledge Graph Reasoning via Jointly Modeling of Event Historical Evolution Context and Dynamic Subgraph Context},
  author={Lingfang Chen and Yongpan Sheng and Hongyan Ouyang and Lirong He and Ming Liu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
~~~
