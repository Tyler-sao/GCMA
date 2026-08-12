# GCMA

Official implementation of **GCMA** for circuit representation learning.

GCMA learns circuit representations from And-Inverter Graphs (AIGs) and Verilog-derived semantic features. The repository provides a two-stage workflow:

1. self-supervised representation pretraining with masked gate modeling and Verilog–AIG alignment;
2. few-shot adaptation for logic equivalence identification and quality-of-results (QoR) prediction.

## Highlights

- Graph representation learning over AIG circuit structures.
- Masked gate reconstruction in the latent representation space.
- Cross-modal alignment between AIG graphs and Verilog semantic features.
- Graph encoder based on message passing and cross-modal decoding.
- Few-shot downstream heads for logic equivalence identification and QoR prediction.
- Evaluation scripts for AUC, precision, recall, F1, NDCG, and Top-k% commonality.

## Environment Requirements

GCMA uses the same software environment as [Cross Modal Graph Matching for Verilog–AIG Alignment](https://github.com/zhongmoxingzhe/Cross_Modal_Graph_Matching_for_Verilog_AIG_Alignment).

Install the following package versions:

```text
numpy==1.23.5
pandas==2.0.3
torch==1.10.0+cu113
torch-cluster==1.5.9
torch-geometric==1.7.0
torch-scatter==2.0.9
torch-sparse==0.6.13
torch-spline-conv==1.2.1
torchaudio==0.10.0+cu113
torchvision==0.11.1+cu113
tqdm==4.67.1
transformers==4.20.1
```

The PyTorch build above targets CUDA 11.3. Make sure the CUDA runtime, PyTorch, and PyTorch Geometric binary packages are mutually compatible.

## Data

GCMA uses the same data as [Cross Modal Graph Matching for Verilog–AIG Alignment](https://github.com/zhongmoxingzhe/Cross_Modal_Graph_Matching_for_Verilog_AIG_Alignment).

The generated data is available from Baidu Netdisk:

- [CDCR dataset](https://pan.baidu.com/s/1DHe2vY-Dwhyisinn85waWA?pwd=whm6)
- Extraction code: `whm6`

The original circuit sources are described by the following projects:

- [OpenABC / OpenABC-D](https://github.com/NYU-MLDA/OpenABC)
- [EPFL combinational benchmark suite](https://www.epfl.ch/labs/lsi/benchmarks)

The commands below assume a directory layout similar to:

```text
workspace/
├── GCMA/
├── openabc-d/
│   └── OPENABC2_DATASET/
│       ├── processed/
│       └── merged_train.csv
└── pyg_ver/
    ├── resyn27k_pt/
    ├── resyn27k_npy.npy
    ├── qor_out_fewshot_pt_small/
    ├── eq_out_fewshot_pt_small/
    └── pairs1.csv
```

Paths may be changed to match the local storage layout.

## Pretrained Weights

Pretrained and few-shot adapted model weights are included in the `weight/`
directory:
weight/
├── config.json
├── pytorch_model.bin
└── head/
    ├── config.json
    ├── pytorch_model.bin
    ├── similarity_head.pt
    ├── similarity_head1.pt
    ├── qor_head.pt
    └── qor_head1.pt

## Usage

Run all commands from the repository root unless stated otherwise.

Expose the repository root and the `data/` directory to Python before running the commands:

```bash
export PYTHONPATH="$(pwd):$(pwd)/data:${PYTHONPATH}"
```

This is required because the training entry points import `data_helper.py` and `data_helper_zyc.py` from the `data/` directory.

### 1. Base Representation Pretraining

The pretrained model and its configuration are written to `--output_dir`.

### 2. Few-Shot Downstream-Head Training

The few-shot stage trains the logic-equivalence similarity head and the QoR regression head using a pretrained GCMA checkpoint.

This stage produces the following downstream checkpoints in `--output_dir`:

- `similarity_head.pt` for logic equivalence identification;
- `qor_head.pt` for QoR prediction;
- the saved GCMA model configuration and model weights.

`--pretrained_ckpt` must point to the checkpoint selected for the few-shot experiment. Replace the example path if the base model was saved under a different version name.

## Evaluation

### Logic Equivalence Identification

```bash
python eval/eval_eq.py \
    --pretrained 
    --head_ckpt 
    --graphs_dir /root/autodl-tmp/pyg_ver/output_eq_new \
    --pairs_csv /root/autodl-tmp/pyg_ver/output_eq_new/all_pairs.csv \
    --batch_size 4
```

The pair CSV is expected to contain three columns:

```text
graph1_id,graph2_id,label
```

Each graph ID should correspond to a `{graph_id}.pt` file under `--graphs_dir`. The script reports AUC, precision, recall, F1 score, and the selected decision threshold.

### QoR Prediction

```bash
python eval/eval_qor.py \
    --pretrained 
    --qor_head_ckpt 
    --data_root /root/autodl-tmp/pyg_ver/output_qor_new \
    --batch_size 4 \
    --save_result
```

The QoR evaluator reports regression loss, NDCG@3/5/10, and Top-3%/5%/10% commonality for each circuit and their averages. With `--save_result`, the summary is written under `--data_root`.

## Reproducibility Notes

- Training and evaluation entry points set the random seed to `42`.
- The reported commands disable intermediate evaluation and checkpoint saving during training.
- Change `--save_strategy` if intermediate checkpoints are required.
- Absolute paths in the examples reflect the original experimental setup and should be replaced when necessary.
- Ensure that the pretrained checkpoint and downstream head checkpoints originate from compatible model configurations.

## Acknowledgements

This project uses the same data sources and software environment as [Cross Modal Graph Matching for Verilog–AIG Alignment](https://github.com/zhongmoxingzhe/Cross_Modal_Graph_Matching_for_Verilog_AIG_Alignment).

We also acknowledge the following projects and resources:

- [OpenABC / OpenABC-D](https://github.com/NYU-MLDA/OpenABC)
- [EPFL combinational benchmark suite](https://www.epfl.ch/labs/lsi/benchmarks)
- [Circuit Representation Learning with Masked Gate Modeling and Verilog-AIG Alignment](https://github.com/wuhy68/MGVGA)

## License

This project is released under the [MIT License](LICENSE).

For questions, bug reports, or contributions, please open an issue or pull request.
