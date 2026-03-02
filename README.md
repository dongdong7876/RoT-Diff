**Note to Reviewers:** 

> This implementation is part of our submission to **PVLDB 2026 (Vol. 19)**. To protect the intellectual property of our novel **Diffusion-based Time Series Anomaly Detection** model, we currently restrict the usage of this code to academic evaluation only.

This repository contains the source code for the paper `RoT-Diff: A Contamination-Resilient and Regime-Adaptive Discrete Diffusion Framework for Time Series Anomaly Detection`. The code is provided **solely for the purpose of peer review** by the PVLDB program committee.

-----

### 📂 Code Description

The project structure is organized as follows:

- `stage1.py`: Training script for Stage 1 (Robust Representation Distillation).
- `stage2.py`: Training script for Stage 2 (Regime-Adaptive Conditional Generation).
- `evaluate.py`: Evaluation script for performing inference and computing metrics.
- `config/`: Contains the configuration file `config.yaml` for hyperparameters and dataset settings.
- `model/`: Contains the implementation of RoT-Diff (Stage 1 VQ-VAE and Stage 2 Diffusion).
- `data_factory/`: Scripts for data loading and preprocessing.
- `dataset/`: Folder to store the raw datasets.
- `cpt/`: Directory for saving and loading trained model weights (Checkpoints).
- `result/`: Automatically saves experiment results, logs, and evaluation metrics.
- `utils/`: Utility functions (metrics, plotting, etc.).

------

### ⚡ Quick Start

#### 1. Requirements

- Python >= 3.8
- PyTorch >= 1.12
- NVIDIA GPU + CUDA

Install dependencies:

Bash

```
pip install -r requirements.txt
```

#### 2. Data Preparation

Unzip the dataset files into the `dataset/` folder. Ensure the directory structure matches the configuration paths defined in `config/config.yaml`.

#### 3. Training & Evaluation

The training process consists of two sequential stages, followed by evaluation. You can reproduce the results for any dataset (e.g., `SMD`, `MSL`, `SWaT`, `WADI`, `PSM`) using the following commands:

**Step 1: Train Stage 1 (Robust Distillation)**

Train the VQ-VAE to learn robust prototypes from contaminated data.

Bash

```
python stage1.py --data_name SMD
```

**Step 2: Train Stage 2 (Prior Learning)**

Train the diffusion model using the frozen Stage 1 model as a tokenizer.

Bash

```
python stage2.py --data_name SMD
```

**Step 3: Inference & Evaluation**

Evaluate the trained model using the robust scoring protocol.

Bash

```
python evaluate.py --data_name SMD
```

*(Note: Replace `SMD` with other dataset names as needed.)*

------

### 💬 Citation

If you find this repository useful for your research, please cite our paper:

```

```

------

### 📧 Contact

If you have any questions, please feel free to contact [] at [].