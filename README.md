# RoT-Diff

## A Contamination-Resilient and Regime-Adaptive Discrete Diffusion Framework for Time Series Anomaly Detection

**RoT-Diff** is a novel unsupervised anomaly detection framework designed to robustly handle data contamination and complex operating regimes in industrial time series.

### 📝 Abstract

Time series anomaly detection is pivotal for safeguarding industrial systems. However, mainstream unsupervised methods often rely on the unrealistic assumption of contamination-free training data. In real-world settings, where complex operating regimes are intermingled with anomalies, models frequently suffer severe degradation—either by overfitting to anomalies or failing to capture diverse normal patterns.

To address this, we propose **RoT-Diff**, a contamination-resilient and regime-adaptive discrete diffusion framework featuring a **“Robust Distillation followed by Controlled Generation”** paradigm.

1. **Robust Representation Distillation**: A Robust Regime-Aware VQ-VAE optimized with the Student-t distribution fundamentally mitigates data contamination, distilling purified feature prototypes into a *Global Concept Codebook (GCC)* and a *Local Content Codebook (LCC)*.
2. **Regime-Adaptive Conditional Generation**: A Discrete Diffusion Model generates samples strictly adhering to the current global operating regime via an AdaLN-Zero modulation mechanism guided by the GCC.
3. **Inference**: Anomaly detection is achieved via a stochastic masking strategy, augmented by *Shift-based Inference* to mitigate patching artifacts and enhance detection robustness.

Extensive experiments on five real-world datasets validate RoT-Diff’s robustness, significantly outperforming state-of-the-art baselines across varying levels of data contamination.

------

### 🚀 Overall Architecture

|                  ![Figure1](pics/model.png)                  |
| :----------------------------------------------------------: |
| *Figure 1. Overview of RoT-Diff.* |


------

### 📖 Main Results

We compare RoT-Diff against 14 representative baselines on five benchmark datasets (SMD, MSL, SWaT, WADI, PSM). Extensive experiments show that RoT-Diff achieves the best performance on five benchmark datasets compared to state-of-the-arts.

|![Figure2](pics/mainres.png)|
|:--:| 
| *Table 1. Overall results on five benchmark datasets, with performance ranked from lowest to highest. "Average" denotes the mean value across all datasets. All results are in $\%$, the best ones are in \textbf{Bold}, and the second ones are \underline{underlined}.* |

------

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