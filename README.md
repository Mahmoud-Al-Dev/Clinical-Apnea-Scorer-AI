# Clinical-Apnea-Scorer-AI: Dual-Expert DPO & RLHF Pipeline

A complete end-to-end Machine Learning pipeline designed to process raw Polysomnography (PSG) sleep data and autonomously classify Central Apnea (CA) and Obstructive Sleep Apnea (OSA). 

This project progresses from classical digital signal processing to Supervised Fine-Tuning (SFT) of dual Bidirectional LSTMs, culminating in Direct Preference Optimization (DPO) and Reinforcement Learning from Human Feedback (RLHF) to align the AI's diagnostic behavior with clinical standards.

> ** Data Privacy Notice (Datenschutz):** Raw `.csv` clinical patient data, `.npy` feature arrays, and MLflow local databases (`mlruns/`) have been excluded from this repository in compliance with GDPR and medical data privacy standards.

---

##  System Architecture & Workflow

### 1. Advanced Signal Processing & Batch Engineering (`Scripts_Addons/`)
Raw biological signals are inherently noisy and highly variable between patients. The data pipeline standardizes and extracts features using physiological principles:
* **Automated Batch Processing:** `apnea_signal_processing_full.py` dynamically handles dozens of patient nights simultaneously, mapping 7-channel or 8-channel PSG recordings and automatically filtering out dead sensor data (e.g., disconnected SaO2 probes).
* **Signal Cleaning:** Applied bidirectional Butterworth Band-Pass filters (`filtfilt`) to airflow and respiratory effort channels to remove artifact noise without inducing phase shifts.
* **Amplitude Enveloping:** Extracted instantaneous tidal volume using the Hilbert Transform to calculate upper and lower signal envelopes.
* **Physiological Logic:** Engineered domain-specific features, including *Thorax-Abdomen Cross-Correlation* (to detect paradox breathing) and *Effort-Flow Ratio* (the primary OSA identifier).

![Signal Processing Pipeline](docs/signal_processing_plot.png)

### 2. Data Integrity & Label Noise Detection (`Supervised_Learning/`)
Clinical ground-truth labels are frequently noisy. Before training, the dataset is audited using **Cleanlab** (`run_cleanlab.py`) to systematically identify and quarantine human-annotator mistakes, ensuring the Test datasets remain pristine for high-integrity evaluation.

### 3. Supervised Fine-Tuning (SFT) & Bayesian Optimization (`Supervised_Learning/`)
The foundational "brain" consists of two independent Bidirectional LSTMs trained on stitched, full-night datasets.
* **Multi-GPU Distributed Training:** Implemented PyTorch's `nn.DataParallel` to distribute massive batch loads across dual workstation GPUs (e.g., Quadro GV100s), dramatically reducing epoch times.
* **Bayesian Hyperparameter Optimization (HPO):** Integrated **Optuna** (`tune_lstm.py`) with persistent SQLite logging to algorithmically sweep for the optimal learning rates, L2 weight decay, and Simulated PU (Positive-Unlabeled) discount factors on a dense proxy dataset.
* **Cost-Sensitive Learning:** Addressed severe class imbalances (e.g., nights with 150 OSA events vs. 0 CA events) by applying heavy penalty weights to the `CrossEntropyLoss` function, forcing the model to hunt for rare anomalies.

<img width="1518" height="804" alt="image" src="https://github.com/user-attachments/assets/28b0bb50-8a85-4d3c-a7ab-2024ddb963a0" />

### 4. Direct Preference Optimization (DPO) & RLHF (`DPO/` & `RLHF-PPO/`)
To bridge the gap between messy clinical labels and physiological reality, the SFT weights are behaviorally aligned using cutting-edge preference tuning.
* **Automated Pair-Mining:** The `dpo_auto_miner.py` scans the SFT model's predictions to automatically generate "Chosen" vs. "Rejected" mask pairs.
* **Clinical Behavioral Alignment:** Using `train_dpo.py`, the AI is mathematically penalized for hallucinations (False Alarms) and rewarded for catching borderline events, acting as a high-precision scalpel to push Global F1 scores above 80%.
* **Actor-Critic PPO:** For granular, event-by-event Active Learning, the RLHF pipeline utilizes a `detach()` Critic Firewall to prevent the visual cortex from being overwritten during value gradient optimization.

<img width="2071" height="1757" alt="image" src="https://github.com/user-attachments/assets/69af1a5f-8b26-4d1b-8edd-21521169aae2" />

### 5. MLOps & Event-Based Clinical Evaluation
To ensure rigorous academic reproducibility, the training loop is deeply integrated with **MLflow** and custom clinical metrics.
* **Experiment Tracking:** MLflow automatically logs all PPO/DPO hyperparameters, Optuna trials, entropy losses, and metrics per run.
* **Event-Based Validation:** Replaced rigid frame-by-frame accuracy with a clinical "30% Overlap" rule. If the AI prediction overlaps a doctor's labeled event by 30%, it is counted as a successful Recall.
* **Unlabeled Discovery Tracking:** AI predictions that do not overlap with doctor labels are not immediately penalized as False Positives. They are tracked as "Unlabeled AI Discoveries" to be manually reviewed, allowing the AI to catch apneas the human clinician missed.

---

##  Key Technologies
* **PyTorch:** Deep Learning, LSTM, `nn.DataParallel`, DPO Loss Functions.
* **Optuna:** Bayesian Hyperparameter Optimization (HPO).
* **Cleanlab:** Confident Learning for label error detection.
* **MLflow:** MLOps, Experiment Tracking, Artifact Registry, Metric Logging.
* **SciPy & Scikit-Learn:** Butterworth filters, Hilbert transforms, Morphological operations, Standardization.
* **Gymnasium (OpenAI Gym):** Custom RL environment (`apnea_env.py`) simulation for sequential medical data.
* **Pandas & NumPy:** Big-data manipulation, array stitching, and batch feature synthesis.

---

## Repository Structure

```text
CLINICAL-APNEA-SCORER-AI/
│
├── Data/                       # Raw clinical .csv and .TXT files (Ignored in Git)
├── docs/                       # Diagrams, plots, and pipeline documentation
├── DPO/                        # Direct Preference Optimization Pipeline
│   ├── dpo_auto_miner.py       # Algorithmic generation of Chosen/Rejected pairs
│   ├── dpo_data_collector.py
│   ├── train_dpo.py            # DPO model alignment script
│   └── train_lstm.py
│
├── mlruns/                     # Local MLflow tracking database (Ignored in Git)
├── Nights/                     # Processed .npy AI arrays (OSA Targets)
├── Nights_Vitalog/             # Alternative target arrays (CA Targets)
│
├── RLHF-PPO/                   # Proximal Policy Optimization Pipeline
│   ├── actor_critic_lstm.py
│   ├── apnea_env.py            # Custom Gymnasium environment
│   ├── calculate_clinical_metrics.py
│   ├── clean_anchor_labels.py
│   ├── train_rlhf_ppo.py
│   ├── visualize_full_rlhf.py
│   └── visualize_rlhf.py
│
├── Scripts_Addons/             # Preprocessing & Data Generation
│   ├── adjust_event_boundaries.py
│   ├── apnea_signal_processing_full.py # Batch processing for dynamic channels
│   ├── apnea-signal-processing.py
│   ├── generate_clinical_y_labels.py
│   ├── Matlab_Signal_Processing.m
│   └── ultimate_test.py
│
└── Supervised_Learning/        # Foundational SFT Pipeline & Data Auditing
    ├── calculate_clinical_metrics_sft.py
    ├── manual_ai_reviewer.py
    ├── overwrite_clinical_txt.py
    ├── penta_lstm_CA_weights.pth
    ├── review_cleanlab_flags.py
    ├── run_cleanlab.py         # Label noise detection
    ├── train_lstm.py           # Multi-GPU SFT execution
    ├── tune_lstm.py            # Optuna Bayesian Hyperparameter Sweeps
    ├── update_clinical_txt.py
    ├── visualize_full_dataset.py
    └── visualize_results.py
