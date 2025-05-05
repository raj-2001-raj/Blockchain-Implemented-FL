# Blockchain-Implemented-FL
# 📦 Federated Learning + Blockchain (BCFL)

> **Reproducible experimental suite for MNIST, Fashion‑MNIST & CIFAR‑10 combining robust federated averaging with an optional Blockchain + IPFS trust layer**

---

## 🗺️ Table of Contents

1. [Project Overview](#project-overview)
2. [Hardware Testbed](#hardware-testbed)
3. [Datasets & Non‑IID Partitioning](#datasets--non-iid-partitioning)
4. [Training Protocol](#training-protocol)
5. [Model Architecture](#model-architecture)
6. [Robust Aggregation & Outlier Handling](#robust-aggregation--outlier-handling)
7. [Blockchain ✕ IPFS Layer](#blockchain--ipfs-layer)
8. [Reproducibility ✔︎](#reproducibility-)
9. [Running the Experiments](#running-the-experiments)
10. [Key Results](#key-results)
11. [Security Insights](#security-insights)
12. [Repository Layout](#repository-layout)
13. [References](#references)

---

## Project Overview

This repository accompanies **§ 3.7 – § 3.8** of the thesis and offers a self‑contained implementation of a *blockchain‑enabled federated learning* (BCFL) framework.

* **Baseline:** Robust Federated Averaging with outlier detection.
* **Enhancement:** Optional logging of every client update on a private Ethereum chain (Ganache) plus decentralized weight storage on IPFS, backed by a lightweight hash‑based zero‑knowledge proof (ZKP).
* **Goal:** Evaluate the **trust vs. overhead** trade‑off across three image‑classification datasets.

---

## Hardware Testbed

| Component | Spec                                                                    |
| --------- | ----------------------------------------------------------------------- |
| **CPU**   | 11th Gen Intel® Core™ i5‑1135G7 (4 P‑cores / 8 threads, 2.4 → 2.53 GHz) |
| **GPU**   | Integrated Intel® Iris Xe                                               |
| **RAM**   | 8 GB                                                                    |
| **OS**    | Tested on Ubuntu 22.04 LTS & Windows 11                                 |

> *The code runs entirely on CPU; no CUDA required.*

---

## Datasets & Non‑IID Partitioning

* **CIFAR‑10** – 32 × 32 × 3 colour (10 classes)
* **MNIST** / **Fashion‑MNIST** – 28 × 28 × 1 greyscale (10 classes)

```text
Normalization : images / 255.0  →  [0, 1]
Labels        : one‑hot vectors of length 10
Partitioning  : Dirichlet(α = 0.5)   # realistic, skewed client data
Clients       : 50 total  (10 selected per round)
```

---

## Training Protocol

| Dataset            | Rounds × Epochs | Runs  | Batch | LR    | Seed behaviour                        |
| ------------------ | --------------- | ----- | ----- | ----- | ------------------------------------- |
| **CIFAR‑10**       | 100 × 2         | **5** | 32    | 0.001 | Same seed pair per *No‑BC* & *BC* run |
| **MNIST / FMNIST** | 60 × 1          | 1     | 32    | 0.001 | Fixed global seed                     |

*At each round, 10/50 clients are sampled without replacement.*

---

## Model Architecture

```text
[Input 32×32×3 / 28×28×1]
↓ Conv(3×3, 32 f, ReLU)
↓ Conv(3×3, 64 f, ReLU)
↓ MaxPool(2×2)
↓ Flatten
↓ Dense(128, ReLU)
↓ Dense(10, Softmax)
```

Lightweight enough to fit into < 1 MB of parameters, keeping memory under 8 GB even for 50 simultaneous client replicas.

---

## Robust Aggregation & Outlier Handling

1. **Euclidean distance** between each client update and the mean update.
2. Updates > 2 σ from the mean ⇒ *outliers*.
3. If > 50 % clients are outliers, the smart contract can **rollback** to a trusted checkpoint.

---

## Blockchain ✕ IPFS Layer

| Component                                    | Purpose                                                                |
| -------------------------------------------- | ---------------------------------------------------------------------- |
| **Ganache** (local Ethereum)                 | Immutable ledger of updates.                                           |
| **Smart Contract (`ModelRegistryEnhanced`)** | Stores `(hash(weights), CID, ZKP, timestamp)` + provides rollback.     |
| **IPFS**                                     | Decentralized, content‑addressed storage of serialized weight tensors. |
| **ZKP (simulated)**                          | `keccak(weights)` ensures tamper‑free uploads without revealing data.  |

Toggle with `use_blockchain=True | False`.

---

## Reproducibility ✔︎

* Global seeds set for **NumPy, PyTorch, random**.
* Identical client subsets for paired (*No‑BC* vs *BC*) runs.
* Every CIFAR‑10 experiment repeated **5×**; we report **mean ± std**.

---

## Running the Experiments

```bash
# 1 – clone & install deps
$ git clone https://github.com/<you>/federated-fl-experiments.git
$ cd federated-fl-experiments
$ pip install -r requirements.txt

# 2 – (optional) start Ganache + IPFS daemon in two terminals
$ ganache --deterministic --port 7545
$ ipfs daemon

# 3 – run the script (CPU‑only)
$ python federated_experimental_setup.py
```

The script will:

1. Download datasets automatically.
2. Train *No‑Blockchain* and *Blockchain* variants per schedule.
3. Print per‑round logs and final tables + save plots to `./results/`.

> **Tip:** Use `CTRL‑C` safely at any time; intermediate stats are printed after each round.

---

## Key Results

| Dataset           | Metric        | No‑BC         | BC              | Δ (BC – No) |
| ----------------- | ------------- | ------------- | --------------- | ----------- |
| **MNIST**         | Accuracy      | 98.2 – 98.8 % | ± 0.1 %         | negligible  |
| **Fashion‑MNIST** | Accuracy      | 86 – 90 %     | −0.99 → +1.07 % | mixed       |
| **CIFAR‑10**      | Accuracy      | 51.96 ± 3.1 % | 51.58 ± 3.3 %   | −0.38 pp    |
|                   | Time overhead | —             | **+≈10 %**      |             |

Detailed tables & box‑plots are reproduced in **`docs/results/`**.

---

## Security Insights

* **Immutability** – every update hashed & timestamped on‑chain.
* **Data integrity** – IPFS CIDs change if weights are tampered with.
* **Rollback** – contract owner can revert if > 50 % outliers.
* **Outlier screening** – Euclidean filtering thwarts many poisoning attacks.

*(See Chapter 3.8 for a full discussion.)*

---

---

## References

* **\[24]** Ganache – [https://trufflesuite.com/ganache](https://trufflesuite.com/ganache)
* **\[25]** Benet, J. *IPFS – Content Addressed, Versioned, P2P File System*, 2014.
* **\[26]** Sasson, E.B. *et al.* *ZK‑SNARKs for Privacy‑Preserving Proofs*, 2014.
* **\[27]** Ben‑Sasson, E. *et al.* *Scalable, transparent STARK proofs*, 2018.

---

> © 2025 Rajatkant Nayak — MIT Licence
