
import os
import time
import random
import gc
from copy import deepcopy
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score

# Optional chain / storage imports (only needed if use_blockchain=True)
from web3 import Web3
import solcx
import ipfshttpclient

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS (modifiable if you need to tweak quickly)
# -----------------------------------------------------------------------------
NUM_CLIENTS           = 50
CLIENTS_PER_ROUND     = 10
BATCH_SIZE            = 32
LEARNING_RATE         = 1e-3
DIRICHLET_ALPHA       = 0.5
STD_FACTOR            = 2.0
ROLLBACK_THRESHOLD    = 0.5   # fraction of outliers triggering rollback

# Per‑dataset schedule
SCHEDULE = {
    "CIFAR10":       {"rounds": 100, "epochs": 2, "runs": 5},
    "MNIST":         {"rounds": 60,  "epochs": 1, "runs": 1},
    "FashionMNIST":  {"rounds": 60,  "epochs": 1, "runs": 1},
}

# -----------------------------------------------------------------------------
# Utility – deterministic behaviour
# -----------------------------------------------------------------------------
BASE_SEED = 42

def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # No GPU usage here, but set anyway
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Dataset loading helpers
# -----------------------------------------------------------------------------

TRANSFORM_TO_TENSOR = transforms.Compose([transforms.ToTensor()])

def load_dataset(name: str):
    """Returns (x_train, y_train_onehot, x_test, y_test_onehot)."""
    if name == "CIFAR10":
        train_ds = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=TRANSFORM_TO_TENSOR)
        test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=TRANSFORM_TO_TENSOR)
        x_train = train_ds.data.astype("float32") / 255.0  # (N, 32, 32, 3)
        x_test  = test_ds.data.astype("float32") / 255.0
        y_train = np.array(train_ds.targets)
        y_test  = np.array(test_ds.targets)
        x_train = x_train  # already (H,W,C)
        x_test  = x_test
    else:
        # MNIST or FashionMNIST – need to convert PIL -> numpy
        DatasetClass = {
            "MNIST": torchvision.datasets.MNIST,
            "FashionMNIST": torchvision.datasets.FashionMNIST,
        }[name]
        train_ds = DatasetClass(root="./data", train=True,  download=True, transform=TRANSFORM_TO_TENSOR)
        test_ds  = DatasetClass(root="./data", train=False, download=True, transform=TRANSFORM_TO_TENSOR)
        x_train = np.array([img.numpy().squeeze() for img, _ in train_ds], dtype="float32")  # (N, 28, 28)
        x_test  = np.array([img.numpy().squeeze() for img, _ in test_ds],  dtype="float32")
        y_train = np.array([label for _, label in train_ds])
        y_test  = np.array([label for _, label in test_ds])
        # add channel dim
        x_train = np.expand_dims(x_train, -1) / 255.0
        x_test  = np.expand_dims(x_test,  -1) / 255.0
    # one‑hot labels
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot  = np.eye(num_classes)[y_test]
    return x_train, y_train_onehot, x_test, y_test_onehot

# -----------------------------------------------------------------------------
# Dirichlet non‑IID partitioning
# -----------------------------------------------------------------------------

def partition_data_dirichlet(y_train_onehot: np.ndarray, num_clients: int, alpha: float):
    num_classes = y_train_onehot.shape[1]
    client_indices = {i: [] for i in range(num_clients)}
    for k in range(num_classes):
        class_idx = np.where(np.argmax(y_train_onehot, axis=1) == k)[0]
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        split_sizes = (proportions * len(class_idx)).astype(int)
        diff = len(class_idx) - split_sizes.sum()
        for i in range(diff):
            split_sizes[i % num_clients] += 1
        start = 0
        for i in range(num_clients):
            end = start + split_sizes[i]
            client_indices[i].extend(class_idx[start:end])
            start = end
    return client_indices

# -----------------------------------------------------------------------------
# Lightweight CNN model (PyTorch)
# -----------------------------------------------------------------------------

class CNNModel(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 10):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.flatten_dim = 64 * (h // 2) * (w // 2)
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.flatten_dim)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# -----------------------------------------------------------------------------
# Helper functions for weights
# -----------------------------------------------------------------------------

def get_model_weights(model: nn.Module):
    return [p.detach().cpu().numpy() for p in model.parameters()]

def set_model_weights(model: nn.Module, weights: List[np.ndarray]):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(torch.tensor(w, dtype=p.dtype))

# -----------------------------------------------------------------------------
# Robust aggregation with outlier detection
# -----------------------------------------------------------------------------

def euclidean_distance(a: List[np.ndarray], b: List[np.ndarray]):
    return sum(np.linalg.norm(x - y) for x, y in zip(a, b))

def robust_fedavg(all_client_weights: List[List[np.ndarray]], std_factor: float):
    if not all_client_weights:
        return None, 0.0
    n_clients = len(all_client_weights)
    n_layers  = len(all_client_weights[0])
    # mean
    mean_w = [np.zeros_like(all_client_weights[0][l]) for l in range(n_layers)]
    for cw in all_client_weights:
        for l in range(n_layers):
            mean_w[l] += cw[l] / n_clients
    # distances
    dists = [euclidean_distance(cw, mean_w) for cw in all_client_weights]
    thr = np.mean(dists) + std_factor * np.std(dists)
    mask = [d <= thr for d in dists]
    n_out = mask.count(False)
    if n_out == n_clients:
        return mean_w, 1.0  # avoid div0
    # re‑average without outliers
    final_w = [np.zeros_like(mean_w[l]) for l in range(n_layers)]
    valid = 0
    for keep, cw in zip(mask, all_client_weights):
        if keep:
            valid += 1
            for l in range(n_layers):
                final_w[l] += cw[l]
    for l in range(n_layers):
        final_w[l] /= valid
    return final_w, n_out / n_clients

# -----------------------------------------------------------------------------
# Blockchain / IPFS integration (unchanged from reference scripts)
# -----------------------------------------------------------------------------

class IPFSClient:
    def __init__(self, address="/ip4/127.0.0.1/tcp/5001"):
        try:
            self.client = ipfshttpclient.connect(address)
            print(f"[IPFS] Connected → {address}")
        except ipfshttpclient.exceptions.VersionMismatch as e:
            print("[IPFS] Version mismatch – patching", e)
            import ipfshttpclient.client as _cm
            _cm.assert_version = lambda *a, **kw: None
            self.client = ipfshttpclient.connect(address)
        except Exception as e:
            raise ConnectionError(f"Cannot connect to IPFS: {e}")

    def add_str(self, data: str):
        return self.client.add_str(data)


def generate_zkp(weights):
    return Web3.keccak(text=str(weights))

def verify_zkp(weights, proof):
    return proof == Web3.keccak(text=str(weights))


class Blockchain:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
        if not self.web3.is_connected():
            raise ConnectionError("Ganache not reachable at :7545")
        self.account = self.web3.eth.accounts[0]
        self.contract = None

    def deploy(self):
        solcx.install_solc("0.8.17")
        solcx.set_solc_version("0.8.17")
        source = open(os.path.join(os.path.dirname(__file__), "ModelRegistryEnhanced.sol"), "r").read() if os.path.exists("ModelRegistryEnhanced.sol") else MODEL_REGISTRY_SOURCE
        compiled = solcx.compile_source(source, output_values=["abi", "bin"])
        _, iface = compiled.popitem()
        Contract = self.web3.eth.contract(abi=iface["abi"], bytecode=iface["bin"])
        tx = Contract.constructor().transact({"from": self.account, "gas": 3_000_000})
        receipt = self.web3.eth.wait_for_transaction_receipt(tx)
        self.contract = self.web3.eth.contract(address=receipt.contractAddress, abi=iface["abi"])
        print(f"[BC] Deployed → {receipt.contractAddress}")

    def add_update(self, client_id: int, model_hash: bytes, ipfs_hash: bytes, zkp: bytes):
        tx = self.contract.functions.addUpdate(model_hash, ipfs_hash, zkp).transact({"from": self.account, "gas": 500_000})
        self.web3.eth.wait_for_transaction_receipt(tx)
        print(f"[BC] Logged client {client_id}")

    def rollback(self, index: int):
        tx = self.contract.functions.rollback(index).transact({"from": self.account, "gas": 500_000})
        self.web3.eth.wait_for_transaction_receipt(tx)
        print(f"[BC] Rollback → {index}")

MODEL_REGISTRY_SOURCE = '''
pragma solidity 0.8.17;
contract ModelRegistryEnhanced {
    address public owner;
    struct ModelUpdate {
        address client;
        bytes32 modelHash;
        bytes32 ipfsHash;
        bytes32 zkpProof;
        uint256 timestamp;
    }
    ModelUpdate[] public updates;
    uint256 public currentIndex;
    event UpdateAdded(address indexed client, bytes32 modelHash, bytes32 ipfsHash, bytes32 zkpProof, uint256 timestamp, uint256 index);
    event Rollback(uint256 fromIndex, uint256 toIndex);
    modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; }
    constructor() { owner = msg.sender; currentIndex = 0; }
    function addUpdate(bytes32 m, bytes32 i, bytes32 z) external {
        updates.push(ModelUpdate(msg.sender, m, i, z, block.timestamp));
        currentIndex = updates.length - 1;
        emit UpdateAdded(msg.sender, m, i, z, block.timestamp, currentIndex);
    }
    function rollback(uint256 idx) external onlyOwner {
        require(idx < updates.length, "bad idx");
        uint256 from = currentIndex; currentIndex = idx; emit Rollback(from, idx);
    }
}
'''

# -----------------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
    n = len(loader.dataset)
    loss = loss_sum / n
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(np.eye(10)[y_true], np.eye(10)[y_pred], multi_class='ovo')
    except ValueError:
        auc = 0.0
    model.train()
    return loss, acc, precision, recall, auc

# -----------------------------------------------------------------------------
# Federated training one run (No‑BC or BC)
# -----------------------------------------------------------------------------

def federated_run(
    x_train: np.ndarray,
    y_train_onehot: np.ndarray,
    x_test: np.ndarray,
    y_test_onehot: np.ndarray,
    rounds: int,
    epochs_per_round: int,
    use_blockchain: bool,
    seed: int,
):
    set_global_seed(seed)

    device = torch.device('cpu')
    num_classes = 10
    # convert to torch tensors, channels first
    x_train_t = torch.tensor(x_train).permute(0, 3, 1, 2)
    x_test_t  = torch.tensor(x_test).permute(0, 3, 1, 2)
    y_train_t = torch.tensor(np.argmax(y_train_onehot, 1))
    y_test_t  = torch.tensor(np.argmax(y_test_onehot, 1))

    # build data partition
    client_idxs = partition_data_dirichlet(y_train_onehot, NUM_CLIENTS, DIRICHLET_ALPHA)
    clients_loaders = []
    for idxs in client_idxs.values():
        ds = TensorDataset(x_train_t[idxs], y_train_t[idxs])
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        clients_loaders.append(loader)

    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    input_shape = (x_train_t.shape[1], x_train_t.shape[2], x_train_t.shape[3])
    global_model = CNNModel(input_shape).to(device)

    # optional chain
    chain, ipfs = None, None
    if use_blockchain:
        chain = Blockchain(); chain.deploy()
        ipfs  = IPFSClient()

    hist = {k: [] for k in ["loss", "accuracy", "precision", "recall", "auc", "outlier_frac"]}
    criterion = nn.CrossEntropyLoss()

    for r in range(rounds):
        sel_clients = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)
        local_w = []
        for cid in sel_clients:
            client_model = CNNModel(input_shape).to(device)
            set_model_weights(client_model, get_model_weights(global_model))
            opt = optim.Adam(client_model.parameters(), lr=LEARNING_RATE)
            for _ in range(epochs_per_round):
                for xb, yb in clients_loaders[cid]:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = criterion(client_model(xb), yb)
                    loss.backward(); opt.step()
            weights = get_model_weights(client_model)
            if use_blockchain:
                proof = generate_zkp(weights)
                if not verify_zkp(weights, proof):
                    continue  # skip bad client
                try:
                    cid_str = ipfs.add_str(str(weights))
                    ipfs_hash  = Web3.keccak(text=cid_str)
                    model_hash = Web3.keccak(text=str(weights))
                    chain.add_update(int(cid), model_hash, ipfs_hash, proof)
                except Exception as e:
                    print("[IPFS] Error, skipping client", cid, e)
                    continue
            local_w.append(weights)
            del client_model
            gc.collect()
        new_w, out_frac = robust_fedavg(local_w, STD_FACTOR)
        set_model_weights(global_model, new_w)
        if use_blockchain and out_frac > ROLLBACK_THRESHOLD:
            chain.rollback(0)
        l, a, p, r_, u = evaluate(global_model, test_loader, device)
        for k, v in zip(hist.keys(), [l, a, p, r_, u, out_frac]):
            hist[k].append(v)
        print(f"[{'BC' if use_blockchain else 'NoBC'}] Round {r+1}/{rounds} – Acc {a:.4f} – outliers {100*out_frac:.1f}%")
        gc.collect()
    return hist

# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def main():
    results_summary = {}
    for dataset in ["CIFAR10", "MNIST", "FashionMNIST"]:
        cfg = SCHEDULE[dataset]
        x_tr, y_tr, x_te, y_te = load_dataset(dataset)
        runs = cfg["runs"]
        rounds = cfg["rounds"]
        epochs = cfg["epochs"]
        print("\n===", dataset, "===")
        # store per‑run metrics
        metrics_no_bc, metrics_bc = [], []
        base_seed = BASE_SEED * 100  # ensure separation from outer seeds
        for run_idx in range(runs):
            seed = base_seed + run_idx
            print(f"\nRun {run_idx+1}/{runs} – seed {seed}")
            start = time.time()
            hist_no = federated_run(x_tr, y_tr, x_te, y_te, rounds, epochs, False, seed)
            t_no = time.time() - start
            start = time.time()
            hist_bc = federated_run(x_tr, y_tr, x_te, y_te, rounds, epochs, True, seed)
            t_bc = time.time() - start
            metrics_no_bc.append({"final_acc": hist_no["accuracy"][-1], "time": t_no})
            metrics_bc.append   ({"final_acc": hist_bc["accuracy"][-1], "time": t_bc})
        df_no = pd.DataFrame(metrics_no_bc)
        df_bc = pd.DataFrame(metrics_bc)
        if runs > 1:
            print("\n–––", dataset, "results across", runs, "runs (No‑BC) –––")
            print(df_no)
            print("Mean ± Std Acc:", df_no["final_acc"].mean(), "±", df_no["final_acc"].std())
            print("–––", dataset, "results across", runs, "runs (BC) –––")
            print(df_bc)
            print("Mean ± Std Acc:", df_bc["final_acc"].mean(), "±", df_bc["final_acc"].std())
        results_summary[dataset] = {
            "NoBC_mean_acc": df_no["final_acc"].mean(),
            "BC_mean_acc":   df_bc["final_acc"].mean(),
        }
    print("\n=== Overall summary (mean accuracy) ===")
    for d, vals in results_summary.items():
        print(f"{d:13s} | No‑BC {vals['NoBC_mean_acc']:.4f} | BC {vals['BC_mean_acc']:.4f}")


if __name__ == "__main__":
    main()
