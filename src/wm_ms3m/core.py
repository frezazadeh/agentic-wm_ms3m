import os
import math
import json
import types
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import TensorDataset, DataLoader

"""
WM--MS3M PRB-Conditioned World Model
-----------------------------------
Self-contained implementation with:

- WMMS3MConfig          (configuration)
- WMMS3MModel           (multi-scale SSM + compact latent world model)
- RewardModule          (scaled-space reward)
- train_and_eval_wm_ms3m
- wm_ms3m_cem_plan_next_prb
- wm_ms3m_demo_planning

Assumptions:
- x.npy: shape (N, L, F)  rolling window of L past KPI vectors
- y.npy: shape (N, F)     next-step KPI
- Feature order matches WMMS3MConfig.feature_names
- PRBs at index 5 are the action channel
"""

# ---------- Paths (default for packaged project) ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CKPT_DIR = os.path.join(BASE_DIR, "artifacts")


# ---------- Config (WM--MS3M style) ----------
class WMMS3MConfig:
    # data / paths
    data_dir = DEFAULT_DATA_DIR
    x_file = "x.npy"
    y_file = "y.npy"

    # KPI feature order (paper Table / notation)
    feature_names = (
        "MCS", "CQI", "RI", "PMI", "Buffer", "PRBs",
        "RSRQ", "RSRP", "RSSI", "SINR", "SE", "BLER", "Delay"
    )

    # Targets (RSRP or all KPIs)
    target = "RSRP"      # "RSRP" or "all"
    rsrp_index = 7       # index of RSRP in feature_names

    # Action: PRBs as control variable
    action_name = "PRBs"
    action_index = 5     # PRB feature index

    # Splits (chronological)
    train_ratio = 0.70
    val_ratio = 0.15

    batch_size = 256
    num_epochs = 100
    lr = 2e-3
    weight_decay = 1e-4

    # WM--MS3M architecture (paper notation)
    d_model = 192          # d: embedding width (deterministic state)
    n_layers = 4           # L_ell: number of MS3M layers
    ssm_state_dim = 64     # N: HiPPO-LegS state size
    n_components = 4       # M: mixture components (time scales)
    z_dim = 48             # d_z: latent world state dimension

    dropout = 0.1

    # HiPPO kernel params
    hippo_dt_init = 0.1

    # Optimization niceties
    seed = 42
    ckpt_dir = DEFAULT_CKPT_DIR
    patience = 30

    # KL annealing (beta schedule)
    beta_kl_start = 0.01
    beta_kl_end = 1.0
    kl_warmup_epochs = 20

    # Auxiliary & robustness
    recon_weight = 1.0          # full KPI reconstruction loss weight
    aux_rsrp_weight = 1.0       # target head weight
    use_heteroscedastic = True  # hetero NLL for target head
    huber_delta = 0.2           # robust term on target mean
    target_consistency = 0.1    # encourage target(mu) ≈ full_decoder[:, rsrp_index]

    # Regularization & augmentation
    feature_channel_dropout = 0.1  # drop entire KPI channels during train
    gaussian_input_noise = 0.01    # std of gaussian noise on inputs (after scaling)

    # Scheduled sampling between posterior and prior during train
    posterior_mix_start = 1.0  # epoch 1: always posterior
    posterior_mix_end = 0.5    # post warmup: 50/50

    # Evaluation
    mc_samples = 8             # test-time Monte Carlo samples from prior

    # Optional multi-step training aid (aligned labels only)
    rollout_train_steps = 0
    rollout_weight = 0.2
    sched_sample_start = 1.0
    sched_sample_end = 0.5

    # Planning / MPC-CEM (paper Alg. 3)
    plan_horizon = 8           # H
    cem_popsize = 256          # P
    cem_elite_frac = 0.1       # rho
    cem_iters = 4              # I
    action_smooth_penalty = 0.05  # λ_sm

    # Reward weights in scaled space
    w_sinr = 1.0
    w_se   = 0.5
    w_bler = 1.0
    w_delay = 0.5
    w_prb_cost = 0.2
    w_rsrp = 0.2

    # Action bounds (%-tile of training distribution, original units)
    action_low_pct = 5.0
    action_high_pct = 95.0

    device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- Scaling ----------
class StandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6):
        self.mean = mean.astype(np.float32)
        self.std = np.clip(std.astype(np.float32), eps, None)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


# ---------- Data loading ----------
from numpy.lib import format as _npfmt


def _inspect_npy_header(path: str):
    with open(path, 'rb') as f:
        magic = _npfmt.read_magic(f)
        if magic[0] == 1:
            shape, fortran, dtype = _npfmt.read_array_header_1_0(f)
        elif magic[0] == 2:
            shape, fortran, dtype = _npfmt.read_array_header_2_0(f)
        else:
            raise RuntimeError("Unsupported NPY version: " + str(magic))
        pos = f.tell()
        f.seek(0, os.SEEK_END)
        total = f.tell()
        data_bytes = total - pos
    return shape, dtype, fortran, data_bytes


def verify_npy_file(path: str):
    shape, dtype, fortran, data_bytes = _inspect_npy_header(path)
    itemsize = np.dtype(dtype).itemsize
    expected_bytes = int(np.prod(shape)) * itemsize
    ok = (expected_bytes == data_bytes)
    return {
        "path": path,
        "shape_in_header": tuple(int(s) for s in shape),
        "dtype": str(dtype),
        "itemsize": itemsize,
        "data_bytes": int(data_bytes),
        "expected_bytes": expected_bytes,
        "byte_match": ok,
    }


def load_data(cfg: WMMS3MConfig):
    x_path = os.path.join(cfg.data_dir, cfg.x_file)
    y_path = os.path.join(cfg.data_dir, cfg.y_file)
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"Expected x.npy and y.npy in data_dir={cfg.data_dir}. "
            f"Got x_exists={os.path.exists(x_path)}, y_exists={os.path.exists(y_path)}"
        )

    # integrity checks before load
    x_info = verify_npy_file(x_path)
    y_info = verify_npy_file(y_path)
    if not x_info["byte_match"]:
        raise RuntimeError(
            (
                f"x.npy appears corrupted. "
                f"Header shape={x_info['shape_in_header']}, dtype={x_info['dtype']}, "
                f"expected_bytes={x_info['expected_bytes']}, actual_bytes={x_info['data_bytes']}."
                " Re-check x.npy."
            )
        )
    if not y_info["byte_match"]:
        raise RuntimeError(
            (
                f"y.npy appears corrupted. "
                f"Header shape={y_info['shape_in_header']}, dtype={y_info['dtype']}, "
                f"expected_bytes={y_info['expected_bytes']}, actual_bytes={y_info['data_bytes']}."
                " Re-check y.npy."
            )
        )

    x = np.load(x_path, allow_pickle=False)  # (N,L,F)
    y = np.load(y_path, allow_pickle=False)  # (N,F)

    assert x.ndim == 3 and y.ndim == 2, f"Expected x (N,L,F) and y (N,F), got {x.shape}, {y.shape}"
    N, L, feat_dim = x.shape
    assert feat_dim == y.shape[1] == len(cfg.feature_names), (
        f"Feature count mismatch: x={feat_dim}, y={y.shape[1]} vs {len(cfg.feature_names)}"
    )

    # chronological split
    n_train = int(N * cfg.train_ratio)
    n_val = int(N * cfg.val_ratio)
    n_test = N - n_train - n_val
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, N)

    x_train, x_val, x_test = x[idx_train], x[idx_val], x[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # per-feature scaling for X over train (across time)
    x_train_flat = x_train.reshape(-1, feat_dim)
    x_mean = x_train_flat.mean(0)
    x_std = x_train_flat.std(0)
    x_scaler = StandardScaler(x_mean, x_std)
    x_train_s = x_scaler.transform(x_train)
    x_val_s = x_scaler.transform(x_val)
    x_test_s = x_scaler.transform(x_test)

    # target selection & scaling
    if cfg.target.lower() == "rsrp":
        t = cfg.rsrp_index
        y_train_t = y_train[:, t:t + 1]
        y_val_t = y_val[:, t:t + 1]
        y_test_t = y_test[:, t:t + 1]
        y_mean = y_train_t.mean(0)
        y_std = y_train_t.std(0)
        y_scaler = StandardScaler(y_mean, y_std)
        y_train_t = y_scaler.transform(y_train_t)
        y_val_t = y_scaler.transform(y_val_t)
        y_test_t = y_scaler.transform(y_test_t)
        out_dim = 1
    else:
        y_mean = y_train.mean(0)
        y_std = y_train.std(0)
        y_scaler = StandardScaler(y_mean, y_std)
        y_train_t = y_scaler.transform(y_train)
        y_val_t = y_scaler.transform(y_val)
        y_test_t = y_scaler.transform(y_test)
        out_dim = feat_dim

    # scaler for full next-step vector (reconstruction)
    y_full_mean = y_train.mean(0)
    y_full_std = y_train.std(0)
    y_full_scaler = StandardScaler(y_full_mean, y_full_std)

    # ACTION STATS (from training set, original units)
    a_idx = cfg.action_index
    a_train = x_train_flat[:, a_idx]
    a_low = float(np.percentile(a_train, cfg.action_low_pct))
    a_high = float(np.percentile(a_train, cfg.action_high_pct))

    # tensors (scaled)
    x_train_t = torch.tensor(x_train_s, dtype=torch.float32)
    x_val_t = torch.tensor(x_val_s, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_t, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_t, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_t, dtype=torch.float32)

    y_train_full = torch.tensor(y_full_scaler.transform(y_train), dtype=torch.float32)
    y_val_full = torch.tensor(y_full_scaler.transform(y_val), dtype=torch.float32)
    y_test_full = torch.tensor(y_full_scaler.transform(y_test), dtype=torch.float32)

    # action sequence (scaled): from x sequences
    a_train_seq = x_train_t[:, :, a_idx:a_idx + 1]
    a_val_seq = x_val_t[:, :, a_idx:a_idx + 1]
    a_test_seq = x_test_t[:, :, a_idx:a_idx + 1]

    train_ds = TensorDataset(x_train_t, y_train_t, y_train_full, a_train_seq)
    val_ds = TensorDataset(x_val_t, y_val_t, y_val_full, a_val_seq)
    test_ds = TensorDataset(x_test_t, y_test_t, y_test_full, a_test_seq)

    meta = {
        "N": int(N),
        "L": int(L),
        "F": int(feat_dim),
        "splits": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
        "target": cfg.target,
        "rsrp_index": cfg.rsrp_index,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "y_full_mean": y_full_mean.tolist(),
        "y_full_std": y_full_std.tolist(),
        "feature_names": list(cfg.feature_names),
        "action_bounds": {"low": a_low, "high": a_high},
    }
    return train_ds, val_ds, test_ds, meta, x_scaler, y_scaler, y_full_scaler


# ---------- HiPPO-LegS Structured SSM Core ----------
def hippo_legs_matrix(N: int, device=None, dtype=torch.float32):
    n = torch.arange(N, device=device, dtype=dtype)
    two_n1 = 2 * n + 1
    sq = torch.sqrt(two_n1[:, None] * two_n1[None, :])
    A = torch.tril(sq, diagonal=-1) * (-1.0)
    A = A + torch.diag(-(n + 1.0))
    B = torch.sqrt(two_n1)
    return A, B


class HiPPOLegSKernel(nn.Module):
    def __init__(self, d_model: int, state_dim: int, L: int, dt_init: float = 0.1):
        super().__init__()
        self.d_model, self.N, self.L = d_model, state_dim, L
        A, B = hippo_legs_matrix(state_dim)
        self.register_buffer("A_ct", A)
        self.register_buffer("B_ref", B)

        B0 = B[None, :].repeat(d_model, 1)
        self.B = nn.Parameter(B0 + 0.01 * torch.randn_like(B0))
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(dt_init, dtype=torch.float32)))

    def _discretize_bilinear(self):
        dt = F_torch.softplus(self.log_dt) + 1e-6
        A = self.A_ct
        N = A.size(0)
        I = torch.eye(N, device=A.device, dtype=A.dtype)
        lhs = I - 0.5 * dt * A
        rhs = I + 0.5 * dt * A
        A_bar = torch.linalg.solve(lhs, rhs)

        B_rhs = (dt * self.B).transpose(0, 1)
        B_bar_T = torch.linalg.solve(lhs, B_rhs)
        B_bar = B_bar_T.transpose(0, 1)
        return A_bar, B_bar

    def forward(self) -> torch.Tensor:
        A_bar, B_bar = self._discretize_bilinear()
        C = self.C
        A_bar_T = A_bar.transpose(0, 1)

        x = B_bar
        out = []
        for _ in range(self.L):
            y = (C * x).sum(dim=-1)
            out.append(y)
            x = x @ A_bar_T

        K = torch.stack(out, dim=1)  # (d_model, L)
        K[:, 0] += self.D
        return K.unsqueeze(1)  # (d_model, 1, L)


class S6MixtureKernel(nn.Module):
    def __init__(self, d_model: int, state_dim: int, L: int, n_components: int = 4, dt_init: float = 0.1):
        super().__init__()
        self.components = nn.ModuleList([
            HiPPOLegSKernel(d_model, state_dim, L, dt_init=dt_init * (1.5 ** i))
            for i in range(n_components)
        ])

    def forward(self) -> torch.Tensor:
        Ks = [comp() for comp in self.components]  # list of (d_model,1,L)
        return torch.stack(Ks, dim=0).sum(0)       # (d_model,1,L)


# ---------- S6 Block (MS3M layer) ----------
class ChannelMix(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 2, use_glu: bool = True):
        super().__init__()
        hidden = mult * d_model
        self.use_glu = use_glu
        if use_glu:
            self.up = nn.Linear(d_model, 2 * hidden)
            self.act = nn.GELU()
            self.down = nn.Linear(hidden, d_model)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model)
            )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.use_glu:
            u = self.up(x)
            a, g = torch.chunk(u, 2, dim=-1)
            y = self.down(self.act(a) * torch.sigmoid(g))
        else:
            y = self.net(x)
        y = self.dropout(y)
        return self.norm(x + y)


class SEGate(nn.Module):
    def __init__(self, d_model: int, r: int = 8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, max(1, d_model // r)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, d_model // r), d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.squeeze(x.transpose(1, 2)).squeeze(-1)  # (B,C)
        g = self.fc(s).unsqueeze(1)
        return x * g


class S6Block(nn.Module):
    def __init__(self, d_model: int, state_dim: int, L: int, dropout: float = 0.1,
                 dt_init: float = 0.1, n_components: int = 4):
        super().__init__()
        self.kernel = S6MixtureKernel(d_model, state_dim, L, n_components=n_components, dt_init=dt_init)
        self.dropout = nn.Dropout(dropout)
        self.se = SEGate(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.mix = ChannelMix(d_model, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        k = self.kernel()               # (C,1,Lk)
        Lk = k.shape[-1]
        x_dw = x.transpose(1, 2)        # (B,C,T)
        y = F_torch.conv1d(
            F_torch.pad(x_dw, (Lk - 1, 0)),
            k,
            groups=C
        ).transpose(1, 2)               # (B,T,C)
        y = self.se(y)
        y = self.dropout(y)
        y = self.norm1(x + y)
        z = self.mix(y)
        return self.norm2(y + z)


# ---------- Reward Module (scaled space) ----------
class RewardModule(nn.Module):
    def __init__(self, feat_idx: dict, cfg: WMMS3MConfig):
        super().__init__()
        self.idx = feat_idx
        # Learnable linear reward (initialized from config) in scaled space
        self.weight = nn.Parameter(torch.tensor([
            cfg.w_sinr,        # SINR
            cfg.w_se,          # SE
            -cfg.w_bler,       # BLER
            -cfg.w_delay,      # Delay
            -cfg.w_prb_cost,   # PRB cost
            cfg.w_rsrp,        # RSRP
        ], dtype=torch.float32))

    def forward(self, y_full_scaled: torch.Tensor, a_scaled: torch.Tensor) -> torch.Tensor:
        # y_full_scaled: (B, F_scaled), a_scaled: (B,1) scaled PRB
        s = y_full_scaled[:, self.idx['SINR']:self.idx['SINR'] + 1]
        se = y_full_scaled[:, self.idx['SE']:self.idx['SE'] + 1]
        bl = y_full_scaled[:, self.idx['BLER']:self.idx['BLER'] + 1]
        de = y_full_scaled[:, self.idx['Delay']:self.idx['Delay'] + 1]
        rsrp = y_full_scaled[:, self.idx['RSRP']:self.idx['RSRP'] + 1]
        feats = torch.cat([s, se, bl, de, a_scaled, rsrp], dim=-1)
        r = (feats * self.weight).sum(dim=-1, keepdim=True)  # (B,1)
        return r


# ---------- WM--MS3M Model (deterministic S6 + stochastic z + actions) ----------
class WMMS3MModel(nn.Module):
    def __init__(self,
                 in_dim: int,
                 act_dim: int,
                 d_model: int,
                 z_dim: int,
                 L: int,
                 n_layers: int,
                 state_dim: int,
                 dropout: float = 0.1,
                 dt_init: float = 0.1,
                 n_components: int = 4,
                 out_dim: int = 1,
                 full_dim: int = 13,
                 rsrp_index: int = 7,
                 feature_channel_dropout: float = 0.0,
                 gaussian_input_noise: float = 0.0,
                 use_hetero: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.act_dim = act_dim
        self.rsrp_index = rsrp_index
        self.feature_channel_dropout = feature_channel_dropout
        self.gaussian_input_noise = gaussian_input_noise
        self.use_hetero = use_hetero

        # Input projection: [X, A] -> d_model
        self.in_proj = nn.Linear(in_dim + act_dim, d_model)

        # MS3M / S6 blocks
        self.blocks = nn.ModuleList([
            S6Block(d_model, state_dim, L, dropout=dropout, dt_init=dt_init, n_components=n_components)
            for _ in range(n_layers)
        ])

        # Prior p(z | d) and posterior q(z | d, y_full)
        self.prior_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * z_dim)
        )
        self.post_net = nn.Sequential(
            nn.Linear(d_model + full_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * z_dim)
        )

        # Decoders
        self.full_decoder = nn.Sequential(  # full KPI reconstruction (scaled)
            nn.Linear(d_model + z_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, full_dim)
        )
        out_t = 2 * out_dim if use_hetero else out_dim
        self.target_head = nn.Sequential(  # target head (mu[, logvar])
            nn.Linear(d_model + z_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_t)
        )

        # AR skip from last step features
        self.ar_head = nn.Sequential(
            nn.Linear(in_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim)
        )
        self.skip_gain = nn.Parameter(torch.tensor(0.0))

    def _apply_input_augs(self, x):
        if not self.training:
            return x
        if self.gaussian_input_noise > 0:
            x = x + self.gaussian_input_noise * torch.randn_like(x)
        if self.feature_channel_dropout > 0:
            B, T, F_dim = x.shape
            mask = (torch.rand(B, 1, F_dim, device=x.device) > self.feature_channel_dropout).float()
            x = x * mask
        return x

    def forward_d(self, x_seq: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B,L,F) scaled; a_seq: (B,L,A) scaled
        h = self.in_proj(torch.cat([x_seq, a_seq], dim=-1))
        for blk in self.blocks:
            h = blk(h)
        return h[:, -1, :]  # deterministic summary of history

    @staticmethod
    def _split_gauss(x):
        mu, logvar = torch.chunk(x, 2, dim=-1)
        logvar = torch.clamp(logvar, -8.0, 8.0)
        return mu, logvar

    def forward(self,
                x_seq: torch.Tensor,
                a_seq: torch.Tensor,
                y_full_next: torch.Tensor = None,
                use_posterior: bool = True):
        x_seq = self._apply_input_augs(x_seq)
        d = self.forward_d(x_seq, a_seq)

        prior_mu, prior_logvar = self._split_gauss(self.prior_net(d))

        if use_posterior and y_full_next is not None:
            post_mu, post_logvar = self._split_gauss(self.post_net(torch.cat([d, y_full_next], dim=-1)))
            std = (0.5 * post_logvar).exp()
            eps = torch.randn_like(std)
            z = post_mu + std * eps
            stats = {"prior": (prior_mu, prior_logvar), "post": (post_mu, post_logvar)}
        else:
            std = (0.5 * prior_logvar).exp()
            eps = torch.randn_like(std)
            z = prior_mu + std * eps
            stats = {"prior": (prior_mu, prior_logvar), "post": None}

        hz = torch.cat([d, z], dim=-1)
        y_full_hat = self.full_decoder(hz)
        target_out = self.target_head(hz)
        if self.use_hetero:
            mu_t, logvar_t = torch.chunk(target_out, 2, dim=-1)
        else:
            mu_t, logvar_t = target_out, None

        # AR baseline from last-step features (scaled)
        x_last = x_seq[:, -1, :]
        base_t = self.ar_head(x_last)
        mu_t = mu_t + self.skip_gain.tanh() * base_t

        return (mu_t, logvar_t), y_full_hat, stats


# ---------- Losses & evaluation ----------
def kl_divergence_diag(mu_q, logvar_q, mu_p, logvar_p):
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return kl.sum(dim=-1)  # (B,)


@torch.no_grad()
def evaluate(model, dl, cfg, y_scaler, y_full_scaler, mc_samples: int = None):
    if mc_samples is None:
        mc_samples = cfg.mc_samples
    device = cfg.device
    model.eval()

    mse = mae = 0.0
    n = 0
    preds, trues = [], []

    for xb, yb_target, yb_full, ab in dl:
        xb = xb.to(device)
        ab = ab.to(device)
        acc = 0.0
        for _ in range(mc_samples):
            (mu_t, _), _, _ = model(xb, ab, y_full_next=None, use_posterior=False)
            acc = acc + mu_t
        yhat_t = acc / mc_samples

        p = yhat_t.cpu().numpy()
        y = yb_target.cpu().numpy()
        p_inv = y_scaler.inverse_transform(p)
        y_inv = y_scaler.inverse_transform(y)
        preds.append(p_inv)
        trues.append(y_inv)

        mse += ((p_inv - y_inv) ** 2).sum()
        mae += np.abs(p_inv - y_inv).sum()
        n += p_inv.size

    mse /= n
    mae /= n
    return float(mse), float(mae), np.vstack(preds), np.vstack(trues)


# ---------- Train & Evaluate ----------
def train_and_eval_wm_ms3m(cfg: WMMS3MConfig):
    def todev(t): return t.to(cfg.device)

    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    train_ds, val_ds, test_ds, meta, x_scaler, y_scaler, y_full_scaler = load_data(cfg)
    L = meta["L"]
    feat_dim = meta["F"]
    out_dim = 1 if cfg.target.lower() == "rsrp" else feat_dim

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

    model = WMMS3MModel(
        in_dim=feat_dim,
        act_dim=1,
        d_model=cfg.d_model,
        z_dim=cfg.z_dim,
        L=L,
        n_layers=cfg.n_layers,
        state_dim=cfg.ssm_state_dim,
        dropout=cfg.dropout,
        dt_init=cfg.hippo_dt_init,
        n_components=cfg.n_components,
        out_dim=out_dim,
        full_dim=feat_dim,
        rsrp_index=cfg.rsrp_index,
        feature_channel_dropout=cfg.feature_channel_dropout,
        gaussian_input_noise=cfg.gaussian_input_noise,
        use_hetero=cfg.use_heteroscedastic
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.device == "cuda"))

    best_val = float("inf")
    patience = cfg.patience
    best_path = os.path.join(cfg.ckpt_dir, "best.pt")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        tr_sum, ntr = 0.0, 0
        beta = cfg.beta_kl_start + (cfg.beta_kl_end - cfg.beta_kl_start) * \
            min(1.0, epoch / max(1, cfg.kl_warmup_epochs))
        p_post = cfg.posterior_mix_start + (cfg.posterior_mix_end - cfg.posterior_mix_start) * \
            min(1.0, epoch / max(1, cfg.kl_warmup_epochs))

        for xb, yb_target, yb_full, ab in train_loader:
            xb = todev(xb)
            yb_target = todev(yb_target)
            yb_full = todev(yb_full)
            ab = todev(ab)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(cfg.device == "cuda")):
                use_post = (torch.rand((), device=xb.device).item() < p_post)
                (mu_t, logvar_t), yhat_full, stats = model(
                    xb, ab,
                    y_full_next=yb_full if use_post else None,
                    use_posterior=use_post
                )

                # Reconstruction loss over all features (scaled space)
                recon_loss = F_torch.mse_loss(yhat_full, yb_full)

                # Target loss
                if cfg.use_heteroscedastic and logvar_t is not None:
                    nll = 0.5 * (logvar_t + (yb_target - mu_t) ** 2 / logvar_t.exp() + math.log(2 * math.pi))
                    target_loss = nll.mean()
                else:
                    target_loss = F_torch.mse_loss(mu_t, yb_target)

                # Huber on target mean
                huber = F_torch.smooth_l1_loss(mu_t, yb_target, beta=cfg.huber_delta)

                # Consistency between target and full decoder RSRP
                rsrp_from_full = yhat_full[:, cfg.rsrp_index:cfg.rsrp_index + 1]
                consistency = F_torch.mse_loss(mu_t, rsrp_from_full)

                # KL(q||p)
                mu_p, logvar_p = stats["prior"]
                if stats["post"] is not None:
                    mu_q, logvar_q = stats["post"]
                else:
                    mu_q, logvar_q = mu_p.detach(), logvar_p.detach()
                kl = kl_divergence_diag(mu_q, logvar_q, mu_p, logvar_p).mean()

                loss = (cfg.recon_weight * recon_loss
                        + cfg.aux_rsrp_weight * target_loss
                        + 0.5 * huber
                        + cfg.target_consistency * consistency
                        + beta * kl)

                # Optional multi-step rollout loss (disabled by default)
                if cfg.rollout_train_steps > 0:
                    window_scaled = xb.clone()
                    a_win = ab.clone()
                    roll_mse = 0.0
                    tf_prob = cfg.sched_sample_start + (cfg.sched_sample_end - cfg.sched_sample_start) * \
                        min(1.0, epoch / max(1, cfg.kl_warmup_epochs))
                    for _k in range(cfg.rollout_train_steps):
                        (mu_t_k, _), y_full_k, _ = model(
                            window_scaled, a_win,
                            y_full_next=None,
                            use_posterior=False
                        )
                        roll_mse = roll_mse + F_torch.mse_loss(y_full_k, yb_full)
                        use_tf = (torch.rand((), device=xb.device).item() < tf_prob)
                        next_frame = yb_full if use_tf else y_full_k.detach()
                        window_scaled = torch.cat(
                            [window_scaled[:, 1:, :], next_frame[:, None, :]],
                            dim=1
                        )
                        a_win = torch.cat(
                            [a_win[:, 1:, :], a_win[:, -1:, :]],
                            dim=1
                        )
                    loss = loss + cfg.rollout_weight * (roll_mse / max(1, cfg.rollout_train_steps))

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            tr_sum += loss.item() * xb.size(0)
            ntr += xb.size(0)
            sched.step(epoch - 1 + ntr / max(1, len(train_loader.dataset)))

        tr_loss = tr_sum / ntr

        # ----- Validation -----
        model.eval()
        with torch.no_grad():
            va_sum, nva = 0.0, 0
            for xb, yb_target, yb_full, ab in val_loader:
                xb = todev(xb)
                yb_target = todev(yb_target)
                yb_full = todev(yb_full)
                ab = todev(ab)

                (mu_t, logvar_t), yhat_full, stats = model(
                    xb, ab, y_full_next=yb_full, use_posterior=True
                )
                recon_loss = F_torch.mse_loss(yhat_full, yb_full)
                if cfg.use_heteroscedastic and logvar_t is not None:
                    nll = 0.5 * (logvar_t + (yb_target - mu_t) ** 2 / logvar_t.exp() + math.log(2 * math.pi))
                    target_loss = nll.mean()
                else:
                    target_loss = F_torch.mse_loss(mu_t, yb_target)
                huber = F_torch.smooth_l1_loss(mu_t, yb_target, beta=cfg.huber_delta)
                rsrp_from_full = yhat_full[:, cfg.rsrp_index:cfg.rsrp_index + 1]
                consistency = F_torch.mse_loss(mu_t, rsrp_from_full)
                mu_p, logvar_p = stats["prior"]
                mu_q, logvar_q = stats["post"]
                kl = kl_divergence_diag(mu_q, logvar_q, mu_p, logvar_p).mean()
                vloss = (cfg.recon_weight * recon_loss
                         + cfg.aux_rsrp_weight * target_loss
                         + 0.5 * huber
                         + cfg.target_consistency * consistency
                         + beta * kl)
                va_sum += vloss.item() * xb.size(0)
                nva += xb.size(0)
            va_loss = va_sum / nva

        print(f"Epoch {epoch:03d} | train: {tr_loss:.6f} | val: {va_loss:.6f} | beta {beta:.3f} | p_post {p_post:.2f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "meta": meta,
                "cfg": vars(cfg),
            }, best_path)
            patience = cfg.patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    # ----- Final evaluation -----
    model.load_state_dict(torch.load(best_path, map_location=cfg.device)["model"])

    tr_mse, tr_mae, _, _ = evaluate(model, train_loader, cfg, y_scaler, y_full_scaler,
                                    mc_samples=cfg.mc_samples)
    va_mse, va_mae, _, _ = evaluate(model, val_loader, cfg, y_scaler, y_full_scaler,
                                    mc_samples=cfg.mc_samples)
    te_mse, te_mae, yhat, ytrue = evaluate(model, test_loader, cfg, y_scaler, y_full_scaler,
                                           mc_samples=cfg.mc_samples)

    metrics = {
        "train": {"mse": tr_mse, "rmse": float(math.sqrt(tr_mse)), "mae": tr_mae},
        "val":   {"mse": va_mse, "rmse": float(math.sqrt(va_mse)), "mae": va_mae},
        "test":  {"mse": te_mse, "rmse": float(math.sqrt(te_mse)), "mae": te_mae},
        "best_val_obj": best_val,
    }
    print("Final metrics (original units for target):", json.dumps(metrics, indent=2))

    # Save artifacts
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    np.save(os.path.join(cfg.ckpt_dir, "test_pred.npy"), yhat)
    np.save(os.path.join(cfg.ckpt_dir, "test_true.npy"), ytrue)
    with open(os.path.join(cfg.ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Build reward module + planner helpers
    feat_idx = {name: i for i, name in enumerate(cfg.feature_names)}
    reward_fn = RewardModule(feat_idx, cfg).to(cfg.device)

    bundle = types.SimpleNamespace(
        model=model,
        cfg=cfg,
        meta=meta,
        x_scaler=StandardScaler(
            np.array(meta["x_mean"], dtype=np.float32),
            np.array(meta["x_std"], dtype=np.float32)
        ),
        y_scaler=StandardScaler(
            np.array(meta["y_mean"], dtype=np.float32),
            np.array(meta["y_std"], dtype=np.float32)
        ),
        y_full_scaler=StandardScaler(
            np.array(meta["y_full_mean"], dtype=np.float32),
            np.array(meta["y_full_std"], dtype=np.float32)
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metrics=metrics,
        reward_fn=reward_fn,
        ckpt_path=best_path,
    )
    # convenient handle when used interactively
    globals()["wm_ms3m"] = bundle
    print("Artifacts in:", cfg.ckpt_dir)
    print("In-memory handle:  wm_ms3m")
    return bundle


# ---------- MPC / CEM PLANNER (WM--MS3M) ----------
@torch.no_grad()
def wm_ms3m_cem_plan_next_prb(bundle, x_window_orig: np.ndarray):
    """
    Choose the next PRB value given the *current* window (L,F) in original units.
    Returns (a0_orig, diagnostics).
    """
    model = bundle.model
    cfg = bundle.cfg
    x_scaler = bundle.x_scaler
    y_full_scaler = bundle.y_full_scaler
    reward_fn = bundle.reward_fn

    device = cfg.device
    model.eval()

    a_idx = cfg.action_index

    # Scale input window and extract action history
    x_win_s = torch.tensor(
        x_scaler.transform(x_window_orig[None, ...]),
        dtype=torch.float32,
        device=device
    )  # (1,L,F)
    a_hist_s = x_win_s[:, :, a_idx:a_idx + 1]  # (1,L,1)

    # Action bounds in original units -> scaled
    a_low = bundle.meta["action_bounds"]["low"]
    a_high = bundle.meta["action_bounds"]["high"]
    a_low_s = (a_low - x_scaler.mean[a_idx]) / x_scaler.std[a_idx]
    a_high_s = (a_high - x_scaler.mean[a_idx]) / x_scaler.std[a_idx]

    H = cfg.plan_horizon
    pop = cfg.cem_popsize
    elite_k = max(1, int(cfg.cem_elite_frac * pop))

    # Initialize Gaussian over actions in scaled space
    mu = torch.full((H, 1), a_hist_s[0, -1, 0].item(), device=device)
    sigma = torch.full((H, 1), 0.5, device=device)

    def rollout_with_actions(a_seq_s: torch.Tensor) -> torch.Tensor:
        """
        a_seq_s: (B,H,1) scaled; returns total reward (B,1).
        Deterministic prior-mean rollouts (paper Alg. 3).
        """
        B = a_seq_s.size(0)
        x_seq = x_win_s.repeat(B, 1, 1)   # (B,L,F)
        a_seq = a_hist_s.repeat(B, 1, 1)  # (B,L,1)
        total_R = torch.zeros(B, 1, device=device)
        prev_a = a_seq[:, -1:, :]  # (B,1,1)

        for t in range(H):
            # Deterministic latent (prior mean)
            d = model.forward_d(x_seq, a_seq)
            prior_mu, prior_logvar = model._split_gauss(model.prior_net(d))
            z = prior_mu
            hz = torch.cat([d, z], dim=-1)
            y_full_hat = model.full_decoder(hz)  # scaled (B,F)

            # Candidate action at this step
            a_t = a_seq_s[:, t:t + 1, :]  # (B,1,1)

            # Reward at this step (scaled space)
            r_t = reward_fn(y_full_hat, a_t.squeeze(1))  # (B,1)

            # Action smoothness penalty
            smooth = (a_t - prev_a).abs().mean(dim=-1)  # (B,1)
            total_R = total_R + r_t - cfg.action_smooth_penalty * smooth

            # Advance window
            next_frame = y_full_hat.unsqueeze(1)  # (B,1,F)
            x_seq = torch.cat([x_seq[:, 1:, :], next_frame], dim=1)
            a_seq = torch.cat([a_seq[:, 1:, :], a_t], dim=1)
            prev_a = a_t

        return total_R  # (B,1)

    # CEM iterations
    for _ in range(cfg.cem_iters):
        eps = torch.randn(pop, H, 1, device=device)
        cand = mu.view(1, H, 1) + sigma.view(1, H, 1) * eps
        cand = cand.clamp(min=a_low_s, max=a_high_s)

        scores = rollout_with_actions(cand).squeeze(-1)  # (pop,)
        elite_idx = torch.topk(scores, k=elite_k, dim=0).indices
        elites = cand[elite_idx]

        mu = elites.mean(dim=0)
        sigma = elites.std(dim=0) + 1e-6

    best = mu  # mean of final elites
    a0_s = best[0, 0]
    a0_orig = float(a0_s.cpu().item() * x_scaler.std[cfg.action_index] + x_scaler.mean[cfg.action_index])
    diag = {
        "mu_scaled": mu.detach().cpu().numpy().tolist(),
        "sigma_scaled": sigma.detach().cpu().numpy().tolist(),
        "a0_scaled": float(a0_s.detach().cpu().item()),
        "bounds_scaled": (float(a_low_s), float(a_high_s)),
    }
    return a0_orig, diag


# ---------- Simple demo ----------
@torch.no_grad()
def wm_ms3m_demo_planning(bundle):
    # Grab the first batch from test loader
    test_loader = bundle.test_loader
    for xb, yb_t, yb_full, ab in test_loader:
        x_win_orig = bundle.x_scaler.inverse_transform(xb[0].cpu().numpy())  # (L,F)
        action, diag = wm_ms3m_cem_plan_next_prb(bundle, x_win_orig)
        print("Suggested next PRB (original units):", action)
        return {"suggested_prb": action, "diag": diag}
