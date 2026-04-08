
import argparse, time, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- DATA ----------
def make_synthetic_longtail(n=120_000, n_features=20, n_classes=19, seed=0):
    rng = np.random.default_rng(seed)
    # zipf-ish class sizes
    weights = 1.0 / np.arange(1, n_classes + 1) ** 1.6
    weights /= weights.sum()
    counts = (weights * n).astype(int)
    counts[0] += n - counts.sum()
    Xs, ys = [], []
    for c, k in enumerate(counts):
        if k <= 0: continue
        mean = rng.normal(0, 2.5, n_features)
        cov_diag = rng.uniform(0.3, 1.5, n_features)
        Xs.append(rng.normal(mean, cov_diag, size=(k, n_features)))
        ys.append(np.full(k, c))
    X = np.vstack(Xs); y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    df = pd.DataFrame(X[perm], columns=[f"f{i}" for i in range(n_features)])
    df["label"] = y[perm]
    return df

def load_data(path, label_col):
    if path == "synthetic":
        return make_synthetic_longtail(), "label"
    df = pd.read_csv(path)
    return df, label_col

def stratified_subset(df, label_col, n, min_per_class=2, seed=0):
    """Stratified sample with a floor per class so rare labels survive."""
    rng = np.random.default_rng(seed)
    parts = []
    classes = df[label_col].unique()
    # floor first
    for c in classes:
        sub = df[df[label_col] == c]
        take = min(len(sub), min_per_class)
        parts.append(sub.sample(take, random_state=seed))
    floor = pd.concat(parts)
    remaining = n - len(floor)
    if remaining > 0:
        rest = df.drop(floor.index)
        # proportional fill
        rest_sample = rest.groupby(label_col, group_keys=False).apply(
            lambda g: g.sample(max(1, int(round(remaining * len(g) / len(rest)))),
                               random_state=seed, replace=False)
        )
        out = pd.concat([floor, rest_sample]).sample(frac=1, random_state=seed)
    else:
        out = floor.sample(n, random_state=seed)
    return out.reset_index(drop=True)

# ---------- GENERATORS ----------
class TinyCTGAN:
    """Wrapper around CTGAN for speed."""
    def __init__(self, epochs=80):
        from ctgan import CTGAN
        self.model = CTGAN(epochs=epochs, verbose=False)
    def fit(self, df, discrete_cols):
        self.model.fit(df, discrete_columns=discrete_cols)
    def sample(self, n):
        return self.model.sample(n)

class MLPDiffusion(nn.Module):
    """Tiny conditional Gaussian diffusion on continuous features, label-conditioned."""
    def __init__(self, d, n_classes, hidden=256, T=100):
        super().__init__()
        self.T = T
        self.n_classes = n_classes
        self.emb = nn.Embedding(n_classes, 32)
        self.t_emb = nn.Embedding(T, 32)
        self.net = nn.Sequential(
            nn.Linear(d + 64, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d),
        )
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", torch.cumprod(alphas, 0))

    def forward(self, x, t, y):
        h = torch.cat([x, self.emb(y), self.t_emb(t)], dim=-1)
        return self.net(h)

    def loss(self, x, y):
        b = x.size(0)
        t = torch.randint(0, self.T, (b,), device=x.device)
        noise = torch.randn_like(x)
        abar = self.abar[t].unsqueeze(-1)
        xt = abar.sqrt() * x + (1 - abar).sqrt() * noise
        pred = self(xt, t, y)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, y):
        b = y.size(0)
        d = self.net[-1].out_features
        x = torch.randn(b, d, device=y.device)
        for t in reversed(range(self.T)):
            tt = torch.full((b,), t, device=y.device, dtype=torch.long)
            pred = self(x, tt, y)
            alpha = self.alphas[t]; abar = self.abar[t]
            mean = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - abar).sqrt() * pred)
            if t > 0:
                x = mean + self.betas[t].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x

def train_diffusion(X, y, n_classes, epochs=200, bs=256, lr=2e-3):
    d = X.shape[1]
    model = MLPDiffusion(d, n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    n = len(Xt)
    for ep in range(epochs):
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            loss = model.loss(Xt[b], yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def diffusion_sample(model, class_counts):
    out_x, out_y = [], []
    for c, k in class_counts.items():
        if k <= 0: continue
        y = torch.full((k,), c, dtype=torch.long, device=DEVICE)
        out_x.append(model.sample(y).cpu().numpy())
        out_y.append(np.full(k, c))
    return np.vstack(out_x), np.concatenate(out_y)

class GMMBaseline:
    """Per-class Gaussian — proxy 'LLM-like' fast baseline so the script runs end-to-end without GPU LLMs."""
    def fit(self, X, y):
        self.stats = {}
        for c in np.unique(y):
            Xc = X[y == c]
            mu = Xc.mean(0)
            sd = Xc.std(0) + 1e-3
            self.stats[c] = (mu, sd)
    def sample(self, class_counts):
        rng = np.random.default_rng(0)
        xs, ys = [], []
        for c, k in class_counts.items():
            if k <= 0 or c not in self.stats: continue
            mu, sd = self.stats[c]
            xs.append(rng.normal(mu, sd, size=(k, len(mu))))
            ys.append(np.full(k, c))
        return np.vstack(xs), np.concatenate(ys)

# ---------- METRICS ----------
def kl_label(real_y, syn_y, n_classes):
    p = np.bincount(real_y, minlength=n_classes) + 1e-6
    q = np.bincount(syn_y, minlength=n_classes) + 1e-6
    p /= p.sum(); q /= q.sum()
    return float(entropy(p, q))

def downstream_eval(X_tr, y_tr, X_te, y_te):
    clf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=0)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro", zero_division=0)
    # PR-AUC macro one-vs-rest
    classes = clf.classes_
    y_te_oh = np.zeros((len(y_te), len(classes)))
    for i, c in enumerate(classes):
        y_te_oh[:, i] = (y_te == c).astype(int)
    try:
        prauc = average_precision_score(y_te_oh, proba, average="macro")
    except Exception:
        prauc = float("nan")
    return acc, f1, prauc

# ---------- EXPERIMENT ----------
def run_one(df, label_col, n, seed, results):
    sub = stratified_subset(df, label_col, n, min_per_class=2, seed=seed)
    y_all = sub[label_col].values
    X_all = sub.drop(columns=[label_col]).values.astype(np.float32)
    n_classes = int(df[label_col].max() + 1)

    scaler = StandardScaler().fit(X_all)
    Xs = scaler.transform(X_all)

    # held-out real test from the FULL df (not the subset) to avoid leakage
    test = df.drop(sub.index, errors="ignore").sample(min(20_000, len(df)-len(sub)), random_state=seed)
    X_te = scaler.transform(test.drop(columns=[label_col]).values.astype(np.float32))
    y_te = test[label_col].values

    real_counts = pd.Series(y_all).value_counts().to_dict()

    # ----- Baseline: train on real subset -----
    acc, f1, pr = downstream_eval(Xs, y_all, X_te, y_te)
    results.append(dict(size=n, seed=seed, method="real_only",
                        acc=acc, f1=f1, prauc=pr, kl=0.0, time=0.0))

    # ----- GAN -----
    t0 = time.time()
    gan = TinyCTGAN(epochs=60)
    gan_df = sub.copy()
    gan_df[label_col] = gan_df[label_col].astype(str)
    gan.fit(gan_df, discrete_cols=[label_col])
    syn = gan.sample(n)
    t_gan = time.time() - t0
    sy = syn[label_col].astype(int).values
    sX = scaler.transform(syn.drop(columns=[label_col]).values.astype(np.float32))
    acc, f1, pr = downstream_eval(sX, sy, X_te, y_te)
    results.append(dict(size=n, seed=seed, method="GAN_synonly",
                        acc=acc, f1=f1, prauc=pr,
                        kl=kl_label(y_all, sy, n_classes), time=t_gan))
    # real + synthetic
    acc, f1, pr = downstream_eval(np.vstack([Xs, sX]), np.concatenate([y_all, sy]), X_te, y_te)
    results.append(dict(size=n, seed=seed, method="GAN_real+syn",
                        acc=acc, f1=f1, prauc=pr,
                        kl=kl_label(y_all, sy, n_classes), time=t_gan))

    # ----- Diffusion -----
    t0 = time.time()
    diff = train_diffusion(Xs, y_all, n_classes, epochs=150)
    sX, sy = diffusion_sample(diff, real_counts)
    t_diff = time.time() - t0
    acc, f1, pr = downstream_eval(sX, sy, X_te, y_te)
    results.append(dict(size=n, seed=seed, method="Diffusion_synonly",
                        acc=acc, f1=f1, prauc=pr,
                        kl=kl_label(y_all, sy, n_classes), time=t_diff))
    acc, f1, pr = downstream_eval(np.vstack([Xs, sX]), np.concatenate([y_all, sy]), X_te, y_te)
    results.append(dict(size=n, seed=seed, method="Diffusion_real+syn",
                        acc=acc, f1=f1, prauc=pr,
                        kl=kl_label(y_all, sy, n_classes), time=t_diff))

    # ----- GMM ('LLM-fast' baseline placeholder) -----
    t0 = time.time()
    gmm = GMMBaseline(); gmm.fit(Xs, y_all)
    sX, sy = gmm.sample(real_counts)
    t_gmm = time.time() - t0
    acc, f1, pr = downstream_eval(sX, sy, X_te, y_te)
    results.append(dict(size=n, seed=seed, method="GMM_synonly",
                        acc=acc, f1=f1, prauc=pr,
                        kl=kl_label(y_all, sy, n_classes), time=t_gmm))

    # ----- Rare-class-only augmentation with diffusion -----
    rare_thresh = 0.01 * len(y_all)
    rare = {c: int(rare_thresh - k) for c, k in real_counts.items() if k < rare_thresh}
    if rare:
        sX_r, sy_r = diffusion_sample(diff, rare)
        acc, f1, pr = downstream_eval(np.vstack([Xs, sX_r]),
                                      np.concatenate([y_all, sy_r]), X_te, y_te)
        results.append(dict(size=n, seed=seed, method="Diffusion_rare_aug",
                            acc=acc, f1=f1, prauc=pr,
                            kl=kl_label(y_all, sy_r, n_classes), time=t_diff))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--label", default="label")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    df, label_col = load_data(args.data, args.label)
    # encode labels to 0..K-1
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])
    print(f"Loaded {len(df)} rows, {df[label_col].nunique()} classes")

    results = []
    for n in args.sizes:
        for s in range(args.seeds):
            print(f"\n=== size={n} seed={s} ===")
            run_one(df, label_col, n, s, results)
            pd.DataFrame(results).to_csv(args.out, index=False)

    res = pd.DataFrame(results)
    print("\n=== AGGREGATED (mean ± std over seeds) ===")
    agg = res.groupby(["size", "method"]).agg(
        acc=("acc", "mean"), acc_std=("acc", "std"),
        f1=("f1", "mean"), f1_std=("f1", "std"),
        prauc=("prauc", "mean"), kl=("kl", "mean"), time=("time", "mean")
    ).round(4)
    print(agg)
    agg.to_csv(args.out.replace(".csv", "_agg.csv"))

if __name__ == "__main__":
    main()