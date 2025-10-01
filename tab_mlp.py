# -*- coding: utf-8 -*-
# Simple Tabular MLP with categorical embeddings + Huber loss
# Usage: from tab_mlp import train_mlp_and_predict
import math, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# AMP import compatible with torch<=2.5 and >=2.6
try:
    from torch.amp import autocast, GradScaler  # PyTorch >= 2.6
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast, GradScaler  # PyTorch <= 2.5


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabMLP(nn.Module):
    def __init__(self, n_num, cat_cardinalities, emb_drop=0.05,
                 hidden=(512, 256, 128), p=0.15):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(int(card), int(min(50, max(4, round(1.6 * (card ** 0.56))))))
            for card in (cat_cardinalities or [])
        ])
        emb_dim = sum(e.embedding_dim for e in self.embs)
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_num = nn.BatchNorm1d(n_num) if n_num and n_num > 0 else None

        in_dim = (n_num if n_num else 0) + emb_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(p)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x = None
        if len(self.embs) > 0 and x_cat is not None and x_cat.numel() > 0:
            emb = [emb_(x_cat[:, i]) for i, emb_ in enumerate(self.embs)]
            x = torch.cat(emb, dim=1)
            x = self.emb_drop(x)
        if x_num is not None and x_num.shape[1] > 0:
            if self.bn_num is not None:
                x_num = self.bn_num(x_num)
            x = x_num if x is None else torch.cat([x, x_num], dim=1)
        return self.net(x).squeeze(1)


def _factorize_cats(df_fold: pd.DataFrame, cat_cols):
    if not cat_cols:
        return np.zeros((len(df_fold), 0), dtype="int64"), [], {}
    maps, X_cat = {}, []
    for c in cat_cols:
        vals, idx = np.unique(df_fold[c].astype(str).fillna("__NA__"), return_inverse=True)
        maps[c] = {v: i for i, v in enumerate(vals)}
        X_cat.append(idx.astype("int64"))
    X_cat = np.stack(X_cat, axis=1)
    cards = [len(maps[c]) for c in cat_cols]
    return X_cat, cards, maps


def _apply_cat_maps(df_part: pd.DataFrame, cat_cols, maps):
    if not cat_cols:
        return np.zeros((len(df_part), 0), dtype="int64")
    X_cat = []
    for c in cat_cols:
        m = maps[c]
        arr = (
            df_part[c].astype(str).fillna("__NA__").map(m).fillna(len(m)).astype("int64").to_numpy()
        )
        X_cat.append(arr)
    return np.stack(X_cat, axis=1)


# --- metrics helpers ---
def _rmse_np(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_pred[m] - y_true[m]) ** 2)))


def _mae_np(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_pred[m] - y_true[m])))


def _r2_np(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    ss_res = float(np.sum((yp - yt) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def train_mlp_and_predict(
    Xn_tr: pd.DataFrame, Xn_te: pd.DataFrame,
    Xc_tr: pd.DataFrame, Xc_te: pd.DataFrame,
    y_tr: pd.Series, y_val: pd.Series = None,       # валід. ціль (fold-TE у CV)
    *, max_epochs: int = 40, batch_size: int = 262_144,
    lr: float = 3e-3, weight_decay: float = 1e-5,
    patience: int = 6, hidden=(512, 256, 128),
    dropout: float = 0.15, emb_drop: float = 0.05,
    huber_delta: float = 1.0, num_workers: int = 2,
    standardize: bool = True, early_stop_on: str = "val"  # "val" або "train"
) -> np.ndarray:

    dev = _device()

    # ----- numerics -----
    X_num_tr = Xn_tr.to_numpy(dtype="float32") if Xn_tr is not None and Xn_tr.shape[1] > 0 else None
    X_num_te = Xn_te.to_numpy(dtype="float32") if Xn_te is not None and Xn_te.shape[1] > 0 else None
    if X_num_tr is not None and standardize:
        mu = np.nanmean(X_num_tr, axis=0)
        sd = np.nanstd(X_num_tr, axis=0); sd[sd == 0] = 1.0
        X_num_tr = (np.nan_to_num(X_num_tr) - mu) / sd
        if X_num_te is not None:
            X_num_te = (np.nan_to_num(X_num_te) - mu) / sd

    # ----- categories -----
    cat_cols = list(Xc_tr.columns) if (Xc_tr is not None and len(Xc_tr.columns) > 0) else []
    if cat_cols:
        X_cat_tr_ids, cards, maps = _factorize_cats(Xc_tr, cat_cols)
        X_cat_te_ids = _apply_cat_maps(Xc_te, cat_cols, maps)
        cards = [c + 1 for c in cards]  # unseen bucket
    else:
        X_cat_tr_ids = np.zeros((len(Xn_tr), 0), dtype="int64")
        X_cat_te_ids = np.zeros((len(Xn_te), 0), dtype="int64")
        cards = []

    # ----- tensors (keep on CPU for pin_memory) -----
    tXn_tr = torch.from_numpy(X_num_tr) if X_num_tr is not None else torch.zeros((len(X_cat_tr_ids), 0), dtype=torch.float32)
    tXn_te = torch.from_numpy(X_num_te) if X_num_te is not None else torch.zeros((len(X_cat_te_ids), 0), dtype=torch.float32)
    tXc_tr = torch.from_numpy(X_cat_tr_ids)
    tXc_te = torch.from_numpy(X_cat_te_ids)
    ty_tr  = torch.from_numpy(y_tr.to_numpy(dtype="float32"))

    # ----- loaders -----
    pin = torch.cuda.is_available()
    workers = max(0, int(num_workers))
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
    )
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    ds_tr = TensorDataset(tXn_tr, tXc_tr, ty_tr)
    dl_tr = DataLoader(ds_tr, **loader_kwargs)

    dl_val = None
    if y_val is not None:
        ty_val = torch.from_numpy(y_val.to_numpy(dtype="float32"))
        ds_val = TensorDataset(tXn_te, tXc_te, ty_val)   # валідатор = ваш fold-TE
        val_kwargs = loader_kwargs.copy()
        val_kwargs["shuffle"] = False
        dl_val = DataLoader(ds_val, **val_kwargs)

    # ----- model/opt -----
    model = TabMLP(n_num=tXn_tr.shape[1], cat_cardinalities=cards,
                   emb_drop=emb_drop, hidden=hidden, p=dropout).to(dev)
    print("device:", next(model.parameters()).device, "| CUDA:", torch.cuda.is_available())

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine -> Replace with ReduceLROnPlateau (monitor val RMSE or train loss)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        cooldown=1,
        min_lr=1e-6
    )

    loss_fn = nn.HuberLoss(delta=huber_delta)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_score = float("inf")
    best_state, bad, best_epoch = None, 0, -1

    # ---- training loop ----
    for epoch in range(max_epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        opt.zero_grad(set_to_none=True)
        for xb_num, xb_cat, yb in dl_tr:
            xb_num = xb_num.to(dev, non_blocking=True)
            xb_cat = xb_cat.to(dev, non_blocking=True)
            yb     = yb.to(dev, non_blocking=True)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                pred = model(xb_num, xb_cat)
                loss = loss_fn(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)

            epoch_loss += float(loss.detach().cpu()) * len(yb)

        epoch_time = time.time() - t0
        epoch_loss /= len(ds_tr)

        # ---- validation metrics (if provided) ----
        if dl_val is not None:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb_num, xb_cat, yb in dl_val:
                    xb_num = xb_num.to(dev, non_blocking=True)
                    xb_cat = xb_cat.to(dev, non_blocking=True)
                    yp = model(xb_num, xb_cat).detach().cpu().numpy()
                    preds.append(yp); trues.append(yb.numpy())
            y_pred_val = np.concatenate(preds); y_true_val = np.concatenate(trues)
            val_rmse = _rmse_np(y_true_val, y_pred_val)
            val_mae  = _mae_np(y_true_val, y_pred_val)
            val_r2   = _r2_np(y_true_val, y_pred_val)
            score_for_sched = val_rmse
        else:
            val_rmse = val_mae = val_r2 = None
            score_for_sched = epoch_loss

        # ---- scheduler step (після метрики!) ----
        sched.step(score_for_sched)

        # ---- логи ----
        cur_lr = opt.param_groups[0]["lr"]
        if dl_val is not None:
            print(f"[epoch {epoch+1}] loss={epoch_loss:.5f} | val: RMSE={val_rmse:.3f} "
                  f"MAE={val_mae:.3f} R²={val_r2:.3f} | lr={cur_lr:.2e} | {epoch_time:.1f}s")
        else:
            print(f"[epoch {epoch+1}] loss={epoch_loss:.5f} | lr={cur_lr:.2e} | {epoch_time:.1f}s")

        # ---- early stopping (після логів) ----
        score = (val_rmse if (early_stop_on == "val" and dl_val is not None) else epoch_loss)
        if math.isfinite(score) and (score + 1e-9 < best_score):
            best_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            tag = "val RMSE" if (early_stop_on == "val" and dl_val is not None) else "train loss"
            print(f"[early stop] no improvement for {patience} epochs "
                  f"(best {tag}={best_score:.5f} @ epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- inference on provided TE -----
    model.eval()
    preds, bs = [], 131072
    with torch.no_grad():
        for i in range(0, len(tXn_te), bs):
            xb_num = tXn_te[i:i+bs].to(dev, non_blocking=True)
            xb_cat = tXc_te[i:i+bs].to(dev, non_blocking=True)
            preds.append(model(xb_num, xb_cat).detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)

    return y_pred
