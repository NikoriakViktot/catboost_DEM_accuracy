# -*- coding: utf-8 -*-
# FABDEM error regression with spatial CV (rgt×track)
# Data source: Parquet via DuckDB, consensus features (incl. tan_dem_stream / tan_dem_2000)
# Model: PyTorch MLP with cat embeddings (Huber)

import os, json, duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score

from tab_mlp import train_mlp_and_predict

# ---------------- config ----------------
PARQUET_PATH = "data/NMAD_with_embeddings_cls_features.parquet"  # <-- звідси беремо ДАНІ
SAVE_DIR     = "artifacts_fabdem_mlp"
TARGET_MODE  = "signed"   # "signed" | "absolute"
N_SPLITS     = 5
RANDOM_STATE = 42
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_COLS = {"signed": "delta_fab_dem", "absolute": "abs_delta_fab_dem"}
target_col  = TARGET_COLS[TARGET_MODE]

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_pred[m] - y_true[m]) ** 2)))

def main():
    # ------------- 1) Дані з DuckDB (feature engineering) -------------
    con = duckdb.connect()
    df = con.execute(rf"""
    WITH base AS (
      SELECT
        rgt, track, spot, x, y,
        h_fab_dem, delta_fab_dem, abs(delta_fab_dem) AS abs_delta_fab_dem,
    
        -- heights
        h_alos_dem, h_aster_dem, h_copernicus_dem, h_nasa_dem, h_srtm_dem, h_tan_dem,
    
        -- FAB derivatives
        fab_dem_slope, fab_dem_twi, fab_dem_2000, fab_dem_stream,
    
        -- derivatives for all DEMs (consensus)
        alos_dem_2000, aster_dem_2000, copernicus_dem_2000, nasa_dem_2000, srtm_dem_2000, tan_dem_2000,
        alos_dem_stream, aster_dem_stream, copernicus_dem_stream, nasa_dem_stream, srtm_dem_stream, tan_dem_stream,
        alos_dem_slope,  aster_dem_slope,  copernicus_dem_slope,  nasa_dem_slope,  srtm_dem_slope,  tan_dem_slope,
        alos_dem_twi,    aster_dem_twi,    copernicus_dem_twi,    nasa_dem_twi,    srtm_dem_twi,    tan_dem_twi,
    
        -- terrain features
        aspect_sin, aspect_cos, curvature, roughness, tpi, tri, d8_accum_log1p, fab_breached,
    
        -- categories
        lulc_class, lulc_name, fab_dem_geomorphon, fab_dem_landform,
    
        -- embeddings
        emb_001, emb_002, emb_003, emb_004, emb_005, emb_006, emb_007, emb_008,
        emb_009, emb_010, emb_011, emb_012, emb_013, emb_014, emb_015, emb_016,
        emb_017, emb_018, emb_019, emb_020, emb_021, emb_022, emb_023, emb_024,
        emb_025, emb_026, emb_027, emb_028, emb_029, emb_030, emb_031, emb_032,
        emb_033, emb_034, emb_035, emb_036, emb_037, emb_038, emb_039, emb_040,
        emb_041, emb_042, emb_043, emb_044, emb_045, emb_046, emb_047, emb_048,
        emb_049, emb_050, emb_051, emb_052, emb_053, emb_054, emb_055, emb_056,
        emb_057, emb_058, emb_059, emb_060, emb_061, emb_062, emb_063, emb_064
      FROM read_parquet('{PARQUET_PATH}')
    ),
    
    -- HEIGHT consensus (other DEMs only)
    height_calc AS (
      SELECT
        *,
        list_sort(list_value(h_alos_dem, h_aster_dem, h_copernicus_dem, h_nasa_dem, h_srtm_dem, h_tan_dem)) AS h_others_sorted,
        (h_alos_dem + h_aster_dem + h_copernicus_dem + h_nasa_dem + h_srtm_dem + h_tan_dem) AS h_sum,
        (power(h_alos_dem,2)+power(h_aster_dem,2)+power(h_copernicus_dem,2)+power(h_nasa_dem,2)
         + power(h_srtm_dem,2)+power(h_tan_dem,2)) AS h_sum_sq
      FROM base
    ),
    height_feat AS (
      SELECT
        *,
        h_sum/6.0                                                       AS h_other_mean,
        (h_others_sorted[3] + h_others_sorted[4]) / 2.0                 AS h_other_median,
        sqrt(h_sum_sq/6.0 - power(h_sum/6.0, 2))                        AS h_other_std,
        greatest(h_alos_dem,h_aster_dem,h_copernicus_dem,h_nasa_dem,h_srtm_dem,h_tan_dem)
          - least(h_alos_dem,h_aster_dem,h_copernicus_dem,h_nasa_dem,h_srtm_dem,h_tan_dem) AS h_other_range,
        h_fab_dem - (h_sum/6.0)                                         AS fab_minus_mean,
        h_fab_dem - ((h_others_sorted[3] + h_others_sorted[4]) / 2.0)   AS fab_minus_median,
        (h_fab_dem - (h_sum/6.0)) / nullif(sqrt(h_sum_sq/6.0 - power(h_sum/6.0,2)), 0.0) AS fab_minus_h_z
      FROM height_calc
    ),
    
    -- HAND(2000) consensus
    hand_calc AS (
      SELECT
        *,
        list_sort(list_value(alos_dem_2000, aster_dem_2000, copernicus_dem_2000, nasa_dem_2000, srtm_dem_2000, tan_dem_2000)) AS hand_others_sorted,
        (alos_dem_2000 + aster_dem_2000 + copernicus_dem_2000 + nasa_dem_2000 + srtm_dem_2000 + tan_dem_2000) AS hand_sum,
        (power(alos_dem_2000,2)+power(aster_dem_2000,2)+power(copernicus_dem_2000,2)
         + power(nasa_dem_2000,2)+power(srtm_dem_2000,2)+power(tan_dem_2000,2)) AS hand_sum_sq
      FROM height_feat
    ),
    hand_feat AS (
      SELECT
        *,
        hand_sum/6.0                                                    AS hand_other_mean,
        (hand_others_sorted[3] + hand_others_sorted[4]) / 2.0           AS hand_other_median,
        sqrt(hand_sum_sq/6.0 - power(hand_sum/6.0, 2))                  AS hand_other_std,
        greatest(alos_dem_2000, aster_dem_2000, copernicus_dem_2000, nasa_dem_2000, srtm_dem_2000, tan_dem_2000)
          - least(alos_dem_2000, aster_dem_2000, copernicus_dem_2000, nasa_dem_2000, srtm_dem_2000, tan_dem_2000) AS hand_other_range,
        fab_dem_2000 - (hand_sum/6.0)                                   AS fab_minus_hand_mean,
        fab_dem_2000 - ((hand_others_sorted[3] + hand_others_sorted[4]) / 2.0) AS fab_minus_hand_median,
        (fab_dem_2000 - (hand_sum/6.0)) / nullif(sqrt(hand_sum_sq/6.0 - power(hand_sum/6.0,2)), 0.0) AS fab_minus_hand_z
      FROM hand_calc
    ),
    
    -- STREAM consensus
    stream_calc AS (
      SELECT
        *,
        list_sort(list_value(alos_dem_stream, aster_dem_stream, copernicus_dem_stream, nasa_dem_stream, srtm_dem_stream, tan_dem_stream)) AS stream_others_sorted,
        (alos_dem_stream + aster_dem_stream + copernicus_dem_stream + nasa_dem_stream + srtm_dem_stream + tan_dem_stream) AS stream_sum,
        (power(alos_dem_stream,2)+power(aster_dem_stream,2)+power(copernicus_dem_stream,2)
         + power(nasa_dem_stream,2)+power(srtm_dem_stream,2)+power(tan_dem_stream,2)) AS stream_sum_sq
      FROM hand_feat
    ),
    stream_feat AS (
      SELECT
        *,
        stream_sum/6.0                                                  AS stream_other_mean,
        (stream_others_sorted[3] + stream_others_sorted[4]) / 2.0       AS stream_other_median,
        sqrt(stream_sum_sq/6.0 - power(stream_sum/6.0, 2))              AS stream_other_std,
        greatest(alos_dem_stream, aster_dem_stream, copernicus_dem_stream, nasa_dem_stream, srtm_dem_stream, tan_dem_stream)
          - least(alos_dem_stream, aster_dem_stream, copernicus_dem_stream, nasa_dem_stream, srtm_dem_stream, tan_dem_stream) AS stream_other_range,
        fab_dem_stream - (stream_sum/6.0)                               AS fab_minus_stream_mean,
        fab_dem_stream - ((stream_others_sorted[3] + stream_others_sorted[4]) / 2.0) AS fab_minus_stream_median,
        (fab_dem_stream - (stream_sum/6.0)) / nullif(sqrt(stream_sum_sq/6.0 - power(stream_sum/6.0,2)), 0.0) AS fab_minus_stream_z
      FROM stream_calc
    ),
    
    -- SLOPE consensus
    slope_calc AS (
      SELECT
        *,
        list_sort(list_value(alos_dem_slope, aster_dem_slope, copernicus_dem_slope, nasa_dem_slope, srtm_dem_slope, tan_dem_slope)) AS slope_others_sorted,
        (alos_dem_slope + aster_dem_slope + copernicus_dem_slope + nasa_dem_slope + srtm_dem_slope + tan_dem_slope) AS slope_sum,
        (power(alos_dem_slope,2)+power(aster_dem_slope,2)+power(copernicus_dem_slope,2)
         + power(nasa_dem_slope,2)+power(srtm_dem_slope,2)+power(tan_dem_slope,2)) AS slope_sum_sq
      FROM stream_feat
    ),
    slope_feat AS (
      SELECT
        *,
        slope_sum/6.0                                                   AS slope_other_mean,
        (slope_others_sorted[3] + slope_others_sorted[4]) / 2.0         AS slope_other_median,
        sqrt(slope_sum_sq/6.0 - power(slope_sum/6.0, 2))                AS slope_other_std,
        greatest(alos_dem_slope, aster_dem_slope, copernicus_dem_slope, nasa_dem_slope, srtm_dem_slope, tan_dem_slope)
          - least(alos_dem_slope, aster_dem_slope, copernicus_dem_slope, nasa_dem_slope, srtm_dem_slope, tan_dem_slope) AS slope_other_range,
        fab_dem_slope - (slope_sum/6.0)                                  AS fab_minus_slope_mean,
        fab_dem_slope - ((slope_others_sorted[3] + slope_others_sorted[4]) / 2.0) AS fab_minus_slope_median,
        (fab_dem_slope - (slope_sum/6.0)) / nullif(sqrt(slope_sum_sq/6.0 - power(slope_sum/6.0,2)), 0.0) AS fab_minus_slope_z
      FROM slope_calc
    ),
    
    -- TWI consensus
    twi_calc AS (
      SELECT
        *,
        list_sort(list_value(alos_dem_twi, aster_dem_twi, copernicus_dem_twi, nasa_dem_twi, srtm_dem_twi, tan_dem_twi)) AS twi_others_sorted,
        (alos_dem_twi + aster_dem_twi + copernicus_dem_twi + nasa_dem_twi + srtm_dem_twi + tan_dem_twi) AS twi_sum,
        (power(alos_dem_twi,2)+power(aster_dem_twi,2)+power(copernicus_dem_twi,2)
         + power(nasa_dem_twi,2)+power(srtm_dem_twi,2)+power(tan_dem_twi,2)) AS twi_sum_sq
      FROM slope_feat
    ),
    twi_feat AS (
      SELECT
        *,
        twi_sum/6.0                                                     AS twi_other_mean,
        (twi_others_sorted[3] + twi_others_sorted[4]) / 2.0             AS twi_other_median,
        sqrt(twi_sum_sq/6.0 - power(twi_sum/6.0, 2))                    AS twi_other_std,
        greatest(alos_dem_twi, aster_dem_twi, copernicus_dem_twi, nasa_dem_twi, srtm_dem_twi, tan_dem_twi)
          - least(alos_dem_twi, aster_dem_twi, copernicus_dem_twi, nasa_dem_twi, srtm_dem_twi, tan_dem_twi) AS twi_other_range,
        fab_dem_twi - (twi_sum/6.0)                                     AS fab_minus_twi_mean,
        fab_dem_twi - ((twi_others_sorted[3] + twi_others_sorted[4]) / 2.0) AS fab_minus_twi_median,
        (fab_dem_twi - (twi_sum/6.0)) / nullif(sqrt(twi_sum_sq/6.0 - power(twi_sum/6.0,2)), 0.0) AS fab_minus_twi_z
      FROM twi_calc
    )
    
    SELECT * FROM twi_feat;
    """).fetchdf()
    con.close()

    # ------------- 2) Target & groups -------------
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found.")
    y = df[target_col].astype("float32")
    print(f"[INFO] TARGET_MODE={TARGET_MODE} -> y = '{target_col}'")

    df["group_id"] = df["rgt"].astype(int).astype(str) + "_" + df["track"].astype(int).astype(str)

    # ------------- 3) Feature lists -------------
    base_num = [
        "h_fab_dem","fab_dem_slope","fab_dem_twi","fab_dem_2000","fab_dem_stream",
        "aspect_sin","aspect_cos","curvature","roughness","tpi","tri","d8_accum_log1p",
        "fab_breached"
    ]
    consistency_cols = [c for c in [
        "h_other_mean","h_other_median","h_other_std","h_other_range","fab_minus_mean","fab_minus_median","fab_minus_h_z",
        "hand_other_mean","hand_other_median","hand_other_std","hand_other_range","fab_minus_hand_mean","fab_minus_hand_median","fab_minus_hand_z",
        "stream_other_mean","stream_other_median","stream_other_std","stream_other_range","fab_minus_stream_mean","fab_minus_stream_median","fab_minus_stream_z",
        "slope_other_mean","slope_other_median","slope_other_std","slope_other_range","fab_minus_slope_mean","fab_minus_slope_median","fab_minus_slope_z",
        "twi_other_mean","twi_other_median","twi_other_std","twi_other_range","fab_minus_twi_mean","fab_minus_twi_median","fab_minus_twi_z",
    ] if c in df.columns]

    # FAB - each other DEM (heights)
    for c in [col for col in df.columns if col.startswith("h_") and col.endswith("_dem") and col != "h_fab_dem"]:
        name = f"fab_minus_{c.replace('h_','').replace('_dem','')}"
        if name not in df.columns:
            df[name] = df["h_fab_dem"] - df[c]
    per_dem_diff_cols = [c for c in df.columns if c.startswith("fab_minus_") and c not in consistency_cols]

    # engineered numerics
    df["accum_exp"]   = np.expm1(df["d8_accum_log1p"])
    df["accum_sqrt"]  = np.sqrt(df["accum_exp"])
    df["slope_sq"]    = df["fab_dem_slope"]**2
    df["slope_sqrt"]  = np.sqrt(np.clip(df["fab_dem_slope"], 0, None))
    df["twi_sq"]      = df["fab_dem_twi"]**2
    df["twi_sqrt"]    = np.sqrt(np.clip(df["fab_dem_twi"], 0, None))
    df["slope_x_twi"] = df["fab_dem_slope"] * df["fab_dem_twi"]
    df["slope_x_stream"]= df["fab_dem_slope"] * df["fab_dem_stream"]
    df["twi_x_stream"]= df["fab_dem_twi"]   * df["fab_dem_stream"]
    df["slope_x_accum"]= df["fab_dem_slope"]* df["accum_exp"]
    df["twi_x_accum"] = df["fab_dem_twi"]  * df["accum_exp"]
    df["curv_abs"]    = df["curvature"].abs()
    df["rough_x_tri"] = df["roughness"] * df["tri"]
    df["tpi_x_tri"]   = df["tpi"] * df["tri"]
    df["planar_idx"]  = (df["curvature"].abs() < 0.005).astype("int8")
    df["steep_idx"]   = (df["fab_dem_slope"]  > 20).astype("int8")

    # robust bins
    q  = df["fab_dem_2000"].quantile([0.1,0.3,0.7,0.9]).values
    df["level_bin"] = pd.cut(df["fab_dem_2000"], [-np.inf, q[0], q[1], q[2], q[3], np.inf],
                             labels=["lev_very_low","lev_low","lev_mid","lev_high","lev_very_high"])
    qt = df["fab_dem_twi"].quantile([0.2,0.4,0.6,0.8]).values
    df["twi_bin"] = pd.cut(df["fab_dem_twi"], [-np.inf, qt[0], qt[1], qt[2], qt[3], np.inf],
                           labels=["twi_vlow","twi_low","twi_mid","twi_high","twi_vhigh"])
    qa = df["accum_exp"].replace([np.inf,-np.inf], np.nan).fillna(0).quantile([0.2,0.4,0.6,0.8]).values
    df["accum_bin"] = pd.cut(df["accum_exp"], [-np.inf, qa[0], qa[1], qa[2], qa[3], np.inf],
                             labels=["acc_vlow","acc_low","acc_mid","acc_high","acc_vhigh"])

    # aspect quadrants
    df["aspect_quad_NE"] = ((df["aspect_cos"]>0) & (df["aspect_sin"]>0)).astype("int8")
    df["aspect_quad_NW"] = ((df["aspect_cos"]<0) & (df["aspect_sin"]>0)).astype("int8")
    df["aspect_quad_SE"] = ((df["aspect_cos"]>0) & (df["aspect_sin"]<0)).astype("int8")
    df["aspect_quad_SW"] = ((df["aspect_cos"]<0) & (df["aspect_sin"]<0)).astype("int8")

    added_num = [
        "slope_sq","slope_sqrt","twi_sq","twi_sqrt","accum_exp","accum_sqrt",
        "slope_x_twi","slope_x_stream","twi_x_stream","slope_x_accum","twi_x_accum",
        "curv_abs","rough_x_tri","tpi_x_tri","planar_idx","steep_idx",
        "aspect_quad_NE","aspect_quad_NW","aspect_quad_SE","aspect_quad_SW"
    ]
    num_cols = base_num + added_num + consistency_cols + per_dem_diff_cols

    cat_cols = [c for c in [
        "lulc_class","lulc_name","fab_dem_geomorphon","fab_dem_landform",
        "level_bin","twi_bin","accum_bin"
    ] if c in df.columns]

    emb_cols = [f"emb_{i:03d}" for i in range(1,65) if f"emb_{i:03d}" in df.columns]
    # додаємо ембединги як числові фічі (MLP ок з ними прямо)
    num_cols += emb_cols
    num_cols = list(dict.fromkeys(num_cols))  # унікалізація з збереженням порядку

    # rare-category collapse
    def collapse_rare(s: pd.Series, min_frac=0.01) -> pd.Series:
        vc = s.value_counts(normalize=True, dropna=False)
        rare = vc[vc < min_frac].index
        s = s.astype(str)
        return s.where(~s.isin(rare), "__OTHER__")
    for c in cat_cols:
        df[c] = collapse_rare(df[c], 0.01)

    # float32
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # ------------- 4) StratifiedGroupKFold -------------
    y_clean = y.replace([np.inf,-np.inf], np.nan)
    groups  = df["group_id"].astype(str)
    mask    = y_clean.notna() & groups.notna()

    df_cv = df.loc[mask].reset_index(drop=True)
    y_cv  = y_clean.loc[mask].reset_index(drop=True)
    grp   = groups.loc[mask].reset_index(drop=True)

    try:
        y_bins = pd.qcut(y_cv.abs(), q=10, labels=False, duplicates="drop").astype(int)
    except ValueError:
        y_bins = pd.qcut(y_cv.abs(), q=5, labels=False, duplicates="drop").astype(int)

    sgkf  = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(sgkf.split(df_cv.index, y_bins, groups=grp))

    # ------------- 5) CV train (MLP) -------------
    all_metrics = []
    test_preds_all = []

    for k, (tr_idx, te_idx) in enumerate(folds, 1):
        print(f"\n===== FOLD {k} =====")
        Xn_tr = df_cv.loc[tr_idx, num_cols]
        Xn_te = df_cv.loc[te_idx, num_cols]
        Xc_tr = df_cv.loc[tr_idx, cat_cols].astype(str) if len(cat_cols) else pd.DataFrame(index=tr_idx)
        Xc_te = df_cv.loc[te_idx, cat_cols].astype(str) if len(cat_cols) else pd.DataFrame(index=te_idx)

        y_hat = train_mlp_and_predict(
            Xn_tr=Xn_tr, Xn_te=Xn_te,
            Xc_tr=Xc_tr, Xc_te=Xc_te,
            y_tr=y_cv.iloc[tr_idx],
            y_val=y_cv.iloc[te_idx],  # валідація = te_idx
            max_epochs=60,
            batch_size=256_000,
            lr=3e-3, weight_decay=1e-5,
            patience=12,
            hidden=(1024, 512, 256),
            dropout=0.15, emb_drop=0.05,
            huber_delta=1.0,
            num_workers=6,
            early_stop_on="val"
        )

        # --- метрики моделі на цьому фолді ---
        y_true = y_cv.iloc[te_idx].to_numpy()
        m = np.isfinite(y_true) & np.isfinite(y_hat)
        fold_rmse = rmse(y_true, y_hat)  # твоя функція вже маскує
        fold_mae = float(np.mean(np.abs(y_hat[m] - y_true[m])))
        fold_r2 = float(r2_score(y_true[m], y_hat[m]))
        print(f"FOLD {k}: RMSE={fold_rmse:.3f}  MAE={fold_mae:.3f}  R²={fold_r2:.3f}")

        # --- >>> САМЕ ТУТ: бейслайн FAB на цьому TE-фолді + приріст ---
        # Бейслайн: "залишаємо FAB як є" = передбачаємо нульову поправку, тобто помилка = y_true
        eps = 1e-12
        base_rmse_fold = float(np.sqrt(np.mean(np.square(y_true[m]))))
        base_mae_fold = float(np.mean(np.abs(y_true[m])))
        imp_rmse = 100.0 * (base_rmse_fold - fold_rmse) / max(base_rmse_fold, eps)
        imp_mae = 100.0 * (base_mae_fold - fold_mae) / max(base_mae_fold, eps)
        bias = float(np.mean(y_hat[m] - y_true[m]))

        print(f"FAB baseline (fold {k}): RMSE={base_rmse_fold:.3f} MAE={base_mae_fold:.3f}")
        print(f"Improvement vs FAB: RMSE {imp_rmse:.1f}% | MAE {imp_mae:.1f}% | bias={bias:.3f}")

        # --- зберігаємо в метрики ---
        all_metrics.append({
            "fold": k,
            "rmse": fold_rmse,
            "mae": fold_mae,
            "r2": fold_r2,
            "base_rmse": base_rmse_fold,
            "base_mae": base_mae_fold,
            "imp_rmse_pct": imp_rmse,
            "imp_mae_pct": imp_mae,
            "bias": bias,
        })

        # --- збереження предиктів фолда (як було) ---
        out = df_cv.loc[te_idx, ["x", "y", "rgt", "track", "spot"]].copy()
        out["y_true"] = y_true
        out["y_pred"] = y_hat
        out["abs_error"] = (out["y_pred"] - out["y_true"]).abs()
        out["fold"] = k
        test_preds_all.append(out)

    # ------------- 6) Artifacts -------------
    metrics_df = pd.DataFrame(all_metrics)
    print("\n===== CV SUMMARY =====")
    print("RMSE:", metrics_df["rmse"].mean().round(3), "±", metrics_df["rmse"].std().round(3))
    print("MAE :", metrics_df["mae"].mean().round(3), "±", metrics_df["mae"].std().round(3))
    print("R²  :", metrics_df["r2"].mean().round(3), "±", metrics_df["r2"].std().round(3))

    # усереднений FAB-бейслайн та приріст у %
    if {"base_rmse", "base_mae"}.issubset(metrics_df.columns):
        base_rmse_mean = metrics_df["base_rmse"].mean()
        base_mae_mean = metrics_df["base_mae"].mean()
        model_rmse_mean = metrics_df["rmse"].mean()
        model_mae_mean = metrics_df["mae"].mean()

        eps = 1e-12
        imp_rmse_mean = 100.0 * (base_rmse_mean - model_rmse_mean) / max(base_rmse_mean, eps)
        imp_mae_mean = 100.0 * (base_mae_mean - model_mae_mean) / max(base_mae_mean, eps)

        print(f"FAB baseline (mean over folds): RMSE={base_rmse_mean:.3f} MAE={base_mae_mean:.3f}")
        print(f"Improvement vs FAB (mean): RMSE {imp_rmse_mean:.1f}% | MAE {imp_mae_mean:.1f}%")

    test_preds_df = pd.concat(test_preds_all, axis=0).reset_index(drop=True)
    test_preds_df.to_parquet(os.path.join(SAVE_DIR, "cv_mlp_test_predictions.parquet"), index=False)
    metrics_df.to_csv(os.path.join(SAVE_DIR, "cv_mlp_metrics.csv"), index=False)

    manifest = {
        "target_mode": TARGET_MODE,
        "target_column": target_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "group_field": "group_id",
        "splits": N_SPLITS,
        "data_source": PARQUET_PATH,
        "model": "TabMLP (Huber)"
    }
    with open(os.path.join(SAVE_DIR, "columns_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(" - CV metrics  ->", os.path.join(SAVE_DIR, "cv_mlp_metrics.csv"))
    print(" - Test preds  ->", os.path.join(SAVE_DIR, "cv_mlp_test_predictions.parquet"))
    print(" - Manifest    ->", os.path.join(SAVE_DIR, "columns_manifest.json"))


if __name__ == "__main__":
    main()