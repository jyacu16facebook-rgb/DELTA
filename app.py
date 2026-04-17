from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import statsmodels.api as sm
from scipy.spatial.distance import mahalanobis
from scipy.stats import levene
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Dashboard — Delta Peso Baya",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_FILE = "DELTA_CONSOLIDADO 2022-2025.xlsx"
DEFAULT_SHEET = "DATA"
MAX_LAG = 15
MIN_OBS_BP = 5

CURRENT_TO_INTERNAL = {
    "año": "AÑO", "campaña": "CAMPAÑA", "semana": "SEMANA",
    "semana fenologica": "SEMANA FENOLOGICA", "fundo": "FUNDO", "etapa": "ETAPA",
    "campo": "CAMPO", "turno": "TURNO", "variedad": "VARIEDAD",
    "conteo_flores": "FLORES", "conteo_fruto_cuajado": "FRUTO CUAJADO",
    "conteo_fruto_verde": "FRUTO VERDE", "conteo_total_frutos": "TOTAL DE FRUTOS",
    "conteo_bayas_cremosas": "FRUTO CREMOSO", "conteo_bayas_rosadas": "FRUTO ROSADO",
    "conteo_bayas_maduras": "FRUTO MADURO", "peso_promedio_baya_g": "PESO BAYA (g)",
}

INTERNAL_BASE_TO_CURRENT_LAG_PREFIX = {
    "FLORES": "conteo_flores",
    "FRUTO CUAJADO": "conteo_fruto_cuajado",
    "FRUTO VERDE": "conteo_fruto_verde",
    "TOTAL DE FRUTOS": "conteo_total_frutos",
    "FRUTO CREMOSO": "conteo_bayas_cremosas",
    "FRUTO ROSADO": "conteo_bayas_rosadas",
    "FRUTO MADURO": "conteo_bayas_maduras",
}

ID_COLS_DISPLAY = ["AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]
ENTITY_COLS = ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]
TARGET_COL = "PESO BAYA (g)"
PHENOLOGY_COLS = [
    "FLORES", "FRUTO CUAJADO", "FRUTO VERDE", "TOTAL DE FRUTOS",
    "FRUTO CREMOSO", "FRUTO ROSADO", "FRUTO MADURO",
]
ALL_ANALYSIS_BASE_COLS = ID_COLS_DISPLAY + PHENOLOGY_COLS + [TARGET_COL, "SEMANA FENOLOGICA"]
ALL_X_TRANSFORM_OPTIONS = [
    "Original",
    "log(X+1)",
    "log(X+1)*log(SF+1)",
    "X*SF",
    "sqrt(X)",
    "sqrt(X)/sqrt(SF)",
    "log(X+1)^2 * log(SF+1)",
    "log(X+1)/sqrt(SF)",
]
RANKING_EMPTY_COLS = [
    "ranking", "transformacion", "n", "pearson", "pearson_abs", "direccion",
    "reset_f_stat", "reset_f_pval", "lineal", "bp_f_stat", "bp_f_pval", "homocedastico",
]


# =========================================================
# HELPERS
# =========================================================
def find_excel_file(default_file: str) -> Path:
    """
    Búsqueda robusta para Streamlit local + GitHub/Streamlit Cloud.
    Prioridad:
    1) carpeta del app
    2) ./data
    3) working dir actual
    4) primer .xlsx encontrado en esas rutas
    """
    candidates = [
        APP_DIR / default_file,
        APP_DIR / "data" / default_file,
        Path.cwd() / default_file,
        Path.cwd() / "data" / default_file,
    ]
    for p in candidates:
        if p.exists():
            return p

    search_dirs = [APP_DIR, APP_DIR / "data", Path.cwd(), Path.cwd() / "data"]
    for d in search_dirs:
        if d.exists():
            files = sorted(d.glob("*.xlsx"))
            if files:
                return files[0]

    raise FileNotFoundError(
        f"No se encontró '{default_file}'. "
        f"Colócalo junto al app.py o dentro de una carpeta 'data'."
    )


def load_excel_source(default_file: str, uploaded_file=None) -> tuple[pd.DataFrame, str]:
    """
    Mantiene la lógica del dashboard y solo adecua la carga para Streamlit + Git.
    - Si el usuario sube archivo, usa ese.
    - Si no, busca el Excel dentro del repo / app folder.
    """
    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        df = pd.read_excel(BytesIO(raw), sheet_name=DEFAULT_SHEET, engine="openpyxl")
        return df, f"upload::{uploaded_file.name}"

    fp = find_excel_file(default_file)
    df = pd.read_excel(fp, sheet_name=DEFAULT_SHEET, engine="openpyxl")
    return df, str(fp)


def safe_iso_week_start(year, week):
    try:
        return pd.Timestamp(datetime.fromisocalendar(int(year), int(week), 1))
    except Exception:
        return pd.NaT


def validate_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(missing))


def to_numeric_safe(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_text_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    return df


def build_entity_key(df):
    return (
        df["CAMPAÑA"].astype("string").fillna("") + " | "
        + df["FUNDO"].astype("string").fillna("") + " | "
        + df["ETAPA"].astype("string").fillna("") + " | "
        + df["CAMPO"].astype("string").fillna("") + " | "
        + df["TURNO"].astype("string").fillna("") + " | "
        + df["VARIEDAD"].astype("string").fillna("")
    )


def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def get_series(df, col):
    obj = df[col]
    return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj


def standardize_current_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    validate_columns(df, list(CURRENT_TO_INTERNAL.keys()))
    df = df.rename(columns={k: v for k, v in CURRENT_TO_INTERNAL.items() if k in df.columns})

    for iv, cp in INTERNAL_BASE_TO_CURRENT_LAG_PREFIX.items():
        for lag in range(1, MAX_LAG + 1):
            src = f"{cp}_semana_{lag}_anterior"
            dst = f"{iv}__LAG_{lag}"
            df[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    return df


def smooth_series(series):
    return series.rolling(window=3, center=True, min_periods=1).mean()


def add_delta_only(df):
    df = df.copy()
    validate_columns(df, ["AÑO", "SEMANA", TARGET_COL, *ENTITY_COLS])
    df = df.sort_values(ENTITY_COLS + ["AÑO", "SEMANA"], kind="stable").reset_index(drop=True)
    df["WEEK_START_DATE"] = [safe_iso_week_start(y, w) for y, w in zip(df["AÑO"], df["SEMANA"])]
    df["SEMANA ACTUAL"] = df["SEMANA"]
    df["AÑO-SEMANA ACTUAL"] = (
        df["AÑO"].astype("Int64").astype(str) + "-W" +
        df["SEMANA"].astype("Int64").astype(str).str.zfill(2)
    )

    g = df.groupby(ENTITY_COLS, dropna=False, sort=False)
    df["PESO_BAYA_SUAVIZADO"] = g[TARGET_COL].transform(smooth_series)
    df[TARGET_COL] = df["PESO_BAYA_SUAVIZADO"]

    prev_weight = g[TARGET_COL].shift(1)
    prev_week_date = g["WEEK_START_DATE"].shift(1)

    consecutive_1 = (
        df["WEEK_START_DATE"].notna() &
        prev_week_date.notna() &
        ((df["WEEK_START_DATE"] - prev_week_date) == pd.Timedelta(days=7))
    )

    df["DELTA_BW"] = np.where(
        consecutive_1 & df[TARGET_COL].notna() & prev_weight.notna(),
        df[TARGET_COL] - prev_weight,
        np.nan,
    )

    df["DELTA_BW_%"] = np.where(
        consecutive_1 & df[TARGET_COL].notna() & prev_weight.notna() & (prev_weight > 0),
        (df[TARGET_COL] - prev_weight) / prev_weight,
        np.nan,
    )

    df["PESO_BAYA_SUAVIZADO__LAG_1"] = g["PESO_BAYA_SUAVIZADO"].shift(1)
    df["ENTITY_KEY"] = build_entity_key(df)
    df["SERIES_KEY"] = df["ENTITY_KEY"]
    return df


def mahalanobis_filter(df, x_col, y_col, threshold=3.0):
    x_ser = get_series(df, x_col)
    y_ser = get_series(df, y_col)
    sub = pd.DataFrame({x_col: x_ser, y_col: y_ser}).dropna().copy()

    if len(sub) < 5:
        return pd.Series(True, index=df.index)

    X = sub[[x_col, y_col]].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except Exception:
        return pd.Series(True, index=df.index)

    distances = np.array([mahalanobis(row, mean, inv_cov) for row in X])
    keep_mask = distances < threshold
    valid_index = sub.index[keep_mask]
    return df.index.isin(valid_index)


def transform_x_series(x, semana_fenologica, transform_mode):
    x_num = pd.to_numeric(x, errors="coerce")
    semana_num = pd.to_numeric(semana_fenologica, errors="coerce")

    if transform_mode == "Original":
        return x_num
    if transform_mode == "log(X+1)":
        return np.log1p(x_num.where(x_num >= 0, np.nan))
    if transform_mode == "log(X+1)*log(SF+1)":
        return np.log1p(x_num.where(x_num >= 0, np.nan)) * np.log1p(semana_num.where(semana_num >= 0, np.nan))
    if transform_mode == "X*SF":
        return x_num * semana_num
    if transform_mode == "sqrt(X)":
        return np.sqrt(x_num.where(x_num >= 0, np.nan))
    if transform_mode == "sqrt(X)/sqrt(SF)":
        return np.sqrt(x_num.where(x_num >= 0, np.nan)) / np.sqrt(semana_num.where(semana_num > 0, np.nan))
    if transform_mode == "log(X+1)^2 * log(SF+1)":
        return (np.log1p(x_num.where(x_num >= 0, np.nan)) ** 2) * np.log1p(semana_num.where(semana_num >= 0, np.nan))
    if transform_mode == "log(X+1)/sqrt(SF)":
        return np.log1p(x_num.where(x_num >= 0, np.nan)) / np.sqrt(semana_num.where(semana_num > 0, np.nan))
    return x_num


def get_x_axis_label(base_label, transform_mode):
    if transform_mode == "Original":
        return base_label
    if transform_mode == "log(X+1)":
        return f"log({base_label}+1)"
    if transform_mode == "log(X+1)*log(SF+1)":
        return f"log({base_label}+1)*log(SF+1)"
    if transform_mode == "X*SF":
        return f"{base_label}*SF"
    if transform_mode == "sqrt(X)":
        return f"sqrt({base_label})"
    if transform_mode == "sqrt(X)/sqrt(SF)":
        return f"sqrt({base_label})/sqrt(SF)"
    if transform_mode == "log(X+1)^2 * log(SF+1)":
        return f"log({base_label}+1)^2 * log(SF+1)"
    if transform_mode == "log(X+1)/sqrt(SF)":
        return f"log({base_label}+1)/sqrt(SF)"
    return base_label


def _ols_metrics(sub):
    out = {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan, "aic": np.nan, "bic": np.nan}
    try:
        Xols = sm.add_constant(sub["X"].values, has_constant="add")
        fit = sm.OLS(sub["Y"].values, Xols).fit()
        pred = fit.predict(Xols)
        err = sub["Y"].values - pred
        out["r2"] = float(fit.rsquared)
        out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
        out["mae"] = float(np.mean(np.abs(err)))
        y_true = sub["Y"].values
        good = np.abs(y_true) > 1e-12
        out["mape"] = float(np.mean(np.abs(err[good] / y_true[good])) * 100) if good.any() else np.nan
        out["aic"] = float(fit.aic)
        out["bic"] = float(fit.bic)
    except Exception:
        pass
    return out


def _levene_by_quantiles(x, y):
    try:
        sub = pd.DataFrame({"X": x, "Y": y}).dropna()
        if len(sub) < 8 or sub["X"].nunique() < 4:
            return np.nan
        sub["g"] = pd.qcut(sub["X"], q=4, duplicates="drop")
        groups = [g["Y"].values for _, g in sub.groupby("g") if len(g) >= 2]
        if len(groups) < 2:
            return np.nan
        return float(levene(*groups).pvalue)
    except Exception:
        return np.nan


def get_x_col(base_var, lag):
    return base_var if lag == 0 else f"{base_var}__LAG_{lag}"


def add_model(fig, x, y, model_type):
    xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
    valid = xn.notna() & yn.notna()
    if valid.sum() < 5 or len(np.unique(xn[valid].values)) < 2:
        return fig

    xv, yv = xn[valid].values, yn[valid].values
    xl = np.linspace(xv.min(), xv.max(), 200)

    try:
        if model_type == "Lineal":
            c = np.polyfit(xv, yv, 1)
            yl = c[0] * xl + c[1]
            nm = "Modelo lineal"
        elif model_type == "Polinomial (grado 2)":
            c = np.polyfit(xv, yv, 2)
            yl = c[0] * xl**2 + c[1] * xl + c[2]
            nm = "Polinomial g2"
        elif model_type == "Spline":
            from scipy.interpolate import UnivariateSpline
            idx = np.argsort(xv)
            xs, ys = xv[idx], yv[idx]
            try:
                poly_r = ys - np.polyval(np.polyfit(xs, ys, min(3, len(xs)-2)), xs)
                s = max(len(xs) * float(np.var(poly_r)), 1e-10)
            except Exception:
                s = len(xs)
            spl = UnivariateSpline(xs, ys, k=3, s=s)
            yl = spl(xl)
            nm = "Spline"
        else:
            return fig

        fig.add_trace(go.Scatter(x=xl, y=yl, mode="lines", name=nm))
    except Exception:
        pass
    return fig


def build_transform_ranking(df, x_col, y_col, semana_col="SEMANA FENOLOGICA", apply_outlier=False):
    if x_col not in df.columns or y_col not in df.columns or semana_col not in df.columns:
        return pd.DataFrame(columns=RANKING_EMPTY_COLS)

    x_base = get_series(df, x_col)
    semana = get_series(df, semana_col)
    y = pd.to_numeric(get_series(df, y_col), errors="coerce")
    rows = []

    for mode in ALL_X_TRANSFORM_OPTIONS:
        xt = transform_x_series(x=x_base, semana_fenologica=semana, transform_mode=mode)
        sub = pd.DataFrame({"X": pd.to_numeric(xt, errors="coerce"), "Y": y}).dropna()

        if apply_outlier and len(sub) >= 5:
            sub = sub[mahalanobis_filter(sub, "X", "Y")]

        n = len(sub)
        if n < 2 or sub["X"].nunique() < 2 or sub["Y"].nunique() < 2:
            pearson = np.nan
        else:
            try:
                pearson = sub["X"].corr(sub["Y"], method="pearson")
            except Exception:
                pearson = np.nan

        reset_f_stat = reset_f_pval = lineal = bp_f_stat = bp_f_pval = homocedastico = np.nan
        if n >= MIN_OBS_BP and pd.notna(pearson):
            try:
                Xols = sm.add_constant(sub["X"].values, has_constant="add")
                fit = sm.OLS(sub["Y"].values, Xols).fit()
                try:
                    rr = linear_reset(fit, power=2, use_f=True)
                    reset_f_stat, reset_f_pval = float(rr.fvalue), float(rr.pvalue)
                    lineal = bool(reset_f_pval > 0.05)
                except Exception:
                    pass
                try:
                    _, _, bf, bp = het_breuschpagan(fit.resid, fit.model.exog)
                    bp_f_stat, bp_f_pval = float(bf), float(bp)
                    homocedastico = bool(bp_f_pval > 0.05)
                except Exception:
                    pass
            except Exception:
                pass

        rows.append({
            "transformacion": mode,
            "n": n,
            "pearson": pearson,
            "pearson_abs": abs(pearson) if pd.notna(pearson) else np.nan,
            "direccion": ("Positiva" if pd.notna(pearson) and pearson > 0 else
                          "Negativa" if pd.notna(pearson) and pearson < 0 else "Nula/NA"),
            "reset_f_stat": reset_f_stat,
            "reset_f_pval": reset_f_pval,
            "lineal": lineal,
            "bp_f_stat": bp_f_stat,
            "bp_f_pval": bp_f_pval,
            "homocedastico": homocedastico,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=RANKING_EMPTY_COLS)

    out = out.sort_values(["pearson_abs", "n"], ascending=[False, False], na_position="last").reset_index(drop=True)
    out.insert(0, "ranking", np.arange(1, len(out) + 1))
    return out


def build_global_ranking(df, target, semana_col="SEMANA FENOLOGICA", sort_metric="pearson_abs",
                         sort_ascending=False, apply_outlier=False, vars_filter=None, lags_filter=None):
    if target not in df.columns or semana_col not in df.columns:
        return pd.DataFrame()

    y = pd.to_numeric(df[target], errors="coerce")
    sf = df[semana_col]
    rows = []
    vars_to_use = vars_filter if vars_filter else PHENOLOGY_COLS
    lags_to_use = lags_filter if lags_filter is not None else list(range(MAX_LAG + 1))

    for base_var in vars_to_use:
        for lag in lags_to_use:
            xcol = get_x_col(base_var, lag)
            if xcol not in df.columns:
                continue
            xb = get_series(df, xcol)

            for mode in ALL_X_TRANSFORM_OPTIONS:
                xt = transform_x_series(x=xb, semana_fenologica=sf, transform_mode=mode)
                sub = pd.DataFrame({"X": pd.to_numeric(xt, errors="coerce"), "Y": y}).dropna()

                if apply_outlier and len(sub) >= 5:
                    sub = sub[mahalanobis_filter(sub, "X", "Y")]

                n = len(sub)
                pearson = spearman = reset_f_pval = bp_f_pval = lineal = homocedastico = levene_pval = homocedastico_levene = np.nan
                r2 = rmse = mae = mape = aic = bic = np.nan

                if n >= 2 and sub["X"].nunique() >= 2 and sub["Y"].nunique() >= 2:
                    try:
                        pearson = sub["X"].corr(sub["Y"], method="pearson")
                    except Exception:
                        pass
                    try:
                        spearman = sub["X"].corr(sub["Y"], method="spearman")
                    except Exception:
                        pass

                    levene_pval = _levene_by_quantiles(sub["X"], sub["Y"])
                    homocedastico_levene = bool(pd.notna(levene_pval) and levene_pval > 0.05)

                    om = _ols_metrics(sub)
                    r2, rmse, mae, mape, aic, bic = om["r2"], om["rmse"], om["mae"], om["mape"], om["aic"], om["bic"]

                    if n >= MIN_OBS_BP and pd.notna(pearson):
                        try:
                            Xols = sm.add_constant(sub["X"].values, has_constant="add")
                            fit = sm.OLS(sub["Y"].values, Xols).fit()
                            try:
                                rr = linear_reset(fit, power=2, use_f=True)
                                reset_f_pval = float(rr.pvalue)
                                lineal = bool(reset_f_pval > 0.05)
                            except Exception:
                                pass
                            try:
                                _, _, _, bp = het_breuschpagan(fit.resid, fit.model.exog)
                                bp_f_pval = float(bp)
                                homocedastico = bool(bp_f_pval > 0.05)
                            except Exception:
                                pass
                        except Exception:
                            pass

                rows.append({
                    "variable": base_var,
                    "lag": lag,
                    "transformacion": mode,
                    "n": n,
                    "pearson": pearson,
                    "pearson_abs": abs(pearson) if pd.notna(pearson) else np.nan,
                    "spearman": spearman,
                    "spearman_abs": abs(spearman) if pd.notna(spearman) else np.nan,
                    "direccion": ("Positiva" if pd.notna(pearson) and pearson > 0 else
                                  "Negativa" if pd.notna(pearson) and pearson < 0 else "Nula/NA"),
                    "reset_f_pval": reset_f_pval,
                    "bp_f_pval": bp_f_pval,
                    "levene_pval": levene_pval,
                    "lineal": lineal,
                    "homocedastico": homocedastico,
                    "homocedastico_levene": homocedastico_levene,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "aic": aic,
                    "bic": bic,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by=[sort_metric, "n"], ascending=[sort_ascending, False], na_position="last").reset_index(drop=True)


def compute_matrix(filtered_df, base_var, analysis_target, apply_outlier=False):
    lags = list(range(MAX_LAG + 1))
    mks = ["pearson", "spearman", "reset_f_pval", "bp_f_pval", "levene_pval", "r2", "rmse", "mae", "mape", "aic", "bic", "n"]
    results = {k: pd.DataFrame(np.nan, index=lags, columns=ALL_X_TRANSFORM_OPTIONS) for k in mks}
    sf = filtered_df["SEMANA FENOLOGICA"] if "SEMANA FENOLOGICA" in filtered_df.columns else pd.Series(dtype=float, index=filtered_df.index)
    y = pd.to_numeric(filtered_df[analysis_target], errors="coerce") if analysis_target in filtered_df.columns else pd.Series(dtype=float, index=filtered_df.index)

    for lag in lags:
        xcol = get_x_col(base_var, lag)
        if xcol not in filtered_df.columns:
            continue

        xb = get_series(filtered_df, xcol)
        for tr in ALL_X_TRANSFORM_OPTIONS:
            xt = transform_x_series(x=xb, semana_fenologica=sf, transform_mode=tr)
            sub = pd.DataFrame({"X": pd.to_numeric(xt, errors="coerce"), "Y": y}).dropna()

            if apply_outlier and len(sub) >= 5:
                sub = sub[mahalanobis_filter(sub, "X", "Y")]

            n = len(sub)
            results["n"].at[lag, tr] = n
            if n < 2 or sub["X"].nunique() < 2 or sub["Y"].nunique() < 2:
                continue

            try:
                results["pearson"].at[lag, tr] = sub["X"].corr(sub["Y"], method="pearson")
            except Exception:
                pass
            try:
                results["spearman"].at[lag, tr] = sub["X"].corr(sub["Y"], method="spearman")
            except Exception:
                pass

            results["levene_pval"].at[lag, tr] = _levene_by_quantiles(sub["X"], sub["Y"])
            om = _ols_metrics(sub)
            for mk in ["r2", "rmse", "mae", "mape", "aic", "bic"]:
                results[mk].at[lag, tr] = om[mk]

            if n >= MIN_OBS_BP:
                try:
                    Xols = sm.add_constant(sub["X"].values, has_constant="add")
                    fit = sm.OLS(sub["Y"].values, Xols).fit()
                    try:
                        rr = linear_reset(fit, power=2, use_f=True)
                        results["reset_f_pval"].at[lag, tr] = float(rr.pvalue)
                    except Exception:
                        pass
                    try:
                        _, _, _, bp = het_breuschpagan(fit.resid, fit.model.exog)
                        results["bp_f_pval"].at[lag, tr] = float(bp)
                    except Exception:
                        pass
                except Exception:
                    pass
    return results


@st.cache_data(show_spinner=True)
def load_and_prepare_data(uploaded_name: str | None = None, uploaded_bytes: bytes | None = None):
    uploaded = None
    if uploaded_name is not None and uploaded_bytes is not None:
        class _Uploaded:
            def __init__(self, name, data):
                self.name = name
                self._data = data
            def getvalue(self):
                return self._data
        uploaded = _Uploaded(uploaded_name, uploaded_bytes)

    df, loaded_from = load_excel_source(DEFAULT_FILE, uploaded)
    df.columns = [str(c).strip() for c in df.columns]
    df = standardize_current_data(df)
    df = normalize_text_cols(df, ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"])

    lag_cols = [f"{v}__LAG_{l}" for v in PHENOLOGY_COLS for l in range(1, MAX_LAG + 1)]
    keep_cols = unique_preserve_order([c for c in ALL_ANALYSIS_BASE_COLS + lag_cols if c in df.columns])
    df = df[keep_cols].copy()

    num_cols = ["AÑO", "SEMANA", "SEMANA FENOLOGICA", TARGET_COL] + PHENOLOGY_COLS + lag_cols
    df = to_numeric_safe(df, [c for c in num_cols if c in df.columns])
    df = add_delta_only(df)
    return df, loaded_from


# =========================================================
# UI
# =========================================================
st.markdown(
    """
    <div style='background:#1a1a2e;color:#eee;padding:10px 16px;border-radius:6px;margin-bottom:8px'>
        <span style='font-size:17px;font-weight:bold'>🫐 Dashboard — Delta Peso Baya</span>
        <span style='font-size:11px;margin-left:14px;opacity:0.7'>Filtros → Aplicar → explorar pestañas</span>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader("Excel opcional", type=["xlsx"])
uploaded_name = uploaded_file.name if uploaded_file else None
uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None

try:
    data, loaded_file = load_and_prepare_data(uploaded_name, uploaded_bytes)
except Exception as e:
    st.error(f"Error al cargar o preparar los datos: {e}")
    st.stop()

st.caption(f"Archivo cargado: {loaded_file}")

with st.expander("🔽  Filtros globales", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        f_ca = st.multiselect("CAMPAÑA", sorted(data["CAMPAÑA"].dropna().astype(str).unique()))
        f_an = st.multiselect("AÑO", [str(x) for x in sorted(data["AÑO"].dropna().unique())])
    with c2:
        f_fu = st.multiselect("FUNDO", sorted(data["FUNDO"].dropna().astype(str).unique()))
        f_et = st.multiselect("ETAPA", sorted(data["ETAPA"].dropna().astype(str).unique()))
    with c3:
        f_cm = st.multiselect("CAMPO", sorted(data["CAMPO"].dropna().astype(str).unique()))
        f_tu = st.multiselect("TURNO", sorted(data["TURNO"].dropna().astype(str).unique()))
    with c4:
        f_va = st.multiselect("VARIEDAD", sorted(data["VARIEDAD"].dropna().astype(str).unique()))
        target = st.selectbox("Objetivo", ["DELTA_BW", "DELTA_BW_%"])
        outlier_opt = st.selectbox("Outliers", ["Todos", "Todos excepto outliers"])

filtered = data.copy()
for col, sel in [
    ("CAMPAÑA", f_ca), ("AÑO", f_an), ("FUNDO", f_fu), ("ETAPA", f_et),
    ("CAMPO", f_cm), ("TURNO", f_tu), ("VARIEDAD", f_va),
]:
    if sel:
        filtered = filtered[filtered[col].astype(str).isin(sel)]

st.info(f"Filas filtradas: {len(filtered):,} | Grupos únicos: {filtered['ENTITY_KEY'].nunique():,}")

if filtered.empty:
    st.warning("No hay datos luego de aplicar los filtros.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Scatterplot", "🏆 Ranking", "🔲 Matriz", "📈 Series Temporales"])

# =========================================================
# TAB 1 Scatterplot
# =========================================================
with tab1:
    left, right = st.columns([1.35, 1])
    with left:
        sc_var = st.selectbox("Var A", PHENOLOGY_COLS, index=0)
        lag_opts = [f"Lag {i}" + (" (actual)" if i == 0 else "") for i in range(MAX_LAG + 1)]
        sc_lag_label = st.selectbox("Lag A", lag_opts, index=1)
        sc_lag = int(sc_lag_label.split()[1])
        sc_ta = st.selectbox("Transf. A", ALL_X_TRANSFORM_OPTIONS, index=0)
        sc_cmp = st.toggle("Comparar", value=False)

    with right:
        sc_mod = st.selectbox("Modelo", ["Ninguno", "Lineal", "Polinomial (grado 2)", "Spline"], index=1)
        sc_col = st.selectbox("Color", ["Ninguno", "CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"], index=6)
        if sc_cmp:
            sc_vb = st.selectbox("Var B", PHENOLOGY_COLS, index=0, key="vb")
            sc_lb_label = st.selectbox("Lag B", lag_opts, index=1, key="lb")
            sc_lb = int(sc_lb_label.split()[1])
            sc_tb = st.selectbox("Transf. B", ALL_X_TRANSFORM_OPTIONS, index=0, key="tb")

    def get_panel_data(filtered_df, base_var, lag, transform, target_col, apply_outlier):
        x_col = get_x_col(base_var, lag)
        if x_col not in filtered_df.columns:
            return None, None, x_col

        sf = filtered_df["SEMANA FENOLOGICA"] if "SEMANA FENOLOGICA" in filtered_df.columns else pd.Series(dtype=float, index=filtered_df.index)
        xraw = get_series(filtered_df, x_col)
        yraw = pd.to_numeric(filtered_df[target_col], errors="coerce")

        if apply_outlier:
            tmp = filtered_df.copy()
            tmp["_xt"] = pd.to_numeric(transform_x_series(xraw, sf, transform), errors="coerce")
            mask = mahalanobis_filter(tmp, "_xt", target_col)
            filtered_df = filtered_df[mask]
            sf = filtered_df["SEMANA FENOLOGICA"] if "SEMANA FENOLOGICA" in filtered_df.columns else pd.Series(dtype=float, index=filtered_df.index)
            xraw = get_series(filtered_df, x_col)
            yraw = pd.to_numeric(filtered_df[target_col], errors="coerce")

        xt = transform_x_series(x=xraw, semana_fenologica=sf, transform_mode=transform)
        d = {"X": pd.to_numeric(xt, errors="coerce"), "Y": yraw}
        if sc_col != "Ninguno" and sc_col in filtered_df.columns:
            d["C"] = filtered_df[sc_col].astype(str)
        sub = pd.DataFrame(d, index=filtered_df.index).dropna(subset=["X", "Y"])
        label = get_x_axis_label(x_col, transform)
        return sub, label, x_col

    apply_out = outlier_opt == "Todos excepto outliers"
    sub_a, label_a, x_col_a = get_panel_data(filtered, sc_var, sc_lag, sc_ta, target, apply_out)
    if sub_a is None:
        st.warning(f"'{x_col_a}' no disponible.")
    elif not sc_cmp:
        if sub_a.empty:
            st.warning("Sin datos.")
        else:
            p = sub_a["X"].corr(sub_a["Y"], method="pearson") if len(sub_a) >= 2 else np.nan
            s = sub_a["X"].corr(sub_a["Y"], method="spearman") if len(sub_a) >= 2 else np.nan

            if sc_col != "Ninguno" and "C" in sub_a.columns:
                fig = px.scatter(sub_a, x="X", y="Y", color="C", opacity=0.65, labels={"X": label_a, "Y": target, "C": sc_col})
            else:
                fig = px.scatter(sub_a, x="X", y="Y", opacity=0.65, labels={"X": label_a, "Y": target})

            if sc_mod != "Ninguno":
                fig = add_model(fig, sub_a["X"], sub_a["Y"], sc_mod)

            fig.add_annotation(
                text=f"n={len(sub_a)} | Pearson={p:.3f} | Spearman={s:.3f}",
                xref="paper", yref="paper", x=0, y=1.07, showarrow=False,
                bgcolor="rgba(255,255,255,0.85)", align="left", font=dict(size=12)
            )
            fig.update_layout(height=520, title=f"{x_col_a} [{sc_ta}] vs {target}", margin=dict(t=90))
            st.plotly_chart(fig, use_container_width=True)
    else:
        sub_b, label_b, x_col_b = get_panel_data(filtered, sc_vb, sc_lb, sc_tb, target, apply_out)
        if sub_b is None:
            st.warning(f"'{x_col_b}' no disponible.")
        elif sub_a.empty or sub_b.empty:
            st.warning("Sin datos suficientes para comparar.")
        else:
            pa = sub_a["X"].corr(sub_a["Y"], method="pearson")
            sa = sub_a["X"].corr(sub_a["Y"], method="spearman")
            pb = sub_b["X"].corr(sub_b["Y"], method="pearson")
            sb = sub_b["X"].corr(sub_b["Y"], method="spearman")

            fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                subplot_titles=[f"A: {x_col_a} [{sc_ta}]", f"B: {x_col_b} [{sc_tb}]"],
                                horizontal_spacing=0.05)
            fig.add_trace(go.Scatter(x=sub_a["X"], y=sub_a["Y"], mode="markers",
                                     marker=dict(size=5, opacity=0.65, color="#2196F3"), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=sub_b["X"], y=sub_b["Y"], mode="markers",
                                     marker=dict(size=5, opacity=0.65, color="#e67e22"), showlegend=False), row=1, col=2)

            if sc_mod != "Ninguno":
                def _model_trace(sub_x, sub_y, model_type):
                    xv, yv = sub_x.values, sub_y.values
                    if len(xv) < 5 or len(np.unique(xv)) < 2:
                        return None
                    xl = np.linspace(xv.min(), xv.max(), 200)
                    try:
                        if model_type == "Lineal":
                            c = np.polyfit(xv, yv, 1); return xl, c[0] * xl + c[1]
                        if model_type == "Polinomial (grado 2)":
                            c = np.polyfit(xv, yv, 2); return xl, c[0] * xl**2 + c[1] * xl + c[2]
                        if model_type == "Spline":
                            from scipy.interpolate import UnivariateSpline
                            idx = np.argsort(xv)
                            xs, ys = xv[idx], yv[idx]
                            try:
                                poly_r = ys - np.polyval(np.polyfit(xs, ys, min(3, len(xs)-2)), xs)
                                sval = max(len(xs) * float(np.var(poly_r)), 1e-10)
                            except Exception:
                                sval = len(xs)
                            spl = UnivariateSpline(xs, ys, k=3, s=sval)
                            return xl, spl(xl)
                    except Exception:
                        return None
                    return None

                ra = _model_trace(sub_a["X"], sub_a["Y"], sc_mod)
                if ra:
                    fig.add_trace(go.Scatter(x=ra[0], y=ra[1], mode="lines",
                                             line=dict(color="#0d47a1", width=2), showlegend=False), row=1, col=1)
                rb = _model_trace(sub_b["X"], sub_b["Y"], sc_mod)
                if rb:
                    fig.add_trace(go.Scatter(x=rb[0], y=rb[1], mode="lines",
                                             line=dict(color="#bf360c", width=2), showlegend=False), row=1, col=2)

            fig.update_xaxes(title_text=label_a, row=1, col=1)
            fig.update_xaxes(title_text=label_b, row=1, col=2)
            fig.update_yaxes(title_text=target, row=1, col=1)
            fig.add_annotation(
                text=f"A: n={len(sub_a)}, r={pa:.3f}, ρ={sa:.3f}     B: n={len(sub_b)}, r={pb:.3f}, ρ={sb:.3f}",
                xref="paper", yref="paper", x=0.5, y=1.10, showarrow=False,
                align="center", bgcolor="rgba(255,255,255,0.85)", font=dict(size=12)
            )
            fig.update_layout(height=520, title="Comparación de paneles", margin=dict(t=100))
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 Ranking
# =========================================================
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rk_met = st.selectbox("Métrica", ["|Pearson|", "|Spearman|", "Pearson", "Spearman", "R²", "RMSE", "MAE", "MAPE", "AIC", "BIC", "RESET p-val", "BP p-val", "Levene p-val"])
    with c2:
        rk_top = st.number_input("Top N", min_value=1, max_value=200, value=20, step=1)
    with c3:
        rk_vars = st.multiselect("Variables", PHENOLOGY_COLS, default=PHENOLOGY_COLS)
    with c4:
        rk_lags = st.multiselect("Lags", list(range(MAX_LAG + 1)), default=list(range(MAX_LAG + 1)))

    met_map = {"|Pearson|": "pearson_abs", "|Spearman|": "spearman_abs", "Pearson": "pearson", "Spearman": "spearman",
               "R²": "r2", "RMSE": "rmse", "MAE": "mae", "MAPE": "mape", "AIC": "aic", "BIC": "bic",
               "RESET p-val": "reset_f_pval", "BP p-val": "bp_f_pval", "Levene p-val": "levene_pval"}
    lower_better = {"RMSE", "MAE", "MAPE", "AIC", "BIC"}
    sort_col = met_map[rk_met]
    sort_asc = rk_met in lower_better

    ranking = build_global_ranking(
        filtered, target, sort_metric=sort_col, sort_ascending=sort_asc,
        apply_outlier=apply_out, vars_filter=rk_vars, lags_filter=rk_lags
    )
    if ranking.empty:
        st.warning("Sin datos suficientes.")
    else:
        top = ranking.head(int(rk_top)).copy()
        st.dataframe(top.round(4), use_container_width=True)

        chart_col = sort_col if sort_col in top.columns else "pearson_abs"
        plot_df = top.dropna(subset=[chart_col]).copy()
        if not plot_df.empty:
            plot_df["label"] = plot_df["variable"] + " Lag" + plot_df["lag"].astype(str) + " [" + plot_df["transformacion"] + "]"
            fig = px.bar(plot_df, x="label", y=chart_col, color="direccion", text="n", title=f"Top {int(rk_top)} — {rk_met} — {target}")
            fig.update_layout(height=420, xaxis_title="", yaxis_title=rk_met, xaxis=dict(tickangle=-40))
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 Matriz
# =========================================================
with tab3:
    c1, c2, c3 = st.columns(3)
    with c1:
        mx_var = st.selectbox("Variable", PHENOLOGY_COLS)
    with c2:
        mx_met = st.selectbox("Métrica matriz", ["|Pearson|", "|Spearman|", "Pearson", "Spearman", "R²", "RMSE", "MAE", "MAPE", "AIC", "BIC", "RESET p-val", "BP p-val", "Levene p-val"])
    with c3:
        mx_view = st.selectbox("Vista", ["Normal", "Semáforo"], index=0)

    mx_thr = st.slider("Umbral |r|", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
    mat_res = compute_matrix(filtered, mx_var, target, apply_outlier=apply_out)

    rlbls = [f"Lag {i}" + (" (actual)" if i == 0 else "") for i in range(MAX_LAG + 1)]
    clbls = ALL_X_TRANSFORM_OPTIONS
    mmap = {"|Pearson|": ("pearson", True), "|Spearman|": ("spearman", True), "Pearson": ("pearson", False), "Spearman": ("spearman", False),
            "R²": ("r2", False), "RMSE": ("rmse", False), "MAE": ("mae", False), "MAPE": ("mape", False), "AIC": ("aic", False), "BIC": ("bic", False),
            "RESET p-val": ("reset_f_pval", False), "BP p-val": ("bp_f_pval", False), "Levene p-val": ("levene_pval", False)}
    mkey, take_abs = mmap[mx_met]
    mat = mat_res[mkey].astype(float)
    if take_abs:
        mat = mat.abs()

    if mx_view == "Normal":
        if "p-val" in mx_met:
            def _pval_color(p):
                if pd.isna(p):
                    return np.nan
                if p <= 0.01:
                    return p / 0.01 * 0.10
                if p <= 0.05:
                    return 0.10 + (p - 0.01) / 0.04 * 0.40
                if p <= 0.10:
                    return 0.50 + (p - 0.05) / 0.05 * 0.25
                return 0.75 + min((p - 0.10) / 0.90, 1.0) * 0.25

            cs_pval = [[0.00, "#c0392b"], [0.10, "#c0392b"],
                       [0.10, "#e67e22"], [0.50, "#e67e22"],
                       [0.50, "#f1c40f"], [0.75, "#f1c40f"],
                       [0.75, "#2ecc71"], [1.00, "#2ecc71"]]
            zmap = mat.applymap(_pval_color)
            hover_text = mat.round(3).applymap(
                lambda p: ("Alt. significativo (p≤0.01)" if pd.notna(p) and p <= 0.01 else
                           "Significativo (p≤0.05)" if pd.notna(p) and p <= 0.05 else
                           "Marginal (p≤0.10)" if pd.notna(p) and p <= 0.10 else
                           "No significativo (p>0.10)" if pd.notna(p) else "Sin datos")
            )
            fig = go.Figure(go.Heatmap(
                z=zmap.values, x=clbls, y=rlbls, colorscale=cs_pval, zmin=0, zmax=1,
                text=mat.round(3).astype(str).values, texttemplate="%{text}", textfont=dict(size=9),
                customdata=hover_text.values,
                hovertemplate="%{y} × %{x}<br>p=%{text}<br>%{customdata}<extra></extra>",
                colorbar=dict(title=mx_met)
            ))
        else:
            lower_better_mat = mx_met in {"RMSE", "MAE", "MAPE", "AIC", "BIC"}
            cs_normal = "RdYlGn_r" if lower_better_mat else "RdYlGn"
            fig = go.Figure(go.Heatmap(
                z=mat.values, x=clbls, y=rlbls, colorscale=cs_normal,
                text=mat.round(3).astype(str).values, texttemplate="%{text}", textfont=dict(size=9),
                colorbar=dict(title=mx_met),
                hovertemplate="%{y} × %{x}<br>" + mx_met + "=%{z:.3f}<extra></extra>",
            ))
    else:
        pabs = mat_res["pearson"].abs().astype(float)
        rp = mat_res["reset_f_pval"].astype(float)
        bp = mat_res["bp_f_pval"].astype(float)
        nr = MAX_LAG + 1
        nc = len(ALL_X_TRANSFORM_OPTIONS)
        semaf = np.full((nr, nc), np.nan)
        for i in range(nr):
            for j, tr in enumerate(ALL_X_TRANSFORM_OPTIONS):
                pv = pabs.iloc[i, j]
                rv = rp.iloc[i, j]
                bv = bp.iloc[i, j]
                if pd.isna(pv):
                    semaf[i, j] = np.nan
                    continue
                p_ok = pv >= mx_thr
                l_ok = (not pd.isna(rv)) and rv > 0.05
                h_ok = (not pd.isna(bv)) and bv > 0.05
                semaf[i, j] = 3 if (p_ok and l_ok and h_ok) else 2 if (p_ok and (l_ok or h_ok)) else 1 if p_ok else 0

        fig = go.Figure(go.Heatmap(
            z=semaf,
            x=clbls,
            y=rlbls,
            colorscale=[[0.00, "#e74c3c"], [0.25, "#e74c3c"], [0.25, "#e67e22"], [0.50, "#e67e22"],
                        [0.50, "#f1c40f"], [0.75, "#f1c40f"], [0.75, "#2ecc71"], [1.00, "#2ecc71"]],
            zmin=0, zmax=3,
            text=mat.round(3).astype(str).values,
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="Semáforo", tickvals=[0, 1, 2, 3], ticktext=["🔴", "🟠", "🟡", "🟢"]),
        ))

    fig.update_layout(title=f"Matriz: {mx_var} | {mx_met} | {target}", xaxis_title="Transformación",
                      yaxis_title="Lag", height=620, xaxis=dict(tickangle=-35), margin=dict(b=130))
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4 Series temporales
# =========================================================
with tab4:
    c1, c2, c3 = st.columns(3)
    with c1:
        ts_ent = st.selectbox("Entidad", sorted(filtered["SERIES_KEY"].dropna().unique().tolist()))
    with c2:
        ts_var = st.selectbox("Variable", PHENOLOGY_COLS)
    with c3:
        ts_lag_label = st.selectbox("Lag", lag_opts, index=1, key="tslag")
        ts_lag = int(ts_lag_label.split()[1])

    x_col = get_x_col(ts_var, ts_lag)
    ts_df = filtered[filtered["SERIES_KEY"] == ts_ent].sort_values(["AÑO", "SEMANA"]).copy()
    if ts_df.empty:
        st.warning("Sin datos para el grupo.")
    else:
        ts_df["EJE"] = ts_df["AÑO"].astype("Int64").astype(str) + "-W" + ts_df["SEMANA"].astype("Int64").astype(str).str.zfill(2)

        fig = go.Figure()
        if TARGET_COL in ts_df.columns:
            fig.add_trace(go.Scatter(x=ts_df["EJE"], y=ts_df[TARGET_COL], mode="lines+markers", name="Peso suavizado"))
        if target in ts_df.columns and target != TARGET_COL:
            fig.add_trace(go.Scatter(x=ts_df["EJE"], y=ts_df[target], mode="lines+markers", name=target, yaxis="y2"))
        if x_col in ts_df.columns:
            fig.add_trace(go.Scatter(x=ts_df["EJE"], y=ts_df[x_col], mode="lines+markers", name=x_col, yaxis="y3"))

        fig.update_layout(
            height=560,
            title=ts_ent,
            xaxis=dict(title="Semana"),
            yaxis=dict(title="Peso (g)"),
            yaxis2=dict(title=target, overlaying="y", side="right"),
            yaxis3=dict(title=x_col, anchor="free", overlaying="y", side="right", position=0.92),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

        show_cols = [c for c in ["CAMPAÑA", "AÑO", "SEMANA", "SEMANA FENOLOGICA", TARGET_COL, "PESO_BAYA_SUAVIZADO", target, x_col] if c in ts_df.columns]
        st.dataframe(ts_df[show_cols].reset_index(drop=True), use_container_width=True)
