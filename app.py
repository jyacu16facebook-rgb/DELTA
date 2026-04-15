import glob
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import kruskal


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="EDA Delta semanal de peso de la baya",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_FILE = "DELTA CONSOLIDADO 2022-2025.xlsx"
DEFAULT_SHEET = "DATA"
MAX_LAG = 15
MIN_OBS_STABILITY = 8  # mínimo razonable para calcular correlaciones por grupo

# =========================================================
# MAPEO: DATA ACTUAL -> NOMBRES INTERNOS DEL CÓDIGO ORIGINAL
# =========================================================
CURRENT_TO_INTERNAL = {
    "año": "AÑO",
    "campaña": "CAMPAÑA",
    "semana": "SEMANA",
    "fundo": "FUNDO",
    "etapa": "ETAPA",
    "campo": "CAMPO",
    "turno": "TURNO",
    "variedad": "VARIEDAD",
    "conteo_flores": "FLORES",
    "conteo_fruto_cuajado": "FRUTO CUAJADO",
    "conteo_fruto_verde": "FRUTO VERDE",
    "conteo_total_frutos": "TOTAL DE FRUTOS",
    "conteo_bayas_cremosas": "FRUTO CREMOSO",
    "conteo_bayas_rosadas": "FRUTO ROSADO",
    "conteo_bayas_maduras": "FRUTO MADURO",
    "peso_promedio_baya_g": "PESO BAYA (g)",
}

# variable base interna -> prefijo real de la data actual
INTERNAL_BASE_TO_CURRENT_LAG_PREFIX = {
    "FLORES": "conteo_flores",
    "FRUTO CUAJADO": "conteo_fruto_cuajado",
    "FRUTO VERDE": "conteo_fruto_verde",
    "TOTAL DE FRUTOS": "conteo_total_frutos",
    "FRUTO CREMOSO": "conteo_bayas_cremosas",
    "FRUTO ROSADO": "conteo_bayas_rosadas",
    "FRUTO MADURO": "conteo_bayas_maduras",
}

ID_COLS_DISPLAY = [
    "AÑO",
    "CAMPAÑA",
    "SEMANA",
    "FUNDO",
    "ETAPA",
    "CAMPO",
    "TURNO",
    "VARIEDAD",
]

ENTITY_COLS = [
    "CAMPAÑA",
    "FUNDO",
    "ETAPA",
    "CAMPO",
    "TURNO",
    "VARIEDAD",
]

TARGET_COL = "PESO BAYA (g)"

PHENOLOGY_COLS = [
    "FLORES",
    "FRUTO CUAJADO",
    "FRUTO VERDE",
    "TOTAL DE FRUTOS",
    "FRUTO CREMOSO",
    "FRUTO ROSADO",
    "FRUTO MADURO",
]

CONTROL_COLS = []
OPTIONAL_EXTRA_COLS = []

ALL_ANALYSIS_BASE_COLS = ID_COLS_DISPLAY + PHENOLOGY_COLS + [TARGET_COL]


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def find_excel_file(default_file: str) -> str:
    if default_file in glob.glob("*.xlsx"):
        return default_file

    files = glob.glob("*.xlsx")
    if files:
        return files[0]

    raise FileNotFoundError(
        f"No se encontró el archivo '{default_file}' ni ningún archivo .xlsx en la raíz."
    )


def safe_iso_week_start(year, week):
    try:
        year = int(year)
        week = int(week)
        return pd.Timestamp(datetime.fromisocalendar(year, week, 1))
    except Exception:
        return pd.NaT


def validate_columns(df: pd.DataFrame, required_cols: list[str]):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el archivo: " + ", ".join(missing)
        )


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_text_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def build_entity_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["CAMPAÑA"].astype("string").fillna("")
        + " | " + df["FUNDO"].astype("string").fillna("")
        + " | " + df["ETAPA"].astype("string").fillna("")
        + " | " + df["CAMPO"].astype("string").fillna("")
        + " | " + df["TURNO"].astype("string").fillna("")
        + " | " + df["VARIEDAD"].astype("string").fillna("")
    )


def standardize_current_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas actuales a nombres internos del código
    y crea columnas internas de lag a partir de los lags ya existentes.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_current_cols = list(CURRENT_TO_INTERNAL.keys())
    validate_columns(df, required_current_cols)

    rename_base = {k: v for k, v in CURRENT_TO_INTERNAL.items() if k in df.columns}
    df = df.rename(columns=rename_base)

    for internal_var, current_prefix in INTERNAL_BASE_TO_CURRENT_LAG_PREFIX.items():
        for lag in range(1, MAX_LAG + 1):
            current_lag_col = f"{current_prefix}_semana_{lag}_anterior"
            internal_lag_col = f"{internal_var}__LAG_{lag}"

            if current_lag_col in df.columns:
                df[internal_lag_col] = pd.to_numeric(df[current_lag_col], errors="coerce")
            else:
                df[internal_lag_col] = np.nan

    return df


def add_delta_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula DELTA_BW y DELTA_BW_%.
    No recalcula lags porque la data actual ya los trae.
    """
    df = df.copy()

    df = df.sort_values(ENTITY_COLS + ["AÑO", "SEMANA"], kind="stable").reset_index(drop=True)

    df["WEEK_START_DATE"] = [
        safe_iso_week_start(y, w) for y, w in zip(df["AÑO"], df["SEMANA"])
    ]

    df["SEMANA ACTUAL"] = df["SEMANA"]
    df["AÑO-SEMANA ACTUAL"] = (
        df["AÑO"].astype("Int64").astype(str)
        + "-W"
        + df["SEMANA"].astype("Int64").astype(str).str.zfill(2)
    )

    invalid_weeks = df["WEEK_START_DATE"].isna().sum()
    if invalid_weeks > 0:
        st.warning(
            f"Se detectaron {invalid_weeks} filas con combinación AÑO/SEMANA inválida. "
            "Esas filas no podrán participar correctamente en cálculos temporales."
        )

    dup_mask = df.duplicated(subset=ENTITY_COLS + ["AÑO", "SEMANA"], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        st.error(
            f"Se detectaron {n_dup} filas duplicadas en la llave "
            f"{ENTITY_COLS + ['AÑO', 'SEMANA']}. "
            "El cálculo temporal se detendrá porque debería existir solo una fila por grupo-semana."
        )
        st.stop()

    g = df.groupby(ENTITY_COLS, dropna=False, sort=False)

    prev_weight = g[TARGET_COL].shift(1)
    prev_week_date = g["WEEK_START_DATE"].shift(1)

    consecutive_1 = (
        df["WEEK_START_DATE"].notna()
        & prev_week_date.notna()
        & ((df["WEEK_START_DATE"] - prev_week_date) == pd.Timedelta(days=7))
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

    df["ENTITY_KEY"] = build_entity_key(df)
    return df


def iqr_filter_mask(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index)

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return s.notna()

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return s.between(lower, upper, inclusive="both")


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    filtered = df.copy()

    filter_cols = ["CAMPAÑA", "AÑO", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]
    for col in filter_cols:
        if col in filtered.columns:
            opts = sorted(filtered[col].dropna().astype(str).unique().tolist())
            selected = st.sidebar.multiselect(
                f"{col}",
                options=opts,
                default=opts,
            )
            if selected:
                filtered = filtered[filtered[col].astype(str).isin(selected)]

    return filtered


def add_trendline(fig, x: pd.Series, y: pd.Series, name: str = "Tendencia lineal"):
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = x_num.notna() & y_num.notna()

    if valid.sum() < 2:
        return fig

    x_v = x_num[valid].values
    y_v = y_num[valid].values

    if len(np.unique(x_v)) < 2:
        return fig

    coef = np.polyfit(x_v, y_v, 1)
    x_line = np.linspace(x_v.min(), x_v.max(), 100)
    y_line = coef[0] * x_line + coef[1]

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=name,
        )
    )
    return fig


def compute_group_stability(df: pd.DataFrame, group_col: str, x_col: str, y_col: str, min_obs: int = 8) -> pd.DataFrame:
    """
    Calcula estabilidad de la relación x vs y por grupo.
    Devuelve n, Pearson y Spearman por cada campaña o variedad.
    """
    work = df[[group_col, x_col, y_col]].copy()
    work = work.dropna()

    if work.empty:
        return pd.DataFrame(columns=[group_col, "n", "pearson", "spearman", "abs_pearson", "abs_spearman"])

    rows = []
    for grp, sub in work.groupby(group_col, dropna=False):
        n = len(sub)

        if n < min_obs:
            rows.append({
                group_col: grp,
                "n": n,
                "pearson": np.nan,
                "spearman": np.nan,
                "abs_pearson": np.nan,
                "abs_spearman": np.nan,
                "estado": f"<{min_obs} obs",
            })
            continue

        x = pd.to_numeric(sub[x_col], errors="coerce")
        y = pd.to_numeric(sub[y_col], errors="coerce")
        valid = x.notna() & y.notna()

        if valid.sum() < min_obs:
            pearson = np.nan
            spearman = np.nan
        else:
            try:
                pearson = x[valid].corr(y[valid], method="pearson")
            except Exception:
                pearson = np.nan
            try:
                spearman = x[valid].corr(y[valid], method="spearman")
            except Exception:
                spearman = np.nan

        rows.append({
            group_col: grp,
            "n": valid.sum(),
            "pearson": pearson,
            "spearman": spearman,
            "abs_pearson": abs(pearson) if pd.notna(pearson) else np.nan,
            "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
            "estado": "ok" if valid.sum() >= min_obs else f"<{min_obs} obs",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_spearman", "abs_pearson", "n"], ascending=[False, False, False], na_position="last")
    return out


def build_findings_summary(df: pd.DataFrame, analysis_target: str, base_vars: list[str], max_lag: int) -> pd.DataFrame:
    """
    Resume, para cada variable base, cuál fue el mejor lag
    según |Spearman| y reporta también Pearson y n.
    """
    rows = []

    for base_var in base_vars:
        best_row = None
        best_score = -np.inf

        for lag in range(1, max_lag + 1):
            x_col = f"{base_var}__LAG_{lag}"
            if x_col not in df.columns:
                continue

            sub = df[[x_col, analysis_target]].dropna()
            n = len(sub)

            if n < 8:
                continue

            pearson = sub[x_col].corr(sub[analysis_target], method="pearson")
            spearman = sub[x_col].corr(sub[analysis_target], method="spearman")
            score = abs(spearman) if pd.notna(spearman) else -np.inf

            if score > best_score:
                best_score = score
                best_row = {
                    "variable_base": base_var,
                    "mejor_lag": lag,
                    "columna_lag": x_col,
                    "n": n,
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
                }

        if best_row is not None:
            rows.append(best_row)

    if not rows:
        return pd.DataFrame(columns=["variable_base", "mejor_lag", "columna_lag", "n", "pearson", "spearman", "abs_spearman"])

    out = pd.DataFrame(rows).sort_values(["abs_spearman", "n"], ascending=[False, False]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    file_path = find_excel_file(DEFAULT_FILE)
    df = pd.read_excel(file_path, sheet_name=DEFAULT_SHEET, engine="openpyxl")

    df.columns = [str(c).strip() for c in df.columns]

    # Adaptar estructura actual a estructura interna
    df = standardize_current_data(df)

    # Normalización de textos
    df = normalize_text_cols(
        df,
        ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]
    )

    # Mantener columnas necesarias
    lag_cols = [
        f"{var}__LAG_{lag}"
        for var in PHENOLOGY_COLS
        for lag in range(1, MAX_LAG + 1)
    ]

    keep_cols = [
        c for c in (ALL_ANALYSIS_BASE_COLS + lag_cols)
        if c in df.columns
    ]
    df = df[keep_cols].copy()

    # Conversión numérica
    numeric_cols = [
        "AÑO",
        "SEMANA",
        TARGET_COL,
        *PHENOLOGY_COLS,
        *lag_cols,
    ]
    df = to_numeric_safe(df, [c for c in numeric_cols if c in df.columns])

    # Calcular delta
    df = add_delta_only(df)

    return df, file_path


# =========================================================
# CARGA PRINCIPAL
# =========================================================
st.title("EDA: Delta semanal de peso de la baya")
st.caption("Análisis exploratorio de la relación entre el cambio semanal del peso de la baya y variables fenológicas rezagadas.")

try:
    data, loaded_file = load_and_prepare_data()
except Exception as e:
    st.error(f"Error al cargar o preparar los datos: {e}")
    st.stop()

with st.expander("Verificación técnica del enfoque usado", expanded=False):
    st.markdown(
        """
**Cálculo temporal aplicado**
- Los deltas se calculan por: `CAMPAÑA + FUNDO + ETAPA + CAMPO + TURNO + VARIEDAD`
- El orden temporal es: `CAMPAÑA -> AÑO -> SEMANA`
- `AÑO` se conserva como identificador y filtro, pero no rompe la continuidad temporal dentro de la campaña
- Solo se calcula delta si la semana previa existe realmente y es consecutiva
- Si falta una semana intermedia, el valor queda vacío
- `DELTA_BW_%` queda vacío si el peso previo es nulo o igual a 0
- Los lags ya no se recalculan: se usan directamente desde la data actual
        """
    )

filtered = apply_sidebar_filters(data)

if filtered.empty:
    st.warning("No hay datos luego de aplicar los filtros.")
    st.stop()

# =========================================================
# CONTROLES PRINCIPALES
# =========================================================
st.sidebar.header("Configuración analítica")

analysis_target = st.sidebar.selectbox(
    "Variable objetivo",
    options=["DELTA_BW", "DELTA_BW_%"],
    index=0,
)

base_analysis_vars = PHENOLOGY_COLS
selected_base_var = st.sidebar.selectbox(
    "Variable explicativa base",
    options=base_analysis_vars,
    index=base_analysis_vars.index("FRUTO VERDE") if "FRUTO VERDE" in base_analysis_vars else 0,
)

selected_lag = st.sidebar.slider(
    "Lag a visualizar",
    min_value=1,
    max_value=MAX_LAG,
    value=1,
)

selected_x_col = f"{selected_base_var}__LAG_{selected_lag}"

outlier_mode = st.sidebar.radio(
    "Filtro de outliers en la variable objetivo",
    options=["Todos", "Todos excepto outliers"],
    index=0,
)

show_trend = st.sidebar.checkbox("Mostrar línea de tendencia lineal", value=True)

# Base para visuales
viz_cols = [
    *ID_COLS_DISPLAY,
    "SEMANA ACTUAL",
    "AÑO-SEMANA ACTUAL",
    "ENTITY_KEY",
    TARGET_COL,
    "DELTA_BW",
    "DELTA_BW_%",
    selected_x_col,
]

viz_cols = [c for c in viz_cols if c in filtered.columns]
viz_df = filtered[viz_cols].copy()

if selected_x_col not in viz_df.columns:
    st.error(f"No existe la columna seleccionada para análisis: {selected_x_col}")
    st.stop()

if outlier_mode == "Todos excepto outliers":
    valid_mask = iqr_filter_mask(viz_df[analysis_target])
    viz_df = viz_df[valid_mask].copy()

# =========================================================
# RESUMEN GENERAL
# =========================================================
st.subheader("Resumen general")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Filas filtradas", f"{len(filtered):,}")
c2.metric("Grupos únicos", f"{filtered['ENTITY_KEY'].nunique():,}")
c3.metric("Con peso válido", f"{filtered[TARGET_COL].notna().sum():,}")
c4.metric("Delta válido", f"{filtered['DELTA_BW'].notna().sum():,}")
c5.metric("Delta % válido", f"{filtered['DELTA_BW_%'].notna().sum():,}")
c6.metric("Filas en visual actual", f"{len(viz_df.dropna(subset=[analysis_target, selected_x_col])):,}")

missing_vars = [TARGET_COL] + PHENOLOGY_COLS + ["DELTA_BW", "DELTA_BW_%"]

missing_summary = pd.DataFrame(
    {
        "variable": missing_vars,
        "faltantes_%": [
            filtered[var].isna().mean() * 100 if var in filtered.columns else np.nan
            for var in missing_vars
        ],
    }
).sort_values("faltantes_%", ascending=False)

with st.expander("Ver porcentaje de faltantes por variable", expanded=False):
    st.dataframe(missing_summary, use_container_width=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Scatterplot",
        "Boxplot",
        "Series temporales",
        "Estabilidad",
        "Resumen de hallazgos",
    ]
)

# =========================================================
# TAB 1: SCATTER
# =========================================================
with tab1:
    st.subheader("Scatterplot")

    scatter_cols = [
        selected_x_col,
        analysis_target,
        "CAMPAÑA",
        "AÑO",
        "SEMANA",
        "SEMANA ACTUAL",
        "AÑO-SEMANA ACTUAL",
        "VARIEDAD",
        "CAMPO",
        "FUNDO",
        "ENTITY_KEY",
    ]
    scatter_cols = [c for c in scatter_cols if c in viz_df.columns]

    scatter_df = viz_df[scatter_cols].dropna()

    if scatter_df.empty:
        st.warning("No hay datos suficientes para el scatterplot con la selección actual.")
    else:
        fig = px.scatter(
            scatter_df,
            x=selected_x_col,
            y=analysis_target,
            color="VARIEDAD",
            hover_data=[
                c for c in [
                    "CAMPAÑA",
                    "AÑO",
                    "SEMANA ACTUAL",
                    "AÑO-SEMANA ACTUAL",
                    "FUNDO",
                    "CAMPO",
                    "VARIEDAD",
                    "ENTITY_KEY",
                ] if c in scatter_df.columns
            ],
            opacity=0.65,
        )

        if show_trend:
            fig = add_trendline(fig, scatter_df[selected_x_col], scatter_df[analysis_target])

        fig.update_layout(
            xaxis_title=selected_x_col,
            yaxis_title=analysis_target,
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        corr_p = scatter_df[selected_x_col].corr(scatter_df[analysis_target], method="pearson")
        corr_s = scatter_df[selected_x_col].corr(scatter_df[analysis_target], method="spearman")

        c1, c2, c3 = st.columns(3)
        c1.metric("n", f"{len(scatter_df):,}")
        c2.metric("Pearson", f"{corr_p:.4f}" if pd.notna(corr_p) else "NA")
        c3.metric("Spearman", f"{corr_s:.4f}" if pd.notna(corr_s) else "NA")

# =========================================================
# TAB 2: BOXPLOT
# =========================================================
with tab2:
    st.subheader("Boxplot por niveles de la variable rezagada")

    box_cols = [
        selected_x_col,
        analysis_target,
        "CAMPAÑA",
        "AÑO",
        "SEMANA",
        "SEMANA ACTUAL",
        "AÑO-SEMANA ACTUAL",
        "FUNDO",
        "CAMPO",
        "TURNO",
        "VARIEDAD",
        "ENTITY_KEY",
    ]
    box_cols = [c for c in box_cols if c in viz_df.columns]

    box_df = viz_df[box_cols].dropna().copy()

    if len(box_df) < 10:
        st.warning("No hay suficientes datos para construir bins interpretables.")
    else:
        try:
            n_bins = st.slider("Número de bins", min_value=4, max_value=10, value=5, key="bins_slider")

            box_df["BIN_X"] = pd.qcut(
                box_df[selected_x_col],
                q=n_bins,
                duplicates="drop",
            )

            ordered_bins = sorted(box_df["BIN_X"].dropna().unique())
            ordered_bin_labels = [str(b) for b in ordered_bins]
            box_df["BIN_X_LABEL"] = pd.Categorical(
                box_df["BIN_X"].astype(str),
                categories=ordered_bin_labels,
                ordered=True,
            )

            fig = go.Figure()

            for bin_label in ordered_bin_labels:
                sub = box_df[box_df["BIN_X_LABEL"] == bin_label].copy()

                fig.add_trace(
                    go.Box(
                        y=sub[analysis_target],
                        x=[bin_label] * len(sub),
                        name=bin_label,
                        boxpoints="all",
                        jitter=0.35,
                        pointpos=-1.8,
                        marker=dict(size=5, opacity=0.7),
                        line=dict(width=1.5),
                        customdata=np.stack(
                            [
                                sub["SEMANA ACTUAL"].astype(str),
                                sub["AÑO-SEMANA ACTUAL"].astype(str),
                                sub["CAMPAÑA"].astype(str),
                                sub["FUNDO"].astype(str),
                                sub["CAMPO"].astype(str),
                                sub["TURNO"].astype(str),
                                sub["VARIEDAD"].astype(str),
                                sub["ENTITY_KEY"].astype(str),
                                sub[selected_x_col].astype(float),
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            f"Grupo bin: %{{x}}<br>"
                            f"{analysis_target}: %{{y:.4f}}<br>"
                            "SEMANA ACTUAL: %{customdata[0]}<br>"
                            "AÑO-SEMANA ACTUAL: %{customdata[1]}<br>"
                            "CAMPAÑA: %{customdata[2]}<br>"
                            "FUNDO: %{customdata[3]}<br>"
                            "CAMPO: %{customdata[4]}<br>"
                            "TURNO: %{customdata[5]}<br>"
                            "VARIEDAD: %{customdata[6]}<br>"
                            f"{selected_x_col}: %{{customdata[8]:.4f}}<br>"
                            "ENTITY_KEY: %{customdata[7]}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                xaxis_title=f"Bins de {selected_x_col}",
                yaxis_title=analysis_target,
                height=650,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            summary_box = (
                box_df.groupby("BIN_X_LABEL", dropna=False)[analysis_target]
                .agg(["count", "mean", "median", "std"])
                .reset_index()
                .rename(columns={
                    "BIN_X_LABEL": "GRUPO",
                    "count": "N",
                    "mean": "MEDIA",
                    "median": "MEDIANA",
                    "std": "DESV_STD"
                })
            )

            summary_box["CV(%)"] = np.where(
                summary_box["MEDIA"].abs() > 0,
                (summary_box["DESV_STD"] / summary_box["MEDIA"].abs()) * 100,
                np.nan,
            )

            kruskal_groups = []
            for grp in ordered_bin_labels:
                vals = box_df.loc[box_df["BIN_X_LABEL"] == grp, analysis_target].dropna().values
                if len(vals) > 0:
                    kruskal_groups.append(vals)

            if len(kruskal_groups) >= 2:
                h_stat, p_value = kruskal(*kruskal_groups)
            else:
                h_stat, p_value = np.nan, np.nan

            kc1, kc2, kc3, kc4, kc5 = st.columns(5)
            kc1.metric("Prueba", "Kruskal-Wallis")
            kc2.metric("Estadístico (H)", f"{h_stat:.4f}" if pd.notna(h_stat) else "NA")
            kc3.metric("p-valor", f"{p_value:.6f}" if pd.notna(p_value) else "NA")
            kc4.metric("N", f"{int(summary_box['N'].sum()):,}")
            kc5.metric("Grupos", f"{len(summary_box):,}")

            summary_box_display = summary_box.copy()
            summary_box_display["MEDIA"] = summary_box_display["MEDIA"].round(4)
            summary_box_display["MEDIANA"] = summary_box_display["MEDIANA"].round(4)
            summary_box_display["DESV_STD"] = summary_box_display["DESV_STD"].round(4)
            summary_box_display["CV(%)"] = summary_box_display["CV(%)"].round(2)

            st.dataframe(
                summary_box_display[["GRUPO", "N", "MEDIA", "MEDIANA", "DESV_STD", "CV(%)"]],
                use_container_width=True,
            )

        except Exception as e:
            st.warning(f"No fue posible generar el boxplot por bins: {e}")

# =========================================================
# TAB 3: SERIES TEMPORALES
# =========================================================
with tab3:
    st.subheader("Series temporales por grupo")

    entity_options = sorted(filtered["ENTITY_KEY"].dropna().unique().tolist())
    selected_entity = st.selectbox(
        "Selecciona un grupo temporal",
        options=entity_options,
        index=0 if entity_options else None,
    )

    ts_df = filtered[filtered["ENTITY_KEY"] == selected_entity].copy()
    ts_df = ts_df.sort_values(["AÑO", "SEMANA"])

    if ts_df.empty:
        st.warning("No hay datos para el grupo seleccionado.")
    else:
        ts_df["EJE_TIEMPO"] = (
            ts_df["AÑO"].astype("Int64").astype(str)
            + "-W"
            + ts_df["SEMANA"].astype("Int64").astype(str).str.zfill(2)
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ts_df["EJE_TIEMPO"],
                y=ts_df[TARGET_COL],
                mode="lines+markers",
                name=TARGET_COL,
                customdata=np.stack(
                    [
                        ts_df["SEMANA ACTUAL"].astype(str),
                        ts_df["AÑO-SEMANA ACTUAL"].astype(str),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    f"{TARGET_COL}: %{{y:.4f}}<br>"
                    "SEMANA ACTUAL: %{customdata[0]}<br>"
                    "AÑO-SEMANA ACTUAL: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ts_df["EJE_TIEMPO"],
                y=ts_df["DELTA_BW"],
                mode="lines+markers",
                name="DELTA_BW",
                yaxis="y2",
                customdata=np.stack(
                    [
                        ts_df["SEMANA ACTUAL"].astype(str),
                        ts_df["AÑO-SEMANA ACTUAL"].astype(str),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "DELTA_BW: %{y:.4f}<br>"
                    "SEMANA ACTUAL: %{customdata[0]}<br>"
                    "AÑO-SEMANA ACTUAL: %{customdata[1]}<extra></extra>"
                ),
            )
        )

        if selected_x_col in ts_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=ts_df["EJE_TIEMPO"],
                    y=ts_df[selected_x_col],
                    mode="lines+markers",
                    name=selected_x_col,
                    yaxis="y3",
                    customdata=np.stack(
                        [
                            ts_df["SEMANA ACTUAL"].astype(str),
                            ts_df["AÑO-SEMANA ACTUAL"].astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        f"{selected_x_col}: %{{y:.4f}}<br>"
                        "SEMANA ACTUAL: %{customdata[0]}<br>"
                        "AÑO-SEMANA ACTUAL: %{customdata[1]}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            height=650,
            xaxis=dict(title="Semana"),
            yaxis=dict(title=TARGET_COL),
            yaxis2=dict(
                title="DELTA_BW",
                overlaying="y",
                side="right",
            ),
            yaxis3=dict(
                title=selected_x_col,
                anchor="free",
                overlaying="y",
                side="right",
                position=0.95,
            ),
            legend=dict(orientation="h", y=1.08),
        )

        st.plotly_chart(fig, use_container_width=True)

        display_cols = [
            "AÑO",
            "CAMPAÑA",
            "SEMANA",
            "SEMANA ACTUAL",
            "AÑO-SEMANA ACTUAL",
            "FUNDO",
            "ETAPA",
            "CAMPO",
            "TURNO",
            "VARIEDAD",
            TARGET_COL,
            "DELTA_BW",
            "DELTA_BW_%",
            selected_x_col,
        ]
        display_cols = [c for c in display_cols if c in ts_df.columns]

        st.dataframe(
            ts_df[display_cols],
            use_container_width=True,
        )

# =========================================================
# TAB 4: ESTABILIDAD
# =========================================================
with tab4:
    st.subheader("Estabilidad de la relación")
    st.caption("Se evalúa la consistencia de la relación entre la variable objetivo y el lag seleccionado por CAMPAÑA y VARIEDAD.")

    stability_base = viz_df[[analysis_target, selected_x_col, "CAMPAÑA", "VARIEDAD"]].copy()

    stab_campaign = compute_group_stability(
        df=stability_base,
        group_col="CAMPAÑA",
        x_col=selected_x_col,
        y_col=analysis_target,
        min_obs=MIN_OBS_STABILITY,
    )

    stab_variety = compute_group_stability(
        df=stability_base,
        group_col="VARIEDAD",
        x_col=selected_x_col,
        y_col=analysis_target,
        min_obs=MIN_OBS_STABILITY,
    )

    c1, c2, c3, c4 = st.columns(4)

    valid_cam = stab_campaign["spearman"].notna().sum() if not stab_campaign.empty else 0
    valid_var = stab_variety["spearman"].notna().sum() if not stab_variety.empty else 0

    std_cam = stab_campaign["spearman"].std() if valid_cam > 1 else np.nan
    std_var = stab_variety["spearman"].std() if valid_var > 1 else np.nan

    c1.metric("Campañas evaluables", f"{valid_cam:,}")
    c2.metric("Variedades evaluables", f"{valid_var:,}")
    c3.metric("Desv. Spearman campaña", f"{std_cam:.4f}" if pd.notna(std_cam) else "NA")
    c4.metric("Desv. Spearman variedad", f"{std_var:.4f}" if pd.notna(std_var) else "NA")

    subtab1, subtab2 = st.tabs(["Por campaña", "Por variedad"])

    with subtab1:
        st.markdown("**Tabla de estabilidad por campaña**")
        if stab_campaign.empty:
            st.warning("No hay datos suficientes para estabilidad por campaña.")
        else:
            disp = stab_campaign.copy()
            for col in ["pearson", "spearman", "abs_pearson", "abs_spearman"]:
                if col in disp.columns:
                    disp[col] = disp[col].round(4)
            st.dataframe(disp, use_container_width=True)

            plot_df = stab_campaign.dropna(subset=["spearman"]).copy()
            if not plot_df.empty:
                fig = px.bar(
                    plot_df.sort_values("spearman", ascending=False),
                    x="CAMPAÑA",
                    y="spearman",
                    text="n",
                    title="Spearman por campaña",
                )
                fig.update_layout(height=500, yaxis_title="Spearman", xaxis_title="CAMPAÑA")
                st.plotly_chart(fig, use_container_width=True)

    with subtab2:
        st.markdown("**Tabla de estabilidad por variedad**")
        if stab_variety.empty:
            st.warning("No hay datos suficientes para estabilidad por variedad.")
        else:
            disp = stab_variety.copy()
            for col in ["pearson", "spearman", "abs_pearson", "abs_spearman"]:
                if col in disp.columns:
                    disp[col] = disp[col].round(4)
            st.dataframe(disp, use_container_width=True)

            plot_df = stab_variety.dropna(subset=["spearman"]).copy()
            if not plot_df.empty:
                fig = px.bar(
                    plot_df.sort_values("spearman", ascending=False),
                    x="VARIEDAD",
                    y="spearman",
                    text="n",
                    title="Spearman por variedad",
                )
                fig.update_layout(height=500, yaxis_title="Spearman", xaxis_title="VARIEDAD")
                st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 5: RESUMEN DE HALLAZGOS
# =========================================================
with tab5:
    st.subheader("Resumen de hallazgos")
    st.caption("Resumen cuantitativo para identificar qué variable y qué lag muestran la señal más fuerte.")

    findings_df = build_findings_summary(
        df=viz_df,
        analysis_target=analysis_target,
        base_vars=PHENOLOGY_COLS,
        max_lag=MAX_LAG,
    )

    if findings_df.empty:
        st.warning("No hay datos suficientes para construir el resumen de hallazgos.")
    else:
        disp = findings_df.copy()
        for col in ["pearson", "spearman", "abs_spearman"]:
            if col in disp.columns:
                disp[col] = disp[col].round(4)

        st.markdown("**Mejor lag por variable base**")
        st.dataframe(disp, use_container_width=True)

        top_row = findings_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variable con mayor señal", str(top_row["variable_base"]))
        c2.metric("Mejor lag", f"Lag {int(top_row['mejor_lag'])}")
        c3.metric("Spearman", f"{top_row['spearman']:.4f}" if pd.notna(top_row["spearman"]) else "NA")
        c4.metric("n", f"{int(top_row['n']):,}")

        fig = px.bar(
            findings_df.sort_values("abs_spearman", ascending=False),
            x="variable_base",
            y="abs_spearman",
            text="mejor_lag",
            title="Magnitud de señal por variable base (|Spearman| del mejor lag)",
        )
        fig.update_layout(
            height=500,
            xaxis_title="Variable base",
            yaxis_title="|Spearman|",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Resumen corto para ayudarte luego en Confluence
        st.markdown("**Lectura rápida**")
        best_name = str(top_row["variable_base"])
        best_lag = int(top_row["mejor_lag"])
        best_s = top_row["spearman"]
        st.info(
            f"La variable con mayor señal en la configuración actual es **{best_name}** "
            f"con **lag {best_lag}**, mostrando una correlación de Spearman de "
            f"**{best_s:.4f}**."
            if pd.notna(best_s)
            else f"La variable con mayor señal en la configuración actual es **{best_name}** con **lag {best_lag}**."
        )
