import io
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="EDA Delta semanal de peso de baya",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_FILE = "DELTA CONSOLIDADO 2022-2025.xlsx"
DEFAULT_SHEET = "DATA"
MAX_LAG = 12

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

# Grupo real para cálculos temporales
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
    "FRUTO CREMOSO",
    "FRUTO ROSADO",
    "FRUTO MADURO",
]

CONTROL_COLS = [
    "CALIBRE BAYA (mm)",
    "KG/HA",
    "DENSIDAD",
]

OPTIONAL_EXTRA_COLS = [
    "kilogramos",
    "Ha COSECHADA",
    "Ha TURNO",
    "TOTAL DE FRUTOS",
]

ALL_ANALYSIS_BASE_COLS = ID_COLS_DISPLAY + PHENOLOGY_COLS + CONTROL_COLS + [TARGET_COL]
KEEP_IF_EXISTS = ALL_ANALYSIS_BASE_COLS + OPTIONAL_EXTRA_COLS


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def find_excel_file(default_file: str) -> str:
    """
    Busca primero el archivo exacto en la raíz del repo.
    Si no existe, intenta encontrar un .xlsx disponible.
    """
    if default_file in glob.glob("*.xlsx"):
        return default_file

    files = glob.glob("*.xlsx")
    if files:
        return files[0]

    raise FileNotFoundError(
        f"No se encontró el archivo '{default_file}' ni ningún archivo .xlsx en la raíz del repositorio."
    )


def safe_iso_week_start(year, week):
    """
    Convierte (AÑO, SEMANA) a la fecha del lunes de esa semana ISO.
    Si la semana es inválida, devuelve NaT.
    """
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
            "Faltan columnas obligatorias en el archivo: "
            + ", ".join(missing)
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


def add_delta_and_lags(df: pd.DataFrame, lag_max: int = 12) -> pd.DataFrame:
    """
    Calcula:
    - DELTA_BW
    - DELTA_BW_%
    - lags de 1..lag_max para variables fenológicas y de control

    Reglas:
    - El cálculo es por ENTITY_COLS
    - Orden por CAMPAÑA + AÑO + SEMANA
    - Solo se calcula si la continuidad semanal es real:
      fecha_actual - fecha_pasada == 7 * lag días
    - Si falta semana intermedia, queda NaN
    - Si falta peso previo, delta queda NaN
    - Si peso previo <= 0, DELTA_BW_% queda NaN
    """
    df = df.copy()

    # Orden temporal dentro de la campaña
    df = df.sort_values(ENTITY_COLS + ["AÑO", "SEMANA"], kind="stable").reset_index(drop=True)

    # Fecha inicio de semana ISO
    df["WEEK_START_DATE"] = [
        safe_iso_week_start(y, w) for y, w in zip(df["AÑO"], df["SEMANA"])
    ]

    # Validación de semanas ISO
    invalid_weeks = df["WEEK_START_DATE"].isna().sum()
    if invalid_weeks > 0:
        st.warning(
            f"Se detectaron {invalid_weeks} filas con combinación AÑO/SEMANA inválida. "
            "Esas filas no podrán participar correctamente en cálculos temporales."
        )

    # Verificación de duplicados por grupo-semana
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

    # Delta absoluto y delta %
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

    # Variables para las que sí construiremos lags
    lag_vars = [c for c in PHENOLOGY_COLS + CONTROL_COLS if c in df.columns]

    for var in lag_vars:
        for lag in range(1, lag_max + 1):
            lag_val = g[var].shift(lag)
            lag_date = g["WEEK_START_DATE"].shift(lag)

            consecutive_lag = (
                df["WEEK_START_DATE"].notna()
                & lag_date.notna()
                & ((df["WEEK_START_DATE"] - lag_date) == pd.Timedelta(days=7 * lag))
            )

            new_col = f"{var}__LAG_{lag}"
            df[new_col] = np.where(
                consecutive_lag & lag_val.notna(),
                lag_val,
                np.nan,
            )

    df["ENTITY_KEY"] = build_entity_key(df)
    return df


def iqr_filter_mask(series: pd.Series) -> pd.Series:
    """
    Devuelve máscara True para valores NO outliers según IQR.
    NaN se devuelve como False para visuales que necesiten datos válidos.
    """
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
    """
    Agrega línea de tendencia lineal simple.
    """
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = x_num.notna() & y_num.notna()

    if valid.sum() < 2:
        return fig

    x_v = x_num[valid].values
    y_v = y_num[valid].values

    # Evita error si x tiene un solo valor único
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


def build_correlation_table(df: pd.DataFrame, target_col: str, base_vars: list[str], lag_max: int) -> pd.DataFrame:
    rows = []

    for var in base_vars:
        for lag in range(1, lag_max + 1):
            lag_col = f"{var}__LAG_{lag}"
            if lag_col not in df.columns:
                continue

            tmp = df[[target_col, lag_col]].dropna()
            n = len(tmp)
            if n < 3:
                continue

            pearson = tmp[target_col].corr(tmp[lag_col], method="pearson")
            spearman = tmp[target_col].corr(tmp[lag_col], method="spearman")

            rows.append(
                {
                    "variable_base": var,
                    "lag": lag,
                    "variable_lag": lag_col,
                    "n": n,
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["variable_base", "lag", "variable_lag", "n", "pearson", "spearman", "abs_spearman"])

    out = pd.DataFrame(rows).sort_values(
        ["abs_spearman", "n"], ascending=[False, False]
    ).reset_index(drop=True)
    return out


def build_stability_table(df: pd.DataFrame, x_col: str, y_col: str, group_col: str) -> pd.DataFrame:
    rows = []

    if group_col not in df.columns:
        return pd.DataFrame()

    for grp, sub in df.groupby(group_col, dropna=False):
        tmp = sub[[x_col, y_col]].dropna()
        n = len(tmp)
        if n < 3:
            continue

        rows.append(
            {
                group_col: grp,
                "n": n,
                "pearson": tmp[x_col].corr(tmp[y_col], method="pearson"),
                "spearman": tmp[x_col].corr(tmp[y_col], method="spearman"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[group_col, "n", "pearson", "spearman"])

    return pd.DataFrame(rows).sort_values(["n", "spearman"], ascending=[False, False]).reset_index(drop=True)


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "DATA") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.read()


@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    file_path = find_excel_file(DEFAULT_FILE)
    df = pd.read_excel(file_path, sheet_name=DEFAULT_SHEET, engine="openpyxl")

    # Limpieza básica de nombres y textos
    df.columns = [str(c).strip() for c in df.columns]
    df = normalize_text_cols(df, ENTITY_COLS + ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"])

    required_cols = list(set(ID_COLS_DISPLAY + PHENOLOGY_COLS + CONTROL_COLS + [TARGET_COL]))
    validate_columns(df, required_cols)

    # Mantener columnas clave + extras si existen
    keep_cols = [c for c in KEEP_IF_EXISTS if c in df.columns]
    df = df[keep_cols].copy()

    # Conversión numérica
    numeric_cols = [
        "AÑO",
        "SEMANA",
        TARGET_COL,
        *PHENOLOGY_COLS,
        *CONTROL_COLS,
        *OPTIONAL_EXTRA_COLS,
    ]
    df = to_numeric_safe(df, [c for c in numeric_cols if c in df.columns])

    # Cálculo analítico
    df = add_delta_and_lags(df, lag_max=MAX_LAG)

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

st.success(f"Archivo cargado correctamente: {loaded_file} | Hoja: {DEFAULT_SHEET}")

with st.expander("Verificación técnica del enfoque usado", expanded=False):
    st.markdown(
        """
**Cálculo temporal aplicado**
- Los deltas y lags se calculan por: `CAMPAÑA + FUNDO + ETAPA + CAMPO + TURNO + VARIEDAD`
- El orden temporal es: `CAMPAÑA -> AÑO -> SEMANA`
- `AÑO` se conserva como identificador y filtro, pero no rompe la continuidad temporal dentro de la campaña
- Solo se calcula delta/lag si la semana previa existe realmente y es consecutiva
- Si falta una semana intermedia, el valor queda vacío
- `DELTA_BW_%` queda vacío si el peso previo es nulo o igual a 0
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

base_analysis_vars = PHENOLOGY_COLS + CONTROL_COLS
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

st.sidebar.markdown("---")
st.sidebar.write("Descargas")

# Dataset analítico completo filtrado
csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
xlsx_bytes = to_excel_bytes(filtered, sheet_name="ANALITICO_FILTRADO")

st.sidebar.download_button(
    "Descargar dataset analítico filtrado (CSV)",
    data=csv_bytes,
    file_name="dataset_analitico_filtrado.csv",
    mime="text/csv",
)

st.sidebar.download_button(
    "Descargar dataset analítico filtrado (Excel)",
    data=xlsx_bytes,
    file_name="dataset_analitico_filtrado.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# Base para visuales
viz_df = filtered.copy()
if selected_x_col not in viz_df.columns:
    st.error(f"No existe la columna seleccionada para análisis: {selected_x_col}")
    st.stop()

viz_df = viz_df[[*ID_COLS_DISPLAY, "ENTITY_KEY", TARGET_COL, "DELTA_BW", "DELTA_BW_%", selected_x_col]].copy()

if outlier_mode == "Todos excepto outliers":
    valid_mask = iqr_filter_mask(viz_df[analysis_target])
    viz_df = viz_df[valid_mask].copy()

# =========================================================
# RESUMEN
# =========================================================
st.subheader("Resumen general")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Filas filtradas", f"{len(filtered):,}")
c2.metric("Grupos únicos", f"{filtered['ENTITY_KEY'].nunique():,}")
c3.metric("Con peso válido", f"{filtered[TARGET_COL].notna().sum():,}")
c4.metric("Delta válido", f"{filtered['DELTA_BW'].notna().sum():,}")
c5.metric("Delta % válido", f"{filtered['DELTA_BW_%'].notna().sum():,}")
c6.metric("Filas en visual actual", f"{len(viz_df.dropna(subset=[analysis_target, selected_x_col])):,}")

missing_summary = pd.DataFrame(
    {
        "variable": [TARGET_COL] + PHENOLOGY_COLS + CONTROL_COLS + ["DELTA_BW", "DELTA_BW_%"],
        "faltantes_%": [
            filtered[TARGET_COL].isna().mean() * 100,
            *[(filtered[c].isna().mean() * 100) for c in PHENOLOGY_COLS],
            *[(filtered[c].isna().mean() * 100) for c in CONTROL_COLS],
            filtered["DELTA_BW"].isna().mean() * 100,
            filtered["DELTA_BW_%"].isna().mean() * 100,
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
        "Correlaciones",
        "Estabilidad",
        "Series temporales",
    ]
)

# =========================================================
# TAB 1: SCATTER
# =========================================================
with tab1:
    st.subheader("Scatterplot")

    scatter_df = viz_df[[selected_x_col, analysis_target, "CAMPAÑA", "VARIEDAD", "CAMPO", "FUNDO", "ENTITY_KEY"]].dropna()

    if scatter_df.empty:
        st.warning("No hay datos suficientes para el scatterplot con la selección actual.")
    else:
        fig = px.scatter(
    scatter_df,
    x=selected_x_col,
    y=analysis_target,
    color="VARIEDAD",
    hover_data=["CAMPAÑA", "FUNDO", "CAMPO", "ENTITY_KEY"],
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

    box_df = viz_df[[selected_x_col, analysis_target]].dropna().copy()

    if len(box_df) < 10:
        st.warning("No hay suficientes datos para construir bins interpretables.")
    else:
        try:
            n_bins = st.slider("Número de bins", min_value=4, max_value=10, value=5, key="bins_slider")
            box_df["BIN_X"] = pd.qcut(
                box_df[selected_x_col],
                q=n_bins,
                duplicates="drop",
            ).astype(str)

            fig = px.box(
                box_df,
                x="BIN_X",
                y=analysis_target,
                points="outliers",
            )
            fig.update_layout(
                xaxis_title=f"Bins de {selected_x_col}",
                yaxis_title=analysis_target,
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

            summary_box = (
                box_df.groupby("BIN_X", dropna=False)[analysis_target]
                .agg(["count", "mean", "median", "std"])
                .reset_index()
            )
            st.dataframe(summary_box, use_container_width=True)
        except Exception as e:
            st.warning(f"No fue posible generar el boxplot por bins: {e}")

# =========================================================
# TAB 3: CORRELACIONES
# =========================================================
with tab3:
    st.subheader("Ranking de correlaciones por variable y lag")

    corr_base_df = filtered.copy()
    if outlier_mode == "Todos excepto outliers":
        mask = iqr_filter_mask(corr_base_df[analysis_target])
        corr_base_df = corr_base_df[mask].copy()

    corr_table = build_correlation_table(
        corr_base_df,
        target_col=analysis_target,
        base_vars=base_analysis_vars,
        lag_max=MAX_LAG,
    )

    if corr_table.empty:
        st.warning("No hay suficientes datos para calcular correlaciones.")
    else:
        st.dataframe(corr_table, use_container_width=True)

        top_corr = corr_table.head(20).copy()

        fig = px.bar(
            top_corr,
            x="variable_lag",
            y="spearman",
            hover_data=["n", "pearson"],
        )
        fig.update_layout(
            xaxis_title="Variable rezagada",
            yaxis_title="Spearman",
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4: ESTABILIDAD
# =========================================================
with tab4:
    st.subheader("Estabilidad de la relación observada")

    stability_group = st.selectbox(
        "Evaluar estabilidad por",
        options=["CAMPAÑA", "VARIEDAD", "CAMPO", "FUNDO", "ETAPA", "TURNO"],
        index=0,
    )

    stab_df = filtered[[stability_group, selected_x_col, analysis_target]].copy().dropna()

    if outlier_mode == "Todos excepto outliers" and not stab_df.empty:
        mask = iqr_filter_mask(stab_df[analysis_target])
        stab_df = stab_df[mask].copy()

    stab_table = build_stability_table(
        stab_df,
        x_col=selected_x_col,
        y_col=analysis_target,
        group_col=stability_group,
    )

    if stab_table.empty:
        st.warning("No hay suficientes datos para evaluar estabilidad con la selección actual.")
    else:
        st.dataframe(stab_table, use_container_width=True)

        top_n = min(20, len(stab_table))
        fig = px.bar(
            stab_table.head(top_n),
            x=stability_group,
            y="spearman",
            hover_data=["n", "pearson"],
        )
        fig.update_layout(
            xaxis_title=stability_group,
            yaxis_title="Spearman",
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 5: SERIES TEMPORALES
# =========================================================
with tab5:
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
        ts_df["EJE_TIEMPO"] = ts_df["AÑO"].astype("Int64").astype(str) + "-W" + ts_df["SEMANA"].astype("Int64").astype(str).str.zfill(2)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ts_df["EJE_TIEMPO"],
                y=ts_df[TARGET_COL],
                mode="lines+markers",
                name=TARGET_COL,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ts_df["EJE_TIEMPO"],
                y=ts_df["DELTA_BW"],
                mode="lines+markers",
                name="DELTA_BW",
                yaxis="y2",
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

        st.dataframe(
            ts_df[
                [
                    "AÑO",
                    "CAMPAÑA",
                    "SEMANA",
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
            ],
            use_container_width=True,
        )
