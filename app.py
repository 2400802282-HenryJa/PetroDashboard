# ============================================================
# PETRODASHBOARD
# Premium Petroleum Engineering & Well Log Analytics Platform
# ============================================================

import sqlite3
from io import StringIO
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PetroDashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. APPLICATION CONSTANTS
# ============================================================

APP_TITLE = "PETRODASHBOARD"

APP_SUBTITLE = (
    "Petroleum Engineering • Well Log Interpretation • "
    "Formation Evaluation • Subsurface Analytics"
)

DATABASE_FILE = "petroleum_dashboard.db"

MAX_VISUAL_POINTS = 4000

RESISTIVITY_KEYWORDS = [
    "RT",
    "RES",
    "ILD",
    "LLD",
    "RDEP",
    "AT90",
    "AT60",
    "RESD",
    "LLS",
    "MSFL",
    "RXO",
    "RILD",
]

GR_CANDIDATES = [
    "GR",
    "GAM",
    "GRC",
    "SGR",
    "CGR",
]

RHOB_CANDIDATES = [
    "RHOB",
    "RHOZ",
    "DEN",
    "DENS",
]

NPHI_CANDIDATES = [
    "NPHI",
    "NPHL",
    "TNPH",
    "NEUT",
]

DEPTH_CANDIDATES = [
    "DEPT",
    "DEPTH",
    "MD",
    "MEASURED_DEPTH",
    "TDEP",
]


# ============================================================
# 3. DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / DATABASE_FILE


# ============================================================
# 4. PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------
       GLOBAL
    ------------------------------------------------------ */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(0, 140, 170, 0.12),
                transparent 35%
            ),
            #071116;
        color: #E8F1F5;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ------------------------------------------------------
       HEADER
    ------------------------------------------------------ */

    .petro-header {
        background:
            linear-gradient(
                135deg,
                #0B2028 0%,
                #0E313D 55%,
                #0A1A20 100%
            );

        border: 1px solid rgba(96, 196, 222, 0.18);
        border-radius: 18px;

        padding: 24px 28px;
        margin-bottom: 20px;

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.28);
    }

    .petro-header-inner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
    }

    .petro-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #F3FAFC;
    }

    .petro-subtitle {
        margin-top: 6px;
        color: #91AEB8;
        font-size: 13px;
        letter-spacing: 0.3px;
    }

    .status-online {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;

        background: rgba(38, 202, 135, 0.12);
        border: 1px solid rgba(38, 202, 135, 0.35);

        color: #52E0A1;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        white-space: nowrap;
    }

    /* ------------------------------------------------------
       SECTION HEADERS
    ------------------------------------------------------ */

    .section-label {
        color: #74CFE8;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .section-title {
        color: #F0F7F9;
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* ------------------------------------------------------
       KPI CARDS
    ------------------------------------------------------ */

    .kpi-card {
        background:
            linear-gradient(
                145deg,
                rgba(17, 48, 59, 0.95),
                rgba(8, 26, 33, 0.95)
            );

        border: 1px solid rgba(111, 205, 228, 0.16);
        border-radius: 15px;

        padding: 15px 18px;

        min-height: 90px;

        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.18);
    }

    .kpi-label {
        color: #7896A0;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .kpi-value {
        color: #F3FAFC;
        font-size: 24px;
        font-weight: 800;
        margin-top: 6px;
    }

    .kpi-caption {
        color: #6D8993;
        font-size: 10px;
        margin-top: 2px;
    }

    /* ------------------------------------------------------
       INFO CARDS
    ------------------------------------------------------ */

    .info-card {
        background: rgba(11, 32, 40, 0.82);
        border: 1px solid rgba(111, 205, 228, 0.12);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }

    .info-label {
        color: #6F8D97;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-value {
        color: #EAF5F8;
        font-size: 16px;
        font-weight: 700;
        margin-top: 4px;
        word-break: break-word;
    }

    /* ------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------ */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #07161C 0%,
                #091E26 100%
            );
        border-right: 1px solid rgba(111, 205, 228, 0.10);
    }

    /* ------------------------------------------------------
       FILE UPLOADER
    ------------------------------------------------------ */

    [data-testid="stFileUploader"] {
        background: rgba(12, 39, 48, 0.55);
        border-radius: 14px;
    }

    /* ------------------------------------------------------
       METRIC
    ------------------------------------------------------ */

    [data-testid="stMetric"] {
        background: rgba(10, 31, 39, 0.70);
        border: 1px solid rgba(111, 205, 228, 0.12);
        padding: 12px;
        border-radius: 12px;
    }

    /* ------------------------------------------------------
       BUTTONS
    ------------------------------------------------------ */

    .stButton > button {
        border-radius: 9px;
        border: 1px solid rgba(83, 202, 232, 0.25);
        background: #0D3440;
        color: #EAF7FA;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #55D5EF;
        color: #FFFFFF;
        background: #124653;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="petro-header">
        <div class="petro-header-inner">

            <div>
                <div class="petro-title">
                    🛢️ PETRODASHBOARD
                </div>

                <div class="petro-subtitle">
                    Petroleum Engineering • Well Log Interpretation •
                    Formation Evaluation • Subsurface Analytics
                </div>
            </div>

            <div>
                <span class="status-online">
                    ● SYSTEM ONLINE
                </span>
            </div>

        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS wells (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            well_name TEXT,
            company TEXT,
            depth_samples INTEGER,
            average_vsh REAL,
            net_to_gross REAL
        )
        """
    )

    conn.commit()
    conn.close()


def save_well_analysis(
    well_name,
    company,
    depth_samples,
    average_vsh,
    net_to_gross,
):
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO wells (
            well_name,
            company,
            depth_samples,
            average_vsh,
            net_to_gross
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(well_name),
            str(company),
            int(depth_samples),
            float(average_vsh),
            float(net_to_gross),
        ),
    )

    conn.commit()
    conn.close()


def load_saved_wells():
    conn = sqlite3.connect(DB_PATH)

    wells = pd.read_sql_query(
        """
        SELECT
            id,
            well_name,
            company,
            depth_samples,
            average_vsh,
            net_to_gross
        FROM wells
        ORDER BY id DESC
        """,
        conn,
    )

    conn.close()

    return wells


# ============================================================
# 7. LAS PROCESSING
# ============================================================

def load_las_file(uploaded_file):
    raw_bytes = uploaded_file.getvalue()

    las_string = raw_bytes.decode(
        "utf-8",
        errors="ignore",
    )

    return lasio.read(
        StringIO(las_string),
        engine="normal",
    )


def las_to_dataframe(las):
    df = las.df().reset_index()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def get_header_value(las, mnemonic, default="Unknown"):
    try:
        item = las.well[mnemonic]

        value = getattr(item, "value", item)

        if value is None or str(value).strip() == "":
            return default

        return str(value).strip()

    except Exception:
        return default


def get_well_information(las):
    well_name = get_header_value(
        las,
        "WELL",
        "Unknown Well",
    )

    company = get_header_value(
        las,
        "COMP",
        "Unknown Company",
    )

    return well_name, company


# ============================================================
# 8. CURVE DETECTION
# ============================================================

def find_curve(df, candidates):
    normalized_columns = {
        str(column).upper().strip(): column
        for column in df.columns
    }

    for candidate in candidates:

        candidate_upper = candidate.upper()

        if candidate_upper in normalized_columns:
            return normalized_columns[candidate_upper]

    for column in df.columns:

        column_upper = str(column).upper()

        for candidate in candidates:

            if candidate.upper() in column_upper:
                return column

    return None


def get_depth_column(df):
    exact = find_curve(
        df,
        DEPTH_CANDIDATES,
    )

    if exact is not None:
        return exact

    return df.columns[0]


def find_resistivity_curve(df):
    return find_curve(
        df,
        RESISTIVITY_KEYWORDS,
    )


# ============================================================
# 9. NUMERIC CURVE CLEANING
# ============================================================

def numeric_curve(df, column):
    if column is None or column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# 10. FORMATION EVALUATION
# ============================================================

def calculate_vsh(df, gr_column):
    if gr_column is None:
        df["VSH"] = np.nan
        return df, 0.0

    gr = numeric_curve(
        df,
        gr_column,
    )

    valid = gr.dropna()

    if valid.empty:
        df["VSH"] = np.nan
        return df, 0.0

    gr_min = float(valid.min())
    gr_max = float(valid.max())

    if gr_max == gr_min:
        df["VSH"] = 0.0
        return df, 0.0

    df["VSH"] = (
        (gr - gr_min)
        /
        (gr_max - gr_min)
    ).clip(0, 1)

    average_vsh = float(
        df["VSH"].mean()
    )

    return df, average_vsh


def calculate_reservoir_flag(
    df,
    vsh_cutoff,
):
    if "VSH" not in df.columns:
        df["RES_FLAG"] = 0
        return df

    df["RES_FLAG"] = np.where(
        df["VSH"] < vsh_cutoff,
        1,
        0,
    )

    return df


def calculate_ntg(df):
    if "RES_FLAG" not in df.columns:
        return 0.0

    return float(
        df["RES_FLAG"].mean()
    )


# ============================================================
# 11. VISUALIZATION DECIMATION
# ============================================================

def min_max_decimate(
    df,
    x_column,
    depth_column,
    max_points=MAX_VISUAL_POINTS,
):
    if (
        x_column not in df.columns
        or depth_column not in df.columns
    ):
        return df.copy()

    work = df[
        [
            depth_column,
            x_column,
        ]
    ].copy()

    work[x_column] = pd.to_numeric(
        work[x_column],
        errors="coerce",
    )

    work[depth_column] = pd.to_numeric(
        work[depth_column],
        errors="coerce",
    )

    work = work.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    work = work.dropna(
        subset=[
            depth_column,
            x_column,
        ]
    )

    if len(work) <= max_points:
        return work.sort_values(
            depth_column
        )

    bucket_count = max(
        1,
        max_points // 2,
    )

    boundaries = np.linspace(
        0,
        len(work),
        bucket_count + 1,
        dtype=int,
    )

    selected = []

    for index in range(
        len(boundaries) - 1
    ):

        start = boundaries[index]
        end = boundaries[index + 1]

        section = work.iloc[
            start:end
        ]

        if section.empty:
            continue

        minimum = section[
            x_column
        ].idxmin()

        maximum = section[
            x_column
        ].idxmax()

        selected.extend(
            [
                minimum,
                maximum,
            ]
        )

    selected = list(
        dict.fromkeys(selected)
    )

    result = work.loc[
        selected
    ].sort_values(
        depth_column
    )

    return result


# ============================================================
# 12. GAUGE COMPONENT
# ============================================================

def create_gauge(
    title,
    value,
    maximum,
    suffix="",
    bar_color="#43D17B",
):
    safe_value = float(
        value
        if pd.notna(value)
        else 0
    )

    safe_maximum = max(
        float(maximum),
        1.0,
    )

    safe_value = min(
        max(safe_value, 0),
        safe_maximum,
    )

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=safe_value,
            title={
                "text": title,
                "font": {
                    "size": 13,
                    "color": "#9BB4BC",
                },
            },
            number={
                "suffix": suffix,
                "font": {
                    "size": 25,
                    "color": "#F2F8FA",
                },
            },
            gauge={
                "axis": {
                    "range": [
                        0,
                        safe_maximum,
                    ],
                    "tickwidth": 1,
                    "tickcolor": "#6D8790",
                },
                "bar": {
                    "color": bar_color,
                    "thickness": 0.28,
                },
                "bgcolor": "#14262D",
                "borderwidth": 1,
                "bordercolor": "#29444D",
                "steps": [
                    {
                        "range": [
                            0,
                            safe_maximum * 0.35,
                        ],
                        "color": "#101D22",
                    },
                    {
                        "range": [
                            safe_maximum * 0.35,
                            safe_maximum * 0.70,
                        ],
                        "color": "#16272E",
                    },
                    {
                        "range": [
                            safe_maximum * 0.70,
                            safe_maximum,
                        ],
                        "color": "#1B3038",
                    },
                ],
            },
        )
    )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=235,
        margin=dict(
            l=20,
            r=20,
            t=45,
            b=15,
        ),
    )

    return figure


# ============================================================
# 13. PETREL-STYLE WELL LOG CANVAS
# ============================================================

@st.fragment
def render_well_log_canvas(
    df,
    depth_column,
    gr_column,
    rhob_column,
    nphi_column,
    gr_cutoff,
    vsh_cutoff,
    resistivity_curve,
):
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        horizontal_spacing=0.018,
        subplot_titles=(
            "GAMMA RAY",
            "RESISTIVITY",
            "DENSITY / NEUTRON",
            "VSH",
        ),
    )

    # --------------------------------------------------------
    # GAMMA RAY
    # --------------------------------------------------------

    if gr_column is not None:

        gr_df = min_max_decimate(
            df,
            gr_column,
            depth_column,
        )

        fig.add_trace(
            go.Scatter(
                x=gr_df[gr_column],
                y=gr_df[depth_column],
                mode="lines",
                name="Gamma Ray",
                line=dict(
                    color="#39D98A",
                    width=1.5,
                ),
                hovertemplate=(
                    "<b>Depth:</b> %{y:.2f}"
                    "<br><b>GR:</b> %{x:.2f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        fig.add_vline(
            x=gr_cutoff,
            line_width=1.2,
            line_dash="dash",
            line_color="#FF5C5C",
            row=1,
            col=1,
        )

    # --------------------------------------------------------
    # RESISTIVITY
    # --------------------------------------------------------

    if resistivity_curve is not None:

        res_df = min_max_decimate(
            df,
            resistivity_curve,
            depth_column,
        )

        fig.add_trace(
            go.Scatter(
                x=res_df[resistivity_curve],
                y=res_df[depth_column],
                mode="lines",
                name="Resistivity",
                line=dict(
                    color="#FF6B6B",
                    width=1.5,
                ),
                hovertemplate=(
                    "<b>Depth:</b> %{y:.2f}"
                    "<br><b>Resistivity:</b> %{x:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(
            type="log",
            row=1,
            col=2,
        )

    # --------------------------------------------------------
    # RHOB
    # --------------------------------------------------------

    if rhob_column is not None:

        rhob_df = min_max_decimate(
            df,
            rhob_column,
            depth_column,
        )

        fig.add_trace(
            go.Scatter(
                x=rhob_df[rhob_column],
                y=rhob_df[depth_column],
                mode="lines",
                name="RHOB",
                line=dict(
                    color="#43A5FF",
                    width=1.5,
                ),
                hovertemplate=(
                    "<b>Depth:</b> %{y:.2f}"
                    "<br><b>RHOB:</b> %{x:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=3,
        )

    # --------------------------------------------------------
    # NPHI
    # --------------------------------------------------------

    if nphi_column is not None:

        nphi_df = min_max_decimate(
            df,
            nphi_column,
            depth_column,
        )

        fig.add_trace(
            go.Scatter(
                x=nphi_df[nphi_column],
                y=nphi_df[depth_column],
                mode="lines",
                name="NPHI",
                line=dict(
                    color="#B477FF",
                    width=1.5,
                ),
                hovertemplate=(
                    "<b>Depth:</b> %{y:.2f}"
                    "<br><b>NPHI:</b> %{x:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=3,
        )

    # --------------------------------------------------------
    # VSH
    # --------------------------------------------------------

    if "VSH" in df.columns:

        vsh_df = min_max_decimate(
            df,
            "VSH",
            depth_column,
        )

        fig.add_trace(
            go.Scatter(
                x=vsh_df["VSH"],
                y=vsh_df[depth_column],
                mode="lines",
                name="VSH",
                line=dict(
                    color="#F2F4F5",
                    width=1.6,
                ),
                hovertemplate=(
                    "<b>Depth:</b> %{y:.2f}"
                    "<br><b>VSH:</b> %{x:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=4,
        )

        fig.add_vline(
            x=vsh_cutoff,
            line_width=1.2,
            line_dash="dash",
            line_color="#FFB84D",
            row=1,
            col=4,
        )

    # --------------------------------------------------------
    # SHARED DEPTH
    # --------------------------------------------------------

    fig.update_yaxes(
        autorange="reversed",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#8FE8FF",
        spikethickness=1,
        gridcolor="rgba(120,160,170,0.12)",
        zeroline=False,
    )

    # --------------------------------------------------------
    # X AXES
    # --------------------------------------------------------

    fig.update_xaxes(
        title_text="GR",
        gridcolor="rgba(120,160,170,0.12)",
        zeroline=False,
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Resistivity",
        gridcolor="rgba(120,160,170,0.12)",
        zeroline=False,
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="RHOB / NPHI",
        gridcolor="rgba(120,160,170,0.12)",
        zeroline=False,
        row=1,
        col=3,
    )

    fig.update_xaxes(
        title_text="VSH",
        range=[0, 1],
        gridcolor="rgba(120,160,170,0.12)",
        zeroline=False,
        row=1,
        col=4,
    )

    # --------------------------------------------------------
    # PLOTLY WORKSPACE
    # --------------------------------------------------------

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#08151B",
        plot_bgcolor="#0A1B22",
        hovermode="y unified",
        height=850,
        margin=dict(
            l=55,
            r=30,
            t=65,
            b=45,
        ),
        dragmode="pan",
        showlegend=False,
        font=dict(
            family="Arial",
            color="#D9E7EB",
        ),
    )

    # Hide internal depth labels
    for column in range(2, 5):

        fig.update_yaxes(
            showticklabels=False,
            row=1,
            col=column,
        )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
            ],
        },
    )


# ============================================================
# 14. AI LITHOLOGY
# ============================================================

def run_ai_lithology(
    df,
    gr_column,
    rhob_column,
    nphi_column,
    gr_cutoff,
):
    required = [
        gr_column,
        rhob_column,
        nphi_column,
    ]

    if any(
        column is None
        for column in required
    ):
        return None, None, (
            "GR, RHOB and NPHI curves are required "
            "for the AI prototype."
        )

    ai_df = pd.DataFrame(
        {
            "GR": numeric_curve(
                df,
                gr_column,
            ),
            "RHOB": numeric_curve(
                df,
                rhob_column,
            ),
            "NPHI": numeric_curve(
                df,
                nphi_column,
            ),
        }
    )

    ai_df = ai_df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(ai_df) < 20:
        return None, None, (
            "Not enough valid samples for AI training."
        )

    ai_df["Lithology"] = np.where(
        ai_df["GR"] < gr_cutoff,
        1,
        0,
    )

    X = ai_df[
        [
            "GR",
            "RHOB",
            "NPHI",
        ]
    ]

    y = ai_df["Lithology"]

    if y.nunique() < 2:
        return None, None, (
            "Only one lithology class exists. "
            "Adjust the GR interpretation cutoff."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_test,
    )

    accuracy = accuracy_score(
        y_test,
        prediction,
    )

    ai_df["Predicted_Lithology"] = model.predict(
        X,
    )

    return (
        model,
        ai_df,
        accuracy,
    )


def display_ai_lithology(
    df,
    gr_column,
    rhob_column,
    nphi_column,
    gr_cutoff,
):
    st.markdown(
        '<div class="section-label">AI SUBSURFACE ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">🤖 Lithology Classification Prototype</div>',
        unsafe_allow_html=True,
    )

    result = run_ai_lithology(
        df,
        gr_column,
        rhob_column,
        nphi_column,
        gr_cutoff,
    )

    if result[0] is None:
        st.warning(result[2])
        return

    model, ai_df, accuracy = result

    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(
            create_gauge(
                "Prototype Accuracy",
                accuracy * 100,
                100,
                suffix="%",
                bar_color="#43D17B",
            ),
            width="stretch",
        )

    with col2:
        sand_count = int(
            (
                ai_df["Predicted_Lithology"] == 1
            ).sum()
        )

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    Predicted Sand Samples
                </div>
                <div class="kpi-value">
                    {sand_count:,}
                </div>
                <div class="kpi-caption">
                    AI classification output
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        shale_count = int(
            (
                ai_df["Predicted_Lithology"] == 0
            ).sum()
        )

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    Predicted Shale Samples
                </div>
                <div class="kpi-value">
                    {shale_count:,}
                </div>
                <div class="kpi-caption">
                    AI classification output
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Educational prototype: training labels are derived "
        "from the Gamma Ray cutoff and are not equivalent to "
        "expert-labelled core or facies data."
    )


# ============================================================
# 15. MULTI-WELL ANALYSIS
# ============================================================

def display_multi_well_analysis():
    st.markdown(
        '<div class="section-label">SUBSURFACE DATABASE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">🗄️ Multi-Well Workspace</div>',
        unsafe_allow_html=True,
    )

    wells = load_saved_wells()

    if wells.empty:
        st.info(
            "No well analyses have been saved yet."
        )
        return

    # --------------------------------------------------------
    # Arrow-safe display copy
    # --------------------------------------------------------

    display_wells = wells.copy()

    display_wells["average_vsh"] = display_wells[
        "average_vsh"
    ].round(3)

    display_wells["net_to_gross"] = display_wells[
        "net_to_gross"
    ].round(3)

    st.dataframe(
        display_wells,
        width="stretch",
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        ntg_fig = go.Figure()

        ntg_fig.add_trace(
            go.Bar(
                x=wells["well_name"].astype(str),
                y=wells["net_to_gross"],
                name="NTG",
                marker_color="#43D17B",
            )
        )

        ntg_fig.update_layout(
            title="Net-to-Gross Comparison",
            xaxis_title="Well",
            yaxis_title="NTG",
            paper_bgcolor="#08151B",
            plot_bgcolor="#0A1B22",
            font=dict(color="#D9E7EB"),
            height=380,
        )

        st.plotly_chart(
            ntg_fig,
            width="stretch",
        )

    with col2:

        vsh_fig = go.Figure()

        vsh_fig.add_trace(
            go.Bar(
                x=wells["well_name"].astype(str),
                y=wells["average_vsh"],
                name="Average VSH",
                marker_color="#43A5FF",
            )
        )

        vsh_fig.update_layout(
            title="Average VSH Comparison",
            xaxis_title="Well",
            yaxis_title="Average VSH",
            paper_bgcolor="#08151B",
            plot_bgcolor="#0A1B22",
            font=dict(color="#D9E7EB"),
            height=380,
        )

        st.plotly_chart(
            vsh_fig,
            width="stretch",
        )


# ============================================================
# 16. INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# 17. SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:18px;
        font-weight:800;
        color:#EAF7FA;
        letter-spacing:1px;
        margin-bottom:4px;
    ">
        PETRO WORKSPACE
    </div>

    <div style="
        font-size:11px;
        color:#75919A;
        margin-bottom:20px;
    ">
        Formation evaluation controls
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "### Interpretation Parameters"
)

gr_cutoff = st.sidebar.number_input(
    "Gamma Ray Sand/Shale Cutoff",
    min_value=0.0,
    max_value=250.0,
    value=75.0,
    step=1.0,
)

vsh_cutoff = st.sidebar.number_input(
    "VSH Reservoir Cutoff",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
    format="%.2f",
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload LAS Well Log",
    type=["las"],
)

st.sidebar.caption(
    "Supported format: LAS well-log files"
)


# ============================================================
# 18. MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # LOAD LAS
        # ----------------------------------------------------

        las = load_las_file(
            uploaded_file
        )

        df = las_to_dataframe(
            las
        )

        depth_column = get_depth_column(
            df
        )

        gr_column = find_curve(
            df,
            GR_CANDIDATES,
        )

        rhob_column = find_curve(
            df,
            RHOB_CANDIDATES,
        )

        nphi_column = find_curve(
            df,
            NPHI_CANDIDATES,
        )

        resistivity_curve = find_resistivity_curve(
            df
        )

        well_name, company = get_well_information(
            las
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        st.success(
            f"LAS loaded successfully • {well_name}"
        )

        # ----------------------------------------------------
        # WELL INFORMATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-label">WELL IDENTIFICATION</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">🏭 Well Overview</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">Well</div>
                    <div class="info-value">
                        {well_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">Operator / Company</div>
                    <div class="info-value">
                        {company}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">Depth Samples</div>
                    <div class="info-value">
                        {len(df):,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # CURVE STATUS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-label">LOG CURVE INVENTORY</div>',
            unsafe_allow_html=True,
        )

        curve_col1, curve_col2, curve_col3, curve_col4 = st.columns(4)

        with curve_col1:
            st.metric(
                "Gamma Ray",
                str(gr_column)
                if gr_column
                else "Not Found",
            )

        with curve_col2:
            st.metric(
                "Resistivity",
                str(resistivity_curve)
                if resistivity_curve
                else "Not Found",
            )

        with curve_col3:
            st.metric(
                "Density",
                str(rhob_column)
                if rhob_column
                else "Not Found",
            )

        with curve_col4:
            st.metric(
                "Neutron",
                str(nphi_column)
                if nphi_column
                else "Not Found",
            )

        # ----------------------------------------------------
        # FORMATION EVALUATION
        # ----------------------------------------------------

        df, average_vsh = calculate_vsh(
            df,
            gr_column,
        )

        df = calculate_reservoir_flag(
            df,
            vsh_cutoff,
        )

        ntg = calculate_ntg(
            df,
        )

        # ----------------------------------------------------
        # HIGH-DENSITY KPI / GAUGE AREA
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-label">FORMATION EVALUATION</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">📡 Reservoir Interpretation Spectrum</div>',
            unsafe_allow_html=True,
        )

        gauge1, gauge2, gauge3 = st.columns(3)

        with gauge1:

            vsh_color = (
                "#43D17B"
                if average_vsh < 0.35
                else "#F2A93B"
                if average_vsh < 0.60
                else "#FF5C5C"
            )

            st.plotly_chart(
                create_gauge(
                    "Average VSH",
                    average_vsh * 100,
                    100,
                    suffix="%",
                    bar_color=vsh_color,
                ),
                width="stretch",
            )

        with gauge2:

            ntg_color = (
                "#FF5C5C"
                if ntg < 0.30
                else "#F2A93B"
                if ntg < 0.60
                else "#43D17B"
            )

            st.plotly_chart(
                create_gauge(
                    "Net-to-Gross",
                    ntg * 100,
                    100,
                    suffix="%",
                    bar_color=ntg_color,
                ),
                width="stretch",
            )

        with gauge3:

            depth_samples = len(df)

            st.plotly_chart(
                create_gauge(
                    "Log Sample Density",
                    min(
                        depth_samples,
                        10000,
                    ),
                    10000,
                    suffix="",
                    bar_color="#43A5FF",
                ),
                width="stretch",
            )

        # ----------------------------------------------------
        # MAIN WELL LOG CANVAS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-label">INTERACTIVE SUBSURFACE CANVAS</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">🖥️ Petrel-Style Well Log Workspace</div>',
            unsafe_allow_html=True,
        )

        render_well_log_canvas(
            df=df,
            depth_column=depth_column,
            gr_column=gr_column,
            rhob_column=rhob_column,
            nphi_column=nphi_column,
            gr_cutoff=gr_cutoff,
            vsh_cutoff=vsh_cutoff,
            resistivity_curve=resistivity_curve,
        )

        # ----------------------------------------------------
        # FORMATION SUMMARY
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-label">INTERPRETATION SUMMARY</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">📋 Formation Evaluation Summary</div>',
            unsafe_allow_html=True,
        )

        # IMPORTANT:
        # Every Value is deliberately converted to STRING.
        # This prevents the PyArrow:
        # "Expected bytes, got a int object"
        # error caused by mixed object types.

        summary_df = pd.DataFrame(
            {
                "Parameter": [
                    "Well",
                    "Company",
                    "Depth Samples",
                    "Depth Curve",
                    "Gamma Ray Curve",
                    "Resistivity Curve",
                    "Density Curve",
                    "Neutron Curve",
                    "GR Cutoff",
                    "VSH Cutoff",
                    "Average VSH",
                    "Net-to-Gross",
                ],
                "Value": [
                    str(well_name),
                    str(company),
                    f"{len(df):,}",
                    str(depth_column),
                    str(gr_column)
                    if gr_column
                    else "Not Found",
                    str(resistivity_curve)
                    if resistivity_curve
                    else "Not Found",
                    str(rhob_column)
                    if rhob_column
                    else "Not Found",
                    str(nphi_column)
                    if nphi_column
                    else "Not Found",
                    f"{gr_cutoff:.1f}",
                    f"{vsh_cutoff:.2f}",
                    f"{average_vsh:.3f}",
                    f"{ntg:.3f}",
                ],
            }
        )

        summary_df["Parameter"] = (
            summary_df["Parameter"].astype(str)
        )

        summary_df["Value"] = (
            summary_df["Value"].astype(str)
        )

        st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # AVAILABLE CURVES
        # ----------------------------------------------------

        with st.expander(
            "📚 View Available LAS Curves"
        ):

            curve_df = pd.DataFrame(
                {
                    "Curve": [
                        str(column)
                        for column in df.columns
                    ]
                }
            )

            st.dataframe(
                curve_df,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # RAW DATA PREVIEW
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Raw LAS Data Preview"
        ):

            preview_df = df.head(25).copy()

            # Convert display-only copy to strings.
            # This prevents Arrow errors from mixed LAS object
            # columns without changing the analytical dataframe.

            for column in preview_df.columns:
                preview_df[column] = (
                    preview_df[column]
                    .map(
                        lambda value:
                        ""
                        if pd.isna(value)
                        else str(value)
                    )
                )

            st.dataframe(
                preview_df,
                width="stretch",
                hide_index=True,
            )

        # ----------------------------------------------------
        # SAVE WELL
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-label">DATABASE</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">💾 Archive Well Interpretation</div>',
            unsafe_allow_html=True,
        )

        save_col1, save_col2 = st.columns(
            [1, 3]
        )

        with save_col1:

            if st.button(
                "Save Well Analysis",
                type="primary",
                width="stretch",
            ):

                save_well_analysis(
                    well_name=well_name,
                    company=company,
                    depth_samples=len(df),
                    average_vsh=average_vsh,
                    net_to_gross=ntg,
                )

                st.success(
                    f"{well_name} successfully archived."
                )

        with save_col2:

            st.caption(
                "Saved records are stored in the local "
                "PetroDashboard SQLite database."
            )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        st.divider()

        display_ai_lithology(
            df=df,
            gr_column=gr_column,
            rhob_column=rhob_column,
            nphi_column=nphi_column,
            gr_cutoff=gr_cutoff,
        )

        # ----------------------------------------------------
        # DATABASE
        # ----------------------------------------------------

        st.divider()

        display_multi_well_analysis()

    except Exception as error:

        st.error(
            "PetroDashboard could not process this LAS file."
        )

        st.exception(error)

else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <div style="
            background:
                linear-gradient(
                    135deg,
                    rgba(12,43,53,0.95),
                    rgba(7,22,28,0.95)
                );

            border:1px solid rgba(111,205,228,0.15);
            border-radius:18px;

            padding:35px;

            text-align:center;

            margin-top:30px;
        ">

            <div style="
                font-size:45px;
                margin-bottom:10px;
            ">
                🛢️
            </div>

            <div style="
                font-size:25px;
                font-weight:800;
                color:#F0F8FA;
            ">
                PETRODASHBOARD
            </div>

            <div style="
                font-size:13px;
                color:#77949E;
                margin-top:8px;
            ">
                Interactive Petroleum Engineering
                & Subsurface Analytics Workspace
            </div>

            <div style="
                font-size:12px;
                color:#8EAAB3;
                margin-top:22px;
                line-height:1.8;
            ">
                Upload a LAS well-log file from the sidebar
                to initialize the subsurface workspace.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">PLATFORM CAPABILITIES</div>',
        unsafe_allow_html=True,
    )

    empty_col1, empty_col2, empty_col3, empty_col4 = st.columns(4)

    with empty_col1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">
                    Well Logs
                </div>
                <div class="kpi-value">
                    LAS
                </div>
                <div class="kpi-caption">
                    Automated curve discovery
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with empty_col2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">
                    Formation Evaluation
                </div>
                <div class="kpi-value">
                    VSH / NTG
                </div>
                <div class="kpi-caption">
                    Reservoir interval screening
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with empty_col3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">
                    Visualization
                </div>
                <div class="kpi-value">
                    4 TRACKS
                </div>
                <div class="kpi-caption">
                    Interactive shared-depth canvas
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with empty_col4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">
                    Analytics
                </div>
                <div class="kpi-value">
                    AI
                </div>
                <div class="kpi-caption">
                    Lithology prototype
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
