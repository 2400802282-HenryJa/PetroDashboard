# ============================================================
# PETRODASHBOARD
# Advanced Petroleum Engineering & Subsurface Workspace
# ============================================================

# ============================================================
# 1. IMPORTS
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
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PetroDashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. APPLICATION CONSTANTS
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
    "RESD",
    "LLS",
    "MSFL",
    "RXO",
    "RILD"
]

DEPTH_KEYWORDS = [
    "DEPT",
    "DEPTH",
    "MD",
    "TVD"
]

GR_KEYWORDS = [
    "GR",
    "GAM",
    "GAMMA"
]

RHOB_KEYWORDS = [
    "RHOB",
    "RHOZ",
    "DEN"
]

NPHI_KEYWORDS = [
    "NPHI",
    "TNPH",
    "NPOR",
    "CNPOR"
]


# ============================================================
# 4. DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / DATABASE_FILE


# ============================================================
# 5. PREMIUM APPLICATION STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(17, 80, 105, 0.20),
                transparent 32%
            ),
            #081116;
        color: #E8EEF2;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 17, 22, 0.90);
    }

    [data-testid="stSidebar"] {
        background: #0B171D;
        border-right: 1px solid #20343D;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    /* --------------------------------------------------------
       TYPOGRAPHY
    -------------------------------------------------------- */

    h1, h2, h3 {
        color: #F2F7FA !important;
    }

    p, span, label {
        color: #B9C7CE;
    }

    /* --------------------------------------------------------
       HEADER
    -------------------------------------------------------- */

    .petro-header {
        background:
            linear-gradient(
                135deg,
                #102832 0%,
                #0B1A21 55%,
                #0A151B 100%
            );

        border: 1px solid #25434F;
        border-radius: 12px;

        padding: 20px 24px;

        margin-bottom: 16px;

        box-shadow:
            0 8px 30px rgba(0, 0, 0, 0.30);
    }

    .petro-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #F5F8FA;
    }

    .petro-subtitle {
        font-size: 13px;
        letter-spacing: 0.4px;
        color: #8EA6B1;
        margin-top: 5px;
    }

    .status-online {
        display: inline-block;

        background: rgba(42, 184, 114, 0.12);
        border: 1px solid rgba(42, 184, 114, 0.40);

        color: #5BE39A;

        border-radius: 20px;

        padding: 5px 12px;

        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.7px;
    }

    /* --------------------------------------------------------
       SECTION HEADERS
    -------------------------------------------------------- */

    .section-label {
        color: #6D8A96;

        font-size: 11px;
        font-weight: 700;

        letter-spacing: 1.6px;
        text-transform: uppercase;

        margin-top: 14px;
        margin-bottom: 8px;
    }

    /* --------------------------------------------------------
       KPI CARDS
    -------------------------------------------------------- */

    .kpi-card {
        background:
            linear-gradient(
                145deg,
                #11242C,
                #0D1B21
            );

        border: 1px solid #24404B;

        border-radius: 10px;

        min-height: 115px;

        padding: 15px 16px;

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.025),
            0 6px 20px rgba(0,0,0,0.20);
    }

    .kpi-label {
        color: #718B96;

        font-size: 10px;

        text-transform: uppercase;

        letter-spacing: 1.2px;

        font-weight: 700;
    }

    .kpi-value {
        color: #F2F7FA;

        font-size: 26px;

        font-weight: 700;

        margin-top: 9px;
    }

    .kpi-unit {
        color: #75919D;

        font-size: 11px;

        margin-top: 2px;
    }

    .kpi-status {
        color: #58D995;

        font-size: 10px;

        margin-top: 9px;

        letter-spacing: 0.5px;
    }

    /* --------------------------------------------------------
       TECHNICAL PANELS
    -------------------------------------------------------- */

    .technical-panel {
        background: #0D1C23;

        border: 1px solid #203A45;

        border-radius: 10px;

        padding: 10px;

        box-shadow:
            0 6px 25px rgba(0,0,0,0.18);
    }

    /* --------------------------------------------------------
       STATUS BOXES
    -------------------------------------------------------- */

    .status-box {
        background: #0D1D24;

        border: 1px solid #23404B;

        border-radius: 8px;

        padding: 10px 12px;

        text-align: center;
    }

    .status-title {
        color: #78929D;

        font-size: 10px;

        text-transform: uppercase;

        letter-spacing: 1px;
    }

    .status-value {
        color: #EAF2F5;

        font-size: 17px;

        font-weight: 700;

        margin-top: 5px;
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    .sidebar-title {
        color: #E8F0F3;

        font-size: 15px;

        font-weight: 700;

        letter-spacing: 1px;

        margin-bottom: 3px;
    }

    .sidebar-subtitle {
        color: #6F8994;

        font-size: 10px;

        letter-spacing: 0.6px;

        margin-bottom: 20px;
    }

    /* --------------------------------------------------------
       FILE UPLOADER
    -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: #0C1A20;

        border: 1px dashed #31515D;

        border-radius: 9px;

        padding: 8px;
    }

    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    [data-testid="stMetric"] {
        background: #0D1C23;

        border: 1px solid #203A45;

        border-radius: 8px;

        padding: 10px;
    }

    /* --------------------------------------------------------
       DATAFRAME
    -------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #203A45;
        border-radius: 8px;
    }

    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {
        border-radius: 7px;

        border: 1px solid #2A5867;

        background: #12313C;

        color: #DCEBF0;

        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #4E91A6;

        background: #17404D;

        color: #FFFFFF;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    """
    Create the SQLite database and required tables.
    """

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


# ============================================================
# 7. DATABASE SAVE
# ============================================================

def save_well_analysis(
    well_name,
    company,
    depth_samples,
    average_vsh,
    net_to_gross
):
    """
    Save well formation evaluation results.
    """

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
            float(net_to_gross)
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# 8. LOAD DATABASE
# ============================================================

def load_saved_wells():

    conn = sqlite3.connect(DB_PATH)

    wells = pd.read_sql_query(
        "SELECT * FROM wells",
        conn
    )

    conn.close()

    return wells


# ============================================================
# 9. ARROW-SAFE DATAFRAME
# ============================================================

def make_arrow_safe_dataframe(df):

    safe_df = df.copy()

    for column in safe_df.columns:

        if safe_df[column].dtype == "object":

            safe_df[column] = safe_df[column].map(
                lambda value:
                ""
                if pd.isna(value)
                else str(value)
            )

    return safe_df


# ============================================================
# 10. LAS LOADING
# ============================================================

def load_las_file(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:

        raise ValueError(
            "The uploaded LAS file is empty."
        )

    las_string = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    las_stream = StringIO(
        las_string
    )

    return lasio.read(
        las_stream,
        engine="normal"
    )


# ============================================================
# 11. LAS → DATAFRAME
# ============================================================

def las_to_dataframe(las):

    df = las.df()

    df = df.reset_index()

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    for column in df.columns:

        if column == df.columns[0]:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# 12. WELL INFORMATION
# ============================================================

def get_well_information(las):

    well_name = "Unknown"
    company = "Unknown"

    try:

        if "WELL" in las.well:

            value = las.well.WELL.value

            if value is not None:
                well_name = str(value).strip()

    except Exception:
        pass

    try:

        if "COMP" in las.well:

            value = las.well.COMP.value

            if value is not None:
                company = str(value).strip()

    except Exception:
        pass

    if not well_name:
        well_name = "Unknown"

    if not company:
        company = "Unknown"

    return well_name, company


# ============================================================
# 13. DEPTH DETECTION
# ============================================================

def get_depth_column(df):

    columns = list(df.columns)

    for keyword in DEPTH_KEYWORDS:

        for column in columns:

            if str(column).upper().strip() == keyword:

                return column

    for column in columns:

        column_upper = str(
            column
        ).upper()

        if any(
            keyword in column_upper
            for keyword in DEPTH_KEYWORDS
        ):

            return column

    return columns[0]


# ============================================================
# 14. GENERIC CURVE DETECTION
# ============================================================

def find_curve(
    df,
    keywords
):

    columns = list(df.columns)

    for keyword in keywords:

        for column in columns:

            if str(column).upper().strip() == keyword:

                return column

    for column in columns:

        column_upper = str(
            column
        ).upper()

        for keyword in keywords:

            if keyword in column_upper:

                return column

    return None


# ============================================================
# 15. CURVE DETECTION HELPERS
# ============================================================

def find_gamma_ray_curve(df):

    return find_curve(
        df,
        GR_KEYWORDS
    )


def find_density_curve(df):

    return find_curve(
        df,
        RHOB_KEYWORDS
    )


def find_neutron_curve(df):

    return find_curve(
        df,
        NPHI_KEYWORDS
    )


def find_resistivity_curve(df):

    return find_curve(
        df,
        RESISTIVITY_KEYWORDS
    )


# ============================================================
# 16. VSH
# ============================================================

def calculate_vsh(
    df,
    gr_column=None
):

    if gr_column is None:

        gr_column = find_gamma_ray_curve(
            df
        )

    if gr_column is None:

        df["VSH"] = np.nan

        return df, 0.0

    gr = pd.to_numeric(
        df[gr_column],
        errors="coerce"
    )

    gr_min = gr.min()
    gr_max = gr.max()

    if pd.isna(gr_min) or pd.isna(gr_max):

        df["VSH"] = np.nan

        return df, 0.0

    if gr_max == gr_min:

        df["VSH"] = 0.0

        return df, 0.0

    df["VSH"] = (
        (gr - gr_min)
        /
        (gr_max - gr_min)
    )

    df["VSH"] = df["VSH"].clip(
        0,
        1
    )

    average_vsh = float(
        df["VSH"].mean()
    )

    return df, average_vsh


# ============================================================
# 17. RESERVOIR FLAG
# ============================================================

def calculate_reservoir_flag(
    df,
    vsh_cutoff
):

    if "VSH" not in df.columns:

        df["RES_FLAG"] = 0

        return df

    df["RES_FLAG"] = np.where(
        df["VSH"] < vsh_cutoff,
        1,
        0
    )

    return df


# ============================================================
# 18. NTG
# ============================================================

def calculate_ntg(df):

    if "RES_FLAG" not in df.columns:

        return 0.0

    valid_flags = pd.to_numeric(
        df["RES_FLAG"],
        errors="coerce"
    ).dropna()

    if valid_flags.empty:

        return 0.0

    return float(
        valid_flags.mean()
    )


# ============================================================
# 19. MIN-MAX DECIMATION
# ============================================================

def min_max_decimate(
    df,
    x_column,
    depth_column,
    max_points=MAX_VISUAL_POINTS
):

    if x_column not in df.columns:

        return df.copy()

    if depth_column not in df.columns:

        return df.copy()

    work_df = df[
        [
            depth_column,
            x_column
        ]
    ].copy()

    work_df[x_column] = pd.to_numeric(
        work_df[x_column],
        errors="coerce"
    )

    work_df[depth_column] = pd.to_numeric(
        work_df[depth_column],
        errors="coerce"
    )

    work_df = work_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    work_df = work_df.dropna(
        subset=[
            depth_column,
            x_column
        ]
    )

    if work_df.empty:

        return work_df

    if len(work_df) <= max_points:

        return work_df.sort_values(
            depth_column
        )

    bucket_count = max(
        1,
        max_points // 2
    )

    boundaries = np.linspace(
        0,
        len(work_df),
        bucket_count + 1
    ).astype(int)

    selected_positions = set()

    for i in range(
        len(boundaries) - 1
    ):

        start = boundaries[i]
        end = boundaries[i + 1]

        if end <= start:
            continue

        section = work_df.iloc[
            start:end
        ]

        if section.empty:
            continue

        selected_positions.add(
            section[x_column].idxmin()
        )

        selected_positions.add(
            section[x_column].idxmax()
        )

    result = work_df.loc[
        sorted(selected_positions)
    ]

    return result.sort_values(
        depth_column
    )


# ============================================================
# 20. RADIAL GAUGE
# ============================================================

def create_radial_gauge(
    title,
    value,
    minimum,
    maximum,
    suffix="",
    color="#35C98A",
    decimals=2
):
    """
    Create a premium Plotly radial gauge.
    """

    if value is None or pd.isna(value):

        value = minimum

    value = float(value)

    value = max(
        minimum,
        min(
            maximum,
            value
        )
    )

    if decimals == 0:

        number_format = ",.0f"

    elif decimals == 1:

        number_format = ",.1f"

    else:

        number_format = ",.2f"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={
                "text": title,
                "font": {
                    "size": 13,
                    "color": "#A8BBC3"
                }
            },
            number={
                "suffix": suffix,
                "valueformat": number_format,
                "font": {
                    "size": 25,
                    "color": "#F1F6F8"
                }
            },
            gauge={
                "axis": {
                    "range": [
                        minimum,
                        maximum
                    ],
                    "tickwidth": 1,
                    "tickcolor": "#627983",
                    "tickfont": {
                        "size": 9,
                        "color": "#718993"
                    }
                },
                "bar": {
                    "color": color,
                    "thickness": 0.25
                },
                "bgcolor": "#182A31",
                "borderwidth": 1,
                "bordercolor": "#304952",
                "steps": [
                    {
                        "range": [
                            minimum,
                            minimum
                            + (
                                maximum
                                - minimum
                            ) * 0.33
                        ],
                        "color": "#111D22"
                    },
                    {
                        "range": [
                            minimum
                            + (
                                maximum
                                - minimum
                            ) * 0.33,
                            minimum
                            + (
                                maximum
                                - minimum
                            ) * 0.66
                        ],
                        "color": "#17262C"
                    },
                    {
                        "range": [
                            minimum
                            + (
                                maximum
                                - minimum
                            ) * 0.66,
                            maximum
                        ],
                        "color": "#1D3037"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=225,
        margin=dict(
            l=15,
            r=15,
            t=35,
            b=10
        )
    )

    return fig


# ============================================================
# 21. PETREL-STYLE WELL LOG CANVAS
# ============================================================

@st.fragment
def render_well_log_canvas(
    df,
    depth_column,
    gr_cutoff,
    vsh_cutoff,
    resistivity_curve,
    gr_curve=None,
    rhob_curve=None,
    nphi_curve=None
):

    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        subplot_titles=(
            "GAMMA RAY",
            "RESISTIVITY",
            "DENSITY / POROSITY",
            "VSH"
        )
    )

    # --------------------------------------------------------
    # GAMMA RAY
    # --------------------------------------------------------

    if gr_curve is not None:

        gr_df = min_max_decimate(
            df,
            gr_curve,
            depth_column
        )

        if not gr_df.empty:

            fig.add_trace(
                go.Scatter(
                    x=gr_df[gr_curve],
                    y=gr_df[depth_column],
                    mode="lines",
                    name="GR",
                    line=dict(
                        color="#36C275",
                        width=1.5
                    ),
                    hovertemplate=(
                        "Depth: %{y:.2f}<br>"
                        "GR: %{x:.2f}"
                        "<extra></extra>"
                    )
                ),
                row=1,
                col=1
            )

            fig.add_vline(
                x=gr_cutoff,
                line_width=1,
                line_dash="dash",
                line_color="#F1A33C",
                row=1,
                col=1
            )

    # --------------------------------------------------------
    # RESISTIVITY
    # --------------------------------------------------------

    if resistivity_curve is not None:

        res_df = min_max_decimate(
            df,
            resistivity_curve,
            depth_column
        )

        if not res_df.empty:

            res_df = res_df[
                res_df[resistivity_curve] > 0
            ]

            if not res_df.empty:

                fig.add_trace(
                    go.Scatter(
                        x=res_df[
                            resistivity_curve
                        ],
                        y=res_df[
                            depth_column
                        ],
                        mode="lines",
                        name="RES",
                        line=dict(
                            color="#E35B5B",
                            width=1.5
                        ),
                        hovertemplate=(
                            "Depth: %{y:.2f}<br>"
                            "Resistivity: %{x:.3f}"
                            "<extra></extra>"
                        )
                    ),
                    row=1,
                    col=2
                )

                fig.update_xaxes(
                    type="log",
                    row=1,
                    col=2
                )

    # --------------------------------------------------------
    # RHOB
    # --------------------------------------------------------

    if rhob_curve is not None:

        rhob_df = min_max_decimate(
            df,
            rhob_curve,
            depth_column
        )

        if not rhob_df.empty:

            fig.add_trace(
                go.Scatter(
                    x=rhob_df[rhob_curve],
                    y=rhob_df[depth_column],
                    mode="lines",
                    name="RHOB",
                    line=dict(
                        color="#42A5F5",
                        width=1.5
                    ),
                    hovertemplate=(
                        "Depth: %{y:.2f}<br>"
                        "RHOB: %{x:.3f}"
                        "<extra></extra>"
                    )
                ),
                row=1,
                col=3
            )

    # --------------------------------------------------------
    # NPHI
    # --------------------------------------------------------

    if nphi_curve is not None:

        nphi_df = min_max_decimate(
            df,
            nphi_curve,
            depth_column
        )

        if not nphi_df.empty:

            fig.add_trace(
                go.Scatter(
                    x=nphi_df[nphi_curve],
                    y=nphi_df[depth_column],
                    mode="lines",
                    name="NPHI",
                    line=dict(
                        color="#A46EDC",
                        width=1.5
                    ),
                    hovertemplate=(
                        "Depth: %{y:.2f}<br>"
                        "NPHI: %{x:.3f}"
                        "<extra></extra>"
                    )
                ),
                row=1,
                col=3
            )

    # --------------------------------------------------------
    # VSH
    # --------------------------------------------------------

    if "VSH" in df.columns:

        vsh_df = min_max_decimate(
            df,
            "VSH",
            depth_column
        )

        if not vsh_df.empty:

            fig.add_trace(
                go.Scatter(
                    x=vsh_df["VSH"],
                    y=vsh_df[depth_column],
                    mode="lines",
                    name="VSH",
                    line=dict(
                        color="#F0F4F5",
                        width=1.5
                    ),
                    hovertemplate=(
                        "Depth: %{y:.2f}<br>"
                        "VSH: %{x:.3f}"
                        "<extra></extra>"
                    )
                ),
                row=1,
                col=4
            )

            fig.add_vline(
                x=vsh_cutoff,
                line_width=1,
                line_dash="dash",
                line_color="#F1A33C",
                row=1,
                col=4
            )

    # --------------------------------------------------------
    # DEPTH / CROSSHAIR
    # --------------------------------------------------------

    fig.update_yaxes(
        autorange="reversed",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#DCEAF0",
        spikethickness=1,
        gridcolor="#243A43",
        zeroline=False
    )

    # --------------------------------------------------------
    # X AXES
    # --------------------------------------------------------

    fig.update_xaxes(
        title_text="GR",
        gridcolor="#243A43",
        zeroline=False,
        row=1,
        col=1
    )

    fig.update_xaxes(
        title_text="RES",
        gridcolor="#243A43",
        zeroline=False,
        row=1,
        col=2
    )

    fig.update_xaxes(
        title_text="RHOB / NPHI",
        gridcolor="#243A43",
        zeroline=False,
        row=1,
        col=3
    )

    fig.update_xaxes(
        title_text="VSH",
        range=[0, 1],
        gridcolor="#243A43",
        zeroline=False,
        row=1,
        col=4
    )

    # --------------------------------------------------------
    # INTERACTION
    # --------------------------------------------------------

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1C23",
        plot_bgcolor="#0D1C23",
        hovermode="y unified",
        height=900,
        margin=dict(
            l=55,
            r=25,
            t=55,
            b=45
        ),
        dragmode="pan",
        showlegend=False,
        font=dict(
            color="#B9C8CE"
        )
    )

    for column in range(
        2,
        5
    ):

        fig.update_yaxes(
            showticklabels=False,
            row=1,
            col=column
        )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d"
            ]
        }
    )


# ============================================================
# 22. AI LITHOLOGY MODEL
# ============================================================

def run_ai_lithology(
    df,
    gr_curve,
    rhob_curve,
    nphi_curve,
    gr_cutoff
):

    if gr_curve is None:

        return None, None, (
            "Gamma Ray curve is required."
        )

    if rhob_curve is None:

        return None, None, (
            "Density curve is required."
        )

    if nphi_curve is None:

        return None, None, (
            "Neutron porosity curve is required."
        )

    ai_df = df[
        [
            gr_curve,
            rhob_curve,
            nphi_curve
        ]
    ].copy()

    ai_df.columns = [
        "GR",
        "RHOB",
        "NPHI"
    ]

    ai_df = ai_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    ai_df = ai_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    ai_df = ai_df.dropna()

    if len(ai_df) < 20:

        return None, None, (
            "Not enough valid samples for AI training."
        )

    ai_df["Lithology"] = np.where(
        ai_df["GR"] < gr_cutoff,
        1,
        0
    )

    X = ai_df[
        [
            "GR",
            "RHOB",
            "NPHI"
        ]
    ]

    y = ai_df["Lithology"]

    if y.nunique() < 2:

        return None, None, (
            "Only one lithology class exists. "
            "Adjust the GR cutoff."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    ai_df["Predicted_Lithology"] = model.predict(
        X
    )

    return (
        model,
        ai_df,
        accuracy
    )


# ============================================================
# 23. AI DISPLAY
# ============================================================

def display_ai_lithology(
    df,
    gr_curve,
    rhob_curve,
    nphi_curve,
    gr_cutoff
):

    result = run_ai_lithology(
        df,
        gr_curve,
        rhob_curve,
        nphi_curve,
        gr_cutoff
    )

    if result[0] is None:

        st.warning(
            result[2]
        )

        return None

    model, ai_df, accuracy = result

    sand_count = int(
        (
            ai_df["Predicted_Lithology"] == 1
        ).sum()
    )

    shale_count = int(
        (
            ai_df["Predicted_Lithology"] == 0
        ).sum()
    )

    return {
        "accuracy": float(accuracy),
        "sand_count": sand_count,
        "shale_count": shale_count
    }


# ============================================================
# 24. MULTI-WELL ANALYSIS
# ============================================================

def display_multi_well_analysis():

    wells = load_saved_wells()

    if wells.empty:

        st.info(
            "No saved wells are currently available."
        )

        return

    st.markdown(
        '<div class="section-label">MULTI-WELL DATABASE</div>',
        unsafe_allow_html=True
    )

    display_wells = make_arrow_safe_dataframe(
        wells
    )

    st.dataframe(
        display_wells,
        width="stretch",
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:

        ntg_fig = go.Figure()

        ntg_fig.add_trace(
            go.Bar(
                x=wells["well_name"].astype(str),
                y=pd.to_numeric(
                    wells["net_to_gross"],
                    errors="coerce"
                ),
                marker_color="#35C98A"
            )
        )

        ntg_fig.update_layout(
            title="Net-to-Gross",
            template="plotly_dark",
            paper_bgcolor="#0D1C23",
            plot_bgcolor="#0D1C23",
            height=350
        )

        st.plotly_chart(
            ntg_fig,
            width="stretch"
        )

    with col2:

        vsh_fig = go.Figure()

        vsh_fig.add_trace(
            go.Bar(
                x=wells["well_name"].astype(str),
                y=pd.to_numeric(
                    wells["average_vsh"],
                    errors="coerce"
                ),
                marker_color="#A46EDC"
            )
        )

        vsh_fig.update_layout(
            title="Average VSH",
            template="plotly_dark",
            paper_bgcolor="#0D1C23",
            plot_bgcolor="#0D1C23",
            height=350
        )

        st.plotly_chart(
            vsh_fig,
            width="stretch"
        )


# ============================================================
# 25. INITIALIZE
# ============================================================

initialize_database()


# ============================================================
# 26. SESSION STATE
# ============================================================

if "saved_message" not in st.session_state:

    st.session_state.saved_message = None


# ============================================================
# 27. APPLICATION HEADER
# ============================================================

st.markdown(
    f"""
    <div class="petro-header">

        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:20px;
        ">

            <div>

                <div class="petro-title">
                    🛢️ {APP_TITLE}
                </div>

                <div class="petro-subtitle">
                    {APP_SUBTITLE}
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
    unsafe_allow_html=True
)


# ============================================================
# 28. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">SUBSURFACE WORKSPACE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'TECHNICAL INTERPRETATION CONSOLE'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📂 WELL DATA"
    )

    uploaded_file = st.file_uploader(
        "Upload LAS Well Log",
        type=["las"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "### ⚙️ INTERPRETATION"
    )

    gr_cutoff = st.number_input(
        "GR Sand/Shale Cutoff",
        min_value=0.0,
        max_value=150.0,
        value=75.0,
        step=1.0
    )

    vsh_cutoff = st.number_input(
        "VSH Reservoir Cutoff",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05
    )

    st.divider()

    st.markdown(
        "### WORKSPACE MODULES"
    )

    st.markdown(
        """
        **● Well Logs**

        **● Formation Evaluation**

        **● AI Interpretation**

        **● Multi-Well Analysis**

        **○ Volumetrics & Uncertainty**

        **○ 3D Subsurface**

        **○ Production Analytics**
        """
    )


# ============================================================
# 29. EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.markdown(
        '<div class="section-label">SYSTEM READY</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    empty_cards = [
        (
            "WELL LOGS",
            "LAS processing",
            "READY"
        ),
        (
            "FORMATION",
            "VSH / NTG",
            "READY"
        ),
        (
            "AI ENGINE",
            "Lithology prototype",
            "READY"
        ),
        (
            "DATABASE",
            "Multi-well storage",
            "READY"
        )
    ]

    for column, card in zip(
        [col1, col2, col3, col4],
        empty_cards
    ):

        with column:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        {card[0]}
                    </div>

                    <div class="kpi-value">
                        {card[1]}
                    </div>

                    <div class="kpi-status">
                        ● {card[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.info(
        "Upload a LAS well log from the workspace panel "
        "to initialize the subsurface interpretation canvas."
    )

    st.markdown(
        """
        <div class="technical-panel">

        ### PetroDashboard Workspace

        The current platform combines:

        - Interactive LAS well-log processing
        - Petrel-style multi-track visualization
        - Shared-depth crosshair interaction
        - Resistivity logarithmic scaling
        - VSH and Net-to-Gross evaluation
        - AI lithology prototype
        - Multi-well comparison
        - High-density technical KPI visualization

        **Planned workspace expansion**

        Monte Carlo volumetrics, P90/P50/P10 uncertainty,
        sensitivity analysis, 3D subsurface visualization,
        production analytics and advanced AI interpretation.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# 30. MAIN APPLICATION
# ============================================================

try:

    # ========================================================
    # LOAD LAS
    # ========================================================

    las = load_las_file(
        uploaded_file
    )

    well_name, company = get_well_information(
        las
    )

    df = las_to_dataframe(
        las
    )

    if df.empty:

        st.error(
            "The LAS file contains no usable log samples."
        )

        st.stop()

    depth_column = get_depth_column(
        df
    )

    # ========================================================
    # CURVE DETECTION
    # ========================================================

    gr_curve = find_gamma_ray_curve(
        df
    )

    rhob_curve = find_density_curve(
        df
    )

    nphi_curve = find_neutron_curve(
        df
    )

    resistivity_curve = find_resistivity_curve(
        df
    )

    # ========================================================
    # FORMATION EVALUATION
    # ========================================================

    df, average_vsh = calculate_vsh(
        df,
        gr_curve
    )

    df = calculate_reservoir_flag(
        df,
        vsh_cutoff
    )

    ntg = calculate_ntg(
        df
    )

    # ========================================================
    # AI
    # ========================================================

    ai_result = display_ai_lithology(
        df=df,
        gr_curve=gr_curve,
        rhob_curve=rhob_curve,
        nphi_curve=nphi_curve,
        gr_cutoff=gr_cutoff
    )

    # ========================================================
    # SECTION — WELL STATUS
    # ========================================================

    st.markdown(
        '<div class="section-label">ACTIVE WELL / RESERVOIR STATUS</div>',
        unsafe_allow_html=True
    )

    status1, status2, status3, status4 = st.columns(4)

    status_cards = [
        (
            "ACTIVE WELL",
            well_name,
            "LAS LOADED"
        ),
        (
            "OPERATOR",
            company,
            "METADATA"
        ),
        (
            "DEPTH SAMPLES",
            f"{len(df):,}",
            "VALID DATA"
        ),
        (
            "CURVE STATUS",
            f"{len(df.columns) - 1} CURVES",
            "DETECTED"
        )
    ]

    for column, card in zip(
        [status1, status2, status3, status4],
        status_cards
    ):

        with column:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        {card[0]}
                    </div>

                    <div class="kpi-value">
                        {card[1]}
                    </div>

                    <div class="kpi-status">
                        ● {card[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # ========================================================
    # SECTION — RESERVOIR KPIs
    # ========================================================

    st.markdown(
        '<div class="section-label">'
        'LIVE FORMATION EVALUATION'
        '</div>',
        unsafe_allow_html=True
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:

        st.plotly_chart(
            create_radial_gauge(
                "AVERAGE VSH",
                average_vsh,
                0,
                1,
                suffix="",
                color="#A46EDC",
                decimals=2
            ),
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    with kpi2:

        st.plotly_chart(
            create_radial_gauge(
                "NET-TO-GROSS",
                ntg,
                0,
                1,
                suffix="",
                color="#35C98A",
                decimals=2
            ),
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    with kpi3:

        detected_count = sum(
            curve is not None
            for curve in [
                gr_curve,
                resistivity_curve,
                rhob_curve,
                nphi_curve
            ]
        )

        st.plotly_chart(
            create_radial_gauge(
                "CURVE COVERAGE",
                detected_count,
                0,
                4,
                suffix=" / 4",
                color="#42A5F5",
                decimals=0
            ),
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    with kpi4:

        ai_accuracy = (
            ai_result["accuracy"]
            if ai_result is not None
            else 0.0
        )

        st.plotly_chart(
            create_radial_gauge(
                "AI PROTOTYPE",
                ai_accuracy,
                0,
                1,
                suffix="",
                color="#F1A33C",
                decimals=2
            ),
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    # ========================================================
    # SECTION — CURVE STATUS
    # ========================================================

    st.markdown(
        '<div class="section-label">CURVE DETECTION MATRIX</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    curve_status = [
        (
            "GAMMA RAY",
            gr_curve
        ),
        (
            "RESISTIVITY",
            resistivity_curve
        ),
        (
            "DENSITY",
            rhob_curve
        ),
        (
            "NEUTRON",
            nphi_curve
        )
    ]

    for column, item in zip(
        [c1, c2, c3, c4],
        curve_status
    ):

        with column:

            curve_name = item[1]

            if curve_name:

                st.markdown(
                    f"""
                    <div class="status-box">

                        <div class="status-title">
                            {item[0]}
                        </div>

                        <div class="status-value">
                            ✓ {curve_name}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="status-box">

                        <div class="status-title">
                            {item[0]}
                        </div>

                        <div class="status-value"
                             style="color:#F1A33C;">
                            — NOT FOUND
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ========================================================
    # SECTION — WELL LOG CANVAS
    # ========================================================

    st.markdown(
        '<div class="section-label">'
        'SUBSURFACE INTERPRETATION CANVAS'
        '</div>',
        unsafe_allow_html=True
    )

    render_well_log_canvas(
        df=df,
        depth_column=depth_column,
        gr_cutoff=gr_cutoff,
        vsh_cutoff=vsh_cutoff,
        resistivity_curve=resistivity_curve,
        gr_curve=gr_curve,
        rhob_curve=rhob_curve,
        nphi_curve=nphi_curve
    )

    # ========================================================
    # SECTION — FORMATION GAUGES
    # ========================================================

    st.markdown(
        '<div class="section-label">'
        'FORMATION QUALITY INDICATORS'
        '</div>',
        unsafe_allow_html=True
    )

    fg1, fg2, fg3 = st.columns(3)

    with fg1:

        st.plotly_chart(
            create_radial_gauge(
                "RESERVOIR QUALITY",
                ntg,
                0,
                1,
                color="#35C98A",
                decimals=2
            ),
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    with fg2:

        clean_fraction = (
            1.0 - average_vsh
        )

        st.plotly_chart(
            create_radial_gauge(
                "CLEANNESS INDEX",
                clean_fraction,
                0,
                1,
                color="#42A5F5",
                decimals=2
            ),
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    with fg3:

        valid_curve_count = sum(
            curve is not None
            for curve in [
                gr_curve,
                resistivity_curve,
                rhob_curve,
                nphi_curve
            ]
        )

        data_quality = (
            valid_curve_count / 4
        )

        st.plotly_chart(
            create_radial_gauge(
                "DATA QUALITY",
                data_quality,
                0,
                1,
                color="#35C98A",
                decimals=2
            ),
            width="stretch",
            config={
                "displayModeBar": False
            }
        )

    # ========================================================
    # SECTION — AI
    # ========================================================

    st.markdown(
        '<div class="section-label">'
        'AI INTERPRETATION ENGINE'
        '</div>',
        unsafe_allow_html=True
    )

    ai_col1, ai_col2 = st.columns(
        [2, 1]
    )

    with ai_col1:

        if ai_result is not None:

            ai_summary = pd.DataFrame(
                {
                    "Classification": [
                        "Predicted Sand",
                        "Predicted Shale"
                    ],
                    "Samples": [
                        ai_result["sand_count"],
                        ai_result["shale_count"]
                    ]
                }
            )

            ai_fig = go.Figure()

            ai_fig.add_trace(
                go.Bar(
                    x=ai_summary["Classification"],
                    y=ai_summary["Samples"],
                    marker_color=[
                        "#35C98A",
                        "#A46EDC"
                    ]
                )
            )

            ai_fig.update_layout(
                title="AI Lithology Distribution",
                template="plotly_dark",
                paper_bgcolor="#0D1C23",
                plot_bgcolor="#0D1C23",
                height=350,
                margin=dict(
                    l=30,
                    r=30,
                    t=60,
                    b=30
                )
            )

            st.plotly_chart(
                ai_fig,
                width="stretch"
            )

        else:

            st.warning(
                "AI lithology requires GR, RHOB and NPHI."
            )

    with ai_col2:

        if ai_result is not None:

            st.plotly_chart(
                create_radial_gauge(
                    "MODEL ACCURACY",
                    ai_result["accuracy"],
                    0,
                    1,
                    color="#F1A33C",
                    decimals=2
                ),
                width="stretch",
                config={
                    "displayModeBar": False
                }
            )

        st.info(
            "Educational prototype: training labels "
            "are generated from the GR cutoff rather "
            "than expert-labelled core or facies data."
        )

    # ========================================================
    # SECTION — FORMATION SUMMARY
    # ========================================================

    with st.expander(
        "📋 View Formation Evaluation Summary"
    ):

        summary_df = pd.DataFrame(
            {
                "Parameter": [
                    "Well",
                    "Company",
                    "Depth Samples",
                    "Depth Curve",
                    "GR Curve",
                    "Resistivity Curve",
                    "Density Curve",
                    "Neutron Curve",
                    "GR Cutoff",
                    "VSH Cutoff",
                    "Average VSH",
                    "Net-to-Gross"
                ],
                "Value": [
                    str(well_name),
                    str(company),
                    str(len(df)),
                    str(depth_column),
                    str(gr_curve or "Not detected"),
                    str(
                        resistivity_curve
                        or "Not detected"
                    ),
                    str(
                        rhob_curve
                        or "Not detected"
                    ),
                    str(
                        nphi_curve
                        or "Not detected"
                    ),
                    str(gr_cutoff),
                    str(vsh_cutoff),
                    str(round(average_vsh, 3)),
                    str(round(ntg, 3))
                ]
            }
        )

        st.dataframe(
            make_arrow_safe_dataframe(
                summary_df
            ),
            width="stretch",
            hide_index=True
        )

    # ========================================================
    # SECTION — SAVE
    # ========================================================

    st.markdown(
        '<div class="section-label">'
        'WELL KNOWLEDGE DATABASE'
        '</div>',
        unsafe_allow_html=True
    )

    save_col1, save_col2 = st.columns(
        [3, 1]
    )

    with save_col1:

        st.markdown(
            f"""
            <div class="technical-panel">

            <strong>{well_name}</strong>

            <br>

            <span style="color:#718B96;">
            {company} • {len(df):,} depth samples •
            NTG {ntg:.3f} • VSH {average_vsh:.3f}
            </span>

            </div>
            """,
            unsafe_allow_html=True
        )

    with save_col2:

        if st.button(
            "💾 SAVE WELL",
            type="primary",
            width="stretch"
        ):

            save_well_analysis(
                well_name=well_name,
                company=company,
                depth_samples=len(df),
                average_vsh=average_vsh,
                net_to_gross=ntg
            )

            st.success(
                f"{well_name} saved."
            )

    # ========================================================
    # SECTION — MULTI-WELL
    # ========================================================

    st.markdown(
        '<div class="section-label">'
        'MULTI-WELL SUBSURFACE ANALYSIS'
        '</div>',
        unsafe_allow_html=True
    )

    display_multi_well_analysis()

    # ========================================================
    # SECTION — DATA PREVIEW
    # ========================================================

    with st.expander(
        "🔬 Raw LAS Data Preview"
    ):

        st.dataframe(
            make_arrow_safe_dataframe(
                df.head(50)
            ),
            width="stretch",
            hide_index=True
        )

except Exception as error:

    st.error(
        "PetroDashboard encountered an error "
        "while processing the LAS file."
    )

    st.exception(
        error
    )
