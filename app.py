# ============================================================
# PETRODASHBOARD
# Interactive Petroleum Engineering & Well Log Platform
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
    layout="wide"
)


# ============================================================
# 3. APPLICATION CONSTANTS
# ============================================================

APP_TITLE = "🛢️ PetroDashboard"

APP_SUBTITLE = (
    "Interactive Petroleum Engineering, "
    "Well Log & Formation Evaluation Platform"
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
# 5. DATABASE INITIALIZATION
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
# 6. DATABASE SAVE FUNCTION
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
# 7. LOAD SAVED WELLS
# ============================================================

def load_saved_wells():
    """
    Retrieve saved well analysis records.
    """

    conn = sqlite3.connect(DB_PATH)

    wells = pd.read_sql_query(
        "SELECT * FROM wells",
        conn
    )

    conn.close()

    return wells


# ============================================================
# 8. ARROW-SAFE DATAFRAME
# ============================================================

def make_arrow_safe_dataframe(df):
    """
    Make mixed/object columns safe for Streamlit and PyArrow.

    This prevents errors such as:
    Expected bytes, got a 'int' object
    """

    safe_df = df.copy()

    for column in safe_df.columns:

        if safe_df[column].dtype == "object":

            safe_df[column] = safe_df[column].map(
                lambda value: (
                    ""
                    if pd.isna(value)
                    else str(value)
                )
            )

    return safe_df


# ============================================================
# 9. LAS FILE LOADING
# ============================================================

def load_las_file(uploaded_file):
    """
    Convert uploaded LAS file into a lasio object.
    """

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

    las = lasio.read(
        las_stream,
        engine="normal"
    )

    return las


# ============================================================
# 10. LAS → DATAFRAME
# ============================================================

def las_to_dataframe(las):
    """
    Convert LAS data into a clean Pandas DataFrame.
    """

    df = las.df()

    df = df.reset_index()

    # Remove duplicate column names if any exist.
    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    # Convert numeric-looking log columns to numeric.
    for column in df.columns:

        if column == df.columns[0]:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# 11. WELL INFORMATION EXTRACTION
# ============================================================

def get_well_information(las):
    """
    Extract basic well metadata safely.
    """

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
# 12. DEPTH COLUMN DETECTION
# ============================================================

def get_depth_column(df):
    """
    Identify the most likely depth column.
    """

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

    # LAS reset_index() normally places depth first.
    return columns[0]


# ============================================================
# 13. GENERIC CURVE DETECTION
# ============================================================

def find_curve(
    df,
    keywords
):
    """
    Find a curve using exact matches first,
    followed by keyword matching.
    """

    columns = list(df.columns)

    # Exact match.
    for keyword in keywords:

        for column in columns:

            if str(column).upper().strip() == keyword:

                return column

    # Keyword match.
    for column in columns:

        column_upper = str(
            column
        ).upper()

        for keyword in keywords:

            if keyword in column_upper:

                return column

    return None


# ============================================================
# 14. SPECIFIC CURVE DETECTION
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
    """
    Automatically identify a resistivity curve.
    """

    return find_curve(
        df,
        RESISTIVITY_KEYWORDS
    )


# ============================================================
# 15. VSH CALCULATION
# ============================================================

def calculate_vsh(
    df,
    gr_column=None
):
    """
    Calculate normalized Gamma Ray derived VSH.
    """

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
# 16. RESERVOIR FLAG
# ============================================================

def calculate_reservoir_flag(
    df,
    vsh_cutoff
):
    """
    Identify reservoir intervals using VSH cutoff.
    """

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
# 17. NET-TO-GROSS
# ============================================================

def calculate_ntg(df):
    """
    Calculate Net-to-Gross from valid reservoir flags.
    """

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
# 18. MIN-MAX DECIMATION
# ============================================================

def min_max_decimate(
    df,
    x_column,
    depth_column,
    max_points=MAX_VISUAL_POINTS
):
    """
    Reduce visualization points while preserving
    local minimum and maximum values.
    """

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

        min_position = section[
            x_column
        ].idxmin()

        max_position = section[
            x_column
        ].idxmax()

        selected_positions.add(
            min_position
        )

        selected_positions.add(
            max_position
        )

    result = work_df.loc[
        sorted(selected_positions)
    ]

    return result.sort_values(
        depth_column
    )


# ============================================================
# 19. PREPARE VISUAL DATA
# ============================================================

def prepare_visual_data(
    df,
    depth_column,
    curve_columns
):
    """
    Prepare a clean dataset for Plotly.
    """

    valid_columns = [
        column
        for column in curve_columns
        if column in df.columns
    ]

    if not valid_columns:

        return pd.DataFrame()

    visual_df = df[
        [
            depth_column
        ]
        +
        valid_columns
    ].copy()

    visual_df = visual_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    visual_df[depth_column] = pd.to_numeric(
        visual_df[depth_column],
        errors="coerce"
    )

    for column in valid_columns:

        visual_df[column] = pd.to_numeric(
            visual_df[column],
            errors="coerce"
        )

    visual_df = visual_df.dropna(
        subset=[
            depth_column
        ]
    )

    return visual_df


# ============================================================
# 20. INTERACTIVE PETREL-STYLE WELL LOG CANVAS
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
    """
    Render the main interactive multi-track
    Petrel-style well log canvas.
    """

    track_count = 4

    fig = make_subplots(
        rows=1,
        cols=track_count,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        subplot_titles=(
            "Gamma Ray",
            "Resistivity",
            "Density / Porosity",
            "VSH"
        )
    )

    # --------------------------------------------------------
    # TRACK 1 — GAMMA RAY
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
                        color="#2ca02c",
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
                line_width=1.5,
                line_dash="dash",
                line_color="#ff4b4b",
                row=1,
                col=1
            )

            # ------------------------------------------------
            # CLEAN-SAND VISUAL INDICATOR
            # ------------------------------------------------
            #
            # Use a separate masked curve so only values
            # below the GR cutoff are visually emphasized.
            # ------------------------------------------------

            clean_sand_df = gr_df.copy()

            clean_sand_df.loc[
                clean_sand_df[gr_curve] >= gr_cutoff,
                gr_curve
            ] = np.nan

            fig.add_trace(
                go.Scatter(
                    x=clean_sand_df[gr_curve],
                    y=clean_sand_df[depth_column],
                    mode="lines",
                    line=dict(
                        color="rgba(0,0,0,0)"
                    ),
                    fill="tozerox",
                    fillcolor="rgba(255,193,7,0.18)",
                    name="Low GR",
                    hoverinfo="skip",
                    showlegend=False,
                    connectgaps=False
                ),
                row=1,
                col=1
            )

    # --------------------------------------------------------
    # TRACK 2 — RESISTIVITY
    # --------------------------------------------------------

    if resistivity_curve is not None:

        res_df = min_max_decimate(
            df,
            resistivity_curve,
            depth_column
        )

        if not res_df.empty:

            positive_res = res_df[
                res_df[resistivity_curve] > 0
            ]

            if not positive_res.empty:

                fig.add_trace(
                    go.Scatter(
                        x=positive_res[
                            resistivity_curve
                        ],
                        y=positive_res[
                            depth_column
                        ],
                        mode="lines",
                        name=str(
                            resistivity_curve
                        ),
                        line=dict(
                            color="#d62728",
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
    # TRACK 3 — DENSITY
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
                        color="#1f77b4",
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
    # TRACK 3 — NPHI
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
                        color="#9467bd",
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
    # TRACK 4 — VSH
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
                        color="#111111",
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
                line_width=1.5,
                line_dash="dash",
                line_color="#ff4b4b",
                row=1,
                col=4
            )

    # --------------------------------------------------------
    # SHARED DEPTH AXIS
    # --------------------------------------------------------

    fig.update_yaxes(
        autorange="reversed",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="white",
        spikethickness=1
    )

    # --------------------------------------------------------
    # X-AXIS TITLES
    # --------------------------------------------------------

    fig.update_xaxes(
        title_text="GR",
        row=1,
        col=1
    )

    fig.update_xaxes(
        title_text="Resistivity",
        row=1,
        col=2
    )

    fig.update_xaxes(
        title_text="RHOB / NPHI",
        row=1,
        col=3
    )

    fig.update_xaxes(
        title_text="VSH",
        range=[0, 1],
        row=1,
        col=4
    )

    # --------------------------------------------------------
    # PETREL-STYLE INTERACTION
    # --------------------------------------------------------

    fig.update_layout(
        template="plotly_dark",
        hovermode="y unified",
        height=900,
        margin=dict(
            l=50,
            r=30,
            t=60,
            b=40
        ),
        dragmode="pan",
        showlegend=False
    )

    # --------------------------------------------------------
    # REMOVE INTERIOR DEPTH AXES
    # --------------------------------------------------------

    for column in range(
        2,
        track_count + 1
    ):

        fig.update_yaxes(
            showticklabels=False,
            row=1,
            col=column
        )

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

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
# 21. AI LITHOLOGY MODEL
# ============================================================

def run_ai_lithology(
    df,
    gr_curve,
    rhob_curve,
    nphi_curve,
    gr_cutoff
):
    """
    Train and run the educational Random Forest
    lithology prototype.
    """

    if gr_curve is None:

        return None, None, (
            "Gamma Ray curve is required for the AI prototype."
        )

    if rhob_curve is None:

        return None, None, (
            "Density curve is required for the AI prototype."
        )

    if nphi_curve is None:

        return None, None, (
            "Neutron porosity curve is required for the AI prototype."
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

    y = ai_df[
        "Lithology"
    ]

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
# 22. AI LITHOLOGY DISPLAY
# ============================================================

def display_ai_lithology(
    df,
    gr_curve,
    rhob_curve,
    nphi_curve,
    gr_cutoff
):
    """
    Display AI lithology interpretation.
    """

    st.subheader(
        "🤖 AI Lithology Interpretation"
    )

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

        return

    model, ai_df, accuracy = result

    st.metric(
        "Prototype Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

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

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Sand Samples",
            sand_count
        )

    with col2:

        st.metric(
            "Predicted Shale Samples",
            shale_count
        )

    st.info(
        "This is an educational AI prototype. "
        "The training labels are generated from the GR cutoff "
        "rather than expert-labelled core or facies data."
    )


# ============================================================
# 23. MULTI-WELL ANALYSIS
# ============================================================

def display_multi_well_analysis():
    """
    Display saved wells and comparison charts.
    """

    st.subheader(
        "🗄️ Well Database"
    )

    wells = load_saved_wells()

    if wells.empty:

        st.info(
            "No saved wells are currently available."
        )

        return

    # Ensure database dataframe is Arrow-safe.
    display_wells = make_arrow_safe_dataframe(
        wells
    )

    st.dataframe(
        display_wells,
        width="stretch",
        hide_index=True
    )

    st.subheader(
        "📊 Multi-Well Comparison"
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
                )
            )
        )

        ntg_fig.update_layout(
            title="Net-to-Gross",
            xaxis_title="Well",
            yaxis_title="NTG"
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
                )
            )
        )

        vsh_fig.update_layout(
            title="Average VSH",
            xaxis_title="Well",
            yaxis_title="Average VSH"
        )

        st.plotly_chart(
            vsh_fig,
            width="stretch"
        )


# ============================================================
# 24. APPLICATION INITIALIZATION
# ============================================================

initialize_database()


# ============================================================
# 25. APPLICATION HEADER
# ============================================================

st.title(
    APP_TITLE
)

st.caption(
    APP_SUBTITLE
)

st.divider()


# ============================================================
# 26. SIDEBAR CONTROLS
# ============================================================

st.sidebar.header(
    "⚙️ Interpretation Controls"
)

gr_cutoff = st.sidebar.slider(
    "GR Sand/Shale Cutoff",
    min_value=0,
    max_value=150,
    value=75
)

vsh_cutoff = st.sidebar.slider(
    "VSH Reservoir Cutoff",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)


# ============================================================
# 27. LAS FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload LAS Well Log",
    type=["las"]
)


# ============================================================
# 28. MAIN APPLICATION WORKFLOW
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # LOAD LAS
        # ----------------------------------------------------

        las = load_las_file(
            uploaded_file
        )

        st.success(
            "✅ LAS file loaded successfully."
        )

        # ----------------------------------------------------
        # WELL INFORMATION
        # ----------------------------------------------------

        well_name, company = get_well_information(
            las
        )

        st.subheader(
            "🏭 Well Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Well",
                well_name
            )

        with col2:

            st.metric(
                "Company",
                company
            )

        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # AVAILABLE CURVES
        # ----------------------------------------------------

        st.subheader(
            "📋 Available Curves"
        )

        curve_list_df = pd.DataFrame(
            {
                "Curve": [
                    str(column)
                    for column in df.columns
                ]
            }
        )

        st.dataframe(
            curve_list_df,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # CURVE DETECTION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        with st.expander(
            "View LAS Data Preview"
        ):

            preview_df = make_arrow_safe_dataframe(
                df.head(20)
            )

            st.dataframe(
                preview_df,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # FORMATION EVALUATION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        st.subheader(
            "🧮 Formation Evaluation"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Average VSH",
                f"{average_vsh:.3f}"
            )

        with col2:

            st.metric(
                "Net-to-Gross",
                f"{ntg:.3f}"
            )

        with col3:

            st.metric(
                "Depth Samples",
                f"{len(df):,}"
            )

        # ----------------------------------------------------
        # CURVE STATUS
        # ----------------------------------------------------

        st.subheader(
            "🔎 Curve Detection"

        )

        status_col1, status_col2, status_col3, status_col4 = st.columns(4)

        with status_col1:

            if gr_curve:

                st.success(
                    f"GR: {gr_curve}"
                )

            else:

                st.warning(
                    "GR not found"
                )

        with status_col2:

            if resistivity_curve:

                st.success(
                    f"RES: {resistivity_curve}"
                )

            else:

                st.warning(
                    "RES not found"
                )

        with status_col3:

            if rhob_curve:

                st.success(
                    f"RHOB: {rhob_curve}"
                )

            else:

                st.warning(
                    "RHOB not found"
                )

        with status_col4:

            if nphi_curve:

                st.success(
                    f"NPHI: {nphi_curve}"
                )

            else:

                st.warning(
                    "NPHI not found"
                )

        # ----------------------------------------------------
        # INTERACTIVE CANVAS
        # ----------------------------------------------------

        st.subheader(
            "🖥️ Interactive Well Log Canvas"
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

        # ----------------------------------------------------
        # FORMATION SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📋 Formation Evaluation Summary"
        )

        summary_df = pd.DataFrame(
            {
                "Parameter": [
                    "Well",
                    "Company",
                    "Depth Samples",
                    "GR Cutoff",
                    "VSH Cutoff",
                    "Average VSH",
                    "Net-to-Gross"
                ],
                "Value": [
                    str(well_name),
                    str(company),
                    str(len(df)),
                    str(gr_cutoff),
                    str(vsh_cutoff),
                    str(round(average_vsh, 3)),
                    str(round(ntg, 3))
                ]
            }
        )

        # IMPORTANT:
        # Value is explicitly converted to string.
        # This prevents the PyArrow mixed-type error.
        summary_df = make_arrow_safe_dataframe(
            summary_df
        )

        st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        st.subheader(
            "💾 Save Well Analysis"
        )

        if st.button(
            "Save Well Analysis",
            type="primary"
        ):

            save_well_analysis(
                well_name=well_name,
                company=company,
                depth_samples=len(df),
                average_vsh=average_vsh,
                net_to_gross=ntg
            )

            st.success(
                f"✅ {well_name} saved successfully."
            )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        st.divider()

        display_ai_lithology(
            df=df,
            gr_curve=gr_curve,
            rhob_curve=rhob_curve,
            nphi_curve=nphi_curve,
            gr_cutoff=gr_cutoff
        )

        # ----------------------------------------------------
        # DATABASE
        # ----------------------------------------------------

        st.divider()

        display_multi_well_analysis()

    except Exception as error:

        st.error(
            "❌ PetroDashboard encountered an error "
            "while processing the LAS file."
        )

        st.exception(
            error
        )


# ============================================================
# 29. EMPTY STATE
# ============================================================

else:

    st.info(
        "👆 Upload a LAS file to begin."
    )

    st.markdown(
        """
        ### PetroDashboard

        **Current capabilities**

        - LAS well-log processing
        - Automatic curve detection
        - Interactive multi-track visualization
        - Gamma Ray analysis
        - Resistivity analysis
        - Density / neutron visualization
        - VSH calculation
        - Net-to-Gross calculation
        - Reservoir interval identification
        - SQLite well database
        - Multi-well comparison
        - AI lithology prototype

        **Visualization architecture**

        PetroDashboard uses an interactive Plotly-based
        well-log canvas with shared depth, crosshair
        interaction, logarithmic resistivity scaling,
        reservoir interpretation and visualization
        decimation.
        """
    )
