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
            well_name,
            company,
            depth_samples,
            average_vsh,
            net_to_gross
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
# 8. LAS FILE LOADING
# ============================================================

def load_las_file(uploaded_file):
    """
    Convert uploaded LAS file into a lasio object.
    """

    las_string = uploaded_file.getvalue().decode(
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
# 9. LAS → DATAFRAME
# ============================================================

def las_to_dataframe(las):
    """
    Convert LAS data into a Pandas DataFrame.
    """

    df = las.df()

    df.reset_index(
        inplace=True
    )

    return df


# ============================================================
# 10. WELL INFORMATION EXTRACTION
# ============================================================

def get_well_information(las):
    """
    Extract basic well metadata.
    """

    if "WELL" in las.well:

        well_name = str(
            las.well.WELL.value
        )

    else:

        well_name = "Unknown"

    if "COMP" in las.well:

        company = str(
            las.well.COMP.value
        )

    else:

        company = "Unknown"

    return well_name, company


# ============================================================
# 11. DEPTH COLUMN DETECTION
# ============================================================

def get_depth_column(df):
    """
    Identify the depth/index column.
    """

    return df.columns[0]


# ============================================================
# 12. RESISTIVITY CURVE DETECTION
# ============================================================

def find_resistivity_curve(df):
    """
    Automatically identify a resistivity curve.
    """

    for column in df.columns:

        column_upper = str(
            column
        ).upper()

        for keyword in RESISTIVITY_KEYWORDS:

            if keyword in column_upper:

                return column

    return None


# ============================================================
# 13. VSH CALCULATION
# ============================================================

def calculate_vsh(
    df,
    gr_column="GR"
):
    """
    Calculate normalized Gamma Ray derived VSH.
    """

    if gr_column not in df.columns:

        return df, 0.0

    gr_min = df[gr_column].min()

    gr_max = df[gr_column].max()

    if gr_max == gr_min:

        return df, 0.0

    df["VSH"] = (
        (df[gr_column] - gr_min)
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
# 14. RESERVOIR FLAG
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
# 15. NET-TO-GROSS
# ============================================================

def calculate_ntg(df):
    """
    Calculate Net-to-Gross.
    """

    if "RES_FLAG" not in df.columns:

        return 0.0

    return float(
        df["RES_FLAG"].mean()
    )


# ============================================================
# 16. MIN-MAX DECIMATION
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

    if len(df) <= max_points:

        return df.copy()

    bucket_count = max(
        1,
        max_points // 2
    )

    indices = np.linspace(
        0,
        len(df) - 1,
        bucket_count
    ).astype(int)

    selected_indices = set()

    for i in range(
        len(indices) - 1
    ):

        start = indices[i]

        end = indices[i + 1]

        section = df.iloc[
            start:end + 1
        ]

        if section.empty:

            continue

        min_index = section[
            x_column
        ].idxmin()

        max_index = section[
            x_column
        ].idxmax()

        selected_indices.add(
            min_index
        )

        selected_indices.add(
            max_index
        )

    selected_indices = sorted(
        selected_indices
    )

    return df.loc[
        selected_indices
    ].sort_values(
        depth_column
    )


# ============================================================
# 17. VISUAL DATA PREPARATION
# ============================================================

def prepare_visual_data(
    df,
    depth_column,
    curve_columns
):
    """
    Prepare decimated datasets for Plotly.
    """

    visual_df = df[
        [
            depth_column
        ]
        +
        curve_columns
    ].copy()

    visual_df = visual_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    visual_df = visual_df.dropna(
        subset=curve_columns,
        how="all"
    )

    return visual_df


# ============================================================
# 18. INTERACTIVE PETREL-STYLE WELL LOG CANVAS
# ============================================================

@st.fragment
def render_well_log_canvas(
    df,
    depth_column,
    gr_cutoff,
    vsh_cutoff,
    resistivity_curve
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
    # DEPTH
    # --------------------------------------------------------

    depth = df[
        depth_column
    ]

    # --------------------------------------------------------
    # TRACK 1 — GAMMA RAY
    # --------------------------------------------------------

    if "GR" in df.columns:

        gr_df = min_max_decimate(
            df,
            "GR",
            depth_column
        )

        fig.add_trace(
            go.Scatter(
                x=gr_df["GR"],
                y=gr_df[depth_column],
                mode="lines",
                name="GR",
                line=dict(
                    color="#2ca02c",
                    width=1.5
                ),
                hovertemplate=(
                    "Depth: %{y}<br>"
                    "GR: %{x:.2f}<extra></extra>"
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

        # ----------------------------------------------------
        # GR CLEAN-SAND SHADING
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=gr_df["GR"],
                y=gr_df[depth_column],
                mode="lines",
                line=dict(
                    width=0
                ),
                fill="tozerox",
                fillcolor="rgba(255, 193, 7, 0.18)",
                name="Low GR",
                hoverinfo="skip",
                showlegend=False
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

        fig.add_trace(
            go.Scatter(
                x=res_df[
                    resistivity_curve
                ],
                y=res_df[
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
                    "Depth: %{y}<br>"
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

    if "RHOB" in df.columns:

        rhob_df = min_max_decimate(
            df,
            "RHOB",
            depth_column
        )

        fig.add_trace(
            go.Scatter(
                x=rhob_df["RHOB"],
                y=rhob_df[
                    depth_column
                ],
                mode="lines",
                name="RHOB",
                line=dict(
                    color="#1f77b4",
                    width=1.5
                ),
                hovertemplate=(
                    "Depth: %{y}<br>"
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

    if "NPHI" in df.columns:

        nphi_df = min_max_decimate(
            df,
            "NPHI",
            depth_column
        )

        fig.add_trace(
            go.Scatter(
                x=nphi_df["NPHI"],
                y=nphi_df[
                    depth_column
                ],
                mode="lines",
                name="NPHI",
                line=dict(
                    color="#9467bd",
                    width=1.5
                ),
                hovertemplate=(
                    "Depth: %{y}<br>"
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

        fig.add_trace(
            go.Scatter(
                x=vsh_df["VSH"],
                y=vsh_df[
                    depth_column
                ],
                mode="lines",
                name="VSH",
                line=dict(
                    color="#111111",
                    width=1.5
                ),
                hovertemplate=(
                    "Depth: %{y}<br>"
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
# 19. AI LITHOLOGY MODEL
# ============================================================

def run_ai_lithology(
    df,
    gr_cutoff
):
    """
    Train and run the educational Random Forest
    lithology prototype.
    """

    required_curves = [
        "GR",
        "RHOB",
        "NPHI"
    ]

    missing = [
        curve
        for curve in required_curves
        if curve not in df.columns
    ]

    if missing:

        return None, None, (
            f"Missing curves: {', '.join(missing)}"
        )

    ai_df = df[
        required_curves
    ].copy()

    ai_df = ai_df.replace(
        [np.inf, -np.inf],
        np.nan
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
# 20. AI LITHOLOGY DISPLAY
# ============================================================

def display_ai_lithology(
    df,
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
# 21. MULTI-WELL ANALYSIS
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

    st.dataframe(
        wells,
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
                x=wells["well_name"],
                y=wells["net_to_gross"]
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
                x=wells["well_name"],
                y=wells["average_vsh"]
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
# 22. APPLICATION INITIALIZATION
# ============================================================

initialize_database()


# ============================================================
# 23. APPLICATION HEADER
# ============================================================

st.title(
    APP_TITLE
)

st.caption(
    APP_SUBTITLE
)

st.divider()


# ============================================================
# 24. SIDEBAR CONTROLS
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
# 25. LAS FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload LAS Well Log",
    type=["las"]
)


# ============================================================
# 26. MAIN APPLICATION WORKFLOW
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

        depth_column = get_depth_column(
            df
        )

        # ----------------------------------------------------
        # AVAILABLE CURVES
        # ----------------------------------------------------

        st.subheader(
            "📋 Available Curves"
        )

        st.write(
            list(df.columns)
        )

        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        with st.expander(
            "View LAS Data Preview"
        ):

            st.dataframe(
                df.head(20),
                width="stretch"
            )

        # ----------------------------------------------------
        # FORMATION EVALUATION
        # ----------------------------------------------------

        df, average_vsh = calculate_vsh(
            df
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
        # RESISTIVITY DETECTION
        # ----------------------------------------------------

        resistivity_curve = find_resistivity_curve(
            df
        )

        if resistivity_curve:

            st.success(
                f"Resistivity detected: "
                f"{resistivity_curve}"
            )

        else:

            st.warning(
                "No resistivity curve detected."
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
            resistivity_curve=resistivity_curve
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
                    well_name,
                    company,
                    len(df),
                    gr_cutoff,
                    vsh_cutoff,
                    round(
                        average_vsh,
                        3
                    ),
                    round(
                        ntg,
                        3
                    )
                ]
            }
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
            df,
            gr_cutoff
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
# 27. EMPTY STATE
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
