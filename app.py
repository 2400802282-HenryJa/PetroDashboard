import sqlite3
from io import StringIO
from pathlib import Path

import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Petroleum Engineering Dashboard",
    page_icon="🛢️",
    layout="wide"
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "petroleum_dashboard.db"


def initialize_database():
    """
    Create the SQLite database and wells table if they do not exist.
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


def save_well_to_database(
    well_name,
    company,
    depth_samples,
    average_vsh,
    net_to_gross
):
    """
    Save formation evaluation results to SQLite.
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


def load_saved_wells():
    """
    Load saved wells from the SQLite database.
    """

    conn = sqlite3.connect(DB_PATH)

    df_wells = pd.read_sql_query(
        "SELECT * FROM wells",
        conn
    )

    conn.close()

    return df_wells


# Initialize database
initialize_database()


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title(
    "🛢️ Petroleum Engineering Dashboard"
)

st.write(
    "Interactive LAS Well Log Viewer "
    "with Formation Evaluation and AI Lithology Prediction"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Formation Evaluation Settings"
)

gr_cutoff = st.sidebar.slider(
    "GR Sand/Shale Cutoff",
    min_value=0,
    max_value=150,
    value=75
)

vsh_cutoff = st.sidebar.slider(
    "Maximum VSH Reservoir Cutoff",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.info(
    "Upload a LAS well log file to begin the analysis."
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload LAS Well Log File",
    type=["las"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    try:

        # ====================================================
        # READ LAS FILE
        # ====================================================

        st.subheader(
            "📥 LAS File Processing"
        )

        las_string = uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

        las_file = StringIO(
            las_string
        )

        las = lasio.read(
            las_file,
            engine="normal"
        )

        st.success(
            "✅ LAS file loaded successfully!"
        )

        # ====================================================
        # WELL INFORMATION
        # ====================================================

        st.subheader(
            "🏭 Well Information"
        )

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

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Well Name",
                well_name
            )

        with col2:

            st.metric(
                "Company",
                company
            )

        # ====================================================
        # AVAILABLE LOGS
        # ====================================================

        st.subheader(
            "📋 Available Logs"
        )

        curve_names = [
            curve.mnemonic
            for curve in las.curves
        ]

        st.write(
            curve_names
        )

        # ====================================================
        # CONVERT LAS TO DATAFRAME
        # ====================================================

        df = las.df()

        df.reset_index(
            inplace=True
        )

        st.subheader(
            "📊 Log Data Preview"
        )

        st.dataframe(
            df.head(20),
            width="stretch"
        )

        st.write(
            f"Number of depth samples: **{len(df)}**"
        )

        # ====================================================
        # FORMATION EVALUATION
        # ====================================================

        st.subheader(
            "🧮 Formation Evaluation"
        )

        avg_vsh = 0.0

        ntg = 0.0

        if "GR" in df.columns:

            gr_min = df["GR"].min()

            gr_max = df["GR"].max()

            if gr_max != gr_min:

                df["VSH"] = (
                    (df["GR"] - gr_min)
                    /
                    (gr_max - gr_min)
                )

                df["VSH"] = df["VSH"].clip(
                    0,
                    1
                )

                df["RES_FLAG"] = np.where(
                    df["VSH"] < vsh_cutoff,
                    1,
                    0
                )

                ntg = float(
                    df["RES_FLAG"].mean()
                )

                avg_vsh = float(
                    df["VSH"].mean()
                )

            else:

                st.warning(
                    "GR values are constant, so VSH cannot be calculated."
                )

        else:

            st.warning(
                "⚠️ No GR log found. VSH and NTG cannot be calculated."
            )

        # ====================================================
        # FORMATION EVALUATION METRICS
        # ====================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Net-to-Gross",
                f"{ntg:.2f}"
            )

        with col2:

            st.metric(
                "Average VSH",
                f"{avg_vsh:.2f}"
            )

        with col3:

            st.metric(
                "Depth Samples",
                f"{len(df):,}"
            )

        # ====================================================
        # INTERACTIVE WELL LOG VISUALIZATION
        # ====================================================

        st.subheader(
            "📈 Interactive Well Log Visualization"
        )

        depth = df.iloc[:, 0]

        fig = make_subplots(
            rows=1,
            cols=4,
            shared_yaxes=True,
            horizontal_spacing=0.03,
            subplot_titles=(
                "Gamma Ray",
                "Resistivity",
                "Density / Porosity",
                "VSH"
            )
        )

        # ====================================================
        # TRACK 1 — GAMMA RAY
        # ====================================================

        if "GR" in df.columns:

            fig.add_trace(
                go.Scatter(
                    x=df["GR"],
                    y=depth,
                    mode="lines",
                    name="GR",
                    line=dict(
                        color="green"
                    )
                ),
                row=1,
                col=1
            )

            fig.add_vline(
                x=gr_cutoff,
                line_width=2,
                line_dash="dash",
                line_color="red",
                row=1,
                col=1
            )

            fig.update_xaxes(
                title_text="GR",
                row=1,
                col=1
            )

        # ====================================================
        # TRACK 2 — RESISTIVITY
        # ====================================================

        resistivity_keywords = [
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

        res_curve = None

        for column in df.columns:

            column_upper = str(
                column
            ).upper()

            for keyword in resistivity_keywords:

                if keyword in column_upper:

                    res_curve = column

                    break

            if res_curve is not None:

                break

        if res_curve is not None:

            st.success(
                f"✅ Detected Resistivity Curve: {res_curve}"
            )

            fig.add_trace(
                go.Scatter(
                    x=df[res_curve],
                    y=depth,
                    mode="lines",
                    name=str(res_curve),
                    line=dict(
                        color="red"
                    )
                ),
                row=1,
                col=2
            )

            fig.update_xaxes(
                type="log",
                title_text="Resistivity",
                row=1,
                col=2
            )

        else:

            st.warning(
                "⚠️ No resistivity log detected."
            )

        # ====================================================
        # TRACK 3 — DENSITY
        # ====================================================

        if "RHOB" in df.columns:

            fig.add_trace(
                go.Scatter(
                    x=df["RHOB"],
                    y=depth,
                    mode="lines",
                    name="RHOB",
                    line=dict(
                        color="blue"
                    )
                ),
                row=1,
                col=3
            )

            fig.update_xaxes(
                title_text="RHOB",
                row=1,
                col=3
            )

        else:

            st.warning(
                "⚠️ No RHOB log detected."
            )

        # ====================================================
        # TRACK 4 — VSH
        # ====================================================

        if "VSH" in df.columns:

            fig.add_trace(
                go.Scatter(
                    x=df["VSH"],
                    y=depth,
                    mode="lines",
                    name="VSH",
                    line=dict(
                        color="black"
                    )
                ),
                row=1,
                col=4
            )

            fig.update_xaxes(
                range=[0, 1],
                title_text="VSH",
                row=1,
                col=4
            )

        # ====================================================
        # FIGURE LAYOUT
        # ====================================================

        fig.update_layout(
            height=900,
            width=1200,
            showlegend=True
        )

        fig.update_yaxes(
            title_text="Depth",
            autorange="reversed",
            row=1,
            col=1
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # ====================================================
        # SAVE ANALYSIS TO DATABASE
        # ====================================================

        st.subheader(
            "💾 Save Well Analysis"
        )

        st.write(
            "Save the current well and formation evaluation "
            "results to the local SQLite database."
        )

        if st.button(
            "💾 Save Well Analysis to Database",
            type="primary"
        ):

            save_well_to_database(
                well_name=well_name,
                company=company,
                depth_samples=int(len(df)),
                average_vsh=float(avg_vsh),
                net_to_gross=float(ntg)
            )

            st.success(
                f"✅ {well_name} successfully saved to the database."
            )

        # ====================================================
        # FORMATION EVALUATION TABLE
        # ====================================================

        st.subheader(
            "📋 Formation Evaluation Summary"
        )

        evaluation_data = {
            "Parameter": [
                "Well Name",
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
                round(avg_vsh, 3),
                round(ntg, 3)
            ]
        }

        evaluation_df = pd.DataFrame(
            evaluation_data
        )

        st.dataframe(
            evaluation_df,
            width="stretch",
            hide_index=True
        )

        # ====================================================
        # AI LITHOLOGY PREDICTION
        # ====================================================

        st.subheader(
            "🤖 AI Lithology Prediction"
        )

        st.write(
            "A Random Forest classifier is trained using "
            "GR, RHOB and NPHI. For this prototype, the "
            "training labels are generated from the selected "
            "GR cutoff."
        )

        required_curves = [
            "GR",
            "RHOB",
            "NPHI"
        ]

        missing_curves = [
            curve
            for curve in required_curves
            if curve not in df.columns
        ]

        if len(missing_curves) == 0:

            ai_df = df[
                required_curves
            ].copy()

            ai_df.replace(
                [np.inf, -np.inf],
                np.nan,
                inplace=True
            )

            ai_df.dropna(
                inplace=True
            )

            if len(ai_df) >= 20:

                # --------------------------------------------
                # GENERATE EDUCATIONAL TRAINING LABELS
                # --------------------------------------------

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

                if y.nunique() >= 2:

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42,
                        stratify=y
                    )

                    # ----------------------------------------
                    # RANDOM FOREST MODEL
                    # ----------------------------------------

                    model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )

                    model.fit(
                        X_train,
                        y_train
                    )

                    # ----------------------------------------
                    # MODEL ACCURACY
                    # ----------------------------------------

                    y_pred = model.predict(
                        X_test
                    )

                    accuracy = accuracy_score(
                        y_test,
                        y_pred
                    )

                    st.metric(
                        "AI Model Accuracy",
                        f"{accuracy * 100:.2f}%"
                    )

                    # ----------------------------------------
                    # WHOLE WELL PREDICTION
                    # ----------------------------------------

                    ai_df["Predicted_Lithology"] = model.predict(
                        X
                    )

                    # ----------------------------------------
                    # PREDICTION SUMMARY
                    # ----------------------------------------

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

                    # ----------------------------------------
                    # AI LITHOLOGY PLOT
                    # ----------------------------------------

                    fig_ai, ax = plt.subplots(
                        figsize=(6, 10)
                    )

                    scatter = ax.scatter(
                        ai_df["Predicted_Lithology"],
                        ai_df.index,
                        c=ai_df["Predicted_Lithology"],
                        cmap="viridis",
                        s=12
                    )

                    ax.set_xlabel(
                        "Predicted Lithology"
                    )

                    ax.set_ylabel(
                        "Sample Index"
                    )

                    ax.set_title(
                        "AI Lithology Prediction"
                    )

                    ax.invert_yaxis()

                    ax.set_xticks(
                        [0, 1]
                    )

                    ax.set_xticklabels(
                        [
                            "Shale",
                            "Sand"
                        ]
                    )

                    st.pyplot(
                        fig_ai
                    )

                    # ----------------------------------------
                    # FEATURE IMPORTANCE
                    # ----------------------------------------

                    st.write(
                        "### Feature Importance"
                    )

                    feature_importance = pd.DataFrame(
                        {
                            "Feature": [
                                "GR",
                                "RHOB",
                                "NPHI"
                            ],
                            "Importance": model.feature_importances_
                        }
                    )

                    feature_importance = feature_importance.sort_values(
                        "Importance",
                        ascending=False
                    )

                    st.dataframe(
                        feature_importance,
                        width="stretch",
                        hide_index=True
                    )

                    st.info(
                        "Note: This AI workflow is an educational prototype. "
                        "The training labels are generated from the GR cutoff "
                        "rather than from laboratory core or expert-labelled "
                        "facies data. Therefore, the reported accuracy should "
                        "not be interpreted as professional geological model "
                        "validation."
                    )

                else:

                    st.warning(
                        "AI prediction requires at least two lithology classes "
                        "in the training data. Adjust the GR cutoff."
                    )

            else:

                st.warning(
                    "Not enough valid samples are available for AI training."
                )

        else:

            st.warning(
                "AI lithology prediction requires these curves: "
                "GR, RHOB and NPHI."
            )

            st.write(
                "Missing curves:",
                missing_curves
            )

        # ====================================================
        # SAVED WELL DATABASE
        # ====================================================

        st.divider()

        st.subheader(
            "🗄️ Saved Well Database"
        )

        saved_wells = load_saved_wells()

        if not saved_wells.empty:

            st.dataframe(
                saved_wells,
                width="stretch",
                hide_index=True
            )

            # -----------------------------------------------
            # MULTI-WELL ANALYSIS
            # -----------------------------------------------

            st.subheader(
                "📊 Multi-Well Analysis"
            )

            col1, col2 = st.columns(2)

            with col1:

                fig_ntg = go.Figure()

                fig_ntg.add_trace(
                    go.Bar(
                        x=saved_wells["well_name"],
                        y=saved_wells["net_to_gross"],
                        name="NTG"
                    )
                )

                fig_ntg.update_layout(
                    title="Net-to-Gross Comparison",
                    xaxis_title="Well",
                    yaxis_title="Net-to-Gross"
                )

                st.plotly_chart(
                    fig_ntg,
                    width="stretch"
                )

            with col2:

                fig_vsh = go.Figure()

                fig_vsh.add_trace(
                    go.Bar(
                        x=saved_wells["well_name"],
                        y=saved_wells["average_vsh"],
                        name="Average VSH"
                    )
                )

                fig_vsh.update_layout(
                    title="Average VSH Comparison",
                    xaxis_title="Well",
                    yaxis_title="Average VSH"
                )

                st.plotly_chart(
                    fig_vsh,
                    width="stretch"
                )

        else:

            st.info(
                "No wells have been saved to the database yet."
            )

    except Exception as e:

        st.error(
            "❌ An error occurred while processing the LAS file."
        )

        st.exception(
            e
        )


# ============================================================
# NO FILE UPLOADED
# ============================================================

else:

    st.info(
        "👆 Upload a LAS file above to start the well log analysis."
    )

    st.markdown(
        """
        ### PetroDashboard Features

        - 📂 LAS well log upload
        - 🏭 Well information extraction
        - 📋 Available log identification
        - 📈 Interactive well log visualization
        - 🧮 VSH calculation
        - 🪨 Net-to-Gross calculation
        - 🛢️ Resistivity detection
        - 💾 SQLite well database
        - 📊 Multi-well comparison
        - 🤖 Random Forest lithology prototype
        """
    )
