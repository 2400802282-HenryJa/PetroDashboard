import streamlit as st
import lasio
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from io import StringIO

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------
# DATABASE CONNECTION
# ----------------------------------------

conn = sqlite3.connect(
    "petroleum_dashboard.db"
)

cursor = conn.cursor()

# ----------------------------------------
# CREATE TABLE
# ----------------------------------------

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

# ----------------------------------------
# DASHBOARD TITLE
# ----------------------------------------

st.title(
    "Petroleum Engineering Dashboard"
)

st.write(
    "Interactive LAS Well Log Viewer "
    "with AI Lithology Prediction"
)

# ----------------------------------------
# FILE UPLOADER
# ----------------------------------------

uploaded_file = st.file_uploader(
    "Upload LAS File",
    type=["las"]
)

# ----------------------------------------
# PROCESS FILE
# ----------------------------------------

if uploaded_file is not None:

    try:

        # ----------------------------------------
        # READ LAS FILE
        # ----------------------------------------

        las_string = uploaded_file.getvalue().decode(
            "utf-8"
        )

        las_file = StringIO(las_string)

        las = lasio.read(
            las_file,
            engine="normal"
        )

        st.success(
            "LAS file loaded successfully!"
        )

        # ----------------------------------------
        # WELL INFORMATION
        # ----------------------------------------

        st.subheader(
            "Well Information"
        )

        well_name = (
            las.well.WELL.value
            if "WELL" in las.well
            else "Unknown"
        )

        company = (
            las.well.COMP.value
            if "COMP" in las.well
            else "Unknown"
        )

        st.write(
            f"Well Name: {well_name}"
        )

        st.write(
            f"Company: {company}"
        )

        # ----------------------------------------
        # AVAILABLE LOGS
        # ----------------------------------------

        st.subheader(
            "Available Logs"
        )

        curve_names = [
            curve.mnemonic
            for curve in las.curves
        ]

        st.write(curve_names)

        # ----------------------------------------
        # DATAFRAME
        # ----------------------------------------

        df = las.df()

        df.reset_index(inplace=True)

        st.subheader(
            "Log Data Preview"
        )

        st.dataframe(
            df.head()
        )

        # ----------------------------------------
        # SIDEBAR SETTINGS
        # ----------------------------------------

        st.sidebar.header(
            "Formation Evaluation Settings"
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
            value=0.5
        )

        # ----------------------------------------
        # VSH CALCULATION
        # ----------------------------------------

        avg_vsh = 0
        ntg = 0

        if "GR" in df.columns:

            gr_min = df["GR"].min()

            gr_max = df["GR"].max()

            df["VSH"] = (
                (df["GR"] - gr_min)
                /
                (gr_max - gr_min)
            )

            # Clamp values
            df["VSH"] = df["VSH"].clip(
                0,
                1
            )

            # Reservoir Flag
            df["RES_FLAG"] = np.where(
                df["VSH"] < vsh_cutoff,
                1,
                0
            )

            # Net-to-Gross
            ntg = df["RES_FLAG"].mean()

            # Average VSH
            avg_vsh = df["VSH"].mean()

        else:

            st.warning(
                "No GR log found."
            )

        # ----------------------------------------
        # FORMATION EVALUATION METRICS
        # ----------------------------------------

        st.subheader(
            "Formation Evaluation Metrics"
        )

        col1, col2 = st.columns(2)

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

        # ----------------------------------------
        # INTERACTIVE VISUALIZATION
        # ----------------------------------------

        st.subheader(
            "Interactive Well Log Visualization"
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

        # ----------------------------------------
        # TRACK 1 — GAMMA RAY
        # ----------------------------------------

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

        # ----------------------------------------
        # TRACK 2 — RESISTIVITY
        # ----------------------------------------

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

            column_upper = column.upper()

            for keyword in resistivity_keywords:

                if keyword in column_upper:

                    res_curve = column

                    break

            if res_curve is not None:

                break

        if res_curve is not None:

            st.success(
                f"Detected Resistivity Curve: "
                f"{res_curve}"
            )

            fig.add_trace(
                go.Scatter(
                    x=df[res_curve],
                    y=depth,
                    mode="lines",
                    name=res_curve,
                    line=dict(
                        color="red"
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

        else:

            st.warning(
                "No resistivity log detected."
            )

        # ----------------------------------------
        # TRACK 3 — DENSITY / POROSITY
        # ----------------------------------------

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
        else:
            st.warning(
                "No RHOB log detected."
            )

        # ----------------------------------------
        # TRACK 4 — VSH
        # ----------------------------------------

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

        # ----------------------------------------
        # GLOBAL FIGURE LAYOUT ADJUSTMENTS
        # ----------------------------------------
        
        fig.update_layout(
            height=800,
            width=1000,
            yaxis=dict(
                title="Depth",
                autorange="reverse"  # Deepest depths at the bottom
            )
        )
        
        st.plotly_chart(
            fig, 
            use_container_width=True
        )

        # ----------------------------------------
        # DB LOGGING STORAGE OPERATION
        # ----------------------------------------
        
        cursor.execute(
            """
            INSERT INTO wells (well_name, company, depth_samples, average_vsh, net_to_gross)
            VALUES (?, ?, ?, ?, ?)
            """,
            (well_name, company, int(len(df)), float(avg_vsh), float(ntg))
        )
        conn.commit()
        st.info("Analysis metric details successfully logged to database storage.")

    except Exception as e:
