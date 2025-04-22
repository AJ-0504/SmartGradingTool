import streamlit as st
import time
import plotly.express as px

import pandas as pd
from utils import (
    load_data, clean_data, detect_marks_column, compute_grade_boundaries,
    assign_grades, validate_boundaries, plot_grade_distribution, to_excel,
    smart_ai_grade_suggestion
)


def main():
    """
    Entry point for the Smart Grading Assistant Streamlit app.
    """
    # Page config and dark theme CSS
    st.set_page_config(
        page_title="Smart Grading Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    .stApp {background-color: #121212; color: #e0e0e0;}
    .stSidebar {background-color: #1e1e1e; color: #e0e0e0;}
    .stButton>button {background-color: #bb86fc; color: #121212; border:none; padding:8px 16px; border-radius:4px;}
    h1, h2, h3 {color: #e0e0e0;}
    .main .block-container {padding-top:1rem;}
    </style>
    """, unsafe_allow_html=True)

    # Branding (placeholder logo)
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("logo.png", width=100)  # Adjust the width as needed
    with col2:
        st.title("Smart Grading Assistant")
        st.markdown("_One-Stop Solution for Teachers to Grade & Analyze Exams_")

    # Startup spinner
    with st.spinner("Loading Smart Grading Assistant..."):
        time.sleep(0.5)

    # Sidebar: grade labels
    st.sidebar.header("Grade Settings")
    labels_input = st.sidebar.text_input("Grade Labels (comma-separated)", "A+,A,B,C,D,E,F")
    grade_labels = [lbl.strip() for lbl in labels_input.split(',') if lbl.strip()]
    if len(grade_labels) < 2:
        st.sidebar.error("Enter at least two unique grade labels.")
        st.stop()

    # File upload
    uploaded = st.file_uploader("Upload Marks File (CSV/XLSX)", type=["csv","xlsx"])
    if not uploaded:
        st.info("Awaiting file upload to begin grading.")
        st.stop()

    # Load data and initialize session
    df_orig = load_data(uploaded)
    if 'graded_df' not in st.session_state:
        st.session_state.graded_df = df_orig.copy()
    df = st.session_state.graded_df

    # Fake progress bar
    with st.spinner("Processing data..."):
        prog = st.progress(0)
        for i in range(100):
            time.sleep(0.002)
            prog.progress(i+1)
    st.success("Data loaded!")

    # Select numeric columns
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notna().any()]
    selected_cols = st.multiselect("Select columns to grade", numeric_cols)
    if not selected_cols:
        st.info("Choose at least one column to proceed.")
        st.stop()

    # Sidebar: per-column settings
    col_to_grade = st.sidebar.selectbox("Column to Grade", selected_cols)
    min_val = st.sidebar.number_input(f"Min valid mark for {col_to_grade}", value=0.0, step=1.0)
    max_val = st.sidebar.number_input(f"Max valid mark for {col_to_grade}", value=100.0, step=1.0)
    st.session_state[f"min_{col_to_grade}"] = min_val
    st.session_state[f"max_{col_to_grade}"] = max_val

    # AI suggestion
    if st.sidebar.button("💡 Suggest Centric Grade"):
        sugg = smart_ai_grade_suggestion(df, col_to_grade, grade_labels, min_val, max_val)
        st.sidebar.success(f"AI suggests: {sugg}")
        st.session_state[f"suggest_{col_to_grade}"] = sugg

    default_cent = st.session_state.get(f"suggest_{col_to_grade}", grade_labels[len(grade_labels)//2])
    centric = st.sidebar.selectbox("Centric Grade", grade_labels, index=grade_labels.index(default_cent))

    # Clean data and detect errors
    df_clean = clean_data(df, col_to_grade)
    mask_err = detect_marks_column(df_clean, col_to_grade, min_val, max_val)
    df_valid = df_clean[~mask_err].dropna(subset=[col_to_grade])

    st.info(f"{col_to_grade}: {len(df_valid)} valid, {len(df_clean)-len(df_valid)} invalid entries.")

    # Compute boundaries
    if df_valid.empty:
        st.warning("No valid marks; all will be 'Error'.")
        boundaries = {g:0.0 for g in grade_labels}
    else:
        boundaries = compute_grade_boundaries(df_valid, col_to_grade, grade_labels, centric)

    # Tabs: grading and analysis
    tab1, tab2 = st.tabs(["Grade Assignment","Analysis"])

    with tab1:
        st.sidebar.subheader("Adjust Boundaries")
        manual = {}
        for g in grade_labels[:-1]:
            manual[g] = st.sidebar.slider(f"Min for {g}", 0.0,100.0,float(boundaries[g]),step=0.1)
        manual[grade_labels[-1]] = 0.0
        if validate_boundaries(grade_labels, manual):
            boundaries = manual
            st.sidebar.success("Manual boundaries applied.")
        else:
            st.sidebar.error("Boundaries must strictly decrease.")

        df_graded = assign_grades(df_clean, col_to_grade, f"{col_to_grade}_grade",
                                  grade_labels, boundaries, min_val, max_val)

        c1, c2 = st.columns([1,2])
        with c1:
            st.subheader("Boundaries")
            st.table(pd.DataFrame(list(boundaries.items()), columns=['Grade','Min Mark']))
        with c2:
            st.subheader("Distribution")
            st.plotly_chart(plot_grade_distribution(df_graded, grade_labels, f"{col_to_grade}_grade"), use_container_width=True)

        if st.button(f"Save Grades for {col_to_grade}"):
            st.session_state.graded_df[f"{col_to_grade}_grade"] = df_graded[f"{col_to_grade}_grade"]
            st.success("Grades saved!")

        st.subheader("Preview")
        st.dataframe(st.session_state.graded_df.head(10), use_container_width=True)

        st.subheader("Download Results")
        ca, cb = st.columns(2)
        with ca:
            csv = st.session_state.graded_df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "grades.csv")
        with cb:
            xlsx = to_excel(st.session_state.graded_df)
            st.download_button("Download Excel", xlsx, "grades.xlsx")

    with tab2:
        st.subheader("Statistics")
        stats_cols = st.multiselect("Select columns for statistics", selected_cols, default=[col_to_grade])
        if stats_cols:
            stats = []
            for c in stats_cols:
                mn = st.session_state.get(f"min_{c}",0.0)
                mx = st.session_state.get(f"max_{c}",100.0)
                data = pd.to_numeric(df[c],errors='coerce')
                data = data[(data>=mn)&(data<=mx)]
                stats.append({
                    'Column':c,
                    'Mean':f"{data.mean():.2f}",
                    'Median':f"{data.median():.2f}",
                    'Std Dev':f"{data.std():.2f}",
                    'Min':f"{data.min():.2f}",
                    'Max':f"{data.max():.2f}"
                })
            st.table(pd.DataFrame(stats))
        else:
            st.info("Select at least one column for stats.")

        st.subheader("Visualizations")
        plot_type = st.selectbox("Plot Type",["Histogram","Box Plot","Bar Chart"])
        viz_cols = st.multiselect("Columns to visualize", selected_cols, default=[col_to_grade])
        for c in viz_cols:
            mn = st.session_state.get(f"min_{c}",0.0)
            mx = st.session_state.get(f"max_{c}",100.0)
            data = pd.to_numeric(df[c],errors='coerce')
            data = data[(data>=mn)&(data<=mx)]
            if data.empty:
                st.warning(f"No valid data for {c}; skipping.")
                continue
            if plot_type=="Histogram":
                fig=px.histogram(data,x=c,title=f"{c} Distribution",template='plotly_dark')
            elif plot_type=="Box Plot":
                fig=px.box(data,y=c,title=f"{c} Box Plot",template='plotly_dark')
            else:
                grade_col=f"{c}_grade"
                if grade_col in st.session_state.graded_df.columns:
                    fig=plot_grade_distribution(st.session_state.graded_df,grade_labels,grade_col)
                else:
                    st.warning(f"Grades for {c} not found.")
                    continue
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{c}")


if __name__ == "__main__":
    main()
