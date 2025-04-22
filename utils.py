# utils.py

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objs as go


def load_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


@st.cache_data
def to_excel(df):
    """
    Convert DataFrame to Excel bytes for download.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Grades')
    return output.getvalue()

@st.cache_data
def clean_data(df: pd.DataFrame, marks_column: str) -> pd.DataFrame:
    """
    Convert the marks column to numeric, coercing invalid entries to NaN.
    """
    df = df.copy()
    df[marks_column] = pd.to_numeric(df[marks_column], errors='coerce')
    return df


def detect_marks_column(
    df: pd.DataFrame,
    marks_column: str,
    min_val: float = 0,
    max_val: float = 100
) -> pd.Series:
    """
    Identify rows where marks are NaN or out of the valid range.
    """
    return (
        df[marks_column].isna() |
        (df[marks_column] < min_val) |
        (df[marks_column] > max_val)
    )

@st.cache_data
def compute_grade_boundaries(
    df: pd.DataFrame,
    marks_column: str,
    grade_labels: list[str],
    grade_centric: str,
    manual_boundaries: dict[str, float] | None = None
) -> dict[str, float]:
    """
    Compute grade boundaries for each label.
    Uses manual boundaries if provided, otherwise distributes grades
    around a centric grade using a weighted algorithm.
    """
    df_valid = df.dropna(subset=[marks_column])
    total = len(df_valid)
    center_idx = grade_labels.index(grade_centric)

    if manual_boundaries:
        return manual_boundaries

    # Generate weights peaking at the centric grade
    weights = np.array([abs(i - center_idx) for i in range(len(grade_labels))])
    weights = np.max(weights) - weights + 1
    weights = weights / weights.sum()
    counts = (weights * total).astype(int)
    counts[-1] += total - counts.sum()  # adjust remainder

    df_sorted = df_valid.sort_values(by=marks_column, ascending=False).reset_index(drop=True)
    boundaries = {}
    start = 0
    for i, grade in enumerate(grade_labels[:-1]):
        end = min(start + counts[i], total)
        if start < total:
            boundaries[grade] = df_sorted.iloc[end - 1][marks_column]
        start = end
    boundaries[grade_labels[-1]] = 0.0
    return boundaries

@st.cache_data
def assign_grades(
    df: pd.DataFrame,
    marks_column: str,
    grade_column: str,
    grade_labels: list[str],
    grade_boundaries: dict[str, float],
    min_val: float = 0,
    max_val: float = 100
) -> pd.DataFrame:
    """
    Assign grades to each entry based on boundaries. Marks outside range or NaN
    receive an 'Error' grade.
    """
    df = df.copy()
    mask_error = detect_marks_column(df, marks_column, min_val, max_val)
    numeric = pd.to_numeric(df[marks_column], errors='coerce')

    def get_grade(x):
        if pd.isna(x):
            return 'Error'
        for label, bound in grade_boundaries.items():
            if x >= bound:
                return label
        return grade_labels[-1]

    df[grade_column] = numeric.map(get_grade)
    df.loc[mask_error, grade_column] = 'Error'
    return df


def validate_boundaries(
    grade_labels: list[str],
    boundaries: dict[str, float]
) -> bool:
    """
    Check that manual boundaries strictly decrease from highest to lowest grade.
    """
    values = [boundaries[g] for g in grade_labels[:-1]]
    return all(values[i] > values[i+1] for i in range(len(values)-1))


def plot_grade_distribution(
    df: pd.DataFrame,
    grade_labels: list[str],
    grade_column: str
) -> go.Figure:
    """
    Generate a Plotly bar chart for grade distribution.
    """
    counts = df[grade_column].value_counts().reindex(grade_labels + ['Error'], fill_value=0)
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={'x': 'Grade', 'y': 'Count'},
        title='Grade Distribution',
        template='plotly_dark'
    )
    return fig


def smart_ai_grade_suggestion(
    df: pd.DataFrame,
    marks_column: str,
    grade_labels: list[str],
    min_val: float = 0,
    max_val: float = 100
) -> str:
    """
    Suggest a centric grade via KMeans clustering on valid marks.
    """
    marks = pd.to_numeric(df[marks_column], errors='coerce')
    arr = marks[(marks >= min_val) & (marks <= max_val)].dropna().values.reshape(-1,1)
    if len(arr) < len(grade_labels):
        return grade_labels[len(grade_labels)//2]

    kmeans = KMeans(n_clusters=len(grade_labels), random_state=42, n_init='auto')
    kmeans.fit(arr)
    centroids = sorted(kmeans.cluster_centers_.flatten(), reverse=True)
    mean_val = arr.mean()
    idx = min(range(len(centroids)), key=lambda i: abs(centroids[i]-mean_val))
    return grade_labels[idx]