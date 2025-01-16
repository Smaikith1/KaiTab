import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Statistical Analysis Tool",
    page_icon="📊",
    layout="wide"
)


def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame()
    if 'original_data' not in st.session_state:
        st.session_state.original_data = pd.DataFrame()


def create_empty_table():
    """Create an empty table with adjustable dimensions"""
    cols = st.number_input("Number of columns", min_value=1, value=3)
    rows = st.number_input("Number of rows", min_value=1, value=5)

    if 'temp_df' not in st.session_state or (
            st.session_state.temp_df.shape != (rows, cols)
    ):
        st.session_state.temp_df = pd.DataFrame(
            np.nan,
            index=range(rows),
            columns=[f'Column_{i + 1}' for i in range(cols)]
        )

    return st.session_state.temp_df


def update_session_data(df):
    """Update both the working and original copies of the data"""
    st.session_state.data = df.copy()
    st.session_state.original_data = df.copy()


def data_input_section():
    """Handle data input through file upload or manual entry"""
    st.header("Data Input")

    input_method = st.radio(
        "Choose input method:",
        ["Upload Excel File", "Manual Input"]
    )

    if input_method == "Upload Excel File":
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls']
        )

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                update_session_data(df)
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    else:  # Manual Input
        st.subheader("Manual Data Entry")
        temp_df = create_empty_table()

        # Create an editable dataframe
        edited_df = st.data_editor(
            temp_df,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_input"
        )

        if st.button("Confirm Data"):
            # Remove any completely empty rows
            edited_df = edited_df.dropna(how='all')
            update_session_data(edited_df)
            st.success("Data confirmed!")


def data_viewer_editor():
    """Display and allow editing of the current dataset"""
    if not st.session_state.data.empty:
        st.subheader("Current Dataset")
        edited_df = st.data_editor(
            st.session_state.data,
            use_container_width=True,
            key="data_editor"
        )

        # Update session state if changes were made
        if not edited_df.equals(st.session_state.data):
            st.session_state.data = edited_df
            st.success("Data updated!")


def main():
    st.title("Statistical Analysis Tool")

    # Initialize session state
    initialize_session_state()

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Data Input", "Data Viewer"])

    with tab1:
        data_input_section()

    with tab2:
        data_viewer_editor()


if __name__ == "__main__":
    main()