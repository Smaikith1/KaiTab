import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="KaiTab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state='collapsed'
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

#
# def main():
#     st.title("KaiTab")
#
#     # Initialize session state
#     initialize_session_state()
#
#     # Create tabs for different sections
#     tab1, tab2 = st.tabs(["Data Input", "Data Viewer"])
#
#     with tab1:
#         data_input_section()
#
#     with tab2:
#         data_viewer_editor()


def perform_normality_test(data):
    """Perform Shapiro-Wilk normality test"""
    statistic, p_value = stats.shapiro(data)
    return p_value


def calculate_basic_stats(data):
    """Calculate basic statistics for the data"""
    return {
        'n': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'ci_95': stats.norm.interval(0.95, loc=np.mean(data), scale=stats.sem(data)),
        'ci_99': stats.norm.interval(0.99, loc=np.mean(data), scale=stats.sem(data))
    }


def create_individual_value_plot(data, response_col, group_cols=None):
    """Create individual value plot with optional grouping"""
    fig = go.Figure()

    if group_cols:
        for name, group in data.groupby(group_cols):
            fig.add_trace(go.Box(
                y=group[response_col],
                name=str(name),
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
    else:
        fig.add_trace(go.Box(
            y=data[response_col],
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))

    fig.update_layout(
        title=f"Individual Value Plot of {response_col}",
        yaxis_title=response_col,
        showlegend=True
    )

    return fig


def create_histogram(data, column, add_normal=True, group_cols=None):
    """Create histogram with optional normal curve overlay"""
    fig = go.Figure()


    # Add histogram
    # fig.add_trace(go.Histogram(
    #     x=data[column],
    #     name="Data",
    #     nbinsx=30,
    #     histnorm='probability'
    # ))

    if group_cols:
        for name, group in data.groupby(group_cols):
            fig.add_trace(go.Histogram(
                x=group[column],
                name="Data",
                nbinsx=30,
                histnorm='probability'
            ))
    else:
        fig.add_trace(go.Histogram(
            x=data[column],
            name="Data",
            nbinsx=30,
            histnorm='probability'
        ))

    if add_normal:
        x_range = np.linspace(data[column].min(), data[column].max(), 100)
        mu = data[column].mean()
        sigma = data[column].std()
        normal_curve = stats.norm.pdf(x_range, mu, sigma)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_curve,
            name="Normal Fit",
            line=dict(color='red')
        ))

    fig.update_layout(
        title=f"Histogram of {column}",
        xaxis_title=column,
        yaxis_title="Frequency",
        showlegend=True
    )

    return fig


def perform_ttest(data, group_col, response_col):
    """Perform t-test between two groups"""
    groups = data[group_col].unique()
    if len(groups) != 2:
        return "T-test requires exactly two groups"

    group1_data = data[data[group_col] == groups[0]][response_col]
    group2_data = data[data[group_col] == groups[1]][response_col]

    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
    return t_stat, p_value


def create_regression_plot(data, x_col, y_col):
    """Create regression plot with confidence intervals"""
    fig = px.scatter(data, x=x_col, y=y_col, trendline="ols",
                     trendline_color_override="red")

    # Add confidence intervals
    model = np.polyfit(data[x_col], data[y_col], 1)
    predict = np.poly1d(model)

    x_new = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    y_new = predict(x_new)

    fig.update_layout(
        title=f"Regression Plot: {y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=True
    )

    return fig


def create_multivari_chart(data, response_col, group_cols):
    """Create multi-vari chart"""
    fig = go.Figure()

    for name, group in data.groupby(group_cols[0]):
        fig.add_trace(go.Box(
            y=group[response_col],
            name=str(name),
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))

    fig.update_layout(
        title=f"Multi-Vari Chart of {response_col}",
        xaxis_title=group_cols[0],
        yaxis_title=response_col,
        showlegend=True
    )

    return fig


def create_scatterplot(data, x_col, y_col):
    fig = px.scatter(data, x=x_col, y=y_col)

    fig.update_layout(
        title=f"Regression Plot: {y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=True
    )

    return fig


def create_correlogram(data):
    """Create a correlation heatmap"""
    # Calculate correlation matrix
    corr_matrix = data.select_dtypes(include=[np.number]).corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        zmin=-1,
        zmax=1,
        colorscale='RdBu',
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis_title="Variables",
        yaxis_title="Variables",
        width=800,
        height=800
    )

    return fig, corr_matrix


def perform_anova(data, response_col, factor_cols):
    """Perform one-way or multi-way ANOVA"""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    # Create formula for ANOVA
    formula = f"{response_col} ~ {' + '.join(factor_cols)}"

    # Fit the model
    model = ols(formula, data=data).fit()

    # Perform ANOVA
    anova_table = anova_lm(model, typ=2)

    # Perform Tukey's test for each factor
    tukey_results = {}
    for factor in factor_cols:
        tukey = pairwise_tukeyhsd(data[response_col], data[factor])
        tukey_results[factor] = tukey

    return anova_table, tukey_results, model


def create_anova_plots(data, response_col, factor_cols, model):
    """Create diagnostic plots for ANOVA"""
    # Residual plot
    residuals = model.resid
    fitted = model.fittedvalues

    fig_residuals = go.Figure()
    fig_residuals.add_trace(go.Scatter(
        x=fitted,
        y=residuals,
        mode='markers',
        name='Residuals'
    ))
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    fig_residuals.update_layout(
        title="Residuals vs Fitted Values",
        xaxis_title="Fitted Values",
        yaxis_title="Residuals"
    )

    # Q-Q plot
    from scipy.stats import probplot
    qq_x, qq_y = probplot(residuals, dist="norm")

    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=qq_x[0],
        y=qq_y[0],
        mode='markers',
        name='Q-Q'
    ))
    fig_qq.add_trace(go.Scatter(
        x=qq_x[0],
        y=qq_x[0] * qq_y[1][0] + qq_y[1][1],
        mode='lines',
        name='Reference Line'
    ))
    fig_qq.update_layout(
        title="Normal Q-Q Plot",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles"
    )

    return fig_residuals, fig_qq


def analysis_section():
    """Handle different types of analyses"""
    if st.session_state.data.empty:
        st.warning("Please input data first!")
        return

    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Individual Value Plot", "Histogram", "T-Test",
         "Linear Regression", "Scatter Plot", "Multi-Vari Chart",
         "Correlogram", "ANOVA"]
    )

    with st.sidebar:
        st.header("Plot Controls")
        show_stats = st.checkbox("Show Basic Statistics", value=True)
        show_ci = st.checkbox("Show Confidence Intervals", value=True)
        show_normality = st.checkbox("Show Normality Test", value=True)

    # Select columns for analysis
    numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns

    if analysis_type == "Individual Value Plot":
        response_col = st.selectbox("Select Response Variable", numeric_cols)
        group_cols = st.multiselect("Select Grouping Variables (optional)",
                                    st.session_state.data.columns)

        fig = create_individual_value_plot(st.session_state.data, response_col, group_cols)
        st.plotly_chart(fig, use_container_width=True)

        if show_stats:
            stats_data = calculate_basic_stats(st.session_state.data[response_col])
            st.write("Basic Statistics:")
            st.dataframe(pd.DataFrame(stats_data))

    elif analysis_type == "Histogram":
        col = st.selectbox("Select Column", numeric_cols)
        group_cols = st.multiselect("Select Grouping Variables (optional)",
                                    st.session_state.data.columns)
        add_normal = st.checkbox("Add Normal Curve", value=True)

        fig = create_histogram(st.session_state.data, col, add_normal, group_cols)
        st.plotly_chart(fig, use_container_width=True)

        if show_stats:
            stats_data = calculate_basic_stats(st.session_state.data[col])
            st.write("Basic Statistics:")
            st.dataframe(pd.DataFrame(stats_data))

        if show_normality:
            p_value = perform_normality_test(st.session_state.data[col])
            st.write(f"Normality Test p-value: {p_value:.4f}")

    elif analysis_type == "T-Test":
        group_col = st.selectbox("Select Grouping Variable", st.session_state.data.columns)
        response_col = st.selectbox("Select Response Variable", numeric_cols)

        result = perform_ttest(st.session_state.data, group_col, response_col)
        if isinstance(result, tuple):
            t_stat, p_value = result
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_value:.4f}")

            # Create visualization
            fig = create_individual_value_plot(st.session_state.data, response_col, [group_col])
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Linear Regression":
        x_col = st.selectbox("Select X Variable", numeric_cols)
        y_col = st.selectbox("Select Y Variable", numeric_cols)

        fig = create_regression_plot(st.session_state.data, x_col, y_col)
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Scatter Plot":
        x_col = st.selectbox("Select X Variable", numeric_cols)
        y_col = st.selectbox("Select Y Variable", numeric_cols)

        fig = create_scatterplot(st.session_state.data, x_col, y_col)
        st.plotly_chart(fig, use_container_width=True)


    elif analysis_type == "Multi-Vari Chart":
        response_col = st.selectbox("Select Response Variable", numeric_cols)
        group_cols = st.multiselect("Select Grouping Variables",
                                    st.session_state.data.columns,
                                    max_selections=3)

        if group_cols:
            fig = create_multivari_chart(st.session_state.data, response_col, group_cols)
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Correlogram":
        numeric_data = st.session_state.data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            st.warning("No numeric columns found in the dataset!")
            return

        selected_cols = st.multiselect(
            "Select columns for correlation analysis",
            numeric_data.columns,
            default=list(numeric_data.columns)
        )

        if selected_cols:
            fig, corr_matrix = create_correlogram(st.session_state.data[selected_cols])
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "ANOVA":
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
        categorical_cols = st.session_state.data.select_dtypes(exclude=[np.number]).columns

        response_col = st.selectbox("Select Response Variable", numeric_cols)
        factor_cols = st.multiselect(
            "Select Factor Variables (Categorical)",
            categorical_cols,
            max_selections=3
        )

        if factor_cols:
            try:
                anova_table, tukey_results, model = perform_anova(
                    st.session_state.data,
                    response_col,
                    factor_cols
                )

                # Display ANOVA results
                st.write("ANOVA Results:")
                st.dataframe(anova_table.round(4))

                # Display Tukey results
                st.write("Tukey's HSD Test Results:")
                for factor, tukey in tukey_results.items():
                    st.write(f"\nTukey's test for factor: {factor}")
                    st.write(tukey)

                # Create and display diagnostic plots
                fig_residuals, fig_qq = create_anova_plots(
                    st.session_state.data,
                    response_col,
                    factor_cols,
                    model
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_residuals, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_qq, use_container_width=True)

                # Display model summary
                st.write("Model Summary:")
                st.text(model.summary())

            except Exception as e:
                st.error(f"Error performing ANOVA: {str(e)}")
                st.write("Please ensure your data is properly formatted and contains no missing values.")


def apply_numeric_filter(data, column, operator, value):
    """Apply numeric filtering based on operator and value"""
    if operator == "greater than":
        return data[data[column] > value]
    elif operator == "less than":
        return data[data[column] < value]
    elif operator == "equal to":
        return data[data[column] == value]
    elif operator == "not equal to":
        return data[data[column] != value]
    elif operator == "greater than or equal to":
        return data[data[column] >= value]
    elif operator == "less than or equal to":
        return data[data[column] <= value]
    return data


def apply_categorical_filter(data, column, selected_values):
    """Apply categorical filtering based on selected values"""
    if selected_values:
        return data[data[column].isin(selected_values)]
    return data


def filter_section():
    """Create and manage data filters"""
    st.subheader("Data Filters")

    # Initialize filtered data
    filtered_data = st.session_state.data.copy()

    # Create columns for filter layout
    col1, col2 = st.columns(2)

    with col1:
        # Numeric Filters
        st.write("Numeric Filters")
        numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            num_col = st.selectbox("Select numeric column", numeric_cols, key="num_filter_col")
            operator = st.selectbox(
                "Select operator",
                ["greater than", "less than", "equal to", "not equal to",
                 "greater than or equal to", "less than or equal to"],
                key="num_operator"
            )
            value = st.number_input("Enter value", value=0.0, key="num_value")

            if st.button("Apply Numeric Filter"):
                filtered_data = apply_numeric_filter(filtered_data, num_col, operator, value)
                st.session_state.filtered_data = filtered_data

    with col2:
        # Categorical Filters
        st.write("Categorical Filters")
        categorical_cols = filtered_data.select_dtypes(exclude=[np.number]).columns

        if len(categorical_cols) > 0:
            cat_col = st.selectbox("Select categorical column", categorical_cols, key="cat_filter_col")
            unique_values = filtered_data[cat_col].unique()
            selected_values = st.multiselect(
                "Select values",
                unique_values,
                default=list(unique_values),
                key="cat_values"
            )

            if st.button("Apply Categorical Filter"):
                filtered_data = apply_categorical_filter(filtered_data, cat_col, selected_values)
                st.session_state.filtered_data = filtered_data

    # Add reset button
    if st.button("Reset Filters"):
        filtered_data = st.session_state.data.copy()
        st.session_state.filtered_data = filtered_data

    # Show current filter status
    st.write("Current data shape:", filtered_data.shape)

    return filtered_data


def data_viewer_editor():
    """Display and allow editing of the current dataset with filtering"""
    if not st.session_state.data.empty:
        # Initialize filtered_data in session state if not present
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = st.session_state.data.copy()

        # Add filter section
        filtered_df = filter_section()

        # Show filtered data in editor
        st.subheader("Current Dataset")
        edited_df = st.data_editor(
            filtered_df,
            use_container_width=True,
            key="data_editor",
            column_config={
                col: st.column_config.NumberColumn(
                    col,
                    help=f"Values for {col}",
                    min_value=None,
                    max_value=None,
                    step=0.1
                ) for col in filtered_df.select_dtypes(include=[np.number]).columns
            }
        )

        # Update session state if changes were made
        if not edited_df.equals(st.session_state.filtered_data):
            # Update both filtered and main datasets
            st.session_state.filtered_data = edited_df
            # Update the main dataset with edited values while preserving filtered rows
            mask = st.session_state.data.index.isin(edited_df.index)
            st.session_state.data.loc[mask] = edited_df
            st.success("Data updated!")


def main():
    st.title("KaiTab")

    # Initialize session state
    initialize_session_state()

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data Input", "Data Viewer", "Analysis"])

    with tab1:
        data_input_section()

    with tab2:
        data_viewer_editor()

    with tab3:
        # Use filtered data for analysis if it exists
        if 'filtered_data' in st.session_state and not st.session_state.filtered_data.empty:
            temp_data = st.session_state.data.copy()
            st.session_state.data = st.session_state.filtered_data
            analysis_section()
            st.session_state.data = temp_data
        else:
            analysis_section()


# [Rest of the code remains the same...]

if __name__ == "__main__":
    main()