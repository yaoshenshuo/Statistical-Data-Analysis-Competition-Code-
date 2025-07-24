import streamlit as st
import chardet
import openpyxl
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg




def show_test_results_with_no_effect_name(test_name, statistic, p_value, effect_size=None, effect_name=None, extra_info=None):
    st.markdown(
        f"<div style='text-align: center;'><h4>Test Results: <em>{test_name}</em></h4></div>",
        unsafe_allow_html=True
    )
    # tooltips = {
    #     "Statistic": "Test statistic is a value calculated from sample data that measures how far the sample deviates from the null hypothesis, used to determine whether to reject the null hypothesis.",
    #     "p-value": "P-value is the probability of observing current or more extreme data when the original hypothesis is true.",
    # }

    if test_name == "Pearson correlation":
        tooltips = {
            "Pearson’s r": "Pearson’s r is a measure of the strength and direction of the linear relationship between two continuous variables.",
            "p-value": "P-value is the probability of observing current or more extreme data when the original hypothesis is true.",
        }

        html = """
        <style>
        .tooltip-wrapper {
            position: relative;
            display: inline-block;
        }
        
        .tooltip-trigger {
            border-bottom: 1px dotted #666;
            cursor: help;
        }
        
        .tooltip-content {
            visibility: hidden;
            width: 300px;
            max-width: 80vw;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 4px;
            padding: 8px;
            position: absolute;
            z-index: 9999;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 14px;
            line-height: 1.4;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            word-wrap: break-word;
        }
        
        .tooltip-wrapper:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
        
        /* Arrow for tooltip */
        .tooltip-content::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 600px) {
            .tooltip-content {
                left: 0;
                transform: none;
                right: auto;
            }
            .tooltip-content::after {
                left: 15px;
                margin-left: 0;
            }
        }
        
        /* Table styling */
        .results-table {
            width: 90%;
            border-collapse: collapse;
            margin-bottom: 1rem;
            font-family: sans-serif;
            margin-left: 50px;
        }
        
        .results-table th, 
        .results-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .results-table th {
            background-color: #f5f5f5;
            font-weight: 600;
        }
        
        .results-table tr:hover {
            background-color: #f9f9f9;
        }
        
        .value-cell {
            font-family: monospace;
            text-align: right;
        }
        </style>
        
        <table class="results-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add Statistic row
        html += f"""
        <tr>
            <td>
                <div class="tooltip-wrapper">
                    <span class="tooltip-trigger">Pearson’s r</span>
                    <div class="tooltip-content">{tooltips['Pearson’s r']}</div>
                </div>
            </td>
            <td class="value-cell">{statistic:.4f}</td>
        </tr>
        """
        
        # Add p-value row
        html += f"""
        <tr>
            <td>
                <div class="tooltip-wrapper">
                    <span class="tooltip-trigger">p-value</span>
                    <div class="tooltip-content">{tooltips['p-value']}</div>
                </div>
            </td>
            <td class="value-cell">{p_value:.4f}</td>
        </tr>
        """
        
        st.markdown(html, unsafe_allow_html=True)




    elif test_name == 'Chi-Square test of independence':
        tooltips = {
            "chi2_stat": "你的注释",
            "p-value": "P-value is the probability of observing current or more extreme data when the original hypothesis is true.",
        }
        html = """
        <style>
        .tooltip-wrapper {
            position: relative;
            display: inline-block;
        }
        
        .tooltip-trigger {
            border-bottom: 1px dotted #666;
            cursor: help;
        }
        
        .tooltip-content {
            visibility: hidden;
            width: 300px;
            max-width: 80vw;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 4px;
            padding: 8px;
            position: absolute;
            z-index: 9999;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 14px;
            line-height: 1.4;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            word-wrap: break-word;
        }
        
        .tooltip-wrapper:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
        
        /* Arrow for tooltip */
        .tooltip-content::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 600px) {
            .tooltip-content {
                left: 0;
                transform: none;
                right: auto;
            }
            .tooltip-content::after {
                left: 15px;
                margin-left: 0;
            }
        }
        
        /* Table styling */
        .results-table {
            width: 90%;
            border-collapse: collapse;
            margin-bottom: 1rem;
            font-family: sans-serif;
            margin-left: 50px;
        }
        
        .results-table th, 
        .results-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .results-table th {
            background-color: #f5f5f5;
            font-weight: 600;
        }
        
        .results-table tr:hover {
            background-color: #f9f9f9;
        }
        
        .value-cell {
            font-family: monospace;
            text-align: right;
        }
        </style>
        
        <table class="results-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add Statistic row
        html += f"""
        <tr>
            <td>
                <div class="tooltip-wrapper">
                    <span class="tooltip-trigger">chi2_stat</span>
                    <div class="tooltip-content">{tooltips['chi2_stat']}</div>
                </div>
            </td>
            <td class="value-cell">{statistic:.4f}</td>
        </tr>
        """
        
        # Add p-value row
        html += f"""
        <tr>
            <td>
                <div class="tooltip-wrapper">
                    <span class="tooltip-trigger">p-value</span>
                    <div class="tooltip-content">{tooltips['p-value']}</div>
                </div>
            </td>
            <td class="value-cell">{p_value:.4f}</td>
        </tr>
        """
        st.markdown(html, unsafe_allow_html=True)

    elif test_name == 'Z-test for proportions':
        tooltips = {
            "z_stat": "你的注释",
            "p-value": "P-value is the probability of observing current or more extreme data when the original hypothesis is true.",
        }
        html = """
        <style>
        .tooltip-wrapper {
            position: relative;
            display: inline-block;
        }
        
        .tooltip-trigger {
            border-bottom: 1px dotted #666;
            cursor: help;
        }
        
        .tooltip-content {
            visibility: hidden;
            width: 300px;
            max-width: 80vw;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 4px;
            padding: 8px;
            position: absolute;
            z-index: 9999;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 14px;
            line-height: 1.4;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            word-wrap: break-word;
        }
        
        .tooltip-wrapper:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
        
        /* Arrow for tooltip */
        .tooltip-content::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 600px) {
            .tooltip-content {
                left: 0;
                transform: none;
                right: auto;
            }
            .tooltip-content::after {
                left: 15px;
                margin-left: 0;
            }
        }
        
        /* Table styling */
        .results-table {
            width: 90%;
            border-collapse: collapse;
            margin-bottom: 1rem;
            font-family: sans-serif;
            margin-left: 50px;
        }
        
        .results-table th, 
        .results-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .results-table th {
            background-color: #f5f5f5;
            font-weight: 600;
        }
        
        .results-table tr:hover {
            background-color: #f9f9f9;
        }
        
        .value-cell {
            font-family: monospace;
            text-align: right;
        }
        </style>
        
        <table class="results-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add Statistic row
        html += f"""
        <tr>
            <td>
                <div class="tooltip-wrapper">
                    <span class="tooltip-trigger">z_stat</span>
                    <div class="tooltip-content">{tooltips['z_stat']}</div>
                </div>
            </td>
            <td class="value-cell">{statistic:.4f}</td>
        </tr>
        """
        
        # Add p-value row
        html += f"""
        <tr>
            <td>
                <div class="tooltip-wrapper">
                    <span class="tooltip-trigger">p-value</span>
                    <div class="tooltip-content">{tooltips['p-value']}</div>
                </div>
            </td>
            <td class="value-cell">{p_value:.4f}</td>
        </tr>
        """
        st.markdown(html, unsafe_allow_html=True)
    alpha = 0.05
    if p_value < alpha:
        st.success("Statistically significant (p < 0.05)")
    else:
        st.warning("Not statistically significant (p ≥ 0.05)")
    

    if extra_info is not None:

        expander = st.expander("# Additional Information")
        if isinstance(extra_info, list):
            for info in extra_info:
                expander.write(info)
        else:
            expander.write(extra_info)




st.set_page_config(
    page_title="SmartStat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.highlight {
    border-radius: 0.4rem;
    padding: 0.5rem;
    margin-bottom: 1rem;
}
.highlight.blue {
    background-color: #e6f3ff;
    border-left: 5px solid #1a73e8;
}
</style>
""", unsafe_allow_html=True)

def main_page():
    st.title("SmartStat Selector")
    st.write("Choose the appropriate statistical testing method based on your data type and needs")
    st.markdown("<p style='"
                    "font-size:18px;"
                    "font-weight:bold;"
                    "margin:3em 0 0 0;"
                    "'>1. What is the data type?</p",
                    unsafe_allow_html=True
        )
    st.session_state.data_type = st.selectbox(
        " ",
        ["please select an option","Numerical", "Categorical"]
    )
    
    if st.session_state.data_type == "Numerical":
        st.markdown("<p style='"
                    "font-size:18px;"
                    "font-weight:bold;"
                    "margin:3em 0 0 0;"
                    "'>2. How numerical data is organized?</p",
                    unsafe_allow_html=True
        )
        st.session_state.data_type2 = st.selectbox(
        " ",
        ["please select an option","Two Groups", "Paired Data","More than Two Groups","Relationship between Multiple Variables"]
    )
        
        if st.session_state.data_type2 == "Two Groups":
            st.markdown("<p style='"
                    "font-size:18px;"
                    "font-weight:bold;"
                    "margin:3em 0 0 0;"
                    "'>3. Data distribution assumptions?</p",
                    unsafe_allow_html=True
        )
            st.session_state.data_type3 = st.selectbox(
        " ",
        ["please select an option","Yes (Parametric Test)", "No (Non-Parametric Test)"]
    )
            
            if st.session_state.data_type3 == "Yes (Parametric Test)":
                st.success("Recommended use: **Independent t-test**")
                st.write("Applicable conditions: Normal distribution of data, homogeneity of variance, independent observation")
                st.session_state.selected_test = "Independent t-test"
            elif st.session_state.data_type3 == "No (Non-Parametric Test)":
                st.success("Recommended use: **Mann-Whitney U Test**")
                st.write("Applicable conditions: Sequential data or numerical data with non normal distribution")
                st.session_state.selected_test = "Mann-Whitney U Test"
                
        elif st.session_state.data_type2 == "Paired Data":
            st.markdown("<p style='"
                    "font-size:18px;"
                    "font-weight:bold;"
                    "margin:3em 0 0 0;"
                    "'>3. Data distribution assumptions?</p",
                    unsafe_allow_html=True
        )
            st.session_state.data_type3 = st.selectbox(
        " ",
        ["please select an option","Yes (Parametric Test)", "No (Non-Parametric Test)"]
    )
            
            if  st.session_state.data_type3 == "Yes (Parametric Test)":
                st.success("Recommended use: **Paired t-test**")
                st.write("Applicable conditions: Both sets of data are continuous numerical data")
                st.session_state.selected_test = "Paired t-test"
            elif st.session_state.data_type3 == "No (Non-Parametric Test)":
                st.success("Recommended use: **Wilcoxon signed-rank Test**")
                st.write("Applicable conditions: Two sets of data are either continuous numerical data or ordered categorical data")
                st.session_state.selected_test = "Wilcoxon signed-rank Test"
                
        elif st.session_state.data_type2 == "More than Two Groups":
            st.markdown("<p style='"
                    "font-size:18px;"
                    "font-weight:bold;"
                    "margin:3em 0 0 0;"
                    "'>3. Data distribution assumptions?</p",
                    unsafe_allow_html=True
            )
            st.session_state.data_type3 = st.selectbox(
        " ",
        ["please select an option","Yes (Parametric Test)", "No (Non-Parametric Test)"]
    )
            
            if st.session_state.data_type3 == "Yes (Parametric Test)":
                st.success("Recommended use: **ANOVA**")
                st.write("Applicable conditions: Normal distribution of data, homogeneity of variance, independent observation")
                st.write("If significant differences are found, post hoc testing such as Tukey HSD can be conducted")
                st.session_state.selected_test = "ANOVA"
            elif st.session_state.data_type3 == "No (Non-Parametric Test)":
                st.success("Recommended use: **Kruskal-Wallis Test**")
                st.write("Applicable conditions: Sequential data or numerical data with non normal distribution")
                st.session_state.selected_test = "Kruskal-Wallis Test"

        elif st.session_state.data_type2 == "Relationship between Multiple Variables":
            st.markdown("<p style='"
                    "font-size:18px;"
                    "font-weight:bold;"
                    "margin:3em 0 0 0;"
                    "'>3. Data distribution assumptions?</p",
                    unsafe_allow_html=True
            )
            st.session_state.data_type3 = st.selectbox(
        " ",
        ["please select an option","Yes (Parametric Test)", "No (Non-Parametric Test)"]
    )
            
            if st.session_state.data_type3 == "Yes (Parametric Test)":
                st.success("Recommended use: **Pearson correlation Regression analysis** ")
                st.session_state.selected_test = "Pearson correlation"
            elif st.session_state.data_type3 == "No (Non-Parametric Test)":
                st.success("Recommended use: **Spearman correlation**")
                st.write("Applicable conditions: The relationship between a continuous dependent variable and one or more predictor variables")
                st.session_state.selected_test = "Spearman correlation"
    
    elif st.session_state.data_type == "Categorical":  
        st.markdown("<p style='"
                    "font-size:18px;"
                    "font-weight:bold;"
                    "margin:3em 0 0 0;"
                    "'>2. What is your target task?</p",
                    unsafe_allow_html=True
        )
        st.session_state.data_type4 = st.selectbox(
        " ",
        ["please select an option","Compare two variavles", "Test proportions"]
    )
        
        if st.session_state.data_type4 == "Compare two variavles":
            st.success("Recommended use: **Chi-Square test of independence**")
            st.session_state.selected_test = "Chi-Square test of independence"
                
        elif st.session_state.data_type4 == "Test proportions":
            st.success("Recommended use: **Z-test for proportions Binomial test**")
            st.session_state.selected_test = "Z-test for proportions"
    if st.button("Go to Test!"):
        st.session_state.page = "test_details"

def test_details_page():
    st.title(f"{st.session_state.selected_test}")
    st.write(f"You can use {st.session_state.selected_test} to test your own data")
    uploaded_file = st.file_uploader("Upload your data (CSV or Excel format)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is None:
        st.warning("Please input your file")
        if st.button("Back to Test Selection"):
            st.session_state.page = "main"
        return 
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            rawdata = uploaded_file.read(10000)
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            uploaded_file.seek(0) 
            
            try:
                data = pd.read_csv(uploaded_file, encoding=encoding)
            except:
                encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'big5', 'iso-8859-1', 'latin1']
                for enc in encodings_to_try:
                    try:
                        uploaded_file.seek(0)
                        data = pd.read_csv(uploaded_file, encoding=enc)
                        encoding = enc
                        break
                    except:
                        continue
        
        elif file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(uploaded_file)
            encoding = "Excel file (no encoding specified)"
        
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return
        
        st.write(f"File type: {file_extension.upper()}")
        if file_extension == 'csv':
            st.write(f"Encoding detected: {encoding}")
        
        st.write("Preview of your data:")
        st.dataframe(data.head())

        if file_extension in ['xlsx', 'xls']:
            try:
                import openpyxl
                wb = openpyxl.load_workbook(uploaded_file)
                sheet_names = wb.sheetnames
                if len(sheet_names) > 1:
                    selected_sheet = st.selectbox(
                        "Select worksheet", 
                        sheet_names,
                        index=0
                    )
                    if selected_sheet != wb.active.title:
                        uploaded_file.seek(0)
                        data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                        st.write(f"Data from worksheet: {selected_sheet}")
                        st.dataframe(data.head())
            except:
                pass
            
        if st.session_state.selected_test in ["Independent t-test", "Mann-Whitney U Test"]:
            columns = data.columns.tolist()
            selected_column = st.selectbox(
                "Select a categorical column with exactly 2 groups",
                columns
            )
            
            unique_vals = data[selected_column].dropna().unique().tolist()
            
            if len(unique_vals) != 2:
                st.warning(f"Selected column '{selected_column}' must have exactly 2 groups, but found {len(unique_vals)} groups: {unique_vals}")
                st.stop()
            
            st.write(f"Data from column: ")
            st.dataframe(data[[selected_column]].drop_duplicates().reset_index(drop=True))
            
            col1, col2 = st.columns(2)
            with col1:
                group1_col = st.selectbox("Select Group 1", unique_vals, index=0)
            with col2:
                group2_col = st.selectbox("Select Group 2", unique_vals, index=1)

            filtered = data[data[selected_column].isin([group1_col, group2_col])]

            numeric_cols = data.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns available for comparison.")
                st.stop()
            
            value_col = st.selectbox("Select a numeric column for comparison", numeric_cols)
            
            g1 = filtered.loc[filtered[selected_column] == group1_col, value_col]
            g2 = filtered.loc[filtered[selected_column] == group2_col, value_col]

            col1, col2 = st.columns([3,2])

            with col1:
                #histogram
                
                st.markdown(
                    f"<h4 style='text-align: center;'>Distribution of {value_col} by {selected_column}</h4>",
                    unsafe_allow_html=True
                )
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=g1, 
                    name=str(group1_col),
                    marker_color='#1f77b4',  #choose colors
                    opacity=0.65#1是完全不透明 0是完全透明
                ))
                fig.add_trace(go.Histogram(
                    x=g2, 
                    name=str(group2_col),
                    marker_color='#ff7f0e',  #自己选颜色2
                    opacity=0.65#透明度可调  1是完全不透明 0是完全透明
                ))
                
                #重叠更暗
                fig.update_layout(
                    barmode='overlay',
                    
                    xaxis_title=value_col,
                    yaxis_title="Count",
                    legend_title=selected_column,
                )
                st.plotly_chart(fig)
            
                

            

                if st.button("Perform Test"):
                    group1_data = filtered[filtered[selected_column] == group1_col][value_col].dropna()
                    group2_data = filtered[filtered[selected_column] == group2_col][value_col].dropna()
                    
                    if len(group1_data) < 2 or len(group2_data) < 2:
                        st.error("Each group must have at least 2 observations to perform the test.")
                        st.stop()
                        
                

                    if st.session_state.selected_test == "Independent t-test":
                        _, p1 = stats.shapiro(group1_data)
                        _, p2 = stats.shapiro(group2_data)
                        
                        if p1 < 0.05 or p2 < 0.05:
                            st.warning("Warning: One or both groups may not be normally distributed (Shapiro-Wilk test p < 0.05). Consider using Mann-Whitney U test instead.")
                        



                        with col2:

                            _, p_var = stats.levene(group1_data, group2_data)
                            equal_var = p_var > 0.05
                            
                            t_stat, p_value = stats.ttest_ind(
                                group1_data,
                                group2_data,
                                equal_var=equal_var
                            )
                            
                            cohens_d = pg.compute_effsize(
                                group1_data,
                                group2_data,
                                eftype='cohen'
                            )
                            
                            show_test_results(
                                test_name="Independent t-test",
                                statistic=t_stat,
                                p_value=p_value,
                                effect_size=cohens_d,
                                effect_name="Cohen's d",
                                extra_info=[
                                    f"Group sizes: {len(group1_data)} vs {len(group2_data)}",
                                    f"Group means: {group1_data.mean():.2f} vs {group2_data.mean():.2f}",
                                    f"Equal variance assumed: {'Yes' if equal_var else 'No'} (Levene's test p = {p_var:.4f})"
                                ]
                            )
                            
                    elif st.session_state.selected_test == "Mann-Whitney U Test":
                        u_stat, p_value = stats.mannwhitneyu(
                            group1_data,
                            group2_data,
                            alternative='two-sided'
                        )
                        n1 = len(group1_data)
                        n2 = len(group2_data)
                        r = 1 - (2 * u_stat) / (n1 * n2)
                        show_test_results(
                            test_name="Mann-Whitney U Test",
                            statistic=u_stat,
                            p_value=p_value,
                            effect_size=r,
                            effect_name="Rank-biserial r",
                            extra_info=[
                                f"Group sizes: {n1} vs {n2}",
                                f"Group medians: {group1_data.median():.2f} vs {group2_data.median():.2f}"
                            ]
                        )    

                    
               
        elif st.session_state.selected_test in ["Paired t-test", "Wilcoxon signed-rank Test"]:
            col1, col2 = st.columns(2)
            with col1:
                pre_col = st.selectbox("Select Pre-treatment column", data.columns)
            with col2:
                post_col = st.selectbox("Select Post-treatment column", data.columns)


            if not (np.issubdtype(data[pre_col].dtype, np.number) and np.issubdtype(data[post_col].dtype, np.number)):
                st.error("Error: Both columns must be numeric for paired t-test.")
                st.stop()
            
            if len(data[pre_col]) != len(data[post_col]):
                st.error("Error: Pre and Post columns must have the same number of observations (paired data).")
                st.stop()
            if data[[pre_col, post_col]].isnull().any().any():
                st.warning("Warning: Missing values detected. Rows with missing values will be excluded.")
                data = data.dropna(subset=[pre_col, post_col])






            col1, col2, col3 = st.columns([2,2,2])
            with col1:
                st.markdown(
                    f"<h4 style='text-align: center;'>Pre-Post Comparison</h4>",
                    unsafe_allow_html=True
                )
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data[pre_col].dropna(),
                    name="Pre-treatment",
                    marker_color='#1f77b4',
                    opacity=0.65
                ))
                fig.add_trace(go.Histogram(
                    x=data[post_col].dropna(),
                    name="Post-treatment",
                    marker_color='#ff7f0e',
                    opacity=0.65
                ))
                fig.update_layout(
                    barmode='overlay',
                
                    xaxis_title="Value",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig)

                if st.button("Perform Test"):
                    if st.session_state.selected_test == "Paired t-test":
                        _, p_pre = stats.shapiro(data[pre_col])
                        _, p_post = stats.shapiro(data[post_col])
                        if p_pre < 0.05 or p_post < 0.05:
                            st.warning("Warning: One or both groups may not be normally distributed (Shapiro-Wilk test p < 0.05). Consider using Wilcoxon signed-rank test instead.")





                            with col2:
                                st.markdown(
                                    f"<h4 style='text-align: center;'>Difference Distribution</h4>",
                                    unsafe_allow_html=True
                                )
                                diff = data[post_col] - data[pre_col]
                                fig_diff = go.Figure()
                                fig_diff.add_trace(go.Histogram(
                                    x=diff.dropna(),
                                    name="Difference",
                                    marker_color='#2ca02c'  # Green
                                ))
                                fig_diff.update_layout(
                                
                                    xaxis_title="Post - Pre",
                                    yaxis_title="Count"
                                )
                                st.plotly_chart(fig_diff)
                                



                            with col3:  
                                t_stat, p_value = stats.ttest_rel(
                                    data[pre_col].dropna(), 
                                    data[post_col].dropna()
                                )
                                
                                cohens_d = pg.compute_effsize(
                                    data[pre_col].dropna(), 
                                    data[post_col].dropna(),
                                    paired=True,
                                    eftype='cohen'
                                )
                                
                                show_test_results(
                                    test_name="Paired t-test",
                                    statistic=t_stat,
                                    p_value=p_value,
                                    effect_size=cohens_d,
                                    effect_name="Cohen's d",
                                    extra_info=[
                                        f"Sample size: {len(data)}",
                                        f"Pre-treatment mean: {data[pre_col].mean():.2f}",
                                        f"Post-treatment mean: {data[post_col].mean():.2f}",
                                        f"Mean difference: {diff.mean():.2f}"
                                    ]
                                )
                                    
                    elif st.session_state.selected_test == "Wilcoxon signed-rank Test":
                                w_stat, p_value = stats.wilcoxon(
                                    data[pre_col].dropna(), 
                                    data[post_col].dropna()
                                )
                                
                                show_test_results(
                                    test_name="Wilcoxon signed-rank Test",
                                    statistic=w_stat,
                                    p_value=p_value
                                )
                            
        
        elif st.session_state.selected_test in ["ANOVA", "Kruskal-Wallis Test"]:
            group_column = st.selectbox(
                "Select the grouping column (categorical variable with at least 3 groups)", 
                data.columns
            )
            unique_groups = data[group_column].dropna().unique().tolist()
            if len(unique_groups) < 3:
                st.warning(f"ANOVA/Kruskal-Wallis requires at least 3 groups, but found {len(unique_groups)} groups: {unique_groups}")
                st.stop()
            numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns available for comparison.")
                st.stop()
            
            value_column = st.selectbox("Select the numeric column to compare", numeric_cols)

            group_sizes = data.groupby(group_column)[value_column].count()
            if any(group_sizes < 2):
                small_groups = group_sizes[group_sizes < 2].index.tolist()
                st.error(f"Each group must have at least 2 observations. Groups with insufficient data: {small_groups}")
                st.stop()
            
            # Create histogram for multiple groups

            col1, col2 = st.columns([3,2])

            with col1:
                st.markdown(
                    f"<h4 style='text-align: center;'>Distribution of {value_column} by {group_column}</h4>",
                    unsafe_allow_html=True
                )
                fig = go.Figure()
                # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff']  # Color palette
                for i, group in enumerate(unique_groups):
                    group_data = data[data[group_column] == group][value_column].dropna()
                    fig.add_trace(go.Violin(
                        x=group_data,
                        y=[group] * len(group_data),
                        name=str(group),
                        orientation='h',
                        marker_color=colors[i % len(colors)],
                        line_color=colors[i % len(colors)],
                        showlegend=False
                    ))
                
                
                fig.update_layout(
                    violingap=0,
                    violinmode='overlay',
                    xaxis_title=value_column,
                    yaxis_title=group_column
                )
                st.plotly_chart(fig)
                
                if st.button("Perform Test"):
                    groups_data = [data[data[group_column]==g][value_column].dropna() for g in unique_groups]
                    
                    if st.session_state.selected_test == "ANOVA":
                        normality_results = []
                        for i, g in enumerate(unique_groups):
                            _, p = stats.shapiro(groups_data[i])
                            normality_results.append(p >= 0.05)
                            
                        if not all(normality_results):
                            st.warning("Warning: Some groups may not be normally distributed (Shapiro-Wilk test p < 0.05). Consider using Kruskal-Wallis test instead.")


            
                        _, p_var = stats.levene(*groups_data)
                        equal_var = p_var > 0.05
                        if not equal_var:
                            st.warning("Warning: Groups may not have equal variances (Levene's test p < 0.05). Consider using Welch's ANOVA or Kruskal-Wallis test.")
                        
                        with col2:
                                f_stat, p_value = stats.f_oneway(*groups_data)

                                try:
                                    group_data_numeric = data[group_column].astype(float)
                                except ValueError:
                                    group_data_numeric = pd.factorize(data[group_column])[0]
                                
                                eta_squared = pg.compute_effsize(
                                    data[value_column].dropna(),
                                    group_data_numeric[data[value_column].dropna().index],
                                    eftype='eta-square'
                                )
                                
                                show_test_results(
                                    test_name="ANOVA",
                                    statistic=f_stat,
                                    p_value=p_value,
                                    effect_size=eta_squared,
                                    effect_name="η² (eta squared)",
                                    extra_info=[
                                        f"Group sizes: {', '.join([f'{g}: {len(d)}' for g, d in zip(unique_groups, groups_data)])}",
                                        f"Group means: {', '.join([f'{g}: {d.mean():.2f}' for g, d in zip(unique_groups, groups_data)])}",
                                        f"Equal variance assumed: {'Yes' if equal_var else 'No'} (Levene's test p = {p_var:.4f})"
                                    ]
                                )
                
                    elif st.session_state.selected_test == "Kruskal-Wallis Test":
                                h_stat, p_value = stats.kruskal(*groups_data)
                                n = len(data[value_column].dropna())
                                k = len(unique_groups)
                                epsilon_squared = h_stat / (n * (k + 1))
                                
                                show_test_results(
                                    test_name="Kruskal-Wallis Test",
                                    statistic=h_stat,
                                    p_value=p_value,
                                    effect_size=epsilon_squared,
                                    effect_name="ε² (epsilon squared)",
                                    extra_info=[
                                        f"Group sizes: {', '.join([f'{g}: {len(d)}' for g, d in zip(unique_groups, groups_data)])}",
                                        f"Group medians: {', '.join([f'{g}: {d.median():.2f}' for g, d in zip(unique_groups, groups_data)])}"
                                    ]
                                )
        
        elif st.session_state.selected_test in ["Pearson correlation", "Spearman correlation"]:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X variable", data.columns)
            with col2:
                y_col = st.selectbox("Select Y variable", data.columns)
      


            col1, col2 = st.columns([3,2])
            with col1:
                st.markdown(
                    f"<h4 style='text-align: center;'>Scatter Plot</h4>",
                    unsafe_allow_html=True
                )
                # Scatter plot
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    marker_color='#2ca02c'  # Green
                ))
                if st.session_state.selected_test == "Pearson correlation":
                    fig_scatter.add_trace(go.Scatter(
                        x=data[x_col],
                        y=np.poly1d(np.polyfit(data[x_col].dropna(), data[y_col].dropna(), 1))(data[x_col].dropna()),
                        mode='lines',
                        line_color='red',
                        name='Regression Line'
                    ))
                    fig_scatter.update_layout(
                        xaxis_title=x_col,
                        yaxis_title=y_col
                    )
                    st.plotly_chart(fig_scatter)
                    
            
                if st.button("Perform Test"):
                    if st.session_state.selected_test == "Pearson correlation":
                        x = data[x_col].dropna()
                        y = data[y_col].dropna()
                        r_stat, p_value = stats.pearsonr(x, y)
                        n = len(x)
                        with col2:
                            show_test_results_with_no_effect_name(
                                test_name="Pearson correlation",
                                statistic=r_stat,
                                p_value=p_value,
                                extra_info=[
                                    f"Sample size (n): {n}",
                                    f"Correlation strength: {interpret_correlation(r_stat)}"
                                ]
                            )
                
                
        elif st.session_state.selected_test == "Chi-Square test of independence":
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Select first categorical variable", data.columns)
            with col2:
                var2 = st.selectbox("Select second categorical variable", data.columns)
            
            
            contingency_table = pd.crosstab(data[var1], data[var2])
            st.write("Contingency Table:")
            st.dataframe(contingency_table)
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown(
                    f"<div style='text-align: center;'><h4>Contingency Table Heatmap </h4></div>",
                    unsafe_allow_html=True
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                # ax.set_title("Contingency Table Heatmap")
                st.pyplot(fig)
            
            if st.button("Perform Test"):
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                n = contingency_table.sum().sum()
                phi = np.sqrt(chi2_stat / n)
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                with col2:
                    show_test_results_with_no_effect_name(
                        test_name="Chi-Square test of independence",
                        statistic=chi2_stat,
                        p_value=p_value,
                        extra_info=[
                            f"Degrees of freedom: {dof}",
                            f"Phi coefficient: {phi:.3f}",
                            f"Cramer's V: {cramers_v:.3f}"
                        ]
                    )
        
        elif st.session_state.selected_test == "Z-test for proportions":
            group_col = st.selectbox("Select group column", data.columns)
            outcome_col = st.selectbox("Select outcome column", data.columns)
            summary = data.groupby(group_col)[outcome_col].value_counts().unstack()
            st.write("Summary table:")
            st.dataframe(summary)
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown(
                    f"<div style='text-align: center;'><h4>Proportion Comparison</h4></div>",
                    unsafe_allow_html=True
                )

                success_prop = summary.iloc[:, 0] / summary.sum(axis=1)
                failure_prop = summary.iloc[:, 1] / summary.sum(axis=1)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=summary.index,
                    y=success_prop,
                    name=f'Success ({summary.columns[0]})',
                    marker_color='#1f77b4',
                    text=[f"{p:.1%}" for p in success_prop],
                    textposition='inside'
                ))
                fig.add_trace(go.Bar(
                    x=summary.index,
                    y=failure_prop,
                    name=f'Failure ({summary.columns[1]})',
                    marker_color='#ff7f0e',
                    text=[f"{p:.1%}" for p in failure_prop],
                    textposition='inside'
                ))
                
                fig.update_layout(
                    barmode='stack',
                    yaxis_title="Proportion",
                    yaxis_tickformat=".0%",
                    xaxis_title=group_col,
                    legend_title="Outcome",
                    height=400,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Perform Test"):
                from statsmodels.stats.proportion import proportions_ztest
                
                count = summary.iloc[:, 0].values
                nobs = summary.sum(axis=1).values

                z_stat, p_value = proportions_ztest(count, nobs)

                p1 = count[0] / nobs[0]
                p2 = count[1] / nobs[1]
                h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

                with col2:
                    show_test_results_with_no_effect_name(
                        test_name="Z-test for proportions",
                        statistic=z_stat,
                        p_value=p_value,
                        extra_info=[
                            f"Proportion 1: {p1:.3f}",
                            f"Proportion 2: {p2:.3f}",
                            f"Cohen's h: {abs(h):.3f}"
                        ]
                    )
            
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.write("Please ensure your file is properly formatted.")

        if st.button("Back to Test Selection"):
            st.session_state.page = "main"
        return
    
    if st.button("Back to Test Selection"):
        st.session_state.page = "main"




def show_test_results(test_name, statistic, p_value, effect_size=None, effect_name=None, extra_info=None):
    st.markdown(
    f"<div style='text-align: center;'><h4>Test Results: <em>{test_name}</em></h4></div>",
    unsafe_allow_html=True
)
    # Define tooltips for each metric
    tooltips = {
        "Statistic": "Test statistic is a value calculated from sample data that measures how far the sample deviates from the null hypothesis, used to determine whether to reject the null hypothesis.",
        "p-value": "P-value is the probability of observing current or more extreme data when the original hypothesis is true.",
    }
    if effect_name:
        if effect_name == "Cohen's d":
            tooltips[effect_name] = f"{effect_name}: Effect size in standard deviation units (0.2 small, 0.5 medium, 0.8 large)."
        elif effect_name == "η² (eta squared)":
            tooltips[effect_name] = f"{effect_name}: Proportion of variance in dependent variable explained by independent variable. Interpretation: 0.01 (small), 0.06 (medium), 0.14 (large)."
        elif effect_name == "ε² (epsilon squared)":
            tooltips[effect_name] = f"{effect_name}: Less biased version of eta squared for ANOVA effect size. Interpretation: 0.01 (small), 0.06 (medium), 0.14 (large)."
        elif effect_name == "Rank-biserial r":
            tooltips[effect_name] = f"{effect_name}: Correlation coefficient for rank-based data. Interpretation: 0.1 (small), 0.3 (medium), 0.5 (large)."
        elif effect_name == "Cramer's V":
            tooltips[effect_name] = f"{effect_name}: Measure of association between nominal variables (adjusted chi-square). Interpretation: 0.1 (small), 0.3 (medium), 0.5 (large)."
        


    # Create HTML table with working tooltips
    html = """
    <style>
    .tooltip-wrapper {
        position: relative;
        display: inline-block;
    }
    
    .tooltip-trigger {
        border-bottom: 1px dotted #666;
        cursor: help;
    }
    
    .tooltip-content {
        visibility: hidden;
        width: 300px;
        max-width: 80vw;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 4px;
        padding: 8px;
        position: absolute;
        z-index: 9999;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
        line-height: 1.4;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        word-wrap: break-word;
    }
    
    .tooltip-wrapper:hover .tooltip-content {
        visibility: visible;
        opacity: 1;
    }
    
    /* Arrow for tooltip */
    .tooltip-content::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 600px) {
        .tooltip-content {
            left: 0;
            transform: none;
            right: auto;
        }
        .tooltip-content::after {
            left: 15px;
            margin-left: 0;
        }
    }
    
    /* Table styling */
    .results-table {
        width: 90%;
        border-collapse: collapse;
        margin-bottom: 1rem;
        font-family: sans-serif;
        margin-left: 50px;
    }
    
    .results-table th, 
    .results-table td {
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    
    .results-table th {
        background-color: #f5f5f5;
        font-weight: 600;
    }
    
    .results-table tr:hover {
        background-color: #f9f9f9;
    }
    
    .value-cell {
        font-family: monospace;
        text-align: right;
    }
    </style>
    
    <table class="results-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Add Statistic row
    html += f"""
    <tr>
        <td>
            <div class="tooltip-wrapper">
                <span class="tooltip-trigger">Statistic</span>
                <div class="tooltip-content">{tooltips['Statistic']}</div>
            </div>
        </td>
        <td class="value-cell">{statistic:.4f}</td>
    </tr>
    """
    
    # Add p-value row
    html += f"""
    <tr>
        <td>
            <div class="tooltip-wrapper">
                <span class="tooltip-trigger">p-value</span>
                <div class="tooltip-content">{tooltips['p-value']}</div>
            </div>
        </td>
        <td class="value-cell">{p_value:.4f}</td>
    </tr>
    """
    html += f"""
    <tr>
        <td>
            <div class="tooltip-wrapper">
                <span class="tooltip-trigger">{effect_name}</span>
                <div class="tooltip-content">{tooltips[effect_name]}</div>
            </div>
        </td>
        <td class="value-cell">{effect_size:.3f}</td>
    </tr>
    """
    
    # Display the table
    st.markdown(html, unsafe_allow_html=True)
    
    # Significance interpretation
    alpha = 0.05
    if p_value < alpha:
        st.success("Statistically significant (p < 0.05)")
    else:
        st.warning("Not statistically significant (p ≥ 0.05)")
    
    # Effect size interpretation
    if effect_size is not None and effect_name is not None:
        if effect_name == "Cohen's d":
            st.write(f"**Effect size interpretation**: {interpret_cohens_d(effect_size)}")
        elif effect_name == "η² (eta squared)":
            st.write(f"**Effect size interpretation**: {interpret_eta_squared(effect_size)}")
        elif effect_name == "ε² (epsilon squared)":
            st.write(f"**Effect size interpretation**: {interpret_epsilon_squared(effect_size)}")
        elif effect_name == "Rank-biserial r":
            st.write(f"**Effect size interpretation**: {interpret_rank_biserial(effect_size)}")
        elif effect_name == "Cramer's V":
            st.write(f"**Effect size interpretation**: {interpret_cramers_v(effect_size)}")
    
    # Additional information
    if extra_info is not None:

        expander = st.expander("# Additional Information")
        if isinstance(extra_info, list):
            for info in extra_info:
                expander.write(info)
        else:
            expander.write(extra_info)



def interpret_eta_squared(value):
    abs_value = abs(value)
    if abs_value >= 0.14:
        return "Large effect"
    elif abs_value >= 0.06:
        return "Medium effect"
    elif abs_value >= 0.01:
        return "Small effect"
    else:
        return "Negligible effect"

def interpret_epsilon_squared(value):
    return interpret_eta_squared(value)  # Same interpretation as eta squared

def interpret_rank_biserial(value):
    abs_value = abs(value)
    if abs_value >= 0.5:
        return "Large effect"
    elif abs_value >= 0.3:
        return "Medium effect"
    elif abs_value >= 0.1:
        return "Small effect"
    else:
        return "Negligible effect"

def interpret_cramers_v(value):
    abs_value = abs(value)
    if abs_value >= 0.35:
        return "Strong association"
    elif abs_value >= 0.2:
        return "Moderate association"
    elif abs_value >= 0.1:
        return "Weak association"
    else:
        return "Negligible association"

def interpret_effect_size(effect_name, value):
    """Interpret different types of effect sizes"""
    abs_value = abs(value)
    
    if effect_name == "Cohen's d":
        if abs_value >= 0.8:
            return "Large effect"
        elif abs_value >= 0.5:
            return "Medium effect"
        elif abs_value >= 0.2:
            return "Small effect"
        else:
            return "Negligible effect"
    elif effect_name in ["η² (eta squared)", "ε² (epsilon squared)"]:
        if abs_value >= 0.14:
            return "Large effect"
        elif abs_value >= 0.06:
            return "Medium effect"
        elif abs_value >= 0.01:
            return "Small effect"
        else:
            return "Negligible effect"
    elif effect_name == "Rank-biserial r":
        if abs_value >= 0.5:
            return "Large effect"
        elif abs_value >= 0.3:
            return "Medium effect"
        elif abs_value >= 0.1:
            return "Small effect"
        else:
            return "Negligible effect"
    elif effect_name == "Cramer's V":
        if abs_value >= 0.35:
            return "Strong association"
        elif abs_value >= 0.2:
            return "Moderate association"
        elif abs_value >= 0.1:
            return "Weak association"
        else:
            return "Negligible association"
    else:
        return ""

def interpret_correlation(r):
    """解释相关系数大小"""
    abs_r = abs(r)
    if abs_r >= 0.8:
        return "Very strong"
    elif abs_r >= 0.6:
        return "Strong"
    elif abs_r >= 0.4:
        return "Moderate"
    elif abs_r >= 0.2:
        return "Weak"
    else:
        return "Very weak or none"

def interpret_cohens_d(d):
    """解释Cohen's d效应量"""
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "Large effect"
    elif abs_d >= 0.5:
        return "Medium effect"
    elif abs_d >= 0.2:
        return "Small effect"
    else:
        return "Negligible effect"

if "page" not in st.session_state:
    st.session_state.page = "main"
    st.session_state.selected_test = None

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "test_details":
    test_details_page()

