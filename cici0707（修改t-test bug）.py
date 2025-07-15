import streamlit as st
import chardet
import openpyxl

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


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg

def test_details_page():
    st.title(f"{st.session_state.selected_test}")
    st.write(f"You can use {st.session_state.selected_test} to test your own data")
    uploaded_file = st.file_uploader("Upload your data (CSV or Excel  format)",  type=["csv", "xlsx", "xls"])
    
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
                st.warning(f"Selected column '{selected_column}' must have exactly 2 groups, but found {len(unique_vals)} groups: {unique_vals}")#这里想醒目就用把st.warning改成st.error
                st.stop()
            
            st.write(f"Data from column: {selected_column}")
            st.dataframe(data[[selected_column]].head())
            
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
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                x=selected_column,
                y=value_col,
                data=filtered,
                ax=ax
            )
            ax.set_title("Data Distribution Comparison")
            st.pyplot(fig)
            
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
                        st.warning("Warning: One or both groups may not be normally distributed (Shapiro-Wilk test p < 0.05). Consider using Mann-Whitney U test instead.")#不想要warning直接299-300行删掉就行
                    
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
                    
               
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.write("Please ensure your file is properly formatted.")

        if st.button("Back to Test Selection"):
            st.session_state.page = "main"
        return
    
    # back button
    if st.button("Back to Test Selection"):
        st.session_state.page = "main"



def show_test_results(test_name, statistic, p_value, effect_size=None, effect_name=None, extra_info=None):
    """Display a standardized format for test results"""
    st.subheader("Test Results")
    st.write(f"**Test**: {test_name}")
    st.write(f"**Statistic**: {statistic:.4f}")
    st.write(f"**p-value**: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        st.success("Result is statistically significant (p < 0.05)")
    else:
        st.warning("Result is not statistically significant (p ≥ 0.05)")

    if effect_size is not None:
        st.write(f"**{effect_name}**: {effect_size:.3f}")
        if effect_name == "Cohen's d":
            st.write(f"Effect size interpretation: {interpret_cohens_d(effect_size)}")
    
    if extra_info is not None:
        if isinstance(extra_info, list):
            for info in extra_info:
                st.write(info)
        else:
            st.write(extra_info)

def interpret_correlation(r):
    """Interpret the magnitude of the correlation coefficient"""
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
    """Explain Cohen's d effect sizes"""
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