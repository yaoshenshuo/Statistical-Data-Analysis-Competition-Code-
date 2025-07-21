import streamlit as st
import chardet
import openpyxl
import plotly.graph_objects as go
import numpy as np

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
                st.warning(f"Selected column '{selected_column}' must have exactly 2 groups, but found {len(unique_vals)} groups: {unique_vals}")#st.warning or st.error
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


            fig = go.Figure()
            fig.add_trace(go.Histogram(
                       x=g1, 
                       name=str(group1_col),
                      ))
            fig.add_trace(go.Histogram(
                       x=g2, 
                       name=str(group2_col),
                      ))
            fig.update_layout(
                       barmode='overlay',   
                             
                       xaxis_title=value_col,
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
            
        

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                       x=pre_col, 
                       name=str(pre_col),
                      ))
            fig.add_trace(go.Histogram(
                       x=post_col, 
                       name=str(post_col),
                      ))
            fig.update_layout(
                       barmode='overlay',   
                             
                       xaxis_title=ax1,
                       legend_title=selected_column,
                     )
            st.plotly_chart(fig)
            ax1.set_title("Pre-Post Comparison")
            
            diff = data[post_col] - data[pre_col]
            sns.histplot(diff.dropna(), kde=True, ax=ax2)
            ax2.set_title("Difference Distribution")
            st.pyplot(fig)
            
            if st.button("Perform Test"):
                if st.session_state.selected_test == "Paired t-test":
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
                        effect_name="Cohen's d"
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
            # 选择分组列
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
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                x=group_column, 
                y=value_column, 
                data=data,
                ax=ax
            )
            ax.set_title(f"{value_column} Distribution by {group_column}")
            st.pyplot(fig)
            
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


                    #HSD

                    # if p_value < 0.05:
                    #     st.subheader("Post-hoc Tests (Tukey HSD)")
                    #     try:
                    #         from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    #         tukey = pairwise_tukeyhsd(
                    #             endog=data[value_col].dropna(),
                    #             groups=data[group_col].dropna(),
                    #             alpha=0.05
                    #         )
                    #         st.text(tukey.summary())
                    #     except Exception as e:
                    #         st.warning(f"Could not perform Tukey HSD test: {str(e)}")
                
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

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=x_col, y=y_col, data=data, ax=ax)

            if st.session_state.selected_test == "Pearson correlation":
                sns.regplot(x=x_col, y=y_col, data=data, ax=ax, scatter=False, color='red')
            
            ax.set_title("Scatter Plot with Correlation")
            st.pyplot(fig)
            
            if st.button("Perform Test"):
                if st.session_state.selected_test == "Pearson correlation":
                    r_stat, p_value = stats.pearsonr(
                        data[x_col].dropna(), 
                        data[y_col].dropna()
                    )

                    show_test_results(
                        test_name="Pearson correlation",
                        statistic=r_stat,
                        p_value=p_value,
                        extra_info=f"Correlation strength: {interpret_correlation(r_stat)}"
                    )
                
                elif st.session_state.selected_test == "Spearman correlation":

                    r_stat, p_value = stats.spearmanr(
                        data[x_col].dropna(), 
                        data[y_col].dropna()
                    )
                    
                    show_test_results(
                        test_name="Spearman correlation",
                        statistic=r_stat,
                        p_value=p_value,
                        extra_info=f"Correlation strength: {interpret_correlation(r_stat)}"
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
            

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
            ax.set_title("Contingency Table Heatmap")
            st.pyplot(fig)
            
            if st.button("Perform Test"):

                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                n = contingency_table.sum().sum()
                phi = np.sqrt(chi2_stat / n)
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

                show_test_results(
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
            st.write("Please prepare your data in one of these formats:")
            st.write("1. Raw data with one row per observation")
            st.write("2. Summary data with success counts and totals")
            
            format_choice = st.radio("Data format:", ("Raw data", "Summary data"))
            
            if format_choice == "Raw data":
                group_col = st.selectbox("Select group column", data.columns)
                outcome_col = st.selectbox("Select outcome column", data.columns)

                summary = data.groupby(group_col)[outcome_col].value_counts().unstack()
                st.write("Summary table:")
                st.dataframe(summary)
                
                if st.button("Perform Test"):
                    from statsmodels.stats.proportion import proportions_ztest
                    
                    count = summary.iloc[:, 0].values
                    nobs = summary.sum(axis=1).values

                    z_stat, p_value = proportions_ztest(count, nobs)

                    p1 = count[0] / nobs[0]
                    p2 = count[1] / nobs[1]
                    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

                    show_test_results(
                        test_name="Z-test for proportions",
                        statistic=z_stat,
                        p_value=p_value,
                        extra_info=[
                            f"Proportion 1: {p1:.3f}",
                            f"Proportion 2: {p2:.3f}",
                            f"Cohen's h: {abs(h):.3f}"
                        ]
                    )
            
            else:  
                col1, col2 = st.columns(2)
                with col1:
                    success1 = st.number_input("Successes in Group 1", min_value=0, value=30)
                    total1 = st.number_input("Total in Group 1", min_value=1, value=100)
                with col2:
                    success2 = st.number_input("Successes in Group 2", min_value=0, value=40)
                    total2 = st.number_input("Total in Group 2", min_value=1, value=100)
                
                if st.button("Perform Test"):
                    from statsmodels.stats.proportion import proportions_ztest
                    

                    count = np.array([success1, success2])
                    nobs = np.array([total1, total2])
                    z_stat, p_value = proportions_ztest(count, nobs)

                    p1 = success1 / total1
                    p2 = success2 / total2
                    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
                    

                    show_test_results(
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
    
    # 返回按钮
    if st.button("Back to Test Selection"):
        st.session_state.page = "main"



def show_test_results(test_name, statistic, p_value, effect_size=None, effect_name=None, extra_info=None, extra_info2=None):
    """Displays a uniform format for test results"""
    st.markdown(f"## Test Results: *{test_name}*")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Statistic",
            value=float(statistic),
            format="%.4f",
            disabled=True,
            help="Test statistic is a value calculated from sample data that measures how far the sample deviates from the null hypothesis, used to determine whether to reject the null hypothesis."
        )
    with c2:
        st.number_input(
            "p-value",
            value=float(p_value),
            format="%.4f",
            disabled=True,
            help="P-value is the probability of observing current or more extreme data when the original hypothesis is true."
        )


    alpha = 0.05
    if p_value < alpha:
        st.success("Statistically significant (p < 0.05), ")
    else:
        st.warning("Not statistically significant (p ≥ 0.05)")

    c3, c4 = st.columns(2)
    with c3:
        st.number_input(
            effect_name,
            value=float(effect_size),
            format="%.3f",
            disabled=True,
            help=f"{effect_name}: Effect size in standard deviation units (0.2 small, 0.5 medium, 0.8 large)."
        )


    
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