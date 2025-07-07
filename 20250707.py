import streamlit as st
import chardet
import openpyxl
def main_page():
    st.title("Decision tree selection for statistical testing methods")
    st.write("Choose the appropriate statistical testing method based on your data type and needs")

    # Initialize session state variables
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
        st.session_state.data_type = None
        st.session_state.organization = None
        st.session_state.distribution = None
        st.session_state.task = None
        st.session_state.selected_test = None

    # Step 1: Data type selection (always shown)
    if st.session_state.current_step >= 1:
        st.session_state.data_type = st.selectbox(
            "1. What is the data type?",
            ["", "Numerical", "Categorical"],
            key="step1_data_type"
        )

    # Only show next questions if data type is selected
    if st.session_state.data_type and st.session_state.current_step >= 1:
        st.session_state.current_step = 2
        
        # Step 2: Numerical data organization
        if st.session_state.data_type == "Numerical":
            st.session_state.organization = st.selectbox(
                "2. How numerical data are organized?",
                ["", "Two Groups", "Paired Data", "More than Two Groups", "Relationship between Multiple Variables"],
                key="step2_organization"
            )

            # Only show step 3 if organization is selected
            if st.session_state.organization and st.session_state.current_step >= 2:
                st.session_state.current_step = 3
                
                # Step 3: Distribution assumptions
                if st.session_state.organization in ["Two Groups", "Paired Data", "More than Two Groups", "Relationship between Multiple Variables"]:
                    st.session_state.distribution = st.selectbox(
                        "3. Data distribution assumptions?",
                        ["", "Yes(Parametric Test)", "No(Non-Parametric Test)"],
                        key="step3_distribution"
                    )

                    # Show recommended test based on selections
                    if st.session_state.distribution:
                        if st.session_state.organization == "Two Groups":
                            if st.session_state.distribution == "Yes(Parametric Test)":
                                st.success("Recommended use: **Independent t-test**")
                                st.session_state.selected_test = "Independent t-test"
                            else:
                                st.success("Recommended use: **Mann-Whitney U Test**")
                                st.session_state.selected_test = "Mann-Whitney U Test"
                        
                        elif st.session_state.organization == "Paired Data":
                            if st.session_state.distribution == "Yes(Parametric Test)":
                                st.success("Recommended use: **Paired t-test**")
                                st.session_state.selected_test = "Paired t-test"
                            else:
                                st.success("Recommended use: **Wilcoxon signed-rank Test**")
                                st.session_state.selected_test = "Wilcoxon signed-rank Test"
                        
                        elif st.session_state.organization == "More than Two Groups":
                            if st.session_state.distribution == "Yes(Parametric Test)":
                                st.success("Recommended use: **ANOVA**")
                                st.session_state.selected_test = "ANOVA"
                            else:
                                st.success("Recommended use: **Kruskal-Wallis Test**")
                                st.session_state.selected_test = "Kruskal-Wallis Test"
                        
                        elif st.session_state.organization == "Relationship between Multiple Variables":
                            if st.session_state.distribution == "Yes(Parametric Test)":
                                st.success("Recommended use: **Pearson correlation**")
                                st.session_state.selected_test = "Pearson correlation"
                            else:
                                st.success("Recommended use: **Spearman correlation**")
                                st.session_state.selected_test = "Spearman correlation"

        # Categorical data task selection
        elif st.session_state.data_type == "Categorical":
            st.session_state.task = st.selectbox(
                "2. What is your target task?",
                ["", "Compare two variables", "Test proportions"],
                key="step2_task"
            )

            if st.session_state.task:
                if st.session_state.task == "Compare two variables":
                    st.success("Recommended use: **Chi-Square test of independence**")
                    st.session_state.selected_test = "Chi-Square test of independence"
                elif st.session_state.task == "Test proportions":
                    st.success("Recommended use: **Z-test for proportions**")
                    st.session_state.selected_test = "Z-test for proportions"

    # Only show test button after all steps are completed
    if st.session_state.selected_test:
        if st.button("Go to Test!"):
            st.session_state.page = "test_details"
# def main_page():
#     st.title("Decision tree selection for statistical testing methods")
#     st.write("Choose the appropriate statistical testing method based on your data type and needs")
#     data_type = st.selectbox(
#         "1. What is the data type?",
#         ("Numerical", "Categorical")
#     )
    
#     if data_type == "Numerical":
#         st.markdown("### Numerical Data")
#         organization = st.selectbox(
#             "2. How numerical data are organized?",
#             ("Two Groups", "Paired Data", "More than Two Groups","Relationship between Multiple Variables")
#         )
        
#         if organization == "Two Groups":
#             st.markdown("#### Two Groups")
            
#             distribution = st.selectbox(
#                 "3. Data distribution assumptions?",
#                 ("Yes(Parametric Test)", "No(Non-Parametric Test)")
#             )
            
#             if distribution == "Yes(Parametric Test)":
#                 st.success("Recommended use: **Independent t-test**")
#                 st.write("Applicable conditions: Normal distribution of data, homogeneity of variance, independent observation")
#                 st.session_state.selected_test = "Independent t-test"
#             else:
#                 st.success("Recommended use: **Mann-Whitney U Test**")
#                 st.write("Applicable conditions: Sequential data or numerical data with non normal distribution")
#                 st.session_state.selected_test = "Mann-Whitney U Test"
                
#         elif organization == "Paired Data":
#             st.markdown("#### Paired Data")
#             distribution = st.selectbox(
#                 "3. Data distribution assumptions?",
#                 ("Yes(Parametric Test)", "No(Non-Parametric Test)")
#             )
            
#             if distribution == "Yes(Parametric Test)":
#                 st.success("Recommended use: **Paired t-test**")
#                 st.write("Applicable conditions: Both sets of data are continuous numerical data")
#                 st.session_state.selected_test = "Paired t-test"
#             else:
#                 st.success("Recommended use: **Wilcoxon signed-rank Test**")
#                 st.write("Applicable conditions: Two sets of data are either continuous numerical data or ordered categorical data")
#                 st.session_state.selected_test = "Wilcoxon signed-rank Test"
                
#         elif organization == "More than Two Groups":
#             st.markdown("#### More than Two Groups")
#             relationship_type = st.selectbox(
#                 "3. Data distribution assumptions?",
#                 ("Yes(Parametric Test)", "No(Non-Parametric Test)")
#             )
            
#             if relationship_type == "Yes(Parametric Test)":
#                 st.success("Recommended use: **ANOVA**")
#                 st.write("Applicable conditions: Normal distribution of data, homogeneity of variance, independent observation")
#                 st.write("If significant differences are found, post hoc testing such as Tukey HSD can be conducted")
#                 st.session_state.selected_test = "ANOVA"
#             else:
#                 st.success("Recommended use: **Kruskal-Wallis Test**")
#                 st.write("Applicable conditions: Sequential data or numerical data with non normal distribution")
#                 st.session_state.selected_test = "Kruskal-Wallis Test"

#         elif organization == "Relationship between Multiple Variables":
#             st.markdown("#### Relationship between Multiple Variables")
            
#             relationship_type = st.selectbox(
#                 "3. Data distribution assumptions?",
#                 ("Yes(Parametric Test)", "No(Non-Parametric Test)")
#             )
            
#             if relationship_type == "Yes(Parametric Test)":
#                 st.success("Recommended use: **Pearson correlation Regression analysis** ")
#                 st.session_state.selected_test = "Pearson correlation"
#             else:
#                 st.success("Recommended use: **Spearman correlation**")
#                 st.write("Applicable conditions: The relationship between a continuous dependent variable and one or more predictor variables")
#                 st.session_state.selected_test = "Spearman correlation"
    
#     else:  

#         st.markdown("### Categorical Data")
#         task = st.selectbox(
#             "2. What is your target task?",
#             ("Compare two variavles", "Test proportions")
#         )
        
#         if task == "Compare two variavles":
#             st.markdown("#### Compare two variavles")
#             st.success("Recommended use: **Chi-Square test of independence**")
#             st.session_state.selected_test = "Chi-Square test of independence"
                
#         elif task == "Test proportions":
#             st.success("Recommended use: **Z-test for proportions Binomial test**")
#             st.session_state.selected_test = "Z-test for proportions"
#     if st.button("Go to Test!"):
#         st.session_state.page = "test_details"

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
            col1, col2 = st.columns(2)
            with col1:
                group1_col = st.selectbox("Select column for Group 1", data.columns)
            with col2:
                group2_col = st.selectbox("Select column for Group 2", data.columns)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=data[[group1_col, group2_col]], ax=ax)
            ax.set_title("Data Distribution Comparison")
            st.pyplot(fig)
            
            if st.button("Perform Test"):
                if st.session_state.selected_test == "Independent t-test":
                    t_stat, p_value = stats.ttest_ind(
                        data[group1_col].dropna(), 
                        data[group2_col].dropna(),
                        equal_var=True
                    )
                    cohens_d = pg.compute_effsize(
                        data[group1_col].dropna(), 
                        data[group2_col].dropna(),
                        eftype='cohen'
                    )
                    show_test_results(
                        test_name="Independent t-test",
                        statistic=t_stat,
                        p_value=p_value,
                        effect_size=cohens_d,
                        effect_name="Cohen's d"
                    )
                    
                elif st.session_state.selected_test == "Mann-Whitney U Test":
                    u_stat, p_value = stats.mannwhitneyu(
                        data[group1_col].dropna(), 
                        data[group2_col].dropna(),
                        alternative='two-sided'
                    )
                    show_test_results(
                        test_name="Mann-Whitney U Test",
                        statistic=u_stat,
                        p_value=p_value
                    )
        
        elif st.session_state.selected_test in ["Paired t-test", "Wilcoxon signed-rank Test"]:
            col1, col2 = st.columns(2)
            with col1:
                pre_col = st.selectbox("Select Pre-treatment column", data.columns)
            with col2:
                post_col = st.selectbox("Select Post-treatment column", data.columns)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            sns.boxplot(data=data[[pre_col, post_col]], ax=ax1)
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
            group_col = st.selectbox("Select grouping variable", data.columns)
            value_col = st.selectbox("Select value variable", data.columns)

            groups = data[group_col].unique()
            if len(groups) < 3:
                st.warning("Please select a grouping variable with at least 3 groups")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=group_col, y=value_col, data=data, ax=ax)
                ax.set_title("Comparison Across Groups")
                st.pyplot(fig)
                
                if st.button("Perform Test"):
                    if st.session_state.selected_test == "ANOVA":
                        groups_data = [data[data[group_col]==g][value_col].dropna() for g in groups]
                        f_stat, p_value = stats.f_oneway(*groups_data)

                        try:
                            group_data_numeric = data[group_col].astype(float)
                        except ValueError:
                            group_data_numeric = pd.factorize(data[group_col])[0]
                        
                        eta_squared = pg.compute_effsize(
                            data[value_col].dropna(),
                            group_data_numeric[data[value_col].dropna().index],
                            eftype='eta-square'
                        )
                        
                        show_test_results(
                            test_name="ANOVA",
                            statistic=f_stat,
                            p_value=p_value,
                            effect_size=eta_squared,
                            effect_name="η² (eta squared)"
                        )

                        if p_value < 0.05:
                            st.subheader("Post-hoc Tests (Tukey HSD)")
                            try:
                                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                                tukey = pairwise_tukeyhsd(
                                    endog=data[value_col].dropna(),
                                    groups=data[group_col].dropna(),
                                    alpha=0.05
                                )
                                st.text(tukey.summary())
                            except Exception as e:
                                st.warning(f"Could not perform Tukey HSD test: {str(e)}")
                    
                    elif st.session_state.selected_test == "Kruskal-Wallis Test":
                        h_stat, p_value = stats.kruskal(
                            *[data[data[group_col]==g][value_col].dropna() for g in groups]
                        )

                        show_test_results(
                            test_name="Kruskal-Wallis Test",
                            statistic=h_stat,
                            p_value=p_value
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



def show_test_results(test_name, statistic, p_value, effect_size=None, effect_name=None, extra_info=None):
    """显示统一的测试结果格式"""
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