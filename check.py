import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(
    page_title="Your tool name",
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


st.title("📈 Your tool name")
st.markdown("""
<div class="highlight blue">
    Introduction
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("🔧 数据输入方式")
    data_source = st.radio("选择数据来源", 
                         ["手动输入示例数据", "上传电子表格"],
                         index=0,
                         help="选择手动输入测试数据或上传您的CSV/Excel文件")
    
    st.header("📋 检验方法选择")
    test_type = st.selectbox(
        "选择统计检验类型",
        ["均值差异检验", "比例/分布检验", "相关性检验", "非参数检验"],
        index=0,
        help="根据您的业务问题类型选择检验方法"
    )

if test_type == "均值差异检验":
    sub_test = st.sidebar.selectbox(
        "具体检验方法",
        ["单样本t检验", "独立样本t检验", "配对样本t检验", "ANOVA方差分析"],
        index=0
    )
elif test_type == "比例/分布检验":
    sub_test = st.sidebar.selectbox(
        "具体检验方法",
        ["卡方检验", "二项检验", "McNemar检验"],
        index=0
    )
elif test_type == "相关性检验":
    sub_test = st.sidebar.selectbox(
        "具体检验方法",
        ["Pearson相关系数", "Spearman秩相关", "Cramer's V系数"],
        index=0
    )
else:
    sub_test = st.sidebar.selectbox(
        "具体检验方法",
        ["Mann-Whitney U检验", "Wilcoxon符号秩检验", "Kruskal-Wallis检验"],
        index=0
    )




st.header("📁 数据准备")
if data_source == "手动输入示例数据":
    st.warning("正在使用示例数据，您可以在侧边栏切换为上传自己的数据")
    if sub_test in ["单样本t检验", "独立样本t检验", "配对样本t检验"]:
        group1 = np.random.normal(loc=5, scale=2, size=100)
        group2 = np.random.normal(loc=6, scale=2, size=100)
        df = pd.DataFrame({
            "广告组A": group1,
            "广告组B": group2,
            "促销前": np.random.randint(1, 6, size=100),
            "促销后": np.random.randint(2, 7, size=100)
        })
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'))
        
    elif sub_test == "ANOVA方差分析":
        group1 = np.random.normal(loc=5, scale=1.5, size=30)
        group2 = np.random.normal(loc=6, scale=1.5, size=30)
        group3 = np.random.normal(loc=4, scale=1.5, size=30)
        df = pd.DataFrame({
            "广告版本A": group1,
            "广告版本B": group2,
            "广告版本C": group3
        })
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'))
        
    elif sub_test == "卡方检验":
        contingency_table = pd.DataFrame({
            '点击广告': [120, 80],
            '未点击广告': [30, 70]
        }, index=['男性', '女性'])
        st.dataframe(contingency_table.style.background_gradient(cmap='Blues'))




else:
    uploaded_file = st.file_uploader("上传您的数据文件 (CSV或Excel)", 
                                   type=["csv", "xlsx"],
                                   help="请确保数据格式正确，包含所需的变量列")
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("数据上传成功！")
        st.dataframe(df.head().style.background_gradient(cmap='Blues'))


st.header("⚙️ 检验参数设置")
with st.expander("设置检验参数", expanded=True):
    if sub_test == "单样本t检验":
        test_value = st.number_input("比较的基准值", value=5.0)
        selected_col = st.selectbox("选择分析列", df.columns)
        
    elif sub_test == "独立样本t检验":
        col1, col2 = st.columns(2)
        with col1:
            group1_col = st.selectbox("选择第一组数据", df.columns)
        with col2:
            group2_col = st.selectbox("选择第二组数据", df.columns)
        equal_var = st.checkbox("假设方差齐性", value=True)
        
    elif sub_test == "配对样本t检验":
        col1, col2 = st.columns(2)
        with col1:
            pre_col = st.selectbox("选择前测数据", df.columns)
        with col2:
            post_col = st.selectbox("选择后测数据", df.columns)
            
    elif sub_test == "ANOVA方差分析":
        selected_cols = st.multiselect("选择需要比较的组别", df.columns, default=df.columns.tolist()[:3])
        
    elif sub_test == "卡方检验":
        if data_source == "手动输入示例数据":
            observed = contingency_table.values
        else:
            row_col = st.selectbox("选择行变量", df.columns)
            col_col = st.selectbox("选择列变量", df.columns)
            observed = pd.crosstab(df[row_col], df[col_col]).values
            
    elif sub_test == "二项检验":
        success_count = st.number_input("成功次数", min_value=0, value=120)
        total_count = st.number_input("总试验次数", min_value=1, value=200)
        expected_p = st.slider("预期成功比例", 0.0, 1.0, 0.5)
        
    elif sub_test == "Pearson相关系数" or sub_test == "Spearman秩相关":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("选择X变量", df.columns)
        with col2:
            y_col = st.selectbox("选择Y变量", df.columns)




st.header("🔬 检验结果分析")
if st.button("执行统计检验", type="primary"):
    st.balloons()
    
    # 单样本t检验
    if sub_test == "单样本t检验":
        t_stat, p_value = stats.ttest_1samp(df[selected_col].dropna(), test_value)
        result_df = pd.DataFrame({
            "指标": ["t统计量", "p值", "均值差异", "置信区间(95%)"],
            "值": [
                t_stat, 
                p_value,
                df[selected_col].mean() - test_value,
                stats.t.interval(0.95, len(df[selected_col].dropna())-1, 
                                loc=df[selected_col].mean(), 
                                scale=stats.sem(df[selected_col].dropna()))
            ]
        })
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_col], kde=True, ax=ax, color='skyblue')
        ax.axvline(test_value, color='red', linestyle='--', label=f'基准值 ({test_value})')
        ax.axvline(df[selected_col].mean(), color='green', linestyle='--', label=f'样本均值 ({df[selected_col].mean():.2f})')
        ax.set_title(f"单样本t检验: {selected_col} 分布")
        ax.legend()
        st.pyplot(fig)
        
   
    elif sub_test == "独立样本t检验":
        t_stat, p_value = stats.ttest_ind(df[group1_col].dropna(), 
                                         df[group2_col].dropna(),
                                         equal_var=equal_var)
        result_df = pd.DataFrame({
            "指标": ["t统计量", "p值", "均值差异", "组1均值", "组2均值"],
            "值": [
                t_stat, 
                p_value,
                df[group1_col].mean() - df[group2_col].mean(),
                df[group1_col].mean(),
                df[group2_col].mean()
            ]
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df[[group1_col, group2_col]], ax=ax, palette="Blues")
        ax.set_title(f"独立样本t检验: {group1_col} vs {group2_col}")
        st.pyplot(fig)

    elif sub_test == "配对样本t检验":
        t_stat, p_value = stats.ttest_rel(df[pre_col], df[post_col])
        differences = df[post_col] - df[pre_col]
        
        result_df = pd.DataFrame({
            "指标": ["t统计量", "p值", "平均差异", "前测均值", "后测均值"],
            "值": [
                t_stat, 
                p_value,
                differences.mean(),
                df[pre_col].mean(),
                df[post_col].mean()
            ]
        })
        

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        sns.boxplot(data=df[[pre_col, post_col]], ax=ax[0], palette="Blues")
        ax[0].set_title("前后测对比")
        sns.histplot(differences, kde=True, ax=ax[1], color='skyblue')
        ax[1].axvline(0, color='red', linestyle='--')
        ax[1].set_title("差异值分布")
        st.pyplot(fig)

    elif sub_test == "ANOVA方差分析":
        f_stat, p_value = stats.f_oneway(*[df[col].dropna() for col in selected_cols])
        
        # 事后检验(Tukey HSD)
        tukey_data = pd.melt(df[selected_cols]).dropna()
        tukey_results = pairwise_tukeyhsd(tukey_data['value'], tukey_data['variable'])
        
        result_df = pd.DataFrame({
            "指标": ["F统计量", "p值"],
            "值": [f_stat, p_value]
        })

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        sns.boxplot(data=df[selected_cols], ax=ax[0], palette="Blues")
        ax[0].set_title("P1")
        
        tukey_df = pd.DataFrame(
            data=tukey_results._results_table.data[1:],
            columns=tukey_results._results_table.data[0]
            )
        sig_df = tukey_df[tukey_df['reject']].sort_values('meandiff')
        
        if not sig_df.empty:
            ax[1].errorbar(
                x=sig_df['meandiff'],
                y=range(len(sig_df)),
                xerr=sig_df['upper'] - sig_df['meandiff'],
                fmt='o',
                color='skyblue',
                ecolor='gray',
                capsize=5
            )
            ax[1].axvline(0, color='red', linestyle='--')
            ax[1].set_yticks(range(len(sig_df)))
            ax[1].set_yticklabels([f"{row['group1']} vs {row['group2']}" for _, row in sig_df.iterrows()])
            ax[1].set_title("显著差异组对比 (Tukey HSD)")
        else:
            ax[1].text(0.5, 0.5, '无显著差异组', 
                    ha='center', va='center', fontsize=12)
            ax[1].set_title("Tukey HSD 结果")
        
        st.pyplot(fig)
        
        st.subheader("Tukey HSD事后检验结果")
        numeric_cols = tukey_df.select_dtypes(include=['float64']).columns
        styled_tukey = tukey_df.style.format({col: "{:.4f}" for col in numeric_cols})
        st.dataframe(styled_tukey)
        
    # 卡方检验
    elif sub_test == "卡方检验":
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
        
        result_df = pd.DataFrame({
            "指标": ["卡方统计量", "p值", "自由度"],
            "值": [chi2_stat, p_value, dof]
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(observed/np.sum(observed), annot=True, fmt=".2%", cmap="Blues", ax=ax)
        ax.set_title("观察比例热力图")
        st.pyplot(fig)
        

    elif sub_test == "二项检验":
        p_value = stats.binom_test(success_count, n=total_count, p=expected_p)
        
        result_df = pd.DataFrame({
            "指标": ["观察比例", "预期比例", "p值"],
            "值": [
                success_count/total_count,
                expected_p,
                p_value
            ]
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(['观察值', '期望值'], 
               [success_count/total_count, expected_p], 
               color=['skyblue', 'lightgray'])
        ax.set_ylim(0, 1)
        ax.set_title("观察比例 vs 预期比例")
        st.pyplot(fig)

    elif sub_test in ["Pearson相关系数", "Spearman秩相关"]:
        if sub_test == "Pearson相关系数":
            corr, p_value = stats.pearsonr(df[x_col].dropna(), df[y_col].dropna())
        else:
            corr, p_value = stats.spearmanr(df[x_col].dropna(), df[y_col].dropna())
            
        result_df = pd.DataFrame({
            "指标": ["相关系数", "p值"],
            "值": [corr, p_value]
        })
        

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=df[x_col], y=df[y_col], ax=ax, scatter_kws={'alpha':0.5})
        ax.set_title(f"{sub_test}: {x_col} vs {y_col}")
        st.pyplot(fig)
    
    st.subheader("检验统计结果")
    # st.dataframe(result_df.style.format({"值": "{:.4f}"}).highlight_max(axis=0, color='lightblue'))
    def prepare_results(result_df):
        # 转换所有值为可显示格式
        formatted_values = []
        for val in result_df["值"]:
            if isinstance(val, (tuple, list, np.ndarray)):
                formatted_values.append(f"[{', '.join(f'{v:.4f}' for v in val)}]")
            elif isinstance(val, (int, float)):
                formatted_values.append(val)
            else:
                formatted_values.append(str(val))
        
        # 创建副本避免修改原数据
        display_df = result_df.copy()
        display_df["值"] = formatted_values
        
        numeric_mask = result_df["值"].apply(lambda x: isinstance(x, (int, float)))
        
        return display_df, numeric_mask

    display_df, numeric_mask = prepare_results(result_df)
    styled_df = display_df.style.format({"值": "{:.4f}"}, subset=pd.IndexSlice[numeric_mask, "值"])

    if numeric_mask.any():
        max_val = result_df.loc[numeric_mask, "值"].max()
        styled_df = styled_df.applymap(
            lambda x: 'background-color: lightblue' if x == max_val else '',
            subset=pd.IndexSlice[numeric_mask, "值"]
        )

    st.dataframe(styled_df)



    st.subheader("📝 结果解读")
    if p_value < 0.05:
        st.success(f"p值 = {p_value:.4f} < 0.05，结果具有统计显著性")
    else:
        st.warning(f"p值 = {p_value:.4f} ≥ 0.05，结果不具有统计显著性")

    st.subheader("💡 营销建议")
    if sub_test == "独立样本t检验" and p_value < 0.05:
        st.markdown(f"""
        - advice1
        - advice2
        """)
    elif sub_test == "卡方检验" and p_value < 0.05:
        st.markdown("""
        - advice1
        - advice2
        """)
    elif sub_test == "ANOVA方差分析" and p_value < 0.05:
        st.markdown("""
        - advice1
        - advice2
        """)
        
with st.expander("ℹ️ 使用指南", expanded=False):
    st.markdown("""111
    """)