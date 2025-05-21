import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap  # 导入SHAP库
import matplotlib.pyplot as plt

# 加载预训练的XGBoost模型
model = joblib.load('vote排名前6.pkl')

# 更新后的特征范围定义
feature_ranges = {
    "Sex": {"type": "categorical", "options": [0, 1]},
    'Long-standing illness or disability': {"type": "categorical", "options": [0, 1]},
    "Age": {"type": "numerical"},
    'Number of non-cancer illnesses': {"type": "numerical"},
    'Number of medications taken': {"type": "numerical"},
    "Systolic Blood Pressure": {"type": "numerical"},
    'Cholesterol ratio': {"type": "numerical"},
    "Plasma GDF15": {"type": "numerical"},
    "Plasma MMP12": {"type": "numerical"},
    "Plasma NTproBNP": {"type": "numerical"},
    "Plasma AGER": {"type": "numerical"},
    "Plasma PRSS8": {"type": "numerical"},
    "Plasma PSPN": {"type": "numerical"},
    "Plasma WFDC2": {"type": "numerical"},
    "Plasma LPA": {"type": "numerical"},
    "Plasma CXCL17": {"type": "numerical"},
    "Plasma GAST": {"type": "numerical"},
    "Plasma RGMA": {"type": "numerical"},
    "Plasma EPHA4": {"type": "numerical"},
}

# Streamlit界面标题
st.title("10-Year MACE Risk Prediction")

# 创建两个列，显示输入项
col1, col2 = st.columns(2)

feature_values = []

# 通过 feature_ranges 保持顺序
for i, (feature, properties) in enumerate(feature_ranges.items()):
    if properties["type"] == "numerical":
        # 数值型输入框
        if i % 2 == 0:
            with col1:
                value = st.number_input(
                    label=f"{feature}",
                    value=0.0,  # 默认值为0
                    key=f"{feature}_input"
                )
        else:
            with col2:
                value = st.number_input(
                    label=f"{feature}",
                    value=0.0,  # 默认值
                    key=f"{feature}_input"
                )
    elif properties["type"] == "categorical":
        if feature == "Sex":
            with col1:  # 将"Sex"放在第一个列中
                value = st.radio(
                    label="Sex",
                    options=[0, 1],  # 0 = Female, 1 = Male
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    key=f"{feature}_input"
                )
        elif feature == 'Long-standing illness or disability':
            with col2:  # 将"Long-standing illness or disability"放在第二个列中
                value = st.radio(
                    label="Long-standing illness or disability",
                    options=[0, 1],  # 0 = No, 1 = Yes
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key=f"{feature}_input"
                )
    feature_values.append(value)

# 将特征值转换为模型输入格式
features = np.array([feature_values])

# 预测与SHAP可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

# 获取 MACE 类的概率，确保获取的是 "MACE" 类别的概率
    mace_probability = predicted_proba[1] * 100  # 第二列是 MACE 类别的概率
    # 显示预测结果，使用Matplotlib渲染指定字体
    text = f"Predicted probability of MACE in the next 10 years: {mace_probability:.2f}%."
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.text(
        0.5, 0.1, text,
        fontsize=18,
        ha='center', va='center',
        fontname='Times New Roman',  # 使用Times New Roman字体
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)  # Adjust margins tightly
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=1200)
    st.image("prediction_text.png")

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 打印SHAP值的形状，确保它是二维的
    print(f"SHAP values shape: {np.shape(shap_values)}")
    
    # 获取Class 1（正类）的SHAP值，二分类问题时，shap_values会包含两个元素，分别对应两个类别
    shap_values_class_1 = shap_values  # 获取正类（Class 1）的SHAP值
    expected_value_class_1 = explainer.expected_value  # 获取Class 1的期望值

    # 计算每个特征的绝对SHAP值，按降序排序，选择前6个特征
    shap_values_abs = np.abs(shap_values_class_1).mean(axis=0)
    top_6_indices = np.argsort(shap_values_abs)[-6:]  # 获取前6个特征的索引

    # 只保留前6个特征的SHAP值和特征名称
    shap_values_class_1_top_6 = shap_values_class_1[:, top_6_indices]
    top_6_features = pd.DataFrame([feature_values], columns=feature_ranges.keys()).iloc[:, top_6_indices]

    # 生成仅显示前6个特征的SHAP力图
    shap_fig = shap.force_plot(
        expected_value_class_1,  # 使用Class 1的期望值
        shap_values_class_1_top_6,  # 前6个特征的SHAP值
        top_6_features,  # 仅显示前6个特征的数据
        matplotlib=True,
    )

    # 保存并显示SHAP力图
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)  # Reduce bottom margin, adjust top
    plt.savefig("shap_force_plot_class_1_top_6.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot_class_1_top_6.png")
