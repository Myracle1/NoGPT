import streamlit as st

# 团队介绍页面
def team_introduction_page():
    st.title("😀 团队介绍")
    st.markdown("团队成员聚焦于人工智能与软件工程，在机器学习、深度学习、软件工程领域进行深耕。")
    st.markdown("在大模型时代，我们致力于做出一款关于AI生成文本的检测平台。")
    st.markdown("通过AI赋能，在保障用户安全的前提下，打造强有力的AI工具，实现AI安全。")

    # 团队成员介绍
    st.markdown("### 👨‍👨‍👦‍👦 团队成员")
    st.markdown("下面是我们NoGPT团队的成员")
    st.image('assets/团队介绍.jpg', use_column_width=True, caption='图1.开发团队成员')


    # 作品介绍
    st.markdown("### 📖 我们的作品")
    st.markdown("本团队设计的GPT检测平台是一个创新的在线平台，专门用于识别和分析文本内容是否由GPT模型生成。")
    st.markdown("利用先进的机器学习和自然语言处理技术，通过分析文本的语言风格、结构和内容特征，为用户提供准确的生成概率评估。")
    st.markdown("##### 平台的算法框架如下👇")
    st.image('assets/NoGPT算法框架.png', use_column_width=True, caption='图2.NoGPT算法框架介绍')
    st.markdown("##### 平台的检测效果如下👇")
    st.image('assets/算法指标效果.png', use_column_width=True, caption='图3.算法指标效果图')
    st.markdown("##### 平台的应用领域如下👇")
    st.image('assets/NoGPT应用范围.png', use_column_width=True, caption='图4.产品应用领域')


# 主函数
def main():
    # 设置页面的一些样式
    st.markdown(""" 
    <style>
    body {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    team_introduction_page()

    # 如果有其他页面，可以在这里继续添加条件分支

# 调用主函数
if __name__ == "__main__":
    main()