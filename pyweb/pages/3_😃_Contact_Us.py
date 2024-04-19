import streamlit as st


# 联系我们页面
def contact_us_page():
    st.title("🎁 联系我们")
    st.image('assets/团队生活照.jpg', use_column_width=True, caption='图1.我们团队的日常生活照')
    st.markdown("想要联系我们或获取更多信息，请查看以下链接：")

    # 展示 Github 仓库链接
    st.markdown("### 👨‍💻 Github 仓库")
    st.markdown("- [👉NoGPT 仓库](https://github.com/Jam-Stark/NoGPT) - 查看我们的项目源代码和文档。")
    st.markdown("欢迎大家来仓库给我们点上一颗宝贵的小星星✨")
    st.image('assets/GitHub仓库.jpg', use_column_width=True,  caption='图2.项目开源仓库')

    # 展示 CSDN 博客链接
    st.markdown("### 🎉 CSDN 博客")
    st.markdown("- [👉CSDN 博客](https://blog.csdn.net/weixin_65688914) - 阅读我们的博客文章和教程。")


    # 提供其他联系方式（如果有的话）
    st.markdown("### 📩 其他联系方式")
    st.markdown("- Email: [example@email.com](songhc@mail.dlut.edu.cn)")

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

    contact_us_page()


# 调用主函数
if __name__ == "__main__":
    main()