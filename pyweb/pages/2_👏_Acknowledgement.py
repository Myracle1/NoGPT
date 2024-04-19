import streamlit as st


# 致谢页面
def acknowledgment_page():
    st.title("💕 致谢")

    # 感谢老师的指导
    st.markdown("### 一、感谢老师的指导")
    st.text("""
    在项目的进行过程中，我们深深感激各位老师的悉心指导和宝贵意见。老师们的专业指导和无私帮助，
    为我们提供了丰富的知识和灵感，帮助我们克服了研究过程中的难题。他们的鼓励和支持是我们前进的动力，
    我们对此表示最诚挚的感谢和敬意。
    """)


    # 感谢大模型的支持
    st.markdown("### 二、感谢所有开源大模型的支持")
    st.markdown("在本项目中，我们特别感谢以下大模型的支持：")
    st.image('assets/所使用的大模型.png', use_column_width=True, caption='图1.产品所使用的大模型列表')
    st.markdown("尤为感谢Wenzhong2.0-GPT2-3.5B-chinese模型和T5模型的贡献。")



    # 感谢OCR引擎的使用
    st.markdown("### 三、感谢OCR引擎的使用")
    st.markdown("本项目中使用了先进的OCR引擎，其框架如下：")
    st.image('assets/OCR引擎框架.png', use_column_width=True, caption='图2.OCR引擎框架示意图')

    st.markdown("我们对所有支持和帮助过我们的人表示衷心的感谢。")

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

    acknowledgment_page()

    # 如果有其他页面，可以在这里继续添加条件分支

# 调用主函数
if __name__ == "__main__":
    main()