import base64
import streamlit as st
import time
import cv2
import pytesseract  # OCR:图片转文字
import numpy as np
from model import PPL_LL_based_gpt2_t5  # 确保这个路径正确，并包含了您的模型类
import speech_recognition as sr  # 语音转文字

# 设置页面的一些样式
def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


def sidebar_bg(side_bg):
    side_bg_ext = 'png'

    st.markdown(
        f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )

def set_page_style():
    st.markdown(""" 
    <style>
    body {
        background-color: #f0f0f0;
        font-family: Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# 进度条函数
def show_progress(text='正在检测中'):
    st.progress(0, text=text)
    time.sleep(1)  # 假设的处理时间

# 辅助函数，用于处理文本消息
def process_text_message(user_msg, model_type):
    model = PPL_LL_based_gpt2_t5()
    return model(user_msg, model_type, model_type)

# 辅助函数，用于处理文件
def process_file(file, model_type):
    if file.name.endswith('.txt'):
        texts = read_texts_from_file(file)
        return process_texts(texts, model_type)
    if file.name.endswith('.wav'):
        return process_audio(file, model_type)
    else:
        return process_image(file, model_type)

# 从文件中读取文本
def read_texts_from_file(file):
    texts = []
    for line in file:  # 逐行读取文件
        line = line.decode('utf-8')  # 将字节转换为字符串
        texts.append(line)
    return texts

# 处理文本列表
def process_texts(texts, model_type):
    model = PPL_LL_based_gpt2_t5()
    return [model(text, model_type, model_type) for text in texts]

# 处理图片
def process_image(file, model_type):
    img_data = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(img)
    return process_text_message(text, model_type)

def process_audio(file, model_type):
    # 创建一个Recognizer对象
    r = sr.Recognizer()
    # 使用Google Speech Recognition进行识别
    try:
        with sr.AudioFile(file) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            st.write("识别的文本是:", text)
            return process_text_message(text, model_type)
    except sr.UnknownValueError:
        st.error("Google Speech Recognition引擎无法理解音频")
    except sr.RequestError as e:
        st.error(f"Google Speech Recognition服务出现了错误; {e}")


# 主函数
def main():
    # 调用边框背景
    sidebar_bg('./assets/sidebar.png')
    # 调用背景
    main_bg('./assets/background.png')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col3:
        st.image('assets/logo.png', width=220)
    set_page_style()  # 设置页面样式
    st.title('NoGPT——国产AI生成文本检测平台')
    # 用户输入文本消息
    user_msg = st.text_area("👉输入您的消息：")

    # 文件上传
    uploaded_file = st.file_uploader("📂上传文件", type=["txt", "jpg", "png", "jpeg", "wav"])

    # 选择模型类型
    model_type = st.selectbox('🔑选择模型类型', ['t5-small', 't5-large', 'none'], index=0)

    # 开始检测按钮
    start_detect = st.button('⏳开始检测')
    # 根据用户输入和上传的文件进行处理
    if start_detect:
        if user_msg:
            show_progress()  # 显示进度条
            results = process_text_message(user_msg, model_type)
            st.write("文本检测结果为：", results)
        elif uploaded_file:
            show_progress()  # 显示进度条
            results = process_file(uploaded_file, model_type)
            if isinstance(results, list):
                st.write("文本检测结果为：", results)
            else:
                st.write("文本检测结果为：", results)


# 调用主函数
if __name__ == "__main__":
    main()