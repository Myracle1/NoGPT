"""V3.0：文字+OCR+语音版本+SVM给出分类结果"""
import base64
import streamlit as st
import time
import cv2
import pytesseract  # OCR:图片转文字
import numpy as np
from model import PPL_LL_based_gpt2_t5  # 确保这个路径正确，并包含了您的模型类
import speech_recognition as sr  # 语音转文字
import joblib

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
def predict_svm(data):
    # 加载Python训练的SVM模型
    svm_model = joblib.load('D:/PyCode/NoGPT/models/new_svm_model.pkl')
    # 调整数据的形状以匹配模型的输入
    data = data.reshape(1, -1)
    # 使用模型进行预测
    predictions = svm_model.predict(data)
    probabilities = svm_model.predict_proba(data)
    return predictions, probabilities

# 从结果中提取出关键指标的数值
def extract_crucial_value(results):
    ordered_dict = results[0][0]
    D_LL = ordered_dict['D_LL']
    score = ordered_dict['Score']
    PPL = ordered_dict['Perplexity']
    data_array = np.array([D_LL, score, PPL])
    return data_array

def mul_extract_crucial_value(results):
    results = results[0]
    D_LL = results['D_LL']
    score = results['Score']
    PPL = results['Perplexity']
    data_array = np.array([D_LL, score, PPL])
    return data_array

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
    return model(user_msg, model_type)

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
            data_array = mul_extract_crucial_value(results)
            predictions, probabilities = predict_svm(data_array)
            if predictions == 1:
                st.write("该文本是由AI生成的")
            else:
                st.write("该文本是由人类生成的")
            st.write("0代表人类生成概率，1代表AI生成概率:", probabilities)
        elif uploaded_file:
            show_progress()  # 显示进度条
            results = process_file(uploaded_file, model_type)
            if isinstance(results, list):
                st.write("文本检测结果为：", results)
                data_array = extract_crucial_value(results)
                predictions, probabilities = predict_svm(data_array)
                if predictions==1:
                   st.write("该文本是由AI生成的")
                else:
                   st.write("该文本是由人类生成的")
                st.write("0代表人类生成概率，1代表AI生成概率:", probabilities)
            else:
                st.write("文本检测结果为：", results)
                print(results)
                data_array = mul_extract_crucial_value(results)
                predictions, probabilities = predict_svm(data_array)
                if predictions==1:
                   st.write("该文本是由AI生成的")
                else:
                   st.write("该文本是由人类生成的")
                st.write("0代表人类生成概率，1代表AI生成概率:", probabilities)

# 调用主函数
if __name__ == "__main__":
    main()
    # 使用提供的示例数据调用函数

