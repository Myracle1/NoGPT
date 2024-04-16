import sys
import time

textPath=sys.argv[1] #获取文本路径
version=sys.argv[2] #获取版本
mask_model=sys.argv[3] #获取遮蔽模型类型
n_pertrubations=int(sys.argv[4]) #获取扰动次数

print(f"textPath: {textPath}, version: {version}, mask_model: {mask_model}, n_pertrubations: {n_pertrubations}")

if version=="cpu":
   from model import PPL_LL_based_gpt2_t5
elif version=="gpu":
    from model_web_gpu import PPL_LL_based_gpt2_t5

def read_texts_from_file(filename):
    with open(filename, 'r',encoding='utf-8') as file:
        content = file.read()
    texts = content.split('</text>')
    texts = [text.replace('<text>', '') for text in texts if text.strip() != '']
    texts = [text for text in texts if len(text.split()) >= 100]  # 过滤少于100个词的文本
    return texts

def write_values_to_file(filename, values):
    # 在最后添加内容，不抹去原始内容
    with open(filename, 'a') as file:  # 使用追加模式打开文件
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        #file.write(str(values) + "\n")  # 写入内容并换行
        for x in values:
            file.write(str(x) + "\n")

results = []
predictions = []

model=PPL_LL_based_gpt2_t5()
texts = read_texts_from_file(textPath)

for i in range(len(texts)):
    texts[i]=texts[i].replace("\n","")
    result,prediction1,prediction2 = model(texts[i], mask_model,n_pertrubations)
    print(f"text[{i}]: {prediction1}//text[{i}]: {prediction2}")
    results.append(result)

write_values_to_file('results.txt', results)