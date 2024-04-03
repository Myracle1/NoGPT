from model import PPL_LL_based_gpt2_t5
import time
import os

#textPath=sys.argv[2] #获取文本路径
#textPath="C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\EstimateText.txt" #测试用

"C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\ai_gen.txt"
"C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\man_gen.txt"

def write_values_to_file(filename, values):
    # 在最后添加内容，不抹去原始内容
    with open(filename, 'a') as file:  # 使用追加模式打开文件
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        #file.write(str(values) + "\n")  # 写入内容并换行
        for x in values:
            file.write(str(x) + "\n")

def read_texts_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    texts = content.split('</text>')
    texts = [text.replace('<text>', '') for text in texts if text.strip() != '']
    return texts

model_type=""

print("请输入文本路径(绝对路径)：")
#启动命令行，获得文本路径
textPath=input("(词量不少于100词)")

print("请选择模型类型：t5-small/t5-large(512词以上使用t5-large准确度更高，但是耗时更长)；也可交给系统自动判断(press enter to skip) ")
model_type=input("-->") or "none"
#读取文本
#with open(textPath, 'r', encoding='utf-8') as f:
#    sentence = f.read()
values=[]
texts = read_texts_from_file(textPath)
model=PPL_LL_based_gpt2_t5()

key="c"

while key!="q":
    print("texts in file:",textPath)
    values.append(textPath.split("\\")[-1])
    for i in range(len(texts)):
        print("第",i+1,"次预测")
        #读取文本中<text>标签后的内容为一段文本
        #print("文本内容：",texts[i])
        values.append(model(texts[i],model_type))
        #MultiPerturbedLL(model, texts[i])
        
    print("请输入q退出，其他键继续")
    key=input()
    if key=="q":
        break
    print("请输入文本路径(绝对路径)：")
    textPath=input("(词量不少于100词)")
    texts = read_texts_from_file(textPath)
    model_type=input("t5-small(default)/t5-large(default when len(text)>512)(512词以上使用t5-large准确度更高，但是耗时更长)   ")

write_values_to_file('results.txt', values)
print("测试结束")
os.system("pause")
#results=model(sentence)

#print(results)