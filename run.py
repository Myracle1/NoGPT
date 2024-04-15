version=input("请输入版本号：CPU/GPU(cuda12.1)\n-->") or "gpu"
if version=="cpu":
   from model import PPL_LL_based_gpt2_t5
elif version=="gpu":
    from model_gpu import PPL_LL_based_gpt2_t5
import time
import os

#textPath=sys.argv[2] #获取文本路径
#textPath="C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\EstimateText.txt" #测试用

"C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\ai_gen.txt"
"C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\man_gen.txt"
"C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\formatted_data.txt"

def write_values_to_file(filename, values):
    # 在最后添加内容，不抹去原始内容
    with open(filename, 'a') as file:  # 使用追加模式打开文件
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        #file.write(str(values) + "\n")  # 写入内容并换行
        for x in values:
            file.write(str(x) + "\n")

def read_texts_from_file(filename):
    with open(filename, 'r',encoding='utf-8') as file:
        content = file.read()
    texts = content.split('</text>')
    texts = [text.replace('<text>', '') for text in texts if text.strip() != '']
    texts = [text for text in texts if len(text.split()) >= 100]  # 过滤少于100个词的文本
    return texts

model_type=""
n_perturbations=-1
text_type=-1

print("请输入文本路径(绝对路径)：")
#启动命令行，获得文本路径
textPath=input("(词量不少于100词)\n-->")
while text_type==-1:
    text_type=int(input("该测试文本为AI(1) or Human(0):\n输入0 or 1-->"))

print("请选择模型类型：t5-small/t5-large(512词以上使用t5-large准确度更高，但是耗时更长)；也可交给系统自动判断(press enter to skip) ")
model_type=input("-->") or "none"

n_pertrubations=int(input("请输入扰动次数：（if default press enter）\n-->"))
mark=input("本次预测备注：（无则enter）\n-->") or "none"

values=[]
texts = read_texts_from_file(textPath)
model=PPL_LL_based_gpt2_t5()
key="c"

while key!="q":
    print(f"texts in file: {textPath}, num: {len(texts)}")
    values.append(textPath.split("\\")[-1]+" Mark: "+mark)

    dCount=0
    sCount=0

    for i in range(len(texts)):
        print("第",i+1,"次预测")
        #读取文本中<text>标签后的内容为一段文本
        #print("文本内容：",texts[i])
        #消除文本中的换行符,不确定影响
        texts[i]=texts[i].replace("\n","")
        results=model(texts[i],model_type,n_perturbations)
        if(text_type-results["dPrediction"]==0):
            dCount+=1
        if(text_type==results["sPrediction"]):
            sCount+=1
        values.append(results)
        if(i%100==0):
            write_values_to_file('results.txt', values)
    dAccuracy=dCount/len(texts)
    sAccuracy=sCount/len(texts)
    print(f"dAccuracy: {dAccuracy} D_ll_shreshold: {D_ll_shreshold} sAccuracy: {sAccuracy} Score_threshold: {Score_threshold}")

    values.append("dAccuracy: "+str(dAccuracy))
    values.append("sAccuracy: "+str(sAccuracy))
        
    print("请输入q退出，其他键继续")
    key=input()
    if key=="q":
        break
    print("请输入文本路径(绝对路径)：")
    textPath=input("(词量不少于100词)")
    while text_type==-1:
        text_type=int(input("该测试文本为AI(1) or Human(0):\n输入0 or 1-->"))
    texts = read_texts_from_file(textPath)
    print("请选择模型类型：t5-small/t5-large(512词以上使用t5-large准确度更高，但是耗时更长)；也可交给系统自动判断(press enter to skip) ")
    model_type=input("-->") or "none"
    mark=input("输入备注：") or "none"

write_values_to_file('results.txt', values)
print("测试结束")
os.system("pause")
#results=model(sentence)

#print(results)