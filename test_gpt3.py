from cn import PPL_LL_based_gpt3_t5

#textPath=sys.argv[2] #获取文本路径
#textPath="C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\EstimateText.txt" #测试用

"C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\ai_gen.txt"
"C:\\Users\\28551\\OneDrive\\桌面\\信安\\ssmDemo\\man_gen.txt"

def write_values_to_file(filename, values):
    with open(filename, 'w') as file:
        file.write(str(values))

def read_texts_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    texts = content.split('</text>')
    texts = [text.replace('<text>', '') for text in texts if text.strip() != '']
    return texts

print("请输入文本路径(绝对路径)：")
#启动命令行，获得文本路径
textPath=input("(词量不少于100词)")
#读取文本
#with open(textPath, 'r', encoding='utf-8') as f:
#    sentence = f.read()
values=[]
texts = read_texts_from_file(textPath)
model=PPL_LL_based_gpt3_t5()

key="c"

while key!="q":
    print("texts in file:",textPath)
    values.append(textPath.split("\\")[-1])
    for i in range(len(texts)):
        print("第",i+1,"次预测")
        #读取文本中<text>标签后的内容为一段文本
        #print("文本内容：",texts[i])
        values.append(model(texts[i]))
        #MultiPerturbedLL(model, texts[i])
        
    print("请输入q退出，其他键继续")
    key=input()
    if key=="q":
        break
    print("请输入文本路径(绝对路径)：")
    textPath=input("(词量不少于100词)")
    texts = read_texts_from_file(textPath)

write_values_to_file('3results.txt', values)
print("测试结束")