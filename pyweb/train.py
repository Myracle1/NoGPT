import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib  # 用于保存和加载模型

# 读取human数据
file_path_human = "./human_data_cn.txt"
data_human = pd.read_csv(file_path_human, sep="\t", header=None, names=['D_LL', 'Score', 'Perplexity', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'label'])

# 读取AI数据
file_path_ai = "./AI_data_cn.txt"
data_ai = pd.read_csv(file_path_ai, sep="\t", header=None, names=['D_LL', 'Score', 'Perplexity', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'label'])

# 合并两个数据集
data = pd.concat([data_human, data_ai])

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.3, random_state=123)

# 特征和标签
features = train_data[['D_LL', 'Score', 'Perplexity']]
labels = train_data['label']

# 训练SVM模型
svm_model = SVC(kernel='rbf', probability=True, random_state=123)
svm_model.fit(features, labels)

# 保存模型
joblib.dump(svm_model, 'svm_model.pkl')

# 加载模型
svm_model_loaded = joblib.load('svm_model.pkl')

# 做预测
test_features = test_data[['D_LL', 'Score', 'Perplexity']]
test_labels = test_data['label']
predictions = svm_model_loaded.predict(test_features)

# 获取预测概率
probabilities = svm_model_loaded.predict_proba(test_features)

# 输出预测结果和概率
results = pd.DataFrame({
  'Actual': test_labels,
  'Predicted': predictions,
  'Human_Probability': probabilities[:, 0],
  'AI_Probability': probabilities[:, 1]
})

print(results)

# 混淆矩阵和准确率
conf_matrix = confusion_matrix(test_labels, predictions)
accuracy = accuracy_score(test_labels, predictions)
print(conf_matrix)
print("Accuracy:", accuracy)

# 保存模型到文件
joblib.dump(svm_model, './models/cn_svm_model.pkl')