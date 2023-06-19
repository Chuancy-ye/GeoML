import denoise_ae_util
from bean import autoencoder, nn
import numpy as np
import pandas as pd
import openpyxl
import xlrd
import xlwt

x_matrix = pd.read_csv("C:/Users/7560/Desktop/山东化探数据库/DAN_X.csv", header=None)
x = np.array(x_matrix)

y_matrix = pd.read_csv("C:/Users/7560/Desktop/山东化探数据库/DAN_Y.csv", header=None)
y = np.array(y_matrix)
print(x)
print(y)

# step1 建立autoencoder
# 弄两层autoencoder，其中5为输入的维度
nodes = [39, 20, 10]
# 建立auto框架
ae = denoise_ae_util.aebuilder(nodes)
# 设置部分参数
# 训练，迭代次数为6000
ae = denoise_ae_util.aetrain(ae, x, 10000)

# step2 微调
# 建立完全体的autoencoder，最后层数1为输出的特征数
nodescomplete = np.array([39, 20, 10, 1])
aecomplete = nn(nodescomplete)
# 将之前训练得到的权值赋值给完全体的autoencoder
# 训练得到的权值，即原来的autoencoder网络中每一个autoencoder中的第一层到中间层的权值
for i in range(len(nodescomplete) - 2):
    aecomplete.W[i] = ae.encoders[i].W[0]
# 开始进行神经网络训练，主要就是进行微调
aecomplete = denoise_ae_util.nntrain(aecomplete, x, y, 5000)

# 打印出最后一层的输出
result = pd.DataFrame(aecomplete.values[3])
writer = pd.ExcelWriter('C:/Users/7560/Desktop/山东化探数据库/result.xls') # 写入result文件
result.to_excel(writer,'page_1',float_format='%.5f')
writer.save()
writer.close()


print(aecomplete.values[3])
