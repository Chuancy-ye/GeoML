import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import os
from keras.utils.vis_utils import plot_model
import sys
import xlrd
import xlwt
import imageio,os

file_name = os.path.join(os.getcwd(),'shandong.xls')
work_book = xlrd.open_workbook(file_name)
work_sheet = work_book.sheet_by_index(0)

#total_row = work_sheet.nrow

data_list =[]
key_list =[]
for i in range(1,1581):
	row_data = work_sheet.row_values(i)
	data_list += [row_data[1:-1]]
	if (row_data[-1]) == 1:
		key_list.append(1)
	else:
		key_list.append(0)
X =np.array(data_list)
Y =np.array(key_list)
print(X[0])

#记录过程
saveout = sys.stdout
file = open('variational_autoencoder.txt','w')
sys.stdout = file
#设置模型参数
batch_size = 20
original_dim = 39   #
latent_dim = 2
intermediate_dim = 6
nb_epoch = 40
epsilon_std = 1.0

#编码过程
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
#均值和方差特征
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.0)
	return z_mean + K.exp(z_log_var / 2) * epsilon
# note that "output_shape" isn't necessary with the TensorFlow backend
# my tips:get sample z(encoded)加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#decoder解码部分
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

#建立模型
vae = Model(x, x_decoded_mean)
#loss(restruct X)+KL
def vae_loss(x, x_decoded_mean):
	#my tips:logloss
	xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
	#print(xent_loss)
	#my tips:see paper's appendix B
	kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	return xent_loss + kl_loss
vae.compile(optimizer='rmsprop', loss=vae_loss)
#输入
(x_train, y_train) = (X,Y)
(x_test, y_test) = (X,Y)
history = vae.fit(x_train, y_train,
		shuffle=True,
		nb_epoch=nb_epoch,
		verbose=2,
		batch_size=batch_size,
		validation_data=(x_test, y_test)
		)
#输出loss可视化
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 潜在空间输入模型
encoder = Model(x, z_mean)
# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#np.savetxt("Reconstructedmean.csv",x_test_encoded, delimiter=',')
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
#第一种重建误差算法
#ReconstructedData2= vae.predict(x_train,batch_size=batch_size)
#ReconstructedData3=ReconstructedData2.reshape((ReconstructedData2.shape[0], -1))
#np.savetxt("ReconstructedData.csv", ReconstructedData3, delimiter=',')
#print(vae.predict(x_test, batch_size=batch_size))

#Reconstructed_test  = vae.predict(x_test)
#ReconstructedData1=np.vstack((Reconstructed_train,Reconstructed_test))
#ReconstructedData2=ReconstructedData1.reshape((ReconstructedData1.shape[0], -1))
#np.savetxt("ReconstructedData.csv", ReconstructedData2, delimiter=',')
#a = Input(batch_shape=(1, original_dim))
#xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
#re_loss = original_dim * objectives.binary_crossentropy(a, x_decoded_mean)

# 构建一个数字生成器，它可以从学习到的分布中进行采样 
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)
#另一种想法
grid_x = norm.ppf(np.linspace(0.05, 0.95, 15))
grid_y = norm.ppf(np.linspace(0.05, 0.95, 15))
list1=[]
for i, yi in enumerate(grid_x):
	for j, xi in enumerate(grid_y):
		z_sample = np.array([[xi, yi]])
		x_decoded = generator.predict(z_sample)
		list1 += [x_decoded[0]]

print(list1)
book_name_xlsx = '生成数据39.xlsx'
sheet_name_xlsx = 'redate'
import openpyxl
def write_excel_xlsx(path, sheet_name, value):
	index = len(value)
	workbook = openpyxl.Workbook()
	sheet = workbook.active
	sheet.title = sheet_name
	for i in range(0, index):
		for j in range(0, len(value[i])):
			sheet.cell(row=i+1, column=j+1, value=str(value[i][j]))
	workbook.save(path)
	print("xlsx格式表格写入数据成功！")
write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, list1)
#np.savetxt("Reconstructedgendata.csv",x_test_generator, delimiter=',')
#x_test_generator = generator.predict(x_test_encode                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  d)
#np.savetxt("Reconstructedgendata.csv",x_test_generator, delimiter=',')

#Reconstructed_test = vae.predict(x_test,batch_size=batch_size)
#np.savetxt("ReconstructedData.csv", Reconstructed_test, delimiter=',')

plot_model(vae,to_file='variational_autoencoder_vae.png',show_shapes=True)
plot_model(encoder,to_file='variational_autoencoder_encoder.png',show_shapes=True)
plot_model(generator,to_file='variational_autoencoder_generator.png',show_shapes=True)


sys.stdout.close()
sys.stdout = saveout

