###################################
# written by: Subhadip Dey
# MRSLab, CSRE, Indian Institute of Technology, Bombay
# contact: sdey2307@gmail.com
# citing article: "S. Dey, U. Chaudhuri, D. Mandal, A. Bhattacharya, B. Banerjee and H. McNairn, "BiophyNet: A Regression Network for Joint Estimation of Plant Area Index 
# and Wet Biomass From SAR Data," in IEEE Geoscience and Remote Sensing Letters, doi: 10.1109/LGRS.2020.3008757"
###################################

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import tensorflow_probability as tfp
#%%
data_C11 = np.loadtxt(r"C:\Users\SadTime\Desktop\update1\soybean\calibrationC11.csv", delimiter=",")
data_C22 = np.loadtxt(r"C:\Users\SadTime\Desktop\update1\soybean\calibrationC22.csv", delimiter=",")
data_C33 = np.loadtxt(r"C:\Users\SadTime\Desktop\update1\soybean\calibrationC33.csv", delimiter=",")

data_C11 = data_C11.reshape([len(data_C11),1])
data_C22 = data_C22.reshape([len(data_C22),1])
data_C33 = data_C33.reshape([len(data_C33),1])

min_C11 = np.ndarray.min(data_C11)
max_C11 = np.ndarray.max(data_C11)
min_C22 = np.ndarray.min(data_C22)
max_C22 = np.ndarray.max(data_C22)
min_C33 = np.ndarray.min(data_C33)
max_C33 = np.ndarray.max(data_C33)

data_C11_norm = (data_C11 - min_C11)/(max_C11 - min_C11)
data_C22_norm = (data_C22 - min_C22)/(max_C22 - min_C22)
data_C33_norm = (data_C33 - min_C33)/(max_C33 - min_C33)

# data_C11_norm = data_C11
# data_C22_norm = data_C22
# data_C33_norm = data_C33

# data_C11_norm = data_C11
# data_C22_norm = data_C22
# data_C33_norm = data_C33

C_mat_train = np.hstack([data_C11_norm,data_C22_norm,data_C33_norm])

data_label = np.loadtxt(r"C:\Users\SadTime\Desktop\update1\soybean\calibrationcrop.csv", delimiter=",")

target_label = data_label[:,0:2]

data_label_input_test = np.loadtxt(r"C:\Users\SadTime\Desktop\update1\soybean\validation_cropsigma.csv", delimiter=",")
data_label_target_test = np.loadtxt(r"C:\Users\SadTime\Desktop\update1\soybean\validation_cropparam.csv", delimiter=",")

data_label_target_test = data_label_target_test[:,0:2]

test_C11 = data_label_input_test[:,0]
test_C22 = data_label_input_test[:,1]
test_C33 = data_label_input_test[:,2]

test_C11 = test_C11.reshape([len(test_C11),1])
test_C22 = test_C22.reshape([len(test_C22),1])
test_C33 = test_C33.reshape([len(test_C33),1])

test_C11 = np.float32(test_C11)
test_C22 = np.float32(test_C22)
test_C33 = np.float32(test_C33)

test_C11_norm = (test_C11 - min_C11)/(max_C11 - min_C11)
test_C22_norm = (test_C22 - min_C22)/(max_C22 - min_C22)
test_C33_norm = (test_C33 - min_C33)/(max_C33 - min_C33)

# test_C11_norm = test_C11
# test_C22_norm = test_C22
# test_C33_norm = test_C33

total_C11 = np.vstack([data_C11,test_C11])
total_C22 = np.vstack([data_C22,test_C22])
total_C33 = np.vstack([data_C33,test_C33])

X_input = np.hstack([total_C11,total_C22,total_C33])

total_target = np.vstack([target_label,data_label_target_test])

Y_input = total_target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_input, Y_input, train_size=0.7)

num_test_data = np.size(Ytest[:,1])
test_corr, _ = pearsonr(Ytest[:,0],Ytest[:,1])
print("Test correlation: ",test_corr)

Xtrain = np.float32(Xtrain)
Xtest = np.float32(Xtest)
Ytrain = np.float32(Ytrain)
Ytest = np.float32(Ytest)



print(np.std(Ytrain[:,0]))
print(np.mean(Ytrain[:,0]))

print(np.std(Ytest[:,0]))
print(np.mean(Ytest[:,0]))


data_C11_norm = Xtrain[:,0]
data_C22_norm = Xtrain[:,1]
data_C33_norm = Xtrain[:,2]

data_C11_norm = data_C11_norm.reshape([len(data_C11_norm),1])
data_C22_norm = data_C22_norm.reshape([len(data_C22_norm),1])
data_C33_norm = data_C33_norm.reshape([len(data_C33_norm),1])


test_C11_norm = Xtest[:,0]
test_C22_norm = Xtest[:,1]
test_C33_norm = Xtest[:,2]

test_C11_norm = test_C11_norm.reshape([len(test_C11_norm),1])
test_C22_norm = test_C22_norm.reshape([len(test_C22_norm),1])
test_C33_norm = test_C33_norm.reshape([len(test_C33_norm),1])

target_label = Ytrain
target_label_rf = Ytrain
data_label_target_test = Ytest
data_label_target_test1 = Ytrain

PAI_tar = Ytrain[:,0]
biom_tar = Ytrain[:,1]
target_label_1 = np.vstack([biom_tar,PAI_tar])
target_label_1 = np.transpose(target_label_1)

X_train_Rf = np.hstack([data_C11_norm,data_C22_norm,data_C33_norm])
X_test_Rf = np.hstack([test_C11_norm,test_C22_norm,test_C33_norm])

target_label_SAR = X_train_Rf
test_label_SAR = X_test_Rf
#%%
m = 3
n = 3
p = 10 #10
q = 6 #6

lr = 0.001
epoch = 10000

def init_weights(shape):
 	  return tf.Variable(tf.random_normal(shape, stddev=0.4))
    # return tf.Variable(tf.constant(0.01, shape=shape),)

def init_bias(shape):
 	return tf.Variable(tf.constant(0.2, shape=shape))
    # return tf.Variable(tf.random_normal(shape, stddev=0.2))

def activ_func(dat):
    return tf.nn.selu(dat)

# def activ_func(dat):
#     return tf.nn.leaky_relu(dat)

def activ_func1(dat):
    # return tf.math.exp(-((dat-mu)**2)/sd**2)
    # return tf.math.log(tf.math.abs(dat))*1.05
    return tf.nn.selu(dat)

def activ_func2(dat):
    # return tf.math.exp(-((dat-mu)**2)/sd**2)
    # return tf.math.log(tf.math.abs(dat))*1.05
    return tf.math.exp(dat)
#%%
class biom_regress:
    def __init__(self):
        with tf.variable_scope('bionet'):
            # non-linearity augmentation layers
            # NL layer 1
            self.gWF11 = init_weights([1, m])
            self.gbF11 = init_bias([m])
            self.gWF21 = init_weights([m,n])
            self.gbF21 = init_bias([n])
            
            # NL layer 2
            self.gWF12 = init_weights([1, m])
            self.gbF12 = init_bias([m])
            self.gWF22 = init_weights([m,n])
            self.gbF22 = init_bias([n])
            
            # NL layer 3
            self.gWF13 = init_weights([1, m])
            self.gbF13 = init_bias([m])
            self.gWF23 = init_weights([m,n])
            self.gbF23 = init_bias([n])
            
            # dimensionality embedding layer
            self.gWF3 = init_weights([3*n,p])
            self.gbF3 = init_bias([p])
            
            # Fully connected layer
            self.gWF4 = init_weights([p,q])
            self.gbF4 = init_bias([q])
           
            # Final output layer
            self.gWF5_1 = init_weights([q,1]) #biom output
            self.gbF5_1 = init_bias([1])
            
            self.gWF5_2 = init_weights([q,1]) #PAI output
            self.gbF5_2 = init_bias([1])
            
            self.gWF5_31 = init_weights([1,1]) #biom NL output
            self.gbF5_31 = init_bias([1])
            
            self.gWF5_32 = init_weights([1,1]) #PAI NL output
            self.gbF5_32 = init_bias([1])
            
            self.gWF5_3 = init_weights([1,1]) #PAI output
            self.gbF5_3 = init_bias([1])
            
            self.gWF5_4 = init_weights([1,1]) #biom output
            self.gbF5_4 = init_bias([1])
            
            
            # Decoder
            # Fully connected layer
            self.gWB4 = init_weights([2,q])
            self.gbB4 = init_bias([q]) 
            self.gWB3 = init_weights([q,p])
            self.gbB3 = init_bias([p])
            
            # Decoder NL 1
            self.gWB21 = init_weights([p,n])
            self.gbB21 = init_bias([n])
            self.gWB11 = init_weights([n,m])
            self.gbB11 = init_bias([m])
            
            # Decoder NL 2
            self.gWB22 = init_weights([p,n])
            self.gbB22 = init_bias([n])
            self.gWB12 = init_weights([n,m])
            self.gbB12 = init_bias([m])
            
            # Decoder NL 3
            self.gWB23 = init_weights([p,n])
            self.gbB23 = init_bias([n])
            self.gWB13 = init_weights([n,m])
            self.gbB13 = init_bias([m])
            
            # Decoder output
            self.gWB51 = init_weights([m,1]) # HH output
            self.gbB51 = init_bias([1])
            self.gWB52 = init_weights([m,1]) # VH output
            self.gbB52 = init_bias([1])
            self.gWB53 = init_weights([m,1]) # VV output
            self.gbB53 = init_bias([1])

        
    def forward(self, z1, z2, z3):
        
        # fc1N1 = activ_func1(tf.matmul(z1, self.gWF11) + self.gbF11)
        # fc2N1 = activ_func(tf.matmul(fc1N1, self.gWF21) + self.gbF21) # output of NL 1
        
        # fc1N2 = activ_func1(tf.matmul(z2, self.gWF12) + self.gbF12)
        # fc2N2 = activ_func(tf.matmul(fc1N2, self.gWF22) + self.gbF22)# output of NL 2
        
        # fc1N3 = activ_func1(tf.matmul(z3, self.gWF13) + self.gbF13)
        # fc2N3 = activ_func(tf.matmul(fc1N3, self.gWF23) + self.gbF23)# output of NL 3
        
        fc1N1 = activ_func1(tf.matmul(z1, self.gWF11))
        fc2N1 = activ_func(tf.matmul(fc1N1, self.gWF21)) # output of NL 1 activ_func not here
        
        fc1N2 = activ_func1(tf.matmul(z2, self.gWF12))
        fc2N2 = activ_func(tf.matmul(fc1N2, self.gWF22))# output of NL 2 activ_func not here
        
        fc1N3 = activ_func1(tf.matmul(z3, self.gWF13))
        fc2N3 = activ_func(tf.matmul(fc1N3, self.gWF23))# output of NL 3 activ_func not here
        
        NL_mat = tf.concat([fc2N1,fc2N2,fc2N3],1)
        # print(tf.shape(NL_mat))
        
        # fc1D1 = activ_func(tf.matmul(NL_mat, self.gWF3) + self.gbF3) # output of DL 1
        fc1D1 = activ_func(tf.matmul(NL_mat, self.gWF3) + self.gbF3) # output of DL 1 activ_func not here
        
        # fc1F1 = activ_func(tf.matmul(fc1D1, self.gWF4) + self.gbF4) # output of FC 1
        fc1F1 = activ_func(tf.matmul(fc1D1, self.gWF4) + self.gbF4) # output of FC 1 activ_func not here
        
        self.gWF5_1 = tf.math.abs(self.gWF5_1)
        self.gWF5_2 = tf.math.abs(self.gWF5_2)
        self.gbF5_1 = tf.math.abs(self.gbF5_1)
        self.gbF5_2 = tf.math.abs(self.gbF5_2)
        
        # fc1O1 = activ_func(tf.matmul(fc1F1, self.gWF5) + self.gbF5) # output biom PAI
        fc1O1_biom = (tf.matmul(fc1F1, self.gWF5_1) + self.gbF5_1) # output biom
        fc1O1_PAI = (tf.matmul(fc1F1, self.gWF5_2) + self.gbF5_2) # output PAI
        fc1O1 = tf.concat([fc1O1_PAI,fc1O1_biom],1)
        pred_cov = tfp.stats.correlation(fc1O1_PAI,fc1O1_biom)
        
        # fc1O1_2 = activ_func(tf.matmul(fc1O1_biom, self.gWF5_31)) # NL biom
        # fc1O1_3 = activ_func(tf.matmul(fc1O1_PAI, self.gWF5_32)) # NL PAI
        
        
        # fc1O2_PAI = (tf.matmul(fc1O1_2, self.gWF5_3) + self.gbF5_3) # output PAI
        # fc1O2_biom = (tf.matmul(fc1O1_3, self.gWF5_4) + self.gbF5_4) # output biom
        
        # fc1O1_1 = tf.concat([fc1O2_PAI,fc1O2_biom],1)
        
        # fb1F1 = activ_func(tf.matmul(fc1O1, self.gWB4) + self.gbB4)
        fb1F1 = activ_func(tf.matmul(fc1O1, self.gWB4) + self.gbB4) # activ_func not here
        
        # fb1D1 = activ_func(tf.matmul(fb1F1, self.gWB3) + self.gbB3) # decoder embedded layer
        fb1D1 = activ_func(tf.matmul(fb1F1, self.gWB3) + self.gbB3) # decoder embedded layer activ_func not here
        
        # fb2N1 = activ_func(tf.matmul(fb1D1, self.gWB21) + self.gbB21)
        # fb1N1 = activ_func(tf.matmul(fb2N1, self.gWB11) + self.gbB11)
        fb2N1 = activ_func(tf.matmul(fb1D1, self.gWB21) + self.gbB21) #activ_func not here
        fb1N1 = activ_func(tf.matmul(fb2N1, self.gWB11) + self.gbB11)
        
        # fb2N2 = activ_func(tf.matmul(fb1D1, self.gWB22) + self.gbB22)
        # fb1N2 = activ_func(tf.matmul(fb2N2, self.gWB12) + self.gbB12)
        fb2N2 = activ_func(tf.matmul(fb1D1, self.gWB22) + self.gbB22) #activ_func not here
        fb1N2 = activ_func(tf.matmul(fb2N2, self.gWB12) + self.gbB12)
        
        # fb2N3 = activ_func(tf.matmul(fb1D1, self.gWB23) + self.gbB23)
        # fb1N3 = activ_func(tf.matmul(fb2N3, self.gWB13) + self.gbB13)
        fb2N3 = activ_func(tf.matmul(fb1D1, self.gWB23) + self.gbB23) #activ_func not here
        fb1N3 = activ_func(tf.matmul(fb2N3, self.gWB13) + self.gbB13)
        
        # fb1O2 = activ_func(tf.matmul(fb1N1, self.gWB51)) # Decoded HH
        # fb2O2 = activ_func(tf.matmul(fb1N2, self.gWB52)) # Decoded VH
        # fb3O2 = activ_func(tf.matmul(fb1N3, self.gWB53)) # Decoded VV
        
        fb1O2 = (tf.matmul(fb1N1, self.gWB51)+self.gbB51) # Decoded HH
        fb2O2 = (tf.matmul(fb1N2, self.gWB52)+self.gbB52) # Decoded VH
        fb3O2 = (tf.matmul(fb1N3, self.gWB53)+self.gbB53) # Decoded VV
        
        SAR_out = tf.concat([fb1O2,fb2O2,fb3O2],1)
        
        return fc1O1, SAR_out, NL_mat, pred_cov
#%% 
def cost_Bio_Final(logits1, y1, y2, SAR_in, SAR_out):
 	 # return (tf.reduce_mean(tf.keras.losses.MSE(y,logits)) + tf.norm(tf.math.reduce_mean(weights),ord=1) + tf.norm(tf.math.reduce_mean(biases),ord = 1) + tf.norm(tf.math.subtract(S_loss1,S_loss2),ord=1))
    # msle = tf.keras.losses.MeanSquaredLogarithmicError()
    L1 =  tf.reduce_mean((tf.keras.losses.MSE(y1,logits1)))
#    L4 = tf.norm(tf.math.subtract(y1,logits1),ord=1.5)
    # L3 =  tf.reduce_mean((tf.keras.losses.MSE(y2,logits2)))
    L3 = tf.norm(tf.math.subtract(y2,test_corr),ord=1.5)
    L2 = tf.reduce_mean((tf.keras.losses.MSE(SAR_in,SAR_out)))
#    L5 = tf.norm(tf.math.subtract(SAR_in,SAR_out),ord=1.5)
    # print(L1,L2)
    return (L1+L3, L2)
#%%
biom_net = biom_regress()

phX1 = tf.placeholder(tf.float32, [None, 1])
phX2 = tf.placeholder(tf.float32, [None, 1])
phX3 = tf.placeholder(tf.float32, [None, 1])
# phZ = tf.placeholder(tf.float32, [None, 1])
target_label = tf.dtypes.cast(target_label, tf.float32)

train_vars = tf.trainable_variables()
# train_vars = tf.dtypes.cast(train_vars, tf.float32)

biovars = [var for var in train_vars if 'bionet' in var.name]

# biom_Out1, biom_Out2 = biom_net.forward(phX1, phX2, phX3)
biom_Out1, SAR_pred, _, pred_cov1 = biom_net.forward(phX1, phX2, phX3)

biom_mat = biom_Out1

bio_loss21, bio_loss22 = cost_Bio_Final(target_label, biom_mat, pred_cov1, target_label_SAR, SAR_pred)
#%%
bio_loss2 = bio_loss21+bio_loss22

bio_train = tf.train.AdamOptimizer(lr).minimize(bio_loss2, var_list=biovars)
#%%
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    bio_cost1 = []
    for k in range(epoch):
        _, b_loss1 = sess.run([bio_train, bio_loss2], feed_dict={phX1:data_C11_norm, phX2:data_C22_norm, phX3:data_C33_norm})
        bio_cost1.append(b_loss1)
        if(k%500==0):
            print("Epoch:", (k), "cost =", "{:.5f}".format(b_loss1))
    result2, SAR_pred1, _, cov1 = biom_net.forward(test_C11_norm,test_C22_norm,test_C33_norm)
    result1, SAR_pred_train, _, cov2 = biom_net.forward(data_C11_norm,data_C22_norm,data_C33_norm)
    res = result2
    res = sess.run(res)
    result = result1
    result = sess.run(result)
#%%
#plt.plot(bio_cost1)
#%%
print('Training data')
# from matplotlib import pyplot as plt
# plt.scatter(data_label_target_test1[:,0],result[:,0])

def rmse(predictions, targets):
   return np.sqrt(((predictions - targets) ** 2).mean())

# result = np.hstack([result1,result2])
print('BioPhy1 rmse: {:.2f}'.format(rmse(result[:,0],data_label_target_test1[:,0])))
print('BioPhy2 rmse: {:.2f}'.format(rmse(result[:,1],data_label_target_test1[:,1])))


# from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# print('BioPhy1 r2: {:.2f}'.format(r2_score(result[:,0],data_label_target_test[:,0])))
# print('BioPhy2 r2: {:.2f}'.format(r2_score(result[:,1],data_label_target_test[:,1])))

corr1, _ = pearsonr(result[:,0],data_label_target_test1[:,0])
corr2, _ = pearsonr(result[:,1],data_label_target_test1[:,1])

print('BioPhy1 corr: {:.2f}'.format(corr1))
print('BioPhy2 corr: {:.2f}'.format(corr2))

correlation_matrix = np.corrcoef(result[:,0],data_label_target_test1[:,0])
correlation_xy = correlation_matrix[0,1]
r_squared1 = correlation_xy**2

correlation_matrix = np.corrcoef(result[:,1],data_label_target_test1[:,1])
correlation_xy = correlation_matrix[0,1]
r_squared2 = correlation_xy**2

print(r_squared1)
print(r_squared2)

print('Test data')
# from matplotlib import pyplot as plt
# plt.scatter(data_label_target_test[:,0],res[:,0])

def rmse(predictions, targets):
   return np.sqrt(((predictions - targets) ** 2).mean())

# result = np.hstack([result1,result2])
print('BioPhy1 rmse: {:.2f}'.format(rmse(res[:,0],data_label_target_test[:,0])))
print('BioPhy2 rmse: {:.2f}'.format(rmse(res[:,1],data_label_target_test[:,1])))


# from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# print('BioPhy1 r2: {:.2f}'.format(r2_score(result[:,0],data_label_target_test[:,0])))
# print('BioPhy2 r2: {:.2f}'.format(r2_score(result[:,1],data_label_target_test[:,1])))

corr1, _ = pearsonr(res[:,0],data_label_target_test[:,0])
corr2, _ = pearsonr(res[:,1],data_label_target_test[:,1])

corr3, _ = pearsonr(res[:,0],res[:,1])

print('BioPhy1 corr: {:.2f}'.format(corr1))
print('BioPhy2 corr: {:.2f}'.format(corr2))

print('                     -------')
print("Test correlation: ",test_corr)
print('Predicted data corr: {:.2f}'.format(corr3))
print('                     -------')

correlation_matrix = np.corrcoef(res[:,0],data_label_target_test[:,0])
correlation_xy = correlation_matrix[0,1]
r_squared1 = correlation_xy**2

correlation_matrix = np.corrcoef(res[:,1],data_label_target_test[:,1])
correlation_xy = correlation_matrix[0,1]
r_squared2 = correlation_xy**2

print(r_squared1)
print(r_squared2)
#%%
from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

# regr_multirf = MultiOutputRegressor(SVR(kernel='rbf'))
# regr_multirf.fit(X_train_Rf, target_label_rf)
# Ypred2 = regr_multirf.predict(X_test_Rf)

regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=500))
clf2 = regr_multirf.fit(X_train_Rf, target_label_rf)
# clf2 = MultiOutputRF.fit(X_train_Rf, target_label_rf)
Ypred2 = clf2.predict(X_test_Rf)

print('Test data RF')
# from matplotlib import pyplot as plt
# plt.scatter(data_label_target_test[:,0],res[:,0])

def rmse(predictions, targets):
   return np.sqrt(((predictions - targets) ** 2).mean())

# result = np.hstack([result1,result2])
print('BioPhy1 rmse: {:.2f}'.format(rmse(Ypred2[:,0],data_label_target_test[:,0])))
print('BioPhy2 rmse: {:.2f}'.format(rmse(Ypred2[:,1],data_label_target_test[:,1])))


# from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# print('BioPhy1 r2: {:.2f}'.format(r2_score(result[:,0],data_label_target_test[:,0])))
# print('BioPhy2 r2: {:.2f}'.format(r2_score(result[:,1],data_label_target_test[:,1])))

corr1, _ = pearsonr(Ypred2[:,0],data_label_target_test[:,0])
corr2, _ = pearsonr(Ypred2[:,1],data_label_target_test[:,1])

corr3, _ = pearsonr(Ypred2[:,0],Ypred2[:,1])

print('BioPhy1 corr: {:.2f}'.format(corr1))
print('BioPhy2 corr: {:.2f}'.format(corr2))

print('Predicted RF data corr: {:.2f}'.format(corr3))

correlation_matrix = np.corrcoef(Ypred2[:,0],data_label_target_test[:,0])
correlation_xy = correlation_matrix[0,1]
r_squared1 = correlation_xy**2

correlation_matrix = np.corrcoef(Ypred2[:,1],data_label_target_test[:,1])
correlation_xy = correlation_matrix[0,1]
r_squared2 = correlation_xy**2

print(r_squared1)
print(r_squared2)
