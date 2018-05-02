import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC

#import word vector as features for training
x = np.load('data/x_train_vec.npy')

#import emotion catagories
y = np.load('data/y_train.npy')

#build SVM
model = SVC(kernel='rbf',verbose=True)

#model training
model.fit(x,y)

#save as binary document
joblib.dump(model,'data/svm_model.pkl')

