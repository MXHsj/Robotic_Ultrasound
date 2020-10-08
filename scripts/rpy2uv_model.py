from sklearn.svm import SVR
from joblib import dump, load
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np


# --------------------read data----------------------
marker1u = genfromtxt('Groundtruth/SVM_1_u.csv', delimiter=',')
marker1v = genfromtxt('Groundtruth/SVM_1_v.csv', delimiter=',')
marker2u = genfromtxt('Groundtruth/SVM_2_u.csv', delimiter=',')
marker2v = genfromtxt('Groundtruth/SVM_2_v.csv', delimiter=',')
marker5u = genfromtxt('Groundtruth/SVM_5_u.csv', delimiter=',')
marker5v = genfromtxt('Groundtruth/SVM_5_v.csv', delimiter=',')
marker6u = genfromtxt('Groundtruth/SVM_6_u.csv', delimiter=',')
marker6v = genfromtxt('Groundtruth/SVM_6_v.csv', delimiter=',')

X_1v = marker1v[:, 0].reshape(-1, 1)
y_1v = marker1v[:, 1]
X_2v = marker2v[:, 0].reshape(-1, 1)
y_2v = marker2v[:, 1]
X_5v = marker5v[:, 0].reshape(-1, 1)
y_5v = marker5v[:, 1]
X_6v = marker6v[:, 0].reshape(-1, 1)
y_6v = marker6v[:, 1]

X_1u = marker1u[:, 0].reshape(-1, 1)
y_1u = marker1u[:, 1]
X_2u = marker2u[:, 0].reshape(-1, 1)
y_2u = marker2u[:, 1]
X_5u = marker5u[:, 0].reshape(-1, 1)
y_5u = marker5u[:, 1]
X_6u = marker6u[:, 0].reshape(-1, 1)
y_6u = marker6u[:, 1]


# ----------train and save model------------------
# svr = SVR(kernel='linear', C=1e2, epsilon=0.3)
svr_1v = SVR(kernel='rbf', C=1e2, epsilon=0.3)
svr_1v.fit(X_1v, y_1v)
svr_2v = SVR(kernel='rbf', C=1e2, epsilon=0.3)
svr_2v.fit(X_2v, y_2v)
svr_5v = SVR(kernel='rbf', C=1e2, epsilon=0.3)
svr_5v.fit(X_5v, y_5v)
svr_6v = SVR(kernel='rbf', C=1e2, epsilon=0.3)
svr_6v.fit(X_6v, y_6v)

svr_1u = SVR(kernel='rbf', C=1e2, epsilon=0.3)
svr_1u.fit(X_1u, y_1u)
svr_2u = SVR(kernel='rbf', C=1e2, epsilon=0.3)
svr_2u.fit(X_2u, y_2u)
svr_5u = SVR(kernel='rbf', C=1e2, epsilon=0.3)
svr_5u.fit(X_5u, y_5u)
svr_6u = SVR(kernel='rbf', C=1e2, epsilon=0.3)
svr_6u.fit(X_6u, y_6u)

dump(svr_1v, 'trained_models/marker1v.joblib')
dump(svr_2v, 'trained_models/marker2v.joblib')
dump(svr_5v, 'trained_models/marker5v.joblib')
dump(svr_6v, 'trained_models/marker6v.joblib')
dump(svr_1u, 'trained_models/marker1u.joblib')
dump(svr_2u, 'trained_models/marker2u.joblib')
dump(svr_5u, 'trained_models/marker5u.joblib')
dump(svr_6u, 'trained_models/marker6u.joblib')

svr_1u_load = load('trained_models/marker1u.joblib')
svr_1v_load = load('trained_models/marker1v.joblib')
svr_2u_load = load('trained_models/marker2u.joblib')
svr_2v_load = load('trained_models/marker2v.joblib')
svr_5u_load = load('trained_models/marker5u.joblib')
svr_5v_load = load('trained_models/marker5v.joblib')
svr_6u_load = load('trained_models/marker6u.joblib')
svr_6v_load = load('trained_models/marker6v.joblib')


# ----------------visualize result-------------------
X_new = np.linspace(40, 140, 140-40+1)

y_1u_pred = svr_1u_load.predict(X_new.reshape(-1, 1))
y_1v_pred = svr_1v_load.predict(X_new.reshape(-1, 1))
y_2u_pred = svr_2u_load.predict(X_new.reshape(-1, 1))
y_2v_pred = svr_2v_load.predict(X_new.reshape(-1, 1))
y_5u_pred = svr_5u_load.predict(X_new.reshape(-1, 1))
y_5v_pred = svr_5v_load.predict(X_new.reshape(-1, 1))
y_6u_pred = svr_6u_load.predict(X_new.reshape(-1, 1))
y_6v_pred = svr_6v_load.predict(X_new.reshape(-1, 1))

fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
fig.text(0.5, 0.04, 'rotation around X [deg]', ha='center', va='center')
fig.text(0.06, 0.7, 'u', ha='center', va='center')
fig.text(0.06, 0.3, 'v', ha='center', va='center')

axs[0, 0].plot(X_1u, y_1u, '*', label='training set')
axs[0, 0].plot(X_new, y_1u_pred, label='prediction')
axs[0, 0].legend(loc="upper left")
axs[0, 1].plot(X_2u, y_2u, '*', label='training set')
axs[0, 1].plot(X_new, y_2u_pred, label='prediction')
axs[0, 1].legend(loc="upper left")
axs[0, 2].plot(X_5u, y_5u, '*', label='training set')
axs[0, 2].plot(X_new, y_5u_pred, label='prediction')
axs[0, 2].legend(loc="upper left")
axs[0, 3].plot(X_6u, y_6u, '*', label='training set')
axs[0, 3].plot(X_new, y_6u_pred, label='prediction')
axs[0, 3].legend(loc="upper left")

axs[1, 0].plot(X_1v, y_1v, '*', label='training set')
axs[1, 0].plot(X_new, y_1v_pred, label='prediction')
axs[1, 0].legend(loc="upper left")
axs[1, 1].plot(X_2v, y_2v, '*', label='training set')
axs[1, 1].plot(X_new, y_2v_pred, label='prediction')
axs[1, 1].legend(loc="upper left")
axs[1, 2].plot(X_5v, y_5v, '*', label='training set')
axs[1, 2].plot(X_new, y_5v_pred, label='prediction')
axs[1, 2].legend(loc="upper left")
axs[1, 3].plot(X_6v, y_6v, '*', label='training set')
axs[1, 3].plot(X_new, y_6v_pred, label='prediction')
axs[1, 3].legend(loc="upper left")

plt.show()
