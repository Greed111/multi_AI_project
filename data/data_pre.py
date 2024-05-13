import numpy as np

def feature_nomarlize(X):
    mean = np.nanmean(X)
    std = np.nanstd(X)
    X=(X-mean)/std
    mask = np.isnan(X)
    X[mask] = 0
    return X
def mask(X):
    mask = np.isnan(X)
    X[mask] = 0
    return X

def data_divide(data,ratio_train=0.8,ratio_test=0.3, x_width=12, y_width=1, stride=1):
    X,Y=sliding_window1(data, x_width=x_width, y_width=y_width, stride=stride)
    # print(X.shape,'xxxx')
    # print(Y.shape,'yyyyyyyy')
    number1 = round(ratio_train*np.shape(X)[1])
    number2 = round( (ratio_train+ratio_test) * np.shape(X)[1])
    X_train = X[:, 0:number1, :]
    Y_train = Y[:, 0:number1, :]
    X_test = X[:, number1:number2, :]
    Y_test = Y[:, number1:number2, :]
    return X_train,Y_train,X_test,Y_test

def sliding_window1(Data , x_width=12 , y_width=1 , stride = 1):
    channel=len(Data)
    data1 = np.expand_dims(Data[0], axis=0)
    if channel !=1:
        for i in range(1,channel):
            data = np.expand_dims(Data[i], axis=0)
            data1 = np.concatenate((data1,data), axis=0)
    width=x_width+y_width
    Shape=np.shape(data1)
    T=Shape[1]
    groupA = data1[ :,0:width , :, :]
    group = np.expand_dims(groupA, axis=1)
    for i in range(1,T+1):
        j = i * stride
        if T-j < width:
            break
        tmp = data1[:,j:j + width, :, :]
        tmp = np.expand_dims(tmp, axis=1)
        group = np.concatenate((group, tmp), axis=1)  # 第0维连接
    X = group[: , : , 0:x_width , :, :]
    Y = group[: , : , x_width : x_width+y_width, :, :]
    return X,Y