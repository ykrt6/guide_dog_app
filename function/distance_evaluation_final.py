import numpy as np
from scipy.optimize import curve_fit
from function.input_midas_final import absolute_distances


#双曲関数
def inverse_model(x,a,b):
    return a/x+b


def inverse(d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,depth_value):
    # 深度値データを準備
    relative_values = np.array([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])  # 深度値のリスト

    try:
        # 非線形モデルで最適なパラメータを推定
        params, covariance = curve_fit(inverse_model, relative_values, absolute_distances,maxfev=600,p0=[1,1])
        # 推定されたパラメータを取得
        slope, intercept = params
        predicted_distance = inverse_model(depth_value, slope, intercept) 

    except RuntimeError:
        # 最適なパラメータが見つからない場合はスキップする
        slope, intercept, predicted_distance = None,None,None

    return slope, intercept,predicted_distance, relative_values 
