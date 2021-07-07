# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:11:01 2020

@author: TT
"""
import pandas as pd
import pylab as plt
from datetime import datetime

from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
import numpy as np
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb

# 标准化处理参数
from ann import NerualNetwork
from ann1 import NeuralNetwork
from data_processing import shuffle_split_data
from logistic2 import LogisticRegression
from logisticregression import logisticreg
from randomforest import RandomForestClassifier
from smote import SMOTE
from svm import svm_train, svm_eval

need_standard_processing = 0
# 标准化和归一化2选1
# https://blog.csdn.net/FrankieHello/article/details/79659111
use_standard = 1
use_minmax = 0

# 类别不平衡处理参数
need_balance = 0
use_Balance_SMOTE = 1
use_Balance_Bagging = 0

# 模型选择参数
use_ensemble_learning = 0
use_model_ann = 0
use_model_svm = 0
use_model_randomforest = 0
use_model_logisticregression = 0
use_model_xgboost = 1

# 数据清洗后的结果导入
data_cleaned = pd.read_csv('/Users/oliviachen/Desktop/FinTech1/训练数据集/训练数据集output.csv')

data_x=data_cleaned.drop(['flag'], axis=1)
data_y=data_cleaned['flag']
# 标准化处理
if need_standard_processing == 1:
    print('开始进行标准化处理...')
    if use_standard == 1:
        print('开始进行标准化处理...')
        ss = StandardScaler()
        data_standard = ss.fit_transform(data_x)
    elif use_minmax == 1:
        # 归一化
        print('开始进行归一化处理...')
        mm = MinMaxScaler()
        data_standard = mm.fit_transform(data_x)
        # 归一化还原
        # origin_data = mm.inverse_transform(mm_data)
else:
    print('没有进行标准化处理！！！')
    data_standard=data_x
data = data_standard

# 特征工程
# 进行属性约简

# 类别不平衡处理
if need_balance == 1:  # 判断是否需要进行类别不平衡处理
    print('start Balance processing...')
    if use_Balance_SMOTE == 1:
        # Parameters
        # ----------
        # T: array-like, shape = [n_samps, n_attrs]
        #     Minority class Samples
        # N:
        #     Amount of SMOTE N%. Percentage of new syntethic samples
        # k: int, optional (default = 5)
        #     Number of neighbors to use by default for k_neighbors queries.
        # Returns
        # -------
        # syntethic: array, shape = [(N/100) * T]
        #     Syntethic minority class samples
        N = 10
        k = 2
        smote = SMOTE(data, N, k=k)
        synth = smote.over_sampling()
        print('# Synth Samps: ', synth.shape[0])


    #elif use_Balance_Bagging == 1:
    # 待实现
    else:
        print('没有进行类别不平衡处理！！！')

# 数据区分为x和y
#

x_train, y_train, x_test, y_test = shuffle_split_data(data, data_y, 70)
x_train=x_train.drop(['userid'], axis=1)
userid=x_test['userid']
x_test=x_test.drop(['userid'], axis=1)

# x_train=data
# y_train=data
# x_test=data
# y_test=data
# x_train=x_train.values[:, :]
# y_train=y_train.values[:, :]
# x_test=x_test.values[:, :]
# y_test=y_test.values[:, :]

# 模型选择


if use_model_ann == 1:
    print('开始进行ANN建模...')
    #Initialize Neural Network with the following properties
    # input_layer_size要和x_train的属性数量一致
    # output_layer_size要和y_train的属性数量一致
    # nn = NerualNetwork(input_layer_size=183, hidden_layer_size=6, output_layer_size=1, learning_rate=0.1)
    #
    # #Train NN with data from dataset
    # nn.train(x_train.values, y_train.values, 1)
    # ann_yhat=nn.predict(x_test.values)

    nn = NeuralNetwork([183, 6, 1], 'tanh')
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 1, 1, 0])
    nn.fit(x_train.values, y_train.values)
    # for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    #     print(i, nn.predict(i))
    ann_yhat = nn.predictall(x_test.values)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, ann_yhat)
    ann_auc = auc(false_positive_rate, true_positive_rate)
    print(auc(false_positive_rate, true_positive_rate))
if use_model_svm == 1:
    # https://www.cnblogs.com/luyaoblog/p/6775342.html
    print('开始训练SVM...')
    # kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
    # kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
    # decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
    # decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
    # svm_model = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovr')
    # svm_model.fit(x_train, y_train.ravel())
    # print('训练SVM完成')
    # print
    # svm_model.score(x_train, y_train)  # 精度
    # y_hat = svm_model.predict(x_train)
    # show_accuracy(y_hat, y_train, '训练集')
    # print
    # svm_model.score(x_test, y_test)
    # y_hat = svm_model.predict(x_test)
    # show_accuracy(y_hat, y_test, '测试集')
    model = svm_train(x_train, y_train)
    # svm_eval(x_train, y_train, x_test, y_test)
    svm_yhat=model.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, svm_yhat)
    svm_auc = auc(false_positive_rate, true_positive_rate)
    print(auc(false_positive_rate, true_positive_rate))

if use_model_xgboost == 1:
    print('开始进行xgboost建模...')
    from sklearn.model_selection import GridSearchCV
    xgboster_focal = imb_xgb(special_objective='focal')
    xgboster_weight = imb_xgb(special_objective='weighted')
    CV_focal_booster = GridSearchCV(xgboster_focal, {"focal_gamma": [1.0, 1.5, 2.0, 2.5, 3.0]})
    CV_weight_booster = GridSearchCV(xgboster_weight, {"imbalance_alpha": [1.5, 2.0, 2.5, 3.0, 4.0]})
    print('结束xgboost建模...')
    CV_focal_booster.fit(x_train.values, y_train.values)
    print('aaaa')
    CV_weight_booster.fit(x_train.values, y_train.values)
    opt_focal_booster = CV_focal_booster.best_estimator_
    opt_weight_booster = CV_weight_booster.best_estimator_
    
    raw_output = opt_focal_booster.predict(x_test.values, y=None)
    xgboost_yhat = opt_focal_booster.predict_determine(x_test.values, y=None)
    xgboost_prob = opt_focal_booster.predict_two_class(x_test.values, y=None)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, xgboost_yhat)
    xgboost_auc = auc(false_positive_rate, true_positive_rate)
    print(auc(false_positive_rate, true_positive_rate))
    #print('结束xgboost建模...')

if use_model_randomforest == 1:
    print('开始进行随机森林建模...')
    rf = RandomForestClassifier(n_estimators=5,
                                 max_depth=5,
                                 min_samples_split=6,
                                 min_samples_leaf=2,
                                 min_split_gain=0.0,
                                 colsample_bytree="sqrt",
                                 subsample=0.8,
                                 random_state=66)
    rf.fit(x_train, y_train)
    rf_yhat=rf.predict(x_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, rf_yhat)
    rf_auc = auc(false_positive_rate, true_positive_rate)
    print(auc(false_positive_rate, true_positive_rate))
    # from sklearn import metrics
    #
    # print(metrics.accuracy_score(df.loc[:train_count, 'label'], clf.predict(df.loc[:train_count, feature_list])))
    # print(metrics.accuracy_score(df.loc[train_count:, 'label'], clf.predict(df.loc[train_count:, feature_list])))
if use_model_logisticregression == 1:
    print('开始进行逻辑回归建模...')
    #dataMat, labelMat = loadDataSet("data/ex4.dat")
    #第三个参数是迭代次数
    # theta = logisticreg(x_train, y_train, 100)
    # plotBestFit(dataMat, labelMat, array(theta))

    # Training logistic regression classifier with L2 penalty
    LR = LogisticRegression(learningRate=0.01, numIterations=20, penalty='L2', C=0.01)
    LR.train(x_train, y_train, tol=10 ** -3)

    # Testing fitted model on test data with cutoff probability 50%
    predictions, probs = LR.predict(x_test, 0.5)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
    lr_auc = auc(false_positive_rate, true_positive_rate)
    print(auc(false_positive_rate, true_positive_rate))
    # performance = LR.performanceEval(predictions, y_test)
    # LR.plotDecisionRegions(x_test, y_test)
    # LR.predictionPlot(x_test, y_test)
    #
    # # Print out performance values
    # for key, value in performance.items():
    #     print('%s : %.2f' % (key, value))

# if use_ensemble_learning == 1:
# 待实现


# 模型性能评价
# 计算每个单模型或集成模型的AUC
# 已经放在建模之后立即进行评价

# 选择AUC最高的模型结果作为最终结果
# 待实现
# 

# 测试数据集结果输出
# utf-8,无bom
# UD35456 0.000555624326035
# UD34FCE 0.140826626669
# UD34F30 0.996893765903
final_result = np.vstack((userid, xgboost_prob[:,1]))
final_result = np.transpose(final_result)
final_result = pd.DataFrame(final_result)
final_result.to_csv('result.txt', sep=' ', encoding="utf-8", index=False, header=False)




# 评分数据集导入
x_predict = pd.read_csv('/Users/oliviachen/Desktop/FinTech1/评分数据集/评分数据集output.csv') #此处修改为评分数据集处理后的csv位置
predict_userid=x_predict['userid']#提取userid
x_predict=x_predict.drop(['userid'], axis=1)#去掉userid输入到模型中的评分数据

xgboost_predict_yhat = opt_focal_booster.predict_determine(x_predict.values, y=None)
xgboost_predict_prob = opt_focal_booster.predict_two_class(x_predict.values, y=None)


# 将用户标识和评价结果拼接为一个评分数据集
final_result = np.vstack((predict_userid, xgboost_predict_prob[:,1]))
final_result = np.transpose(final_result)
final_result = pd.DataFrame(final_result)
final_result.to_csv('/Users/oliviachen/Desktop/FinTech-20200512/result.txt', sep=' ', encoding="utf-8", index=False, header=False)


