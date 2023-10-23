import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
# from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from flask import Flask, request
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder, scale
import joblib
import flask
from flask_cors import CORS
import json

from sklearn.svm import SVR

app = Flask(__name__)
cors = CORS(app, resources=r'/*')


def deal_data_reg(excel_path):
    data = pd.read_excel(excel_path, header=0)
    y = np.array(data.iloc[:, 1]).astype(np.float64)
    x = np.array(data.iloc[:, 2:]).astype(np.float64)
    print('deal_data x:', x)
    return x,y
    # _, indices = np.unique(y, return_inverse=True)
    # groups = indices
    # n_groups = len(np.unique(y))
    # return x, y,groups,n_groups

def deal_data_clf(excel_path):
    # def deal_data(excel_path):
    data = pd.read_excel(excel_path, header=None)
    data = np.array(data)  # 直接转化为numpy类型
    y = data[:, 0]  # 第一列为因变量
    x = data[:, 1:]  # 后面的列均为自变量
    return x, y


# 读取csv文件
# def deal_data(csv_path):
#     data = pd.read_csv(csv_path,header=0)
#     data = np.array(data)  # 直接转化为numpy类型
#     y = data[:, 1]  # 第一列为因变量
#     y = y.astype('int')#标签定义为整型
#     x = data[:, 2:]  # 后面的列均为自变量
#     return x, y

# 数据预处理
def MC(x):
    x_mean = np.mean(x, axis=0)
    x_centered = x - x_mean
    # print('x_centered',x_centered)
    return x_centered


def SUM(x):
    # print("预处理方法SUM被选中")
    # x, y = deal_data(excelPath)
    x_sum = np.sum(x, axis=0)
    x_normalized = x / x_sum
    return x_normalized


def SNV(x):
    # print("预处理方法SNV被选中")
    # x, y = deal_data(excelPath)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_snv = (x - x_mean) / x_std
    return x_snv


# 回归任务

def PLSR(preprocessingType, x, y, modelName):
    # 10折分层交叉验证
    preprocessingType(x)
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds)
    r2_scores = {i: [] for i in range(1, 17)}
    rmse_scores = {i: [] for i in range(1, 17)}
    # 每一折交叉验证中，遍历不同主成分数
    for train_idx, test_idx in skf.split(x, y):
        train_X, test_X = x[train_idx], x[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]
        print('trainx:', x)

        # 遍历不同主成分数
        for n_components in range(1, 17):
            model = PLSRegression(n_components=n_components)
            model.fit(train_X, train_y)
            y_pred = model.predict(test_X)
            r2 = model.score(test_X, test_y)
            rmse = np.sqrt(mean_squared_error(test_y, y_pred))

            r2_scores[n_components].append(r2)
            rmse_scores[n_components].append(rmse)

    # 求平均指标
    mean_r2_scores = [np.mean(v) for v in r2_scores.values()]
    mean_rmse_scores = [np.mean(v) for v in rmse_scores.values()]

    print(f'主成分数范围为1到{n_components}，10折交叉验证平均R2分别为:', mean_r2_scores)
    print(f'主成分数范围为1到{n_components}，10折交叉验证平均RMSE分别为:', mean_rmse_scores)
    best_n_components = np.argmax(mean_r2_scores) + 1
    print(f'最佳主成分数: {best_n_components}')
    result = {
        'algorithm': 'PLSR',
        'preprocessingType': preprocessingType.__name__,
        "R2": '{:.2}'.format(mean_r2_scores),
        # "R2":mean_r2_scores,
        # "RMSE": '{:.2}'.format(mean_rmse_scores),
        "RMSE": mean_rmse_scores,
        "modelPath": modelName + "_" + "PLSR" + "_" + preprocessingType.__name__ + ".pkl"

        # "imageName": "PLS_DA" + "_"+preprocessingType.__name__+".jpg"
    }

    return result

    # return mean_r2_scores, mean_rmse_scores, best_n_components


def SSVR(preprocessingType, x, y, modelName):
    preprocessingType(x)
    svr = SVR(kernel='rbf')  # 核函数可根据需要选择，如'linear', 'poly', 'rbf', 'sigmoid'等
    svr.fit(x, y)
    y_pred = svr.predict(x)
    r2 = svr.score(x, y)
    mse_train = mean_squared_error(y, y_pred)
    result = {
        'algorithm': 'SVR',
        'preprocessingType': preprocessingType.__name__,
        "mse":'{:.2}'.format(mse_train),
        "R2" :'{:.2}'.format(r2),
        "modelPath": modelName + "_" + "SVR" + "_" + preprocessingType.__name__ + ".pkl"

        # "imageName": "PLS_DA" + "_"+preprocessingType.__name__+".jpg"
    }
    joblib.dump(svr, f'{modelName}_SVR_{preprocessingType.__name__}.pkl')
    return result


def PCR(preprocessingType, x, y, modelName):
    preprocessingType(x)
    regression = LinearRegression()
    regression.fit(x, y)
    pca = PCA(n_components=16)  # K是你选择的主成分个数
    X_train_pca = pca.fit_transform(x)
    y_pred_train = regression.predict(X_train_pca)
    mse_train = mean_squared_error(y, y_pred_train)
    r2 = pca.score(x, y)
    result = {
    'algorithm': 'PCR',
    'preprocessingType': preprocessingType.__name__,
    "mse": '{:.2}'.format(mse_train),
    "R2": '{:.2}'.format(r2),
    "modelPath": modelName + "_" + "PCR" + "_" + preprocessingType.__name__ + ".pkl"

    # "imageName": "PLS_DA" + "_"+preprocessingType.__name__+".jpg"
            }

    joblib.dump(regression, f'{modelName}_SVM_{preprocessingType.__name__}.pkl')
    return result


# 分类任务

def PLS_DA(preprocessingType, x, y, modelName):
    # print("算法PLS_DA被选中")
    preprocessingType(x)
    component = 16
    n_fold = 10
    k_range = np.linspace(start=1, stop=component, num=component)
    kf = KFold(n_splits=n_fold, random_state=None, shuffle=True)  # n_splits表示要分割为多少个K子集,交叉验证需要
    com_num_score = []  # 存放每个主成分对应的交叉验证平均值
    for j in range(component):  # j∈[0,component-1],j+1∈[1,component]
        p = 0
        # acc = np.zeros((1,n_fold))  # acc表示总的精准度,p表示个数,acc/p平均精确度
        acc = []  # 列表存放交叉验证准确率

        # 下面是交叉验证
        for train_index, test_index in kf.split(x, y):  # 进行n_fold轮交叉验证
            # 划分数据集
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train = pd.get_dummies(y_train)  # 训练数据结果独热编码
            pls_da = PLSRegression(n_components=j + 1)
            pls_da.fit(x_train, y_train)
            y_pred = pls_da.predict(x_test)
            # print('y1',y_pred)
            y_pred = np.array([np.argmax(i1) for i1 in y_pred])
            # print('YYY',y_pred)# 独热编码转化成类别变量
            # acc[:,n_fold-1] = accuracy_score(y_test, y_pred)
            # print(f'主成分个数为{j+1}时的交叉验证准确率为：',acc)
            acc.append(accuracy_score(y_test, y_pred))  # 给出每一折的准确率
            # print(acc)
            # p = p + 1
            # print(p)
        # accuracy_validation[:, j] = acc / p  # 计算j+1个成分的平均精准度
        #     acc[p]= accuracy_score(y_test, y_pred)
        mean = statistics.mean(acc)  # 交叉验证平均准确率
        #
        #     print(f'主成分个数为{j + 1}时的交叉验证平均准确率为：', mean)
        com_num_score.append(mean)
        best_com_num = com_num_score.index(max(com_num_score)) + 1

        #
        # print('最佳主成分个数为：', best_com_num)  # 最佳主成分个数

        # 全部数据训练最优主成分最终模型

        pls_da = PLSRegression(n_components=best_com_num)

        y_label = pd.get_dummies(y)  # 训练数据独热编码

        pls_da.fit(x, y_label)
        y_pred_pls_da = pls_da.predict(x)

        y_pred_pls_da = np.array([np.argmax(i1) for i1 in y_pred_pls_da])
        # print('y_pred',y_pred)
        acc_pls_da = accuracy_score(y, y_pred_pls_da)
        # acc_pls_da = '{:.2%}'.format(accuracy_score(y, y_pred_pls_da))
        # print('pls_da最佳主成分训练准确率为：', '{:.2%}'.format(acc_pls_da))
        joblib.dump(pls_da, f'{modelName}_PLS_DA_{preprocessingType.__name__}.pkl')
        confusion_matrix_pls_da = confusion_matrix(y, y_pred_pls_da)
        confusion_matrix_pls_da = confusion_matrix_pls_da.tolist()
        # print('测试集混淆矩阵为：\n', confusion_matrix_pls_da)
        cm = {"classNames": ['问斯黑咖啡', '美式高因黑咖啡', '钟小粒', '美式黑咖固体', '美式黑卡',
                             '美式冷萃黑咖'], "matrix": confusion_matrix_pls_da}
        # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_pls_da,
        #                               display_labels=['问斯黑咖啡', '美式高因黑咖啡', '钟小粒', '美式黑咖固体',
        #                                               '美式黑卡',
        #                                               '美式冷萃黑咖'])
        # disp.plot(
        #     include_values=True,
        #     cmap='Greens',
        #     ax=None,
        #     xticks_rotation='horizontal',
        # )
        # plt.title(f'pls_da_kfold  accuracy:{accuracy}',loc='center')
        plt.title(f'PLS_DA_KFold -SNV- accuracy:''{:.2%}'.format(acc_pls_da), loc='center')
        plt.xticks(rotation=30)  # 设置字体大小
        plt.yticks(rotation=30)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 添加文字描述
        # plt.text(x=10,y=10,s=f'accuracy:{accuracy}',bbox=dict(facecolor="dimgray", alpha=0.7, boxstyle="round"))
        # plt.show()

        result = {
            'algorithm': 'PLS_DA',
            'preprocessingType': preprocessingType.__name__,
            "confusionMatrix": cm,
            "accuracy": '{:.2}'.format(acc_pls_da),
            "modelPath": modelName + "_" + "PLS_DA" + "_" + preprocessingType.__name__ + ".pkl"

            # "imageName": "PLS_DA" + "_"+preprocessingType.__name__+".jpg"
        }

        return result


def LDA(preprocessingType, x, y, modelName):
    # print("算法LDA被选中")
    preprocessingType(x)
    lda = LinearDiscriminantAnalysis()
    lda.fit(x, y)
    pred_y = lda.predict(x)
    acc_lda = accuracy_score(y, pred_y)
    # print('lda训练分类准确率：', acc_lda)
    score = cross_val_score(lda, x, y, cv=10)
    # print('10折交叉验证得分：', score)
    # print('10折交叉验证平均得分：', score.mean())

    joblib.dump(lda, f'{modelName}_LDA_{preprocessingType.__name__}.pkl')
    confusion_mat_lda = confusion_matrix(y, pred_y)  # 绘制混淆矩阵
    confusion_mat_lda = confusion_mat_lda.tolist()
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_lda,
    #                               display_labels=['问斯黑咖啡', '美式高因黑咖啡', '钟小粒', '美式黑咖固体', '美式黑卡',
    #                                        '美式冷萃黑咖'])
    cm = {"classNames": ['问斯黑咖啡', '美式高因黑咖啡', '钟小粒', '美式黑咖固体', '美式黑卡',
                         '美式冷萃黑咖'], "matrix": confusion_mat_lda}
    # disp.plot(
    #     include_values=True,
    #     cmap='Greens',
    #     ax=None,
    #     xticks_rotation='horizontal',
    # )
    # plt.xticks(rotation=30)  # 设置字体大小
    # plt.yticks(rotation=30)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # acc_lda = accuracy_score(y, pred_y)
    # acc_lda = '{:.2%}'.format(accuracy_score(y, pred_y))
    # plt.title(f'LDA -SNV- accuracy:''{:.2%}'.format(acc_lda), loc='center')
    # plt.show()
    # print('测试集混淆矩阵为：\n', confusion_matrix(y, pred_y))
    # print('平均分类准确率为：\n', acc_lda)

    result = {
        "algorithm": 'LDA',
        "preprocessingType": preprocessingType.__name__,
        "confusionMatrix": cm,
        "accuracy": '{:.2}'.format(acc_lda),
        "modelPath": modelName + "_" + "LDA" + "_" + preprocessingType.__name__ + ".pkl"
        # "imageName": "LDA" +"_"+ preprocessingType.__name__+".jpg"
    }

    return result


def SVM(preprocessingType, x, y, modelName):
    # print("算法SVM被选中")
    preprocessingType(x)
    clf = svm.SVC()
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.1, 1, 10]
    }
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(x, y)
    # print("Best parameters: ", grid_search.best_params_)
    # print("Best score: ", grid_search.best_score_)
    best_clf = grid_search.best_estimator_
    best_clf.fit(x, y)

    pred_y = best_clf.predict(x)

    # clf.fit(x,y)
    # pred_y = clf.predict(x)
    acc_svm = accuracy_score(y, pred_y)
    # print('训练分类准确率：', acc_svm)
    # score = cross_val_score(clf, x, y, cv=10)
    # print('10折交叉验证得分：',score)
    # print('10折交叉验证平均得分：', score.mean())
    joblib.dump(best_clf, f'{modelName}_SVM_{preprocessingType.__name__}.pkl')
    confusion_mat = confusion_matrix(y, pred_y)  # 绘制混淆矩阵
    confusion_mat = confusion_mat.tolist()  # 二维数组转成列表，否则无法进行序列化
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=['芬必得', '对乙酰氨基酚片', '盐酸氨溴锁片', '吗丁啉', '宝塔糖','西地碘含片', '青霉素V钾片', '苯碘酸氨地平片', '阿司匹林肠溶片','盐酸小檗碱片'])
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat,
    #                               display_labels=['问斯黑咖啡', '美式高因黑咖啡', '钟小粒', '美式黑咖固体', '美式黑卡',
    #                                               '美式冷萃黑咖'])
    # disp.plot(
    #     include_values=True,
    #     cmap='Greens',
    #     ax=None,
    #     xticks_rotation='horizontal',
    # )
    # plt.xticks(rotation=30)  # 设置字体大小
    # plt.yticks(rotation=30)
    # plt.title(f'SVM -SNV- accuracy:''{:.2%}'.format(acc_svm), loc='center')
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.show()
    # print('测试集混淆矩阵为：\n',  confusion_mat)
    # print('平均分类准确率为：\n', accuracy_score(y, pred_y))
    cm = {"classNames": ['问斯黑咖啡', '美式高因黑咖啡', '钟小粒', '美式黑咖固体', '美式黑卡',
                         '美式冷萃黑咖'], "matrix": confusion_mat}

    result = {
        "algorithm": "SVM",
        "preprocessingType": preprocessingType.__name__,
        "confusionMatrix": cm,
        "accuracy": '{:.2}'.format(acc_svm),
        "modelPath": modelName + "_" + "LDA" + "_" + preprocessingType.__name__ + ".pkl"
        # "imageName":"SVM"+"_"+ preprocessingType.__name__+".jpg"
    }
    return result


algoruthms_reg_all = {"PLSR": PLSR, "PCR": PCR, "SVR": SSVR}
algoruthms_clf_all = {"PLS_DA": PLS_DA, "LDA": LDA, "SVM": SVM}
# algorithms_all = {"PLS_DA":PLS_DA,"LDA" :LDA, "SVM":SVM}
preprocessingTypes_all = {"MC": MC, "SUM": SUM, "SNV": SNV}


# 回归任务


@app.route('/models/train/', methods=['post'])
def train_model():
    data = request.get_json()
    # # # 发送 POST 请求，并将 JSON 数据作为请求体
    # print("data",data)
    taskType = data['taskType']
    if taskType == "Classification":
        modelName = data['modelName']
        excelPath = data['excelPath']
        x, y = deal_data_clf(excelPath)
        algorithms = data['algorithms']
        # algorithms = json.loads(algorithms)
        # print("algorithms:", type(data['algorithms']))
        # print("sssss",data['algorithms'])
        preprocessingTypes = data['preprocessingTypes']
        # preprocessingTypes = json.loads(preprocessingTypes)
        # # # average = data['average']
        # # taskType = data['taskType']
        # algorithms = [PLS_DA, LDA, SVM]
        # preprocessingTypes = [MC, SUM, SNV]
        # excelPath  = r"C:\Users\Hi-Tronics\Desktop\data\experience_data\0927(1)\0927\data\coffee0927.xlsx"
        # x, y = deal_data(excelPath)
        # # scanInputType = data['scanInputType']
        # # intergrationTime = data['intergrationTime']
        # # operationMode = data['operationMode']
        # # lampMode = data['lampMode']
        # # intensity = data['intensity']

        # 双重循环
        models = []
        for algorithm in algorithms:
            for preprocessingType in preprocessingTypes:
                # algorithm(preprocessingType)
                result = algoruthms_clf_all[algorithm](preprocessingTypes_all[preprocessingType], x, y, modelName)
                models.append(result)
                # print("------------------")
                # result = algorithm(preprocessingType)
                # results[algorithm] = result
                # print(models)
                max_accuracy = float('-inf')  # 初始化最大值为负无穷大
                max_accuracy_model = None
            for model in models:
                accuracy = (model['accuracy'])
                accuracy = float(accuracy.strip("%")) / 100  # 将带百分比的字符串转换为浮点数, 移除百分比符号
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_accuracy_model = model

            # 检查是否找到了最大值
            if max_accuracy_model is not None:
                max_accuracy_model_name = max_accuracy_model['algorithm']
                preprocessingType = max_accuracy_model['preprocessingType']
            # max_accuracy = "{:.2%}".format(max_accuracy)
            # print(f"最佳模型：{max_accuracy_model_name}，预处理方法：{preprocessingType}，准确率：{max_accuracy}")
            else:
                print("没有找到最佳模型")

        result = {"modelName": modelName, "bestAlgorithm": max_accuracy_model_name,
                  "bestPreprocessingType": preprocessingType,
                  "algorithmResults": models}
        print(result)
    else:
        modelName = data['modelName']
        excelPath = data['excelPath']
        x, y = deal_data_reg(excelPath)
        algorithms = data['algorithms']
        # algorithms = json.loads(algorithms)
        # print("algorithms:", type(data['algorithms']))
        # print("sssss",data['algorithms'])
        preprocessingTypes = data['preprocessingTypes']
        models = []
        for algorithm in algorithms:
            for preprocessingType in preprocessingTypes:
                # algorithm(preprocessingType)
                result = algoruthms_reg_all[algorithm](preprocessingTypes_all[preprocessingType], x, y, modelName)
                models.append(result)
                # print("------------------")
                # result = algorithm(preprocessingType)
                # results[algorithm] = result
                # print(models)
                max_r2 = float('-inf')  # 初始化最大值为负无穷大
                max_r2_model = None
            for model in models:
                r2 = (model['R2'])
                print("22222",r2)
                r2= float(r2)
                # accuracy = float(accuracy.strip("%")) / 100  # 将带百分比的字符串转换为浮点数, 移除百分比符号
                if r2 > max_r2:
                    max_accuracy = r2
                    max_r2_model = model

            # 检查是否找到了最大值
            if max_r2_model is not None:
                max_r2_model_name = max_r2_model['algorithm']
                preprocessingType = max_r2_model['preprocessingType']
            # max_accuracy = "{:.2%}".format(max_accuracy)
            # print(f"最佳模型：{max_accuracy_model_name}，预处理方法：{preprocessingType}，准确率：{max_accuracy}")
            else:
                print("没有找到最佳模型")

        result = {"modelName": modelName, "bestAlgorithm": max_r2_model_name,
                  "bestPreprocessingType": preprocessingType,
                  "algorithmResults": models}
        print(result)

    REQUEST_URL = "http://127.0.0.1:9090/api/v1/train/result"  # 服务器服务 ...为app.route路径
    HEADER = {'Content-Type': 'application/json; charset=utf-8'}

    # 测试实现直接在终端输入文字，可取消注释自行实验一下，但需要将下一句赋值注释
    # requestDict = {}
    # requestDict["text"] = input("请输入文本：")

    # 将文本直接赋值
    # requestDict = {
    #     "modelName": "1",
    #     "bestAlgorithm": "xxxx",
    #     "bestPreprocessingType": "aaaaa",
    #     "algorithmResults": [
    #         {
    #             "modelPath": "lda_model.pkl",
    #             "algorithm": "lllllll",
    #             "preprocessingType": "xxxxx",
    #             "accuracy": "100.00%",
    #             "confusionMatrix": {
    #                 "classNames": ["问斯黑咖啡", "美式高因黑咖啡", "钟小粒", "美式黑咖固体", "美式黑卡",
    #                                "美式冷萃黑咖"],
    #                 "matrix": [
    #                     [10, 0, 0, 0, 0, 0],
    #                     [0, 10, 0, 0, 0, 0],
    #                     [0, 0, 10, 0, 0, 0],
    #                     [0, 0, 0, 10, 0, 0],
    #                     [0, 0, 0, 0, 10, 0],
    #                     [0, 0, 0, 0, 0, 10]
    #                 ]
    #             }
    #         },
    #         {
    #             "modelPath": "lda_model.pkl",
    #             "algorithm": "lllllll",
    #             "preprocessingType": "xxxxx",
    #             "accuracy": "100.00%",
    #             "confusionMatrix": {
    #                 "classNames": ["问斯黑咖啡", "美式高因黑咖啡", "钟小粒", "美式黑咖固体", "美式黑卡",
    #                                "美式冷萃黑咖"],
    #                 "matrix": [
    #                     [10, 0, 0, 0, 0, 0],
    #                     [0, 10, 0, 0, 0, 0],
    #                     [0, 0, 10, 0, 0, 0],
    #                     [0, 0, 0, 10, 0, 0],
    #                     [0, 0, 0, 0, 10, 0],
    #                     [0, 0, 0, 0, 0, 10]
    #                 ]
    #             }
    #         }
    #     ]
    # }

    # 实现请求功能
    # rsp = requests.post(REQUEST_URL, json.dumps(result), headers=HEADER)
    # # rspJson = json.loads(rsp.text.encode())
    return json.dumps(result, ensure_ascii=False)

@app.route('/models/test/', methods=['post'])
def test_model():
    data = request.get_json()
    # modelName：ModelName
    taskType = data['taskType']
    modelName = data['modelName']
    modelPath = './'
    # excelPath = data['excelPath']
    # 算法和预处理方法: algorithm \ preprocessingType
    algorithm = data['algorithm']
    preprocessingType = data['preprocessingType']
    # testData: testData(一条数据)
    testData = data['testData']
    # modelPath: C:\Users\Hi-Tronics\Desktop\data
    modelPath = modelPath + modelName + '_' + algorithm + "_" + preprocessingType + '.pkl'
    # 响应
    # 分类：标签
    # 回归：浮点数
    if taskType == "Classification":
        x, y = deal_data_clf(testData)
        model = joblib.load(filename=modelPath)
        label = model.predict(x)
        if algorithm == "PLS_DA":
            label = np.array([np.argmax(i) for i in label])
            label = label.tolist()
        else:
            label = label.tolist()
        result = {"modelName": modelName, "algorithm": algorithm,"preprocessingType":preprocessingType,
              "Label":label}
    else:
        x, y = deal_data_reg(testData)
        model = joblib.load(filename=modelPath)
        # y_pred = '{:.2}'.format(model.predict(x))
        y_pred = model.predict(x)
        y_pred = np.around(y_pred,2)    #保留两位小数
        y_pred = y_pred.tolist()        #转成列表，json序列化
        result = {
            "modelName": modelName,
            'algorithm': algorithm,
            'preprocessingType': preprocessingType,
            "y_pred":y_pred
        }
    return json.dumps(result, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
    # 发送 POST 请求，并将 JSON 数据作为请求体
