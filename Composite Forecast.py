# -*- coding:utf-8 -*-

import sys
import pandas as pd
from db.pgutil import PgUtil
import os
from scpy.logger import get_logger
import matplotlib.pylab as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from PyEMD import EEMD
from pyramid.arima import *
import math
from sklearn.metrics import mean_squared_error
logger = get_logger(__file__)


plt.switch_backend('agg')
PATH=os.getcwd()

sys.path.append("../")  # import上一层




def get_all_id(pgutil): #从数据库中得到筛选后的酒店编码

    sql = ''' select distinct(qiyebianma) from yao_week_clean order by qiyebianma '''
    all_ids = pgutil.query_all_sql(sql)
    ids = []
    [ids.append(all_ids[i]['qiyebianma']) for i in xrange(len(all_ids))]
    return ids




def get_ruzhulv_week(pgutil,id): #输入酒店编码, 得到该酒店以周为单位的入住率


    sql = '''
        select monday, week_rzl::numeric 
        from yao_week_clean
        where qiyebianma = '{}'
        order by monday '''.format(id) #如果需要改为以月为单位的入住率数据,需要修改sql命令中的from clause
    data = pgutil.query_all_sql(sql) #执行sql命令
    n = len(data)
    # 生成日期横坐标
    index = pd.DatetimeIndex(start=data[0]['monday'].replace(tzinfo=None),
                          format='%Y%m%d',
                          periods=len(data),
                          freq='W-MON', # MS：月   # W-MON：每一周周一  D：每天
                          tz=None)

    ruzhulv = []
    for i in xrange(n):
        ruzhulv.append(float(data[i]['week_rzl']))

    return ruzhulv, index




def MAPE(true, pred): #输入测试集真实数据和预测数据, 计算Mean Average Percentage Error
    percentage_err = 0
    for i in range(len(true)):
        diff = np.abs(np.array(true[i]) - np.array(pred[i]))
        percentage_err += np.mean(diff / true)
    return percentage_err/len(true)


def EEMDecompose(length, train_set): #输入需要预测的时间单位长度以及训练集,返回相应的预测数据

    IMF = EEMD().eemd(np.array(train_set[0])) #使用EEMD将原序列分解为数个IMF

    predicts = []
    count = 0
    for imf in IMF:

        model = auto_arima(imf, trace=True, error_action='ignore', stepwise=True, suppress_warnings=True) #为每个IMF单独训练arima模型
        forecast = model.predict(n_periods=length) #生成预测时序

        if count == 0:
            predicts = forecast
        else:
            predicts = [a + b for a, b in zip(predicts, forecast)] #将所有预测的IMF相加得到最终预测时序

        count += 1


    return predicts


def KMP_algorithm(string, substring): #用于季节性分解中使用模式匹配法预测seasonality
    pnext = gen_pnext(substring)
    n = len(string)
    m = len(substring)
    i, j = 0, 0
    while (i < n) and (j < m):
        if (string[i] == substring[j]):
            i += 1
            j += 1
        elif (j != 0):
            j = pnext[j - 1]
        else:
            i += 1
    if (j == m):
        return string[i:i + 12] #生成预测时序,此处为12个时间单位(周),即未来三个月

    else:
        return -1

def gen_pnext(substring): #用于KMP Algorithm
    """
    构造临时数组pnext
    """
    index, m = 0, len(substring)
    pnext = [0] * m
    i = 1
    while i < m:
        if (substring[i] == substring[index]):
            pnext[i] = index + 1
            index += 1
            i += 1
        elif (index != 0):
            index = pnext[index - 1]
        else:
            pnext[i] = 0
            i += 1
    return pnext


def Seasonal_Decompose(timeseries): #输入dataframe格式的训练集时序,返回预测时序


    decomposition = seasonal_decompose(timeseries, model = 'additive',extrapolate_trend='freq') #默认使用additive decomposation, 返回包含三个部分 trend（趋势部分） ， seasonal（季节性部分） 和residual (残留部分)
    seasonal = decomposition.seasonal
    mix = timeseries - seasonal #获得trend和residual的综合时序

    model = auto_arima(mix, trace=True, error_action='ignore', stepwise=True, suppress_warnings=True) #为mix训练arima预测模型
    trend_forecast = model.predict(n_periods=12) #生成预测时序,此处为12个时间单位(周),即未来三个月

    #通过KMP算法匹配seasonality
    search_pattern = pd.Series.tolist(seasonal[len(seasonal) - 12: len(seasonal)])
    seasonal_list = pd.Series.tolist(seasonal)
    seasonality_predict = KMP_algorithm(seasonal_list, search_pattern)


    #将预测得到的seasonality和mix(trend+residual)相加得到最终预测时序
    predict = []
    for i in range(0, 12):
        predict.append(trend_forecast[i] + seasonality_predict[i])


    return predict





if __name__ == '__main__':

    ''' float array数据 '''
    pgutil = PgUtil()


    testSize = 100 #设定一次性运算的样本数量

    errorCount = 0
    successCount = 0

    forecastResults=[]

    RMSETotal = 0
    MAPETotal = 0
    RMSEcount = 0
    MAPEcount = 0

    with open('normal_list.csv') as f:

        for idIndex in range(testSize):


            line = f.readline().replace("\n", "")
            print line
            data, index = get_ruzhulv_week(pgutil, int(line))

            #取数据最后12个时间单位为测试集(此处为12周),其余作为训练集
            train_set = (data[:len(data) - 12:], index[:len(index) - 12:])
            test_set = (data[len(data) - 12::], index[len(index) - 12::])

            try:
                #EEMD-ARIMA模型
                EEMD_results = EEMDecompose(len(test_set[0]), train_set)

                #ARIMA模型
                model = auto_arima(train_set[0], trace=True, error_action='ignore', stepwise=True,suppress_warnings=True)
                ARIMA_results = model.predict(n_periods=12)

                #季节性分解模型
                train_set_dataframe = pd.DataFrame(train_set[0], index=train_set[1], columns=['week_rzl'], dtype='float') #构建dataframe作为输入
                seasonalDecompose_result = Seasonal_Decompose(train_set_dataframe['week_rzl'])

                results = []

                for i in range(12):

                    #results.append((EEMD_results[i] + ARIMA_results[i] + seasonalDecompose_result[i]) / 3) #为三者的预测取平均值

                    results.append((EEMD_results[i] + ARIMA_results[i] / 2)) #仅为EEMD-ARIMA和ARIMA两者的预测取平均值


                #与测试集真实数据计算Root Mean Square Error
                RMSEcount = math.sqrt(mean_squared_error(test_set[0], results))
                RMSETotal += RMSEcount

                print RMSEcount, RMSETotal

                MAPEcount += MAPE(test_set[0], results)
                MAPETotal += MAPEcount
                print MAPETotal, MAPEcount

                successCount += 1



                #保存图表至指定路径
                results = pd.Series(results)
                results.plot(color = "c", label = 'composite')
                test = pd.Series(test_set[0])
                test.plot(color = "g", label = 'observed')
                plt.legend()

                folder_path = PATH + "/Graphical Comparison/Normal/{}".format(line)
                plt_path = folder_path + "/three_models_composite.jpg"
                plt.savefig(plt_path)
                plt.close()





            except Exception as e:
                errorCount += 1
                testSize-= 1
                print e #print error message
                continue



        print "ave MAPE={}, ave RMSE={}".format(MAPETotal / testSize, RMSETotal / testSize) #输出平均准确性指标数据
        print errorCount
        print successCount
