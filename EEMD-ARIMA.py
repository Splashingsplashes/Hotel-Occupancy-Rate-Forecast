#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import pandas as pd
from db.pgutil import PgUtil
import os
from scpy.logger import get_logger
import numpy as np
from PyEMD import EEMD
from pyramid.arima import *
import math
from sklearn.metrics import mean_squared_error
logger = get_logger(__file__)
import csv


PATH=os.getcwd()
#plt.switch_backend('agg')

FIG_PATH1 = os.getcwd() + '/>15%_down/'
FIG_PATH2 = os.getcwd() + '/<15%_down/'


sys.path.append("../")  # import上一层



def get_all_id(pgutil):  #从数据库中得到筛选后的酒店编码

    sql = ''' select distinct(qiyebianma) from yao_week_clean order by qiyebianma '''
    # print sql
    all_ids = pgutil.query_all_sql(sql)
    ids = []
    [ids.append(all_ids[i]['qiyebianma']) for i in xrange(len(all_ids))]
    return ids




def get_ruzhulv_week(pgutil,id):  #输入酒店编码, 得到该酒店以周为单位的入住率

    sql = '''
        select monday, week_rzl::numeric from yao_week_clean 
        where qiyebianma = '{}'
        order by monday '''.format(id) #如果需要改为以月为单位的入住率数据,需要修改sql命令中的from clause
    # print sql
    data = pgutil.query_all_sql(sql)
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


def EEMDecompose(test_set,train_set, id):  #输入需要预测的时间单位长度以及训练集,返回相应的预测数据

    MAPEcount = 0
    RMScount = 0

    for i in range(1):

        IMF = EEMD().eemd(np.array(train_set[0])) #使用EEMD将原序列分解为数个IMF

        predicts = []
        trendDifference = 0
        count = 0
        for imf in IMF:


            model = auto_arima(imf, trace=True, error_action='ignore', stepwise=True, suppress_warnings=True) #为每个IMF单独训练arima模型
            forecast = model.predict(n_periods=len(test_set[0]))  #生成预测时序
            if count == 0:
                predicts = forecast
            else:
                predicts = [a + b for a, b in zip(predicts, forecast)] #将所有预测的IMF相加得到最终预测时序

            count += 1

            if count == len(IMF)-1: #将最后一个频率最小的IMF作为trend并计算落差(原计划通过对比落差和预测准确性寻找规律)
                trendDifference = np.max(imf) - np.min(imf)

        # """画图并保存到指定路径"""
        # tr_pre_ser = pd.Series(test_set[0])
        # te_pre_ser = pd.Series(predicts)
        # tr_pre_ser.plot(color='g', label='observed')
        # te_pre_ser.plot(color='r', label='predict')
        # plt.legend()
        #
        # folder_path = PATH + "/Graphical Comparison/Normal/{}".format(line)
        # # os.makedirs(folder_path)
        # plt_path = folder_path + "/eemd-arima.jpg"
        # plt.savefig(plt_path)
        # plt.close()



        """Summary"""
        MAPEcount += MAPE(test_set[0], predicts)
        RMScount += math.sqrt(mean_squared_error(test_set[0], predicts))

    return (id, MAPEcount,RMScount, trendDifference)


if __name__ == '__main__':

    ''' float array数据 '''
    pgutil = PgUtil()


    errorCount = 0
    forecastResults=[]
    RMSETotal = 0
    MAPETotal = 0
    testSize = 50 #设定一次性测试的酒店样本数量
    with open('normal_list.csv') as f:

        for idIndex in range(testSize):
            # if idIndex < 300:
            #
            #     line=f.readline().replace("\n", "")
            #     continue
            #
            line = f.readline().replace("\n", "")
            print line
            data, index = get_ruzhulv_week(pgutil, int(line))

            #取数据最后12个时间单位为测试集(此处为12周),其余作为训练集
            train_set = (data[:len(data) - 12:], index[:len(index) - 12:])
            test_set = (data[len(data) - 12::], index[len(index) - 12::])

            try:
                results = EEMDecompose(test_set, train_set, line) #返回酒店id, MAPE, RMSE, trend落差
                forecastResults.append(results)
            except:
                errorCount += 1
                continue



        for forecast in forecastResults:
            print "酒店{}: RMS={} 趋势落差={}".format(forecast[0], forecast[2],forecast[3])
            RMSETotal+= forecast[2]
            MAPETotal+= forecast[1]

        print "ave MAPE={}, ave RMSE={}".format(MAPETotal / testSize, RMSETotal / testSize)
        print len(forecastResults)

