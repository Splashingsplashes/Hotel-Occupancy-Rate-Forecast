# coding=UTF-8

import pandas as pd
from db.pgutil import PgUtil
import os
from scpy.logger import get_logger
import matplotlib.pylab as plt
import numpy as np
from pyramid.arima import *
import math
from sklearn.metrics import mean_squared_error
logger = get_logger(__file__)



PATH = os.getcwd()



def get_all_id(pgutil): #从数据库中得到筛选后的酒店编码

    sql = ''' select distinct(qiyebianma) from yao_week_clean order by qiyebianma '''

    all_ids = pgutil.query_all_sql(sql)
    ids = []
    [ids.append(all_ids[i]['qiyebianma']) for i in xrange(len(all_ids))]
    return ids




def get_ruzhulv_week(pgutil,i_n): #输入酒店编码, 得到该酒店以周为单位的入住率

    id = i_n
    sql = '''
        select monday, week_rzl::numeric from yao_week_clean
        where qiyebianma = '{}'
        order by monday '''.format(id) #如果需要改为以月为单位的入住率数据,需要修改sql命令中的from clause

    data = pgutil.query_all_sql(sql)
    n = len(data)
    # 生成日期横坐标
    index = pd.DatetimeIndex(start=data[0]['monday'].replace(tzinfo=None),
                          format='%Y%m%d',
                          periods=len(data),
                          freq='W-MON',
                          tz=None)

    ruzhulv = []
    for i in xrange(n):
        ruzhulv.append(float(data[i]['week_rzl']))
    return ruzhulv, index




def MAPE(true, pred):  #输入测试集真实数据和预测数据, 计算Mean Average Percentage Error
    percentage_err = 0
    for i in range(len(true)):
        diff = np.abs(np.array(true[i]) - np.array(pred[i]))
        percentage_err += np.mean(diff / true)
    return percentage_err/len(true)



if __name__ == '__main__':

    ''' float array数据 '''
    pgutil = PgUtil()


    errorCount = 0
    forecastResults=[]
    RMSETotal = 0
    MAPETotal = 0
    testSize = 155

    with open('down_list.csv') as f:

        for idIndex in range(testSize):
            line = f.readline().replace("\n", "").strip(" ")
            print line


            data, index = get_ruzhulv_week(pgutil, int(line))

            #取数据最后12个时间单位为测试集(此处为12周),其余作为训练集
            train_set = (data[:len(data) - 12:], index[:len(index) - 12:])
            test_set = (data[len(data) - 12::], index[len(index) - 12::])



            try:
                #ARIMA模型
                model = auto_arima(train_set[0], trace=True, error_action='ignore', stepwise=True, suppress_warnings=True)
                results = model.predict(n_periods=12) #生成预测数据


                MAPETotal += MAPE(test_set[0],results)
                RMSETotal += math.sqrt(mean_squared_error(test_set[0], results))

                #画图并保存到指定路径
                results = pd.Series(results)
                test = pd.Series(test_set[0])
                results.plot(color='b', label='predict')
                test.plot(color='g', label='observed')
                plt.legend()
                # plt.show()
                # folder_path = PATH +"/Graphical Comparison/Normal/{}".format(line)
                # if not os.path.exists(folder_path):
                #     os.makedirs(folder_path)
                # plt_path = folder_path + "/pmdarima.jpg"
                # print plt_path
                # plt.savefig(plt_path)
                # plt.close()

            except Exception as e:
                errorCount += 1
                print e
                continue




        print "ave MAPE={}, ave RMSE={}".format(MAPETotal / testSize, RMSETotal / testSize)

        print len(forecastResults)

