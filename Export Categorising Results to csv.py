# -*- coding:utf-8 -*-
import csv
from db.pgutil import PgUtil
import pandas as pd
import numpy as np


def get_all_id(pgutil):

    sql = ''' select distinct(qiyebianma) from yao_week_clean order by qiyebianma '''
    all_ids = pgutil.query_all_sql(sql)
    ids = []
    [ids.append(all_ids[i]['qiyebianma']) for i in xrange(len(all_ids))]
    return ids




def get_ruzhulv_week(pgutil,id):
    sql = '''
        select monday, week_rzl::numeric from yao_week_clean
        where qiyebianma = '{}'
        order by monday '''.format(id)
    data = pgutil.query_all_sql(sql)
    # 时间序列
    index = pd.DatetimeIndex(start=data[0]['monday'].replace(tzinfo=None),
                          format='%Y%m%d',
                          periods=len(data),
                          freq='W-MON', # MS：月   # W-MON：每一周周一  D：每天
                          tz=None)

    ruzhulv = []
    for i in xrange(len(data)):
        ruzhulv.append(float(data[i]['week_rzl']))
    return ruzhulv, index



def Categorisation(): #获得酒店数据,判断趋势类别并输出到各自的csv file

    pgutil = PgUtil()

    writer_up = csv.writer(file('up_list.csv', 'wb'))
    writer_down = csv.writer(file('down_list.csv', 'wb'))
    writer_normal = csv.writer(file('normal_list.csv', 'wb'))

    all_id = get_all_id(pgutil)

    for i_n in range(len(all_id)):
        ts = get_ruzhulv_week(pgutil, all_id[i_n])

        binary = [2]
        for i in range(len(ts[0]) - 1):
            if (ts[0][i] < ts[0][i + 1]):
                binary.append(1)
            elif (ts[0][i] > ts[0][i + 1]):
                binary.append(0)
            else:
                binary.append(2)


        avg = np.mean(ts[0])

        #通过对比整个序列的平均值与最后九个数据判断出该酒店的趋势

        if (ts[0][len(ts) - 1] < avg) & (ts[0][len(ts) - 2] < avg) & \
                (ts[0][len(ts) - 3] < avg) & (ts[0][len(ts) - 4] < avg) & \
                (ts[0][len(ts) - 5] < avg) & (ts[0][len(ts) - 6] < avg) & \
                (ts[0][len(ts) - 7] < avg) & (ts[0][len(ts) - 8] < avg) \
                & (ts[0][len(ts) - 9] < avg):
            writer_down.writerow([all_id[i_n]])


        elif (ts[0][len(ts) - 1] > avg) & (ts[0][len(ts) - 2] > avg) & \
                (ts[0][len(ts) - 3] > avg) & (ts[0][len(ts) - 4] > avg) & \
                (ts[0][len(ts) - 5] > avg) & (ts[0][len(ts) - 6] > avg) & \
                (ts[0][len(ts) - 6] > avg):
            writer_up.writerow([all_id[i_n]])

        else:
            writer_normal.writerow([all_id[i_n]])

if __name__ == '__main__':

    ''' float array数据 '''
    pgutil = PgUtil()
    all_id = get_all_id(pgutil)

    Categorisation()