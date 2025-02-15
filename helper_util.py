import os
import numpy as np
import logging
import sys

dict_logger=dict()
def get_logger(log_path,log_name="xx",debug=False):
    global dict_logger
    
    if log_path+log_name not in dict_logger:
        # 第一步，创建一个logger
        logger = logging.getLogger(log_path+log_name)
        dict_logger[log_path+log_name]=logger
        logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件


        if log_path=="":
            new_log_path = os.path.abspath(".") + '/Logs/'
        else:
            new_log_path=log_path+"/"
        if not os.path.exists(new_log_path):
            os.mkdir(new_log_path)

        logfile = new_log_path+ log_name+'.log'
        print(logfile)
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s- %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面


        #if debug:
        if 0:
            ##再加上一个输出到终端
            ch=logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        logger.addHandler(fh)
        
    # 日志
    return dict_logger[log_path+log_name]


''' 
Author: Angus
Description:
    辅助函数，确保自增
    给定文件夹
        里面的内容是 1， 2，3， 或者1.jpg 2.jpg。得到最大的+1
'''
def get_max(dirpath):
    array=np.array([int(x.split(".")[0]) for x in os.listdir(dirpath) if not x.startswith(".")])
    if len(array)==0:
        return 1
    else:
        return array.max()+1





'''
The ground truth graph of the Danbube river system. 
    GROUNDTRUTH
    Engelke, S. and Hitz, A. S. (2020). Graphical models for extremes. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(4):871–932.
    Lee, J. and Cooley, D. (2022). Partial tail correlation for extremes. arXiv preprint arXiv:2210.02048.
    Gong, Y., Zhong, P., Opitz, T., and Huser, R. (2024). Partial tail-correlation coefficient applied to extremal-network learning. Technometrics, pages 1–16.

'''


start_x=271
start_y=114
if 1:
    position_dict=dict()
    position_x=dict()
    position_x[1]=[1433,421]
    position_x[2]=[1291,375]
    position_x[3]=[1189,341]
    position_x[4]=[1038,298]
    position_x[5]=[960,320]
    position_x[6]=[864,338]
    position_x[7]=[726,358]
    position_x[8]=[541,383]
    position_x[9]=[441,410]
    position_x[10]=[329,452]
    position_x[11]=[460,662]
    position_x[12]=[411,764]
    position_x[13]=[1351,448]
    position_x[14]=[1217,404]
    position_x[15]=[978,480]
    position_x[16]=[882,585]
    position_x[17]=[839,676]
    position_x[18]=[888,750]
    position_x[19]=[908,831]
    position_x[20]=[602,483]
    position_x[21]=[648,600]
    position_x[22]=[611,720]
    position_x[23]=[946,238]
    position_x[24]=[987,172]
    position_x[25]=[1061,171]
    position_x[26]=[1176,179]
    position_x[27]=[1283,176]
    position_x[28]=[1208,774]
    position_x[29]=[1176,827]
    position_x[30]=[1202,583]
    position_x[31]=[1246,668]
    position_dict={"x":[],"y":[]}
    for i in range(1,32):
        position_dict["x"].append(position_x[i][0])
        position_dict["y"].append(position_x[i][1])
    position_dict["x"]=np.array(position_dict["x"])-start_x
    position_dict["y"]=918-np.array(position_dict["y"])


    link_dict={}
    links=[]
    links.append((27,26))
    links.append((26,25))
    links.append((25,4))
    links.append((4,3))
    links.append((3,2))
    links.append((2,1))
    links.append((24,23))
    links.append((23,4))
    links.append((12,11))
    links.append((11,10))
    links.append((10,9))
    links.append((9,8))
    links.append((8,7))
    links.append((7,6))
    links.append((6,5))
    links.append((5,4))
    links.append((22,21))
    links.append((21,20))
    links.append((20,7))
    links.append((19,18))
    links.append((18,17))
    links.append((17,16))
    links.append((16,15))
    links.append((15,14))
    links.append((14,2))
    links.append((29,28))
    links.append((28,31))
    links.append((31,30))
    links.append((30,13))
    links.append((13,1))
    link_dict["GROUNDTRUTH"]=links




    links_gong=[]
    links_gong.append((12,11))
    links_gong.append((27,26))
    links_gong.append((26,25))
    links_gong.append((24,23))
    links_gong.append((10,9))
    links_gong.append((9,8))
    links_gong.append((8,7))
    links_gong.append((7,6))
    links_gong.append((6,5))
    links_gong.append((5,4))
    links_gong.append((4,3))
    links_gong.append((3,2))
    links_gong.append((2,14))
    links_gong.append((14,15))
    links_gong.append((15,16))
    links_gong.append((16,17))
    links_gong.append((17,18))
    links_gong.append((18,19))
    links_gong.append((29,28))
    links_gong.append((28,31))
    links_gong.append((31,30))
    links_gong.append((30,13))
    links_gong.append((13,1))
    links_gong.append((22,21))
    links_gong.append((21,20))

    link_dict["GONG"]=links_gong

    links_cooley=[]

    links_cooley.append((26,25))
    links_cooley.append((24,23))
    links_cooley.append((12,11))
    links_cooley.append((11,10))
    links_cooley.append((10,8))
    links_cooley.append((8,9))
    links_cooley.append((8,4))
    links_cooley.append((9,6))
    links_cooley.append((6,5))
    links_cooley.append((5,3))
    links_cooley.append((3,2))
    links_cooley.append((2,1))
    links_cooley.append((1,13))
    links_cooley.append((13,4))
    links_cooley.append((22,7))
    links_cooley.append((21,20))
    links_cooley.append((17,19))
    links_cooley.append((16,18))
    links_cooley.append((16,15))
    links_cooley.append((15,14))
    links_cooley.append((29,28))

    link_dict["COOLEY"]=links_cooley

    links_Enge=[]
    links_Enge.append((1,2))
    links_Enge.append((2,3))
    links_Enge.append((3,25))
    links_Enge.append((25,27))
    links_Enge.append((25,26))
    links_Enge.append((26,24))
    links_Enge.append((24,23))
    links_Enge.append((3,4))
    links_Enge.append((4,5))
    links_Enge.append((5,6))
    links_Enge.append((6,7))
    links_Enge.append((7,9))
    links_Enge.append((9,8))
    links_Enge.append((9,10))
    links_Enge.append((10,11))
    links_Enge.append((11,12))
    links_Enge.append((6,20))
    links_Enge.append((20,21))
    links_Enge.append((21,22))
    links_Enge.append((2,14))
    links_Enge.append((14,15))
    links_Enge.append((15,16))
    links_Enge.append((16,19))
    links_Enge.append((19,18))
    links_Enge.append((18,17))
    links_Enge.append((1,13))
    links_Enge.append((13,30))
    links_Enge.append((30,31))
    links_Enge.append((31,28))
    links_Enge.append((28,29))

    link_dict["ENGELKE"]=links_Enge
