# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:05:51 2017

@author: LuoJingsheng
"""

import jieba, os, re
import cPickle as pickle
from keras.models import load_model

name = 'Named Entity Recognition (Demo Version 1.0)'
desc = '基于LSTM的命名实体识别。'
examples = ['中华人民共和国中央政府人民大会堂位于中国北京市天安门广场西侧，西长安街南侧。人民大会堂坐西朝东，南北长336米，东西宽206米，高46.5米，占地面积15万平方米，建筑面积17.18万平方米。人民大会堂是中国全国人民代表大会开会地和全国人民代表大会常务委员会的办公场所，是党、国家和各人民团体举行政治活动的重要场所，也是中国党和国家领导人和人民群众举行政治、外交、文化活动的场所。人民大会堂每年举行的全国人民代表大会、中国人民政治协商会议以及五年一届的中国共产党全国代表大会也在此召开。']
port = 11187

Maps = pickle.load(open("MapsW v1.2.pkl", "r"))
jieba.load_userdict('Jieba.txt')
Model = load_model('Model v1.2.h5')
print os.popen('nvidia-smi').read() # 加上 -p 如果要输出更详细信息

def Run(sentence):
    global Maps
    global Model
    ret_val = '***以下是识别的实体及其标签***\n********************\n'
    Before = jieba.lcut(sentence)
    HTTP_input = []
    for x in Before:
        y = re.sub("[A-Za-z0-9\ \-\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\.\<\>\/\?\~\@\#\\\&\*\%\，\。\《\》\；\：\？\、\‘\’\“\”\【\】\（\）\！]", "", x.encode("utf-8"))
        if y == '':
            continue
        HTTP_input.append(x)
    HTTP_input_index = []
    for x in HTTP_input:
        if x.encode("utf-8") in Maps:
            HTTP_input_index.append(Maps[x.encode("utf-8")])
        else:
            HTTP_input_index.append(len(Maps))
    z = Model.predict(HTTP_input_index).tolist()
    word_prob = []
    def maxarg(list):
        length = len(list)
        max = 0
        pos = 0
        for i in range(length):
            if list[i] > max:
                max = list[i]
                pos = i
        return pos
    for line in z:
        max_pos = maxarg(line)
        if max_pos == 0:
            word_prob.append('n')
        if max_pos == 1:
            word_prob.append('机构团体')
        if max_pos == 2:
            word_prob.append('地名')
        if max_pos == 3:
            word_prob.append('其它专有名词')
        if max_pos == 4:
            word_prob.append('实体简称')
        if max_pos == 5:
            word_prob.append('人名')
    i = 0
    Dedupe = {}
    for flag in word_prob:
        if HTTP_input[i] not in Dedupe:
            if (flag == 'n') or (flag == '其它专有名词'):
                i += 1
            else:
                ret_val += str(HTTP_input[i].encode("utf-8")) + '：' + flag + '\n'
                Dedupe[HTTP_input[i]]=i
                i += 1
        else:
            i += 1
    ret_val += '********************\n'
    return ret_val


if __name__ == '__main__':
    Run('这次会议的一项重要成果，是审议通过了澳门特别行政区筹委会组成人员名单。这个名单是按照一国两制精神和澳门基本法的规定，经过充分酝酿协商提出来的，其中澳门委员六十人，来自澳门社会的各个阶层，具有广泛的代表性；内地委员四十人，主要是同澳门事务有较多联系的有关部门的负责人和法律、经济等方面的专家。')
#    Run('今天，我们召开一个深度贫困地区脱贫攻坚座谈会，研究如何做好深度贫困地区脱贫攻坚工作。攻克深度贫困堡垒，是打赢脱贫攻坚战必须完成的任务，全党同志务必共同努力。')
#    Run('中华人民共和国中央政府人民大会堂位于中国北京市天安门广场西侧，西长安街南侧。人民大会堂坐西朝东，南北长336米，东西宽206米，高46.5米，占地面积15万平方米，建筑面积17.18万平方米。人民大会堂是中国全国人民代表大会开会地和全国人民代表大会常务委员会的办公场所，是党、国家和各人民团体举行政治活动的重要场所，也是中国党和国家领导人和人民群众举行政治、外交、文化活动的场所。人民大会堂每年举行的全国人民代表大会、中国人民政治协商会议以及五年一届的中国共产党全国代表大会也在此召开。')



