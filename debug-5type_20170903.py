# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:46:51 2017

@author: LuoJingsheng
"""
import re
import numpy as np
import os
import time
import cPickle as pickle
from sklearn.model_selection import train_test_split as tts
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

pwd = os.getcwd()
pwd = pwd+'\\人民日报utf-8\\'
JiabaFilename = ''
logtimestamp = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
log = open('log\\Debug_log '+str(logtimestamp)+'.txt',"w")
log.write('\n\n\n\n----------------------------\n')
log.write('Start\n')
log.write('Time: '+str(logtimestamp)+'\n-------------\n\n')


def Initialize():
    start_time = time.time()
    f1=open(pwd+"199801.txt","r", encoding ='utf-8')
    f2=open(pwd+"199802.txt","r", encoding ='utf-8')
    f3=open(pwd+"199803.txt","r", encoding ='utf-8')
    f4=open(pwd+"199804.txt","r", encoding ='utf-8')
    f5=open(pwd+"199805.txt","r", encoding ='utf-8')
    f6=open(pwd+"199806.txt","r", encoding ='utf-8')
    data = f1.readlines() + f2.readlines() + f3.readlines() + f4.readlines() + f5.readlines() + f6.readlines()
    print ('Input Raw Data Success!')
    end_time = time.time()
    global log
    log.write('STEP 1 has took '+ str(end_time-start_time) +' seconds to run.\n')
    log.write('Length of Raw Data: '+ str(len(data)) +'.\n')
    return data

def Initialize2():
    start_time = time.time()
    f1=open("199801.txt","r")
    f2=open("199802.txt","r")
    f3=open("199803.txt","r")
    f4=open("199804.txt","r")
    f5=open("199805.txt","r")
    f6=open("199806.txt","r")
    data = f1.readlines() + f2.readlines() + f3.readlines() + f4.readlines() + f5.readlines() + f6.readlines()
    print ('=====================================')
    print ('Input Raw Data Success!')
    end_time = time.time()
    global log
    log.write('STEP 1 has took '+ str(end_time-start_time) +' seconds to run.\n')
    log.write('Length of Raw Data: '+ str(len(data)) +'.\n')
    return data

def PreProcess(data):
    start_time = time.time()
    while '\n' in data:
        data.remove('\n')
    data_time = []
    for line in data:
        line = line[23:]    #删除每一句前的时间
        data_time.append(line)
    data = data_time
    data_time = []
    del data_time
    data_hasentity = []     #删除无实体的句子，提高实体比例
    for line in data:
        traverse_line = {}
        for str in line:
            traverse_line[str] = 1#加速补丁，只遍历一次，20170903
        if ('[' in traverse_line) or ('nr' in traverse_line) or ('ns' in traverse_line) or ('nt' in traverse_line) or ('nz' in traverse_line):#人名补丁，20170902
            data_hasentity.append(line)
        else:
            continue
    data = data_hasentity
    print ('=====================================')
    print ('Data Pre-Process Success!')
    end_time = time.time()
    global log
    log.write('STEP 2 has took '+ str(end_time-start_time) +' seconds to run.\n')
    log.write('Length of Pre-Processed Data: '+ str(len(data)) +'.\n')
    return data

def JiebaPreparation(data):             #STEP 3 把语料库中人工标注的实体提取出来，既可以在分类器训练里用，还可以在Demo里的自定义分词字典用
    start_time = time.time()
    r1 = []
    for classic in data:
        classic_t = re.sub("[0-9\-\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\.\<\>\/\?\~\@\#\\\&\*\%]", "", classic)
        r1.append(classic_t)
    r2 = []
    for classic in r1:
        classic_t = classic.split('[')[1:]
        r2.append(classic_t)
    jieba_word_un = []
    JiebaPOS = []
    for classic in r2:
        if len(classic) == 0:
            continue
        for classic_t in classic:
            if not (']' in classic_t):
                continue
            jieba_temp = classic_t.split(']')
            jieba_word_un.append(jieba_temp[0])
            JiebaPOS.append(jieba_temp[1].split()[0])
    JiebaWords = []
    for classic in jieba_word_un:
        jieba_temp = re.sub("[A-Za-z\ ]", "", classic)
        JiebaWords.append(jieba_temp)
    print ('=====================================')
    print ('Classical Entity Extracted Successfully!')
    end_time = time.time()
    total_time = end_time-start_time
    global log
    log.write('STEP 3 has took '+ str(total_time) +' seconds to run.\n')
    log.write('Length of Classical Entity: '+ str(len(JiebaWords)) +'.\n')
    return JiebaWords, JiebaPOS, total_time


def WordsinData(data):              #STEP 4 构造输入集，输入的是整数，整数代表字典当中该词的位置
                                    #将人工词性标注语料库还原回分词库，这里是包括标点符号的
    start_time = time.time()
    w1=[]
    for l in data:
        l_t = re.sub("[0-9\[\]\-\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\.\<\>\?\~\@\#\&\*\%]", "", l)
        w1.append(l_t)
    words_in_data=[]
    pos_in_data=[]
    for l in w1:
        l_t = l.split()
        for l_tt in l_t:
            if len(l_tt) == 0:
                continue
            if len(l_tt.split('/')) < 2:
                continue
            w2_w = l_tt.split('/')[0]
            w2_l = l_tt.split('/')[1]
            if w2_l in ['nnt','nns','nnr','nnz']:
                w2_l = w2_l[1:3]# Classical Entity Patch, date: 20170903
            words_in_data.append(w2_w)
            pos_in_data.append(w2_l)
    JiebaWords, JiebaPOS, total_time = JiebaPreparation(data)
    words_in_data = words_in_data + JiebaWords
    pos_in_data = pos_in_data + JiebaPOS
    print ('=====================================')
    print ('Spilt Words Extracted Successfully!')
    end_time = time.time()
    global log
    log.write('STEP 4 has took '+ str(end_time-start_time-total_time) +' seconds to run.\n')
    log.write('Length of Spilt Words: '+ str(len(words_in_data)) +'.\n')
    print ('=====================================')
    return words_in_data, pos_in_data

def InputGenerator(data):            #STEP 5 生成输入数据
    ###################  5.1 生成Word to Index字典   ###################
    ###################         Word to Pos字典   ###################
    ###################         Jieba 自定义字典   ###################
    m1 = []
    MAPS_Word_to_Index = {}                     #Word -> Index
    MAPS_Word_to_POS = {}                       #Word -> Part of Speech
    Jieba = []
    Entity = set(['nt','ns','nz','j','jnt','jns','jnz','nr'])
    words_in_data, pos_in_data = WordsinData(data)
    start_time = time.time()
    i = 0
    for d in words_in_data:
        if d not in MAPS_Word_to_Index:
            m1.append(d)                                #首次出现的词才会进入字典
            if pos_in_data[i] in Entity:
                Jieba.append(d+' 50 '+ pos_in_data[i]+'\n')
            MAPS_Word_to_Index[d] = len(m1)             #将该词对应Index
            MAPS_Word_to_POS[d] = pos_in_data[i]
            if (i <= len(pos_in_data)-2 ) and (pos_in_data[i] == 'nr') and (pos_in_data[i+1] == 'nr'):#人名补丁，20170902
                FullName = words_in_data[i] + words_in_data[i+1]
                m1.append(FullName)
                MAPS_Word_to_Index[FullName] = len(m1)
                MAPS_Word_to_POS[FullName] = 'nr'
                Jieba.append(FullName+' 50 nr\n')
            i += 1
        else:
            i += 1

    #日期补丁
    for x in range(1,32):
        Jieba.append(str(x)+'日 20 t\n')
        m1.append(str(x))
        MAPS_Word_to_Index[str(x)] = len(m1)
        MAPS_Word_to_POS[str(x)] = 't'
    ########

    MAPS_Word_to_Index['*'] = len(m1) + 1        #词不在字典
    MAPS_Word_to_POS['*'] = 'unknown'                   #词不在字典


    timestamp = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    JiabaFilename = 'Jieba '+str(timestamp) +'.txt'
    '''
    import codecs
    f = codecs.open(JiabaFilename,"w","utf-8")
    for t in Jieba:
        f.write(t)
    f.close()
    '''
    f = open(JiabaFilename,"w")
    for t in Jieba:
        f.write(t)
    f.close()

    print ('Dictionary and Jieba Manual Generated Successfully!')
    end_time1 = time.time()
    global log
    log.write('STEP 5.1, MAPS Generator, has took '+ str(end_time1-start_time) +' seconds to run.\n')
    log.write ('Length of Dictionary:' + str(len(m1) + 1)+'.\n')
    log.write ('Length of Jieba Manual:' + str(len(Jieba) + 1)+'.\n')
    print ('----------------------------------------------------')
    ###################  5.2 Index到POS的字典   ###################
    MAPS_Index_to_Pos = {}
    for word,index in MAPS_Word_to_Index.items():
        MAPS_Index_to_Pos[index] = MAPS_Word_to_POS[word]
    print ('POS Dictionary Generated Successfully!')
    end_time2 = time.time()
    log.write('STEP 5.2, POS MAPS, has took '+ str(end_time2-end_time1) +' seconds to run.\n')
    print ('----------------------------------------------------')
    #################  5.3 产生输入数据集   #######################
    Input = [MAPS_Word_to_Index[x] for x in words_in_data]
    print ('Input Set Generated Successfully!')
    end_time3 = time.time()
    save_stamp = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    pickle.dump(MAPS_Word_to_Index, open('MAPS_Word_to_Index '+str(save_stamp) +'.pkl','wb'))
    pickle.dump(MAPS_Word_to_POS, open('MAPS_Word_to_POS '+str(save_stamp) +'.pkl','wb'))
    log.write('STEP 5.3, Input Generator, has took '+ str(end_time3-end_time2) +' seconds to run.\n')
    log.write ('Length of Input Set:' + str(len(Input))+'.\n')
    log.write('STEP 5 has took '+ str(end_time3-start_time) +' seconds to run.\n')
    print ('=====================================')

    return MAPS_Word_to_Index, MAPS_Word_to_POS, MAPS_Index_to_Pos, Input


def OutputGenerator(data):                                  # STEP 6 构造输出标签集
    ###################  6.1 为Input匹配标签
    MAPS_Word_to_Index, MAPS_Word_to_POS, MAPS_Index_to_Pos, Input = InputGenerator(data)
    max_Features = len(MAPS_Word_to_Index) + 1
    start_time = time.time()
    Input_Label = []
    for ind in Input:
        Input_Label.append(MAPS_Index_to_Pos[ind])
    end_time1 = time.time()
    global log
    log.write('STEP 6.1, Input Label Generator, has took '+ str(end_time1-start_time) +' seconds to run.\n')
    print ('----------------------------------------------------')
    ###################  6.2 把WordLabel用矩阵/向量表示
    Train_Label = []
    def Label_to_6Cata (l):
        switcher = {
            'nt': 1,                               #机构团体
            'ns': 2,                               #地点
            'nz': 3,                               #其它专有名词
            'j': 4,                                #简略词
            'jnt': 4,                              #简略词
            'jns': 4,                              #简略词
            'jnz': 4,                              #简略词
            'nr':5#人名补丁，20170902
        }
        return switcher.get(l, 0)
    for Label in Input_Label:
        Train_Label.append(Label_to_6Cata(Label))
    '''
    from numpy import array
    from matplotlib import pyplot
    def draw_hist(lenths):
        pyplot.hist(lenths,bins = 2)
        pyplot.xlabel('Catagory')
        pyplot.xlim(0,5)
        pyplot.ylabel('Frequency')
        pyplot.title('SEEEEEEEEEEEe')
        pyplot.show()
    aaa = array(Train_Label)
    draw_hist(aaa)
    '''
    print ('Output Set Generated Successfully!')
    end_time2 = time.time()
    log.write('STEP 6.2, Output Generator, has took '+ str(end_time2-end_time1) +' seconds to run.\n')
    log.write ('Length of Onput Set:' + str(len(Train_Label))+'.\n')
    print ('----------------------------------------------------')
    log.write('STEP 6, has took '+ str(end_time2-start_time) +' seconds to run.\n')
    print ('=====================================')

    return Input, Train_Label, max_Features, MAPS_Word_to_Index

def ModelBuilder1(data):                                     # Step 7 模型搭建
    Input, Train_Label, max_Features, MAPS_Word_to_Index = OutputGenerator(data)
    Train_Label_2dim = keras.utils.to_categorical(np.array(Train_Label), num_classes=6)
    x_train, x_test, y_train, y_test = tts(Input, Train_Label_2dim,train_size=0.8, random_state=1)
    model = Sequential()
    model.add(Embedding(max_Features, output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print ('Model Built Successfully!')
    print ('=====================================')
    global log
    log.write('Model Built Successfully!\n\n')
    log.close()
    return model, x_train, x_test, y_train, y_test, MAPS_Word_to_Index


def ModelBuilder2(data):                                     # Step 7 模型搭建
    Input, Train_Label, max_Features, MAPS_Word_to_Index = OutputGenerator(data)
    import numpy as np
    Input_3dim = np.array(Input).reshape(int(len(Input)/2),2,1)
    import keras
    Train_Label_2dim = keras.utils.to_categorical(np.array(Train_Label), num_classes=6)


    x_train, x_test, y_train, y_test = tts(Input_3dim, Train_Label_2dim,train_size=0.8, random_state=1)

    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

    model = Sequential()
    model.add(Conv1D(64, activation='relu', input_dim=1))
    model.add(Conv1D(64, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, activation='relu'))
    model.add(Conv1D(128, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print ('Model Built Successfully!')
    print ('=====================================')
    global log
    log.write('Model Built Successfully!\n\n')
    log.close()
    return model, x_train, x_test, y_train, y_test, MAPS_Word_to_Index


if __name__ == '__main__':

    data = Initialize()
    data = data[0:10000]
    data = PreProcess(data)
    model, x_train, x_test, y_train, y_test, MAPS_Word_to_Index = ModelBuilder1(data)
    model, x_train, x_test, y_train, y_test, MAPS_Word_to_Index = ModelBuilder2(data)

#1
    hist = model.fit(x_train, y_train, batch_size=512, epochs=1)
    hist = model.fit(x_train, y_train, batch_size=2048, epochs=5)
    score = model.evaluate(x_test, y_test, batch_size=128)

#2
    hist = model.fit(x_train, y_train, batch_size=128, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=16)


    print ('History: ' +str(hist.history))
    print ('Score: ' +str(score))
    stamp = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    fitting_log = open('/log/Fitting_log '+str(stamp)+'.txt',"w")
    fitting_log.write('History: ' +str(hist.history) + '\nScore: ' +str(score) +'\n')
    fitting_log.close()
    model.save('Entry_Model '+str(stamp)+'.h5')
    #model = load_model('my_model.h5')


#model, x_train, x_test, y_train, y_test, MAPS_Word_to_Index = n.ModelBuilder1(n.PreProcess(n.Initialize2()))


    #################### Step 7 预测

    import jieba
    HTTP_input = '今天，我们召开一个深度贫困地区脱贫攻坚座谈会，研究如何做好深度贫困地区脱贫攻坚工作。攻克深度贫困堡垒，是打赢脱贫攻坚战必须完成的任务，全党同志务必共同努力。'
    HTTP_input = '在这一年中，中国的改革开放和现代化建设继续向前迈进。国民经济保持了“高增长、低通胀”的良好发展态势。农业生产再次获得好的收成，企业改革继续深化，人民生活进一步改善。对外经济技术合作与交流不断扩大。民主法制建设、精神文明建设和其他各项事业都有新的进展。我们十分关注最近一个时期一些国家和地区发生的金融风波，我们相信通过这些国家和地区的努力以及有关的国际合作，情况会逐步得到缓解。总的来说，中国改革和发展的全局继续保持了稳定。'
    sentence = '8月31日至9月1日，省委常委、宣传部部长慎海雄到肇庆调研宣传思想文化工作，强调要广泛深入宣传贯彻习近平总书记“7·26”重要讲话和对广东工作重要批示精神，抓牢抓实意识形态工作责任，全力推动精神文明建设上水平上台阶，以昂扬向上、团结奋进的思想舆论氛围迎接党的十九大。'
    sentence = '中华人民共和国中央政府人民大会堂位于中国北京市天安门广场西侧，西长安街南侧。人民大会堂坐西朝东，南北长336米，东西宽206米，高46.5米，占地面积15万平方米，建筑面积17.18万平方米。人民大会堂是中国全国人民代表大会开会地和全国人民代表大会常务委员会的办公场所，是党、国家和各人民团体举行政治活动的重要场所，也是中国党和国家领导人和人民群众举行政治、外交、文化活动的场所。人民大会堂每年举行的全国人民代表大会、中国人民政治协商会议以及五年一届的中国共产党全国代表大会也在此召开。'
    sentence='虽然仍有抵触情绪，但是，近日，爱尔兰还是向欧盟低头了，决定“被动地临时地接收苹果公司补缴的152亿美元税款。”这是税收史上一大奇观。欧盟认定，苹果公司在爱尔兰偷税漏税，但是，爱尔兰认为，苹果公司没这么干，且长时间拒绝接受这笔高达2015年度预算26%的补缴款。苹果公司不是独行侠。数据显示，美国十大企业巨头中有九家均在爱尔兰设立了分公司。这些公司在爱尔兰、荷兰、百慕大、维京群岛设立了令人眼花缭乱的子公司，犹如“蚁穴”，分工明确、各司其职、将一笔笔收入的税率过虑至最低。最新数据显示，截至2015年，美国企业在海外囤积的现金总额增至2.5万亿美元。美国50家最大的跨国企业，通过1751家隐秘的离岸公司“藏了”1.6万亿美元，“每年让美国遭受了1350亿美元的税收损失”。美国总统特朗普誓言，通过税改，“缓解”这一局面，最新消息显示，白宫与美国国会已就税改关键性条款达成一致。不过，这只是税改路上的第一步，前路漫漫。荷兰监管文件显示，2015年，谷歌向百慕达的一家空壳公司转移149亿欧元，避税额达36亿美元。谷歌是透过一个被称为“双层爱尔兰夹荷兰三明治”（Double Irish with a Dutch Sandwich）的避税方式进行操作。2015年，谷歌将大部分国际广告收入转移给谷歌爱尔兰有限公司，而后流向谷歌荷兰分公司。荷兰分公司再将这笔款项转移到谷歌在爱尔兰的另一家分公司——谷歌爱尔兰控股有限公司。而这家公司注册地是百慕达。由于整个流程通过两家爱尔兰公司和一家荷兰公司实现，因此该手法称为“双层爱尔兰夹荷兰三明治”模式。与谷歌公司一样，苹果公司也采用的也是典型的“双层爱尔兰夹荷兰三明治”模式。苹果公司在英属维尔京群岛成立苹果国际运营公司，随后，又在爱尔兰设立苹果国际运营公司子公司——苹果国际销售公司。美国以外地区的所有销售收入进入苹果销售公司，然后，再将利润转入苹果国际运营公司。当然，中间会通过设在荷兰的公司过一下手。调查结果显示，2011年9月30日到2012年9月30日，苹果公司在美国境外所得利润是368亿美元，而它所缴纳的企业所得税仅有7.13亿美元，也就是说，苹果公司美国境外所得税率仅有1.9%。2016年8月,更是有数据指出苹果公司2014年在美国以外的多数地区每获得百万美元收入只需缴纳50美元的税款，税率仅0.005%。'
    jieba.load_userdict('Jieba 2017-09-02 00-02-59')
    HTTP_input = jieba.lcut(HTTP_input)
    HTTP_input_index = []

#    for x in input:
#        if x.encode("utf-8") in MAPS_Word:
#            HTTP_input_index.append(MAPS_Word_to_Index[x.encode("utf-8")])
#        else:
#            HTTP_input_index.append(len(MAPS_Word_to_Index))

    for x in HTTP_input:
        if x in MAPS_Word_to_Index:
            HTTP_input_index.append(MAPS_Word_to_Index[x])
        else:
            HTTP_input_index.append(len(MAPS_Word_to_Index))
    z = model.predict(HTTP_input_index).tolist()
    word_prob = []
    for line in z:
        if int(line[0]) == 0:
            word_prob.append('y')
        else:
            word_prob.append('n')
    i = 0
    for flag in word_prob:
        if flag == 'y':
            print(HTTP_input[i])
            i += 1
        else:
            i += 1

#
#data = n.Initialize2()
#data = n.PreProcess(data)
#model, x_train, x_test, y_train, y_test, MAPS_Word_to_Index = n.ModelBuilder1(data)
#hist = model.fit(x_train, y_train, batch_size=1024, epochs=5)
#score = model.evaluate(x_test, y_test, batch_size=128)