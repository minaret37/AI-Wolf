#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
from numpy.random import *
import six

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L
# cudaがある環境ではコメントアウト解除
from chainer import cuda


import heapq

import read_5v4
import read_15v4
import random
from collections import deque
import csv
import time
import copy

class LSTM5(chainer.Chain):
    def __init__(self, n_st , n_units, n_out):
        super(LSTM5, self).__init__(
            embed=L.EmbedID(n_st, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            #l3=L.LSTM(n_units, n_units),
            #l4=L.LSTM(n_units, n_units),
            #l3=L.Linear(n_units, n_units),
            l5=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h = self.embed(x)
        h = self.l1(h)
        h = self.l2(h)
        #h = self.l3(h)
        #y = self.l4(h)
        y = self.l5(h)
        return y

    def about_Linear(self):
        #lin = "Em"#embed
        #lin = "Adam"#
        #lin = "AdamSoft"
        lin = "SGDmean"
        lin += "LS"#LSTM
        lin += "LS"#LSTM
        #lin += "LS"#LSTM
        #lin += "LS"#LSTM
        #lin += "LS"#LSTM
        lin += "Li"#Linear
        #LSLSLi,LSLSLSLS,LSLS,
        return lin

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        #self.l3.reset_state()
        #self.l4.reset_state()

class LSTM15(chainer.Chain):
    def __init__(self, n_st , n_units, n_out):
        super(LSTM15, self).__init__(
            embed=L.EmbedID(n_st, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            #l3=L.LSTM(n_units, n_units),
            #l4=L.LSTM(n_units, n_units),
            #l3=L.Linear(n_units, n_units),
            #l5=L.LSTM(n_units, n_out),
            l5=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h = self.embed(x)
        h = self.l1(h)
        h = self.l2(h)
        #h = self.l3(h)
        #y = self.l4(h)
        #y = F.softmax(self.l5(h))
        y = self.l5(h)
        #print(y.data)
        return y

    def about_Linear(self):
        #lin = "Em"#embed
        #lin = "AdamMean"
        lin = "SGDMean"
        #lin = "AdaDeltaMean"
        #lin += "Soft"#LSTM
        lin += "LS"#LSTM
        lin += "LS"#LSTM
        #lin += "LS"#LSTM
        #lin += "LS"#LSTM
        #lin += "LS"#LSTM
        #lin += "Li"#Linear
        lin += "Li"#Linear
        return lin

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        #self.l3.reset_state()
        #self.l4.reset_state()
        #self.l5.reset_state()

class Agent():
    def __init__(self, n_st , n_units, n_out):
        if n_out == 5:
            self.lstm = LSTM5(n_st , n_units, n_out)
        if n_out == 15:
            self.lstm = LSTM15(n_st , n_units, n_out)
        # このようにすることで分類タスクを簡単にかける
        # 詳しくはドキュメントを読むとよい

        gpu_device = 0
        cuda.get_device(gpu_device).use()
        self.lstm.to_gpu(gpu_device)
        #self.model = L.Classifier(self.lstm)
        self.model = L.Classifier(self.lstm,lossfun=F.mean_squared_error)
        #self.model = L.Classifier(self.lstm,lossfun=F.softmax_cross_entropy)
        self.model.compute_accuracy = False

        #self.optimizer = optimizers.Adam()
        self.optimizer = optimizers.SGD(lr=0.001)
        #self.optimizer = optimizers.AdaDelta(rho=0.9)
        self.optimizer.setup(self.model)
        #weight_decay = 0.001
        #self.optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
        #grad_clip = 5
        #self.optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
        self.loss = 0
        self.total_loss = 0

    def train_lstm(self,x,t):
        self.loss = 0
        self.total_loss = 0
        #print(x.data,t.data)
        self.loss = self.model(x, t)  # lossの計算
        #self.loss = self.model(x)  # lossの計算
        #print(self.loss)
        self.total_loss += self.loss
        self.model.zerograds()#勾配をゼロ初期化
        self.total_loss.backward()#損失を使って誤差逆電波
        self.total_loss.unchain_backward()#誤差逆伝播した実数や関数へのreferenceを削除
        self.optimizer.update()#最適化ルーチン実行

    def save_model(self, model_dir):
        serializers.save_npz(model_dir + "_model.npz", self.model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir + "_model.npz", self.model)


def main():
    #np.random.seed(seed=32)
    episode_multi = 10
    player_num = 15
    if player_num == 5:
        episode_train = 100000
        episode_test = 10
        view = 50000
        input_day = 2 #何日までのデータか
        train_file = "gat2017log05"
        #train_file = "cedec2017"
        test_file = "gat2017log05"
        test_file = "cedec2018"
        test_file = "cedec2017"
        n_st = 86
        n_st = 111
        n_units = 800
    elif player_num == 15:
        episode_train = 100000
        episode_test = 10
        view = 50000
        input_day = 7 #何日までのデータか Talkする最後の日にち
        train_file = "gat2017log15"
        train_file = "cedec2017"
        #train_file = "cedec2018"
        test_file = "gat2017log15"
        test_file = "cedec2017"
        #test_file = "cedec2018"
        n_st = 571
        #n_st = 796
        #n_st = 1021
        #n_st = 1246
        n_units = 800
    n_out = player_num
    agent = Agent(n_st , n_units, n_out)
    xp = cuda.cupy
    load = False
    #load = True
    #random_agent = Trueinput
    random_agent = False
    #same_name = train_file+str(player_num)+"st"+str(n_st)+"hid"+str(n_units)+agent.lstm.about_Linear()
    same_name = str(player_num)+"st"+str(n_st)+"hid"+str(n_units)+train_file
    if random_agent:
        same_name = "detail_v"+str(player_num)
        episode_train = 1

    model_path = "./model/"+same_name+str(episode_train*episode_multi)
    csv_name = "data/"+same_name+test_file+".csv"
    #agent.load_model(model_path)
    train_results = []#csvに記録するための配列



    when = ["voting","morning"]
    true_or_false = ["F","T"]

    t1 = time.time()#処理前の時刻
    for train_test in range(episode_multi):
        if load:
            print("load",model_path)
            agent.load_model(model_path)
        else:
            t2 = time.time()#処理後の時刻
            time_m=(t2-t1)/60#minits
            time_h=time_m/60#hour
            if train_test!=0:
                print("time",int(time_h),"h",int(time_m%60),"m /",train_test,"loop =",round(time_m/(train_test), 1),"m")
            print(same_name)
            if player_num == 5:
                for epi in range(episode_train):
                    while(True):
                        if test_file == "gat2017log05":
                            log_folder = random.randint(0,799)
                            log_file = random.randint(0,99)
                            file_name = "./"+train_file+"/"+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
                        elif test_file == "cedec2018":
                            log_folder = random.randint(0,200)
                            log_file = random.randint(0,99)
                            file_name = "./"+train_file+"/"+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
                        elif test_file == "cedec2017":
                            log_file = random.randint(0,147999)
                            log_folder = int(log_file/100)+1
                            file_name = "./"+test_file+"/"+str(log_folder).zfill(3)+'_'+str(log_file).zfill(4)+'.log'
                        if read_5v4.distinction(file_name)==player_num:
                            talk_id_input,  true_pos, last_day, death= read_5v4.read_file(file_name,input_day,n_st)
                            if last_day > input_day:#last_dayは終了した日　5人人狼で2日目襲撃・投票なら3日目に終わる
                                break
                    talk_id_input, true_pos, last_day, death= read_5v4.read_file(file_name,100,n_st)
                    talk_id_input =  [int(s) for s in talk_id_input]#[2,30,,,,0,750]
                    agent.lstm.reset_state()  # 前の系列の影響がなくなるようにリセット
                    tp = [np.argmax(true_pos)]
                    for turn in range(len(talk_id_input)):
                        #x = chainer.Variable(np.asarray([talk_id_input[turn]]).astype(np.int32))
                        #t = chainer.Variable(np.asarray(tp).astype(np.int32))
                        x = chainer.Variable(xp.asarray([talk_id_input[turn]]).astype(xp.int32))
                        #t = chainer.Variable(xp.asarray(tp).astype(xp.int32))
                        t = chainer.Variable(xp.asarray([true_pos]).astype(xp.float32))
                        agent.train_lstm(x,t)
                    if epi % int(view) == 0 and  epi!=0:
                        #print("epi",epi,"loss",round(agent.total_loss, 3),"day",last_day,"talk",len(talk_id_input))
                        print("epi",epi,"loss",agent.total_loss.data,"day",last_day,"talk",len(talk_id_input))
                        agent.save_model(model_path)
            elif player_num == 15:
                for epi in range(episode_train):
                    while(True):
                        if test_file == "gat2017log15":
                            log_folder = random.randint(0,799)
                            log_file = random.randint(0,99)
                            file_name = "./"+train_file+"/"+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
                        elif test_file == "cedec2017":
                            #log_file = random.randint(0,147999)
                            log_file = random.randint(0,99999)
                            log_folder = int(log_file/100)+1
                            file_name = "./"+test_file+"/"+str(log_folder).zfill(3)+'_'+str(log_file).zfill(4)+'.log'
                        if read_15v4.distinction(file_name)==player_num:
                            talk_id_input,  true_pos, last_day, death= read_15v4.read_file(file_name,input_day,n_st)
                            if last_day > input_day:#last_dayは終了した日　5人人狼で2日目襲撃・投票なら3日目に終わる
                                break
                    talk_id_input, true_pos, last_day, death= read_15v4.read_file(file_name,100,n_st)
                    talk_id_input =  [int(s) for s in talk_id_input]#[2,30,,,,0,750]
                    #print("talk_id_input",talk_id_input)
                    agent.lstm.reset_state()  # 前の系列の影響がなくなるようにリセット
                    tp = [np.argmax(true_pos)]
                    for turn in range(len(talk_id_input)):
                        #x = chainer.Variable(np.asarray([talk_id_input[turn]]).astype(np.int32))
                        #t = chainer.Variable(np.asarray(tp).astype(np.int32))
                        x = chainer.Variable(xp.asarray([talk_id_input[turn]]).astype(xp.int32))
                        #t = chainer.Variable(xp.asarray(tp).astype(xp.int32))
                        t = chainer.Variable(xp.asarray([true_pos]).astype(xp.float32))
                        agent.train_lstm(x,t)
                    if epi % int(view) == 0 and  epi!=0:
                        #print("epi",epi,"loss",round(agent.total_loss, 3),"day",last_day,"talk",len(talk_id_input))
                        print("epi",epi,"loss",agent.total_loss.data,"day",last_day,"talk",len(talk_id_input))
                        agent.save_model(model_path)



        print("test",train_file,"->",test_file)
        accuracy_rate = [[0]*3 for acc in range(input_day*2)]
        accuracy_rate_death = [[0] for acc in range(input_day*2)]
        for epi in range(episode_test):
            if player_num == 5:
                while(True):
                    if test_file == "gat2017log05":
                        log_folder = random.randint(800,999)
                        log_file = random.randint(0,99)
                        file_name = "./"+test_file+"/"+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
                    elif test_file == "cedec2018":
                        log_folder = random.randint(1,200)
                        log_file = random.randint(0,99)
                        file_name = "./"+test_file+"/"+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
                    elif test_file == "cedec2017":
                        log_file = random.randint(0,147999)
                        log_folder = int(log_file/100)+1
                        file_name = "./"+test_file+"/"+str(log_folder).zfill(3)+'_'+str(log_file).zfill(4)+'.log'

                    if read_5v4.distinction(file_name)==player_num:
                        talk_id_input,  true_pos, last_day, death= read_5v4.read_file(file_name,input_day,n_st)
                        if last_day > input_day:#last_dayは終了した日　5人人狼で2日目襲撃・投票なら3日目に終わる
                            break
            elif player_num == 15:
                while(True):
                    if test_file == "gat2017log15":
                        log_folder = random.randint(800,999)
                        log_file = random.randint(0,99)
                        file_name = "./"+test_file+"/"+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
                    elif test_file == "cedec2018":
                        log_folder = random.randint(101,201)
                        log_file = random.randint(0,99)
                        file_name = "./"+test_file+"/"+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
                    elif test_file == "cedec2017":
                        #log_file = random.randint(0,147999)
                        log_file = random.randint(100000,147999)
                        log_folder = int(log_file/100)+1
                        file_name = "./"+test_file+"/"+str(log_folder).zfill(3)+'_'+str(log_file).zfill(4)+'.log'
                    if read_15v4.distinction(file_name)==player_num:
                        talk_id_input,  true_pos, last_day, death= read_15v4.read_file(file_name,input_day,n_st)
                        if last_day > input_day:#last_dayは終了した日　5人人狼で2日目襲撃・投票なら3日目に終わる
                            break

            talk_id_input =  [int(s) for s in talk_id_input]#[2,30,,,,0,750]
            #talk_id_onehot = np.zeros((len(talk_id_input),n_st), dtype = int)# [[0,1,0,0,0][]]
            answers = []
            agent.lstm.reset_state()  # 前の系列の影響がなくなるようにリセット
            vote = True
            if player_num == 5:
                ans = [[0]*5]
            elif player_num == 15:
                ans = [[0]*15]
            #print("start",talk_id_input)
            for turn in range(len(talk_id_input)):
                #x = chainer.Variable(np.asarray([talk_id_input[turn]]).astype(np.int32))
                x = chainer.Variable(xp.asarray([talk_id_input[turn]]).astype(xp.int32))
                if n_st == 86 and 6 <= talk_id_input[turn] and talk_id_input[turn] <= 10:#before execute
                        ans_slice = copy.copy(ans[0])
                        answers.append(ans_slice)
                        #print("vote",len(answers))
                if n_st == 571 or  n_st == 1021 :
                    if player_num == 15 and 16 <= talk_id_input[turn] and talk_id_input[turn] <= 30:#before execute
                        ans_slice = copy.copy(ans[0])
                        answers.append(ans_slice)
                if n_st == 111 and 86 <= talk_id_input[turn] and talk_id_input[turn] <= 110 and vote:#vote
                    ans_slice = copy.copy(ans[0])
                    answers.append(ans_slice)
                    vote = False
                    #print("vote",len(answers))
                if n_st == 796 and 571 <= talk_id_input[turn] and talk_id_input[turn] <= 795 and vote:#vote
                    ans_slice = copy.copy(ans[0])
                    answers.append(ans_slice)
                    vote = False
                if n_st == 1246 and 1021 <= talk_id_input[turn] and talk_id_input[turn] <= 1245 and vote:#vote
                    ans_slice = copy.copy(ans[0])
                    answers.append(ans_slice)
                    vote = False
                #"""
                #print("l",talk_id_input[turn])
                ans = agent.lstm(x).data
                #ans = [[random.randint(0,99) for ran in range(player_num)]]#random_agentがtrueのとき用だけど、処理が重くなるのでif文にしない
                if player_num == 5 and 1 <= talk_id_input[turn] and talk_id_input[turn] <= 5:#after attack
                    ans_slice = copy.copy(ans[0])
                    answers.append(ans_slice)
                    #print("att",len(answers))

                    vote = True
                    #print("copy")
                elif player_num == 15 and 1 <= talk_id_input[turn] and talk_id_input[turn] <= 15:#after attack
                    ans_slice = copy.copy(ans[0])
                    answers.append(ans_slice)
                    vote = True
            # talk -> copy(voting) ->  vote -> execute -> attack -> copy(morning) -> talk の順
            ans_slice = copy.copy(ans[0])
            answers.append(ans_slice)#最終日　投票時の出力

            cls1 = []
            cls2 = []
            cls3 = []
            cls_death = []
            score1 = []
            score2 = []
            score3 = []
            score_death = []
            death_list = []

            #print("----------------------",len(answers))
            for ans_i in range(len(answers)):
                if player_num == 5:
                    answer_day = int((ans_i+1)/2)+1
                    ans_slice_death = copy.copy(answers[ans_i])
                    cls1.append(np.argmax(answers[ans_i]))
                    if true_pos[int(cls1[ans_i])] == 1:
                        accuracy_rate[ans_i][0] += 1
                        score1.append(1)
                    else:
                        score1.append(0)

                    if answer_day >= 2:# day-1は朝の情報＋初期状態で１つ多いから減らす。　>= 2は２日目から情報入手の意味
                        for i in range(len(death[answer_day])):#死人除いた人狼推定
                            ans_slice_death[death[answer_day][i]-1] = -100#1~5になっているからdeath[day-1][i]-1

                    cls_death.append(np.argmax(ans_slice_death))
                    if true_pos[int(cls_death[ans_i])] == 1:
                        accuracy_rate_death[ans_i][0] += 1
                        score_death.append(1)
                    else:
                        score_death.append(0)
                    if epi%(episode_test/2) == 0:
                        print(epi,"day",answer_day, when[ans_i%2],": cls",cls1[ans_i],",",true_or_false[score1[ans_i]],":",cls_death[ans_i],",",true_or_false[score_death[ans_i]],death[answer_day])

                elif player_num == 15:
                    answer_day = int((ans_i+1)/2)+1
                    ans_slice_death = copy.copy(answers[ans_i])
                    cls1.append(np.argmax(answers[ans_i]))
                    answers[ans_i][cls1[ans_i]] = -100
                    if true_pos[int(cls1[ans_i])] == 1:
                        accuracy_rate[ans_i][0] += 1
                        score1.append(1)
                    else:
                        score1.append(0)
                    cls2.append(np.argmax(answers[ans_i]))
                    answers[ans_i][cls2[ans_i]] = -100
                    if true_pos[int(cls2[ans_i])] == 1:
                        accuracy_rate[ans_i][1] += 1
                        score2.append(1)
                    else:
                        score2.append(0)
                    cls3.append(np.argmax(answers[ans_i]))
                    if true_pos[int(cls3[ans_i])] == 1:
                        accuracy_rate[ans_i][2] += 1
                        score3.append(1)
                    else:
                        score3.append(0)

                    if answer_day >= 2 and  len(death)!= answer_day:# day-1は朝の情報＋初期状態で１つ多いから減らす。　>= 2は２日目から情報入手の意味
                        #print(len(death),answer_day)
                        #print(death)
                        for i in range(len(death[answer_day])):#死人除いた人狼推定
                            ans_slice_death[death[answer_day][i]-1] = -100#1~5になっているからdeath[day-1][i]-1

                    cls_death.append(np.argmax(ans_slice_death))
                    if true_pos[int(cls_death[ans_i])] == 1:
                        accuracy_rate_death[ans_i][0] += 1
                        score_death.append(1)
                    else:
                        score_death.append(0)
                    if epi%(episode_test/2) == 0:
                        print(epi,"day",answer_day, when[ans_i%2],": cls",cls1[ans_i],cls2[ans_i],cls3[ans_i],",",true_or_false[score1[ans_i]],true_or_false[score2[ans_i]],true_or_false[score3[ans_i]],":",cls_death[ans_i],",",true_or_false[score_death[ans_i]])
                        #print(death[answer_day])


        if player_num == 5:
            for ans_i in range(len(answers)):
                answer_day = int((ans_i+1)/2)+1
                print ("day", answer_day,when[ans_i%2], ": ",accuracy_rate[ans_i][0],"/",episode_test,":",accuracy_rate_death[ans_i][0],"/",episode_test)
                train_result = ["train_test,str(answer_day)+when[ans_i%2],accuracy_rate[ans_i][0],accuracy_rate_death[ans_i][0],int(episode_test),ans_i", test_file,str(answer_day)+when[ans_i%2],accuracy_rate[ans_i][0],accuracy_rate_death[ans_i][0],int(episode_test),ans_i]
                train_results.append(train_result)
        elif player_num == 15:
            for ans_i in range(len(answers)):
                answer_day = int((ans_i+1)/2)+1
                print ("day", answer_day,when[ans_i%2], ": ",accuracy_rate[ans_i][0],accuracy_rate[ans_i][1],accuracy_rate[ans_i][2],"/",episode_test,":",accuracy_rate_death[ans_i][0],"/",episode_test)
                train_result = ["train_test,str(answer_day)+when[ans_i%2],accuracy_rate[ans_i][0],accuracy_rate_death[ans_i][0],int(episode_test),ans_i", test_file,str(answer_day)+when[ans_i%2],accuracy_rate[ans_i][0],accuracy_rate[ans_i][1],accuracy_rate[ans_i][2],accuracy_rate_death[ans_i][0],int(episode_test),ans_i]
                train_results.append(train_result)

        with open(csv_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(train_results)


if __name__ == "__main__":
    main()
