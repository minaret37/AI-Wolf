#!/usr/bin/env python
# coding:utf-8
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, optimizers, Variable, serializers

import heapq

import read_village5
import random
from collections import deque
import csv

class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        # クラスの初期化
        # :param in_size: 入力層のサイズ
        # :param hidden_size: 隠れ層のサイズ
        # :param out_size: 出力層のサイズ
        super(LSTM, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh_x = L.Linear(hidden_size,  4 * hidden_size),
            hh_h = L.Linear(hidden_size,  4 * hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )
        self.hidden_size = hidden_size

    def __call__(self, x, t=None, train=False):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        # :param t: 正解の予測値
        # :param train: 学習かどうか
        # :return: 計算した損失 or 予測値
        if self.h is None:
            self.h = Variable(np.zeros((x.shape[0], self.hidden_size), dtype=np.float32))
            self.c = Variable(np.zeros((x.shape[0], self.hidden_size), dtype=np.float32))
        x = Variable(x)
        if train:
            t = Variable(t)
        h = self.xh(x)
        h = self.hh_x(h) + self.hh_h(self.h)
        self.c, self.h = F.lstm(self.c, h)
        y = self.hy(self.h)
        #y2 = F.softmax(y)
        #y2 = F.relu(y)
        if train:
            #多ラベル分類での損失関数は、別にbinary_crossentropyでなくてはいけないというわけではなく、
            #mean_squared_errorでもOKなはずです。
            return F.mean_squared_error(y, t)
        else:
            return y.data

    def reset(self):
        # 勾配の初期化とメモリの初期化
        self.zerograds()
        self.h = None
        self.c = None

class Agent():
    def __init__(self, n_st, n_act):
        self.n_act = n_act
        self.hidden_size = 800
        self.model = LSTM(in_size=n_st, hidden_size=self.hidden_size, out_size=n_act)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.loss = 0
        self.total_loss = 0

    def train_lstm(self,x,t):
        self.loss = 0
        self.total_loss = 0
        self.model.reset() # 勾配とメモリの初期化
        self.loss += self.model(x, t, train=True)
        self.loss.backward()
        self.loss.unchain_backward()
        self.total_loss += self.loss.data
        self.optimizer.update()

    def save_model(self, model_dir):
        serializers.save_npz(model_dir + "_model.npz", self.model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir + "_model.npz", self.model)

def main():
    np.random.seed(seed=32)
    episode_train = 1000
    episode_test = 1000
    player_num = 5
    n_st = 181
    n_out = player_num
    input_day = 3 #何日までのデータか
    agent = Agent(n_st, n_out)
    model_path = "./model/"+"train_cedec2017"
    #agent.load_model(model_path)
    train_results = []

    print("start")
    for epi in range(episode_train):

        log_folder = random.randint(0,799)
        log_file = random.randint(0,99)
        file_name = './gat2017log05/'+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
        talk_id_input, true_pos, last_day, alive= read_village5.read_file(file_name,input_day)
        talk_id_input =  [int(s) for s in talk_id_input]#[2,30,,,,0,750]
        talk_id_onehot = np.zeros((len(talk_id_input),n_st), dtype=np.float32)# [[0,1,0,0,0][]]
        #print(true_pos)

        for turn in range(len(talk_id_input)):
            talk_id_onehot[turn][talk_id_input[turn]] = 1
            x = np.array([talk_id_onehot[turn]], dtype=np.float32)#与えるベクトル　各プレイヤーの行動
            t = np.array([true_pos], dtype=np.float32)#正解ベクトル　プレイヤーのID
            agent.train_lstm(x,t)
            """
            if turn > 10:
                for i in range(2):
                    x = np.array([talk_id_onehot[turn]], dtype=np.float32)#与えるベクトル　各プレイヤーの行動
                    agent.train_lstm(x,t)
            """
        if epi % 1 == 0:
            print("epi",epi,"loss",agent.total_loss,", last_day",last_day,"talk",len(talk_id_input))
            #agent.save_model(model_path)

    for day in range(1,input_day):
        accuracy_rate = [0]*4
        for epi in range(episode_test):
            while(True):
                log_folder = random.randint(800,999)
                log_file = random.randint(0,99)
                file_name = './gat2017log05/'+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
                talk_id_input,  true_pos, last_day, alive= read_village5.read_file(file_name,day)
                if last_day > day:#last_dayは終了した日　5人人狼で2日目襲撃・投票なら3日目に終わる
                    break
            talk_id_input =  [int(s) for s in talk_id_input]#[2,30,,,,0,750]
            talk_id_onehot = np.zeros((len(talk_id_input),n_st), dtype = int)# [[0,1,0,0,0][]]
            for turn in range(len(talk_id_input)):
                talk_id_onehot[turn][talk_id_input[turn]] = 1
                x = np.array([talk_id_onehot[turn]], dtype=np.float32)#与えるベクトル　各プレイヤーの行動
                t = np.array([true_pos], dtype=np.float32)#正解ベクトル　プレイヤーのID
                agent.train_lstm(x,t)
            #print(agent.model(x, t, train=False))
            ans = agent.model(x, t, train=False)
            #print(alive,ans*alive)
            #print(int(ans[0][0]*10),int(ans[0][1]*10),int(ans[0][2]*10),int(ans[0][3]*10),int(ans[0][4]*10),t)
            cls = []
            correct_or = []
            cls.append(np.argmax(ans[0]))
            ans[0][cls[0]] = 0
            if int(t[0][cls[0]]) == 1:
                accuracy_rate[0] += 1
                correct_or.append("correct")
            else:
                correct_or.append("incorrect")
            if epi%(episode_test/10) == 0:
                print("test folder", epi,": cls",cls[0],"is",correct_or,last_day)
        print ("day", day, ": ",accuracy_rate[0],"/",episode_test)
        train_result = [day,last_day,accuracy_rate[0],int(episode_test)]
        train_results.append(train_result)
        agent.save_model(model_path)
        csv_name = "data/cedec2017"+str(episode_train)+"test"+str(episode_test)+"input_day"+str(input_day)+".csv"
        with open(csv_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(train_results)
        print("save model and csv")

if __name__ == "__main__":
    main()
