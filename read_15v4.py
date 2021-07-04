#cd C:\wolf
import numpy as np
import re

def distinction(file_name):
    file = open(file_name, 'r')  #読み込みモードでオープン
    #string = file.read()      #readですべて読み込む
    string_list = file.readlines()     #readlinesでリストとして読み込む
    if len(string_list)>300:
        ret = 15
    else:
        ret = 5
    return ret

def read_file(file_name,input_day,n_st):
    file = open(file_name, 'r')  #読み込みモードでオープン
    #string = file.read()      #readですべて読み込む
    string_list = file.readlines()     #readlinesでリストとして読み込む

    player_num = 15
    #day = 2
    position = {"VILLAGER":0,"SEER":1,"POSSESSED":2,"WEREWOLF":3,"BODYGUARD":4,"MEDIUM":5}
    true_pos = np.zeros((player_num,len(position)), dtype = int)#90

    last_day = 0
    talk_id_input = []
    daily_death = []
    death = []

    for string in string_list:
        string = string.strip('\n')#不要な文字削除
        string = string.replace(' ', ',')
        string = string.replace('(', ',')
        string = string.replace(')', ',')
        text_data = string.split(',')#[日にち　状態　割り当てNo　職業　AI名]になる

        for d in range(len(text_data)):
            #Agent01~を消す作業
            for no in range(1,player_num+1):
                str_no = str(no)
                target_str = 'Agent['+ str(no).zfill(2)
                text_data[d] = text_data[d].replace(target_str, str(no))
                text_data[d] = text_data[d].replace(']', '')
        #print(text_data)

        #教師データ　作成
        if  text_data[0]=='0' and text_data[1] == 'status':#初日の役割欄参照
            #引数は　エージェントNo　役割No
            true_pos[int(text_data[2])-1][position[text_data[3]]] +=1

        talk_id=0
        #入力データ　作成 15
        if int(text_data[0]) > last_day:#last_dayは終了した日時　会話した日の次の日
            last_day = int(text_data[0])
        if int(text_data[0])<input_day:#一日前の情報まで
            if  text_data[1] == 'attack':
                talk_id = int(text_data[2])#1~15
                daily_death.append(int(text_data[2]))
            if  text_data[1] == 'execute':
                talk_id = 15 + int(text_data[2])#16~30
                daily_death.append(int(text_data[2]))
                #print("attack",input_day,int(text_data[0]))
            if  text_data[1] == 'vote':
                if n_st == 796:
                    talk_id = 570 + int(text_data[2])*int(text_data[3])
                if n_st == 1246:
                    talk_id = 1020 + int(text_data[2])*int(text_data[3])
                #print("vote",input_day, int(text_data[2]),int(text_data[3]),talk_id)

        if int(text_data[0])<=input_day:#日数
            #input_day=7のとき７日目まで会話、８日目結果発表、６日目までの処刑・襲撃情報
            if  text_data[1] == 'talk':
                if  text_data[5] == 'COMINGOUT':
                    talk_id = 30 + int(text_data[4])*(position[text_data[7]])#6*15
                #"""
                if  text_data[5] == 'DIVINED' and text_data[7] == 'HUMAN' :
                    talk_id = 120 + int(text_data[4])*int(text_data[6])
                if  text_data[5] == 'DIVINED' and text_data[7] == 'WEREWOLF' :
                    talk_id = 345 + int(text_data[4])*int(text_data[6])
                if  text_data[5] == 'IDENTIFIED' and text_data[7] == 'HUMAN':
                    if n_st == 571 or  n_st == 796 :
                        talk_id = 120 + int(text_data[4])*int(text_data[6])
                    if n_st == 1021 or  n_st == 1246 :
                        talk_id = 570 + int(text_data[4])*int(text_data[6])
                if  text_data[5] == 'IDENTIFIED' and text_data[7] == 'WEREWOLF':
                    if n_st == 571 or  n_st == 796 :
                        talk_id = 345 + int(text_data[4])*int(text_data[6])
                    if n_st == 1021 or  n_st == 1246 :
                        talk_id = 795 + int(text_data[4])*int(text_data[6])
                """
                if  text_data[5] == 'VOTE':
                    talk_id = 805 + int(text_data[4])*int(text_data[6])
                if  text_data[5] == 'REQUEST' and text_data[6] == 'VOTE':
                    talk_id = 805 + int(text_data[4])*int(text_data[7])
                if  text_data[5] == 'ESTIMATE' and text_data[7] == 'WEREWOLF':
                    talk_id = 805 + int(text_data[4])*int(text_data[6])
                if  text_data[5] == 'ESTIMATE' :
                    if text_data[7] == 'WEREWOLF':
                        talk_id = 805 + int(text_data[4])*int(text_data[6])
                        #talk_id = 855 + int(text_data[4])*int(text_data[6])
                    if text_data[7] == 'POSSESSED':
                        talk_id = 805 + int(text_data[4])*int(text_data[6])
                        #talk_id = 880 + int(text_data[4])*int(text_data[6])
                """
        if text_data[1] == 'status' and int(text_data[2]) == 1:#日毎statusが呼ばれる一回だけ使用
            daily_death_slice = daily_death[:]#そのまま入れるとアドレスごと入るので、スライスを使う
            death.append(daily_death_slice)
            #status呼ばれるタイミングでやっているから 0は初期状態　1は初日朝だから死人なし、２は夜だから死人出る　[-1]は最終状態で１つ増加

        if talk_id != 0:
            talk_id_input.append(talk_id)



    target_pos = []
    for num in range(player_num):
        target_pos.append(true_pos[num][3])#特定の役職（人狼）のみの配列
        #target_pos.append(true_pos[num][1])#特定の役職（占い師）のみの配列


    return talk_id_input, target_pos, last_day, death

if __name__ == '__main__':
    for i in range(10):
        file_name = './log/aplain/'+str(i).zfill(3)+'.log'
        read_file(file_name,2)
