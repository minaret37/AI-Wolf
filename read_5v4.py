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
    player_num = 5
    #day = 2
    position = {"VILLAGER":0,"SEER":1,"POSSESSED":2,"WEREWOLF":3,"BODYGUARD":4}#4
    true_pos = np.zeros((player_num,len(position)), dtype = int)#5*4
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
        target_pos = []
        for num in range(player_num):
            target_pos.append(true_pos[num][3])#特定の役職（人狼）のみの配列

        talk_id=0
        #入力データ　作成 5
        if int(text_data[0]) > last_day:#last_dayは終了した日時　会話した日の次の日
            last_day = int(text_data[0])
        #print(int(text_data[0]),input_day)
        if int(text_data[0])<input_day:#一日前の情報まで
            #print("day",int(text_data[0]))
            if  text_data[1] == 'attack':
                talk_id = int(text_data[2])#1~5
                daily_death.append(int(text_data[2]))
            if  text_data[1] == 'execute':
                talk_id = 5 + int(text_data[2])#6~10
                daily_death.append(int(text_data[2]))
                #print("attack",input_day,int(text_data[0]))
            if  text_data[1] == 'vote' and n_st == 111:
                talk_id = 85 + int(text_data[2])*int(text_data[3])
                #print("attack",input_day,int(text_data[0]))
        if int(text_data[0])<=input_day:#日数
            #input_day=7のとき７日目まで会話、８日目結果発表、６日目までの処刑・襲撃情報
            if  text_data[1] == 'talk':
                if  text_data[5] == 'COMINGOUT':
                    talk_id = 10 + int(text_data[4])*(position[text_data[7]])
                if  text_data[5] == 'DIVINED' and text_data[7] == 'HUMAN' :
                    talk_id = 35 + int(text_data[4])*int(text_data[6])
                if  text_data[5] == 'DIVINED' and text_data[7] == 'WEREWOLF' :
                    talk_id = 60 + int(text_data[4])*int(text_data[6])

                """
                if  text_data[5] == 'VOTE':
                    talk_id = 85 + int(text_data[4])*int(text_data[6])
                if  text_data[5] == 'REQUEST' and text_data[6] == 'VOTE':
                    #talk_id = 105 + int(text_data[4])*int(text_data[7])
                    talk_id = 85 + int(text_data[4])*int(text_data[7])
                if  text_data[5] == 'ESTIMATE' :
                    if text_data[7] == 'WEREWOLF':
                        #talk_id = 105 + int(text_data[4])*int(text_data[6])
                        talk_id = 85 + int(text_data[4])*int(text_data[6])
                    #if text_data[7] == 'POSSESSED':
                        #talk_id = 105 + int(text_data[4])*int(text_data[6])
                        #talk_id = 180 + int(text_data[4])*int(text_data[6])
                """
        if text_data[1] == 'status' and int(text_data[2]) == 1:#日毎statusが呼ばれる一回だけ使用
            daily_death_slice = daily_death[:]#そのまま入れるとアドレスごと入るので、スライスを使う
            death.append(daily_death_slice)
            #status呼ばれるタイミングでやっているから 0は初期状態　1は初日朝だから死人なし、２は夜だから死人出る　[-1]は最終状態で１つ増加

        if talk_id > 0:
            talk_id_input.append(talk_id)
    return talk_id_input, target_pos, last_day, death

if __name__ == '__main__':
    v5 = []
    v15 = []
    for i in range(1,203):
        log_folder = i
        log_file = 0
        file_name = './cedec2018/'+str(log_folder).zfill(3)+'/'+str(log_file).zfill(3)+'.log'
        print(log_folder)
        if distinction(file_name)==5:
            v5.append(i)
        else:
            v15.append(i)
    print(len(v5),v5)
    print(len(v15),v15)
