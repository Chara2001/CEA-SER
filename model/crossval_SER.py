import train_ser
from train_ser import parse_arguments
import sys
import pickle
import os
import time


repeat_kfold = 1 # to  perform 10-fold for n-times with different seed
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

#------------PARAMETERS---------------#

#features_file = 'IEMOCAP_multi.pkl'
#features_file = "/home/linping/speech_test/CA-MSER/IEMOCAP_multi/IEMOCAP_multi.pkl" #原来的特征
features_file = "/home/linping/speech_test/CA-MSER/features_extraction/IEMOCAP_multi345200LP.pkl"

#  leave-one-speaker-out
#val_id = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F'] #IEMOCAP
#test_id = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F'] #IEMOCAP
val_id = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F'] #IEMOCAP
test_id = ['2F'] #IEMOCAP
#val_id = ['F','M'] #CASIA
#test_id = ['F','M'] #CASIA

#val_id = ['all'] #SAVEE
#test_id = ['all']#SAVEE

#val_id = ['19', '20', '17', '12', '14', '15', '11', '13', '18', '16'] #ESD
#test_id = ['19', '20', '17', '12', '14', '15', '11', '13', '18', '16']#ESD

#  leave-one-session-out
#val_id = ['1M','2M','3M','4M','5M']
#test_id = ['1F','2F','3F','4F','5F']

num_epochs  = '100'
early_stop = '8'###############8->10
batch_size  = '32'##############64->32
lr          = '0.00002'###########0.00001->0.00003
random_seed = 111
gpu = '1'
gpu_ids = ['1']######
save_label = str_time+'best_net'#'0930_01'#'alexnet_pm_0704'
 

#Start Cross Validation
all_stat = []

for repeat in range(repeat_kfold):

    random_seed +=  (repeat*100)
    seed = str(random_seed)

    for v_id, t_id in list(zip(val_id, test_id)):

        train_ser.sys.argv      = [
                        
                                  'train_ser.py', 
                                  features_file,
                                  '--repeat_idx', str(repeat),
                                  '--val_id',v_id, 
                                  '--test_id', t_id,
                                  '--gpu', gpu,
                                  '--gpu_ids', gpu_ids,
                                  '--num_epochs', num_epochs,
                                  '--early_stop', early_stop,
                                  '--batch_size', batch_size,
                                  '--lr', lr,
                                  '--seed', seed,
                                  '--save_label', save_label,#,
                                  '--pretrained'
                                  ]

    
        stat = train_ser.main(parse_arguments(sys.argv[1:]))   
        all_stat.append(stat)       
        os.remove(save_label+'.pth')
    
    # with open('allstat_iemocap_'+save_label+'_'+str(repeat)+'.pkl', "wb") as fout:
    #     pickle.dump(all_stat, fout)

n_total = repeat_kfold*len(val_id)
total_best_epoch, total_epoch, total_loss, total_wa, total_ua = 0, 0, 0, 0, 0

for i in range(n_total):
    print(i, ": ", all_stat[i][0], all_stat[i][1], all_stat[i][8], all_stat[i][9], all_stat[i][10]) 
    total_best_epoch += all_stat[i][0]
    total_epoch += all_stat[i][1]
    total_loss += float(all_stat[i][8])
    total_wa += float(all_stat[i][9])
    total_ua += float(all_stat[i][10])

print("AVERAGE:", total_best_epoch/n_total, total_epoch/n_total, total_loss/n_total, total_wa/n_total, total_ua/n_total )

print(all_stat)
