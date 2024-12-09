"""
AIO -- All Model in One
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from transformers import  Wav2Vec2Model, AutoTokenizer,AutoModel
#from transformers import AutoModel
from mamba_ssm import Mamba
#from mamba_ssm.modules.mamba2 import Mamba2
from models.WavLM import WavLM, WavLMConfig
from models.ser_efficient import SER_EfficientNet
import numpy as np
from models.efficient_addictive_attention import EfficientAdditiveAttnetion
from models.ser_spec import SER_AlexNet
#from efficientnet_pytorch import EfficientNet
#-*- coding: utf-8 -*-
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# __all__ = ['Ser_Model']
#IEMOCAP是4分类，CASIA是6分类,SAVEE是7分类,ESD是5分类
class Ser_Model(nn.Module):
    def __init__(self):
        super(Ser_Model, self).__init__()
        
        # CNN for Spectrogram
        #self.alexnet_model = SER_AlexNet(num_classes=4, in_ch=3, pretrained=True)
        self.efficientnet_model = SER_EfficientNet(num_classes=4, in_ch=3, pretrained=True)#4->6
        #self.efficientnet_model =  EfficientNet.from_pretrained(model_name="efficientnet-b0",num_classes=4,weights_path="/home/linping/speech_test/efficientNet/efficientnet-b0-355c32eb.pth")
        self.post_spec_dropout = nn.Dropout(p=0.1)
        #self.post_spec_layer = nn.Linear(9216, 128) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        self.post_spec_layer = nn.Linear(1280, 128) # 1280 for cnn, 32768 for ltsm s, 65536 for lstm l
        #self.post_spec_layer = nn.Linear(1280, 128) # 1280 for cnn, 32768 for ltsm s, 65536 for lstm l 128->256 自己的
        self.post_spec_layer_new = nn.Linear(1280, 768) # 1280 for cnn, 32768 for ltsm s, 65536 for lstm l 自己的
        self.post_spec_layer_att=nn.Linear(1280, 384)

        # LSTM for MFCC        
        #self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,bidirectional = True) # bidirectional = True###############
        #Mamba for MFCC
        self.mamba_mfcc  =  Mamba( 
        # 该模块大概使用了 3 * expand * d_model^2 个参数
            d_model = 40 ,  # 模型维度 d_model 
            d_state = 16 ,   # SSM 状态扩展因子，通常为 64 或 128 
            d_conv = 4 , #    局部卷积宽度
            expand = 2 ,     #块扩展因子2->4
        )
        '''self.mamba2_mfcc = Mamba2(
        # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=40, # Model dimension d_model
            d_state=64,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )'''
        '''self.mamba_mfcc_2  =  Mamba( 
        # 该模块大概使用了 3 * expand * d_model^2 个参数
            d_model = 40 ,  # 模型维度 d_model 
            d_state = 16 ,   # SSM 状态扩展因子，通常为 64 或 128 
            d_conv = 4 , #    局部卷积宽度
            expand = 4 ,     #块扩展因子
        )'''

        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        #self.post_mfcc_layer = nn.Linear(153600, 128) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm
        self.post_mfcc_layer = nn.Linear(12000, 128) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 12000 for mamba 128->256
        self.post_mfcc_layer_new = nn.Linear(12000, 768) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 12000 for mamba
        self.post_mfcc_layer_att = nn.Linear(12000, 384) # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 12000 for mamba
        
        # wavLM
        self.post_wavLM_layer = nn.Linear(768, 128) 

        # Spectrogram + MFCC  
        self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        self.post_spec_mfcc_att_layer = nn.Linear(256, 149) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l  256->512
        #self.post_spec_mfcc_att_layer_new = nn.Linear(768, 149) # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l自己加的
                        
        # WAV2VEC 2.0
        #self.wav2vec2_model = Wav2Vec2Model.from_pretrained("/home/heqing001/Coding/SER_0915/features_extraction/pretrained_model/wav2vec2-base-960h")
        #self.wav2vec2_model = Wav2Vec2Model.from_pretrained("/home/linping/speech_test/base_960h")
        #self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        #WavLM
        #self.WavLM_model = AutoModel.from_pretrained("/home/linping/speech_test/WavLM_base")
        checkpoint = torch.load('/home/linping/speech_test/WavLM_base/WavLM-Base+.pt')
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        self.WavLM_model=model


        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(768, 128) # 512 for 1 and 768 for 2  128->256
        self.post_wav_layer_new = nn.Linear(1536, 768) # 512 for 1 and 768 for 2
        
        #Attention 
        self.Eff_Att=EfficientAdditiveAttnetion(token_dim=256, num_heads=3)

        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        #self.post_att_layer_1 = nn.Linear(384, 128)
        self.post_att_layer_1_new = nn.Linear(384, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        #self.post_att_layer_2_new = nn.Linear(768, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)#4->6
        
                                                                     
    def forward(self, audio_spec, audio_mfcc, audio_wav):      
        
        # audio_spec: [batch, 3, 256, 384]
        # audio_mfcc: [batch, 300, 40]
        # audio_wav: [32, 48000]
        
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)
        
        # spectrogram - SER_CNN
        #audio_spec, output_spec_t = self.alexnet_model(audio_spec) # [batch, 256, 6, 6], []
        audio_spec, output_spec_t = self.efficientnet_model(audio_spec) # [batch, 1280,1,1], [] 自己的
        #audio_spec = self.efficientnet_model.extract_features(audio_spec) # [batch, 1280,7,7], []
        #audio_spec = self.efficientnet_model._avg_pooling(audio_spec) # [batch, 1280,7,7], []
        #print("11111111",audio_spec.shape)
        audio_spec = audio_spec.reshape(audio_spec.shape[0], audio_spec.shape[1], -1) # [batch, 1280,49]  自己的
        #print("222222222",audio_spec.shape)'''
        # audio -- MFCC with BiLSTM
        #audio_mfcc, _ = self.lstm_mfcc(audio_mfcc) # [batch, 300, 512]  ############################
        audio_mfcc = self.mamba_mfcc(audio_mfcc) # [batch, 300, 512] #自己加的
        #print("2222222222",audio_mfcc.shape)
        
        audio_spec_ = torch.flatten(audio_spec, 1) # [batch, 1280] 
        
        audio_spec_d = self.post_spec_dropout(audio_spec_) # [batch, 1280]  
        audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=False) # [batch, 128]  ########################
        audio_spec_p_wav_ = F.relu(self.post_spec_layer_new(audio_spec_d), inplace=False) # [batch, 149]  自己加的
        audio_spec_p_wav = audio_spec_p_wav_.reshape(audio_spec_p_wav_.shape[0], 1, -1)# [batch, 1, 149]自己加的'''
        audio_spec_p_att_ = F.relu(self.post_spec_layer_att(audio_spec_d), inplace=False) # [batch, 149]  自己加的
        audio_spec_p_att = audio_spec_p_att_.reshape(audio_spec_p_att_.shape[0], 1, -1)# [batch, 1, 149]自己加的'''
        
        #+ audio_mfcc = self.att(audio_mfcc)
        audio_mfcc_ = torch.flatten(audio_mfcc, 1) # [batch, 153600]  
        #print("333333333",audio_mfcc_.shape)
        audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_) # [batch, 153600]  
        audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False) # [batch, 128]  ##############################
        audio_mfcc_p_wav_ = F.relu(self.post_mfcc_layer_new(audio_mfcc_att_d), inplace=False) # [batch, 149]  自己加的
        audio_mfcc_p_wav = audio_mfcc_p_wav_.reshape(audio_mfcc_p_wav_.shape[0], 1, -1)# [batch, 1, 149]自己加的
        audio_mfcc_p_att_ = F.relu(self.post_mfcc_layer_att(audio_mfcc_att_d), inplace=False) # [batch, 149]  自己加的
        audio_mfcc_p_att = audio_mfcc_p_att_.reshape(audio_mfcc_p_att_.shape[0], 1, -1)# [batch, 1, 149]自己加的
        

        # FOR WAV2VEC2.0 WEIGHTS 
        spec_mfcc = torch.cat([audio_spec_p, audio_mfcc_p], dim=-1) # [batch, 256] 
        audio_spec_mfcc_att_d = self.post_spec_mfcc_att_dropout(spec_mfcc)# [batch, 256] 
        audio_spec_mfcc_att_p = F.relu(self.post_spec_mfcc_att_layer(audio_spec_mfcc_att_d), inplace=False)# [batch, 149] 
        #audio_spec_mfcc_att_p = F.relu(self.post_spec_mfcc_att_layer_new(audio_spec_mfcc_att_d), inplace=False)# [batch, 149] 
        audio_spec_mfcc_att_p = audio_spec_mfcc_att_p.reshape(audio_spec_mfcc_att_p.shape[0], 1, -1)# [batch, 1, 149] 
        #+ audio_spec_mfcc_att_2 = F.softmax(audio_spec_mfcc_att_1, dim=2)'''#############################

        '''# wav2vec 2.0 
        #audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state # [batch, 149, 768] 
        audio_wav = self.wav2vec2_model(audio_wav).last_hidden_state # [batch, 149, 768] 
        audio_wav = torch.matmul(audio_spec_mfcc_att_p, audio_wav) # [batch, 1, 768] 
        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) # [batch, 768] 
        #audio_wav = torch.mean(audio_wav, dim=1)'''###########################################

        # wavLM
        #audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state # [batch, 149, 768] 
        #audio_wav = self.WavLM_model(audio_wav).last_hidden_state # [batch, 149, 768] 

        audio_wav = self.WavLM_model.extract_features(audio_wav)[0]######自己的

        #print("222222222",audio_wav.shape)
        #audio_wav_d = F.relu(self.post_wavLM_layer(audio_wav), inplace=False) # [batch, 149, 128] 自己加的
        #print("222222222",audio_wav_d.shape)
        

        #audio_wav = torch.matmul(audio_spec_mfcc_att_p, audio_wav) # [batch, 1, 768] ########################原来的
        #audio_wav = torch.matmul(audio_mfcc_p_wav, audio_wav) # [batch, 1, 768]#两特征实验
        audio_wav_pre = torch.matmul(audio_spec_mfcc_att_p, audio_wav) # [batch, 1, 768] ########################
        #audio_wav_1 = torch.matmul(audio_spec_p_wav, audio_wav) # [batch, 1, 768] ########################
        #audio_wav_2 = torch.matmul(audio_mfcc_p_wav, audio_wav) # [batch, 1, 768] ########################
        #audio_wav=torch.cat([audio_wav_1, audio_wav_2], dim=-1)
        #audio_wav=torch.add(audio_wav_1,audio_wav_2)'''

        '''# attention
        audio_wav=torch.matmul(audio_mfcc_p_wav,audio_wav_d) / np.sqrt(audio_mfcc_p_wav.size(-1))
        attention_weights = F.softmax(audio_wav,dim=-1)
        #print("333333333",attention_weights.shape)
        audio_wav=torch.matmul(attention_weights.transpose(-2, -1),audio_spec_p_wav)#自己加的'''

        #Effitient Attention
        audio_wav_1=self.Eff_Att(audio_wav,audio_spec_p_att)
        audio_wav_2=self.Eff_Att(audio_wav,audio_mfcc_p_att)
        audio_wav=torch.cat([audio_wav_1, audio_wav_2], dim=-1)
        


        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1) # [batch, 768] 
        audio_wav_pre = audio_wav_pre.reshape(audio_wav_pre.shape[0], -1) # [batch, 768] 自己加的
        #audio_wav_1 = audio_wav_1.reshape(audio_wav_1.shape[0], -1)#两特征实验
        #audio_wav = torch.mean(audio_wav, dim=1)
        
        audio_wav_d = self.post_wav_dropout(audio_wav) # [batch, 768] 
        audio_wav_d_pre = self.post_wav_dropout(audio_wav_pre) # [batch, 768] 自己加的
        #print("444444444",audio_wav_d.shape)
        audio_wav_p = F.relu(self.post_wav_layer(audio_wav_d_pre), inplace=False) # [batch, 128] ####################
        #audio_wav_1 = F.relu(self.post_wav_layer(audio_wav_1), inplace=False)#两特征实验
        audio_wav_att = F.relu(self.post_wav_layer_new(audio_wav_d), inplace=False) # [batch, 256] 自己加的'''


        ## combine()
        #audio_att = torch.cat([audio_spec_p, audio_mfcc_p, audio_wav_p], dim=-1)  # [batch, 384] ######################
        #audio_att = torch.cat([audio_spec_p, audio_mfcc_p, audio_wav_att], dim=-1)
        audio_att = torch.cat([audio_spec_p,audio_mfcc_p,audio_wav_p], dim=-1)
        #audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 384] #############################
        audio_att_d_1 = self.post_att_dropout(audio_att) # [batch, 384] #单模块
        #audio_att_d_1 = self.post_att_dropout(audio_wav_att) # [batch, 384] 
        #audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False) # [batch, 128] 
        audio_att_1 = F.relu(self.post_att_layer_1_new(audio_att_d_1), inplace=False) # [batch, 128] 
        audio_att_d_2 = self.post_att_dropout(audio_att_1) # [batch, 128] 
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128] 
        #audio_att_2 = F.relu(self.post_att_layer_2_new(audio_att_d_2), inplace=False)  # [batch, 128] 
        output_att = self.post_att_layer_3(audio_att_2) # [batch, 4] '''
        #output_att = output_spec_t # [batch, 4] 
  
        output = {
            #'F1': audio_wav_p,#########################
            'F1': audio_wav_p,
            'F2': audio_att_1,
            'F3': audio_att_2,
            'F4': output_att,
            'M': output_att
        }            
        

        return output
    
