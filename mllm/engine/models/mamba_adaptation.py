from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from mamba_ssm import Mamba
from einops import rearrange


class H_MBA(nn.Module):                                              #num_frames=5
    def __init__(self, d_model, n_layer, d_state=8, d_conv=3, expand=2, num_frames=5, dropout=0.):
        super().__init__()
        self.mamba_block = Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.latentMB = Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand)

        self.layer_1 = nn.ModuleList(
            [Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand),
            self.layer_norm,
            ]
        )
        self.reverse_layer_1 = nn.ModuleList(
            [Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand),
            self.layer_norm,
            ]
        )
        self.layer_2 = nn.ModuleList(
            [Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand),
            self.layer_norm,
            ]
        )
        self.reverse_layer_2 = nn.ModuleList(
            [Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand),
            self.layer_norm,
            ]
        )
        self.layer_3 = nn.ModuleList(
            [Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand),
            self.layer_norm,
            ]
        )
        self.reverse_layer_3 = nn.ModuleList(
            [Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand),
            self.layer_norm,
            ]
        )
        self.t_mamba = Mamba(d_model=d_model,d_state=d_state,d_conv=d_conv,expand=expand)
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, d_model))
        self.cls_token = nn.Parameter(torch.ones(1, num_frames, 1, d_model))
        self.v1 = nn.Parameter(torch.ones(1))
        self.v2 = nn.Parameter(torch.ones(1))
        self.v3 = nn.Parameter(torch.ones(1))
        # self.v1 = nn.Parameter(torch.ones(1,5))
        # self.v2 = nn.Parameter(torch.ones(1,5))
        self.fc_q = nn.Linear(d_model,d_model)
        nn.init.eye_(self.fc_q.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        # for param in self.mamba_block.parameters():
        #     print(param.data.dtype)
        #     param.data = param.data.float()
        
    
    def reverse(self,input_features):
        b, l ,c= input_features
        reverse_features = torch.flip(input_features,dim=1)
        return reverse_features
    

    def dual_forward(self,x):
        x_f = x
        x_b = x.flip([1])
        for module in self.layer_2:
            x_f = module(x_f)
        for module in self.reverse_layer_2:
            x_b = module(x_b)

        return x_f + x_b.flip([1])
    
    def forward_JST(self,whole_video): 
        #self.forward_3(whole_video)
        #mean_token = whole_video.mean(dim=1)
        video_token = whole_video + self.time_embed.unsqueeze(dim=2)
        b,t,l,c = whole_video.shape
        video_token = video_token.reshape(b,t*l,c)
        
        # for module in self.layer_3:
        #     video_token = module(video_token)
        video_token = self.dual_forward(video_token)
        video_token = self.layer_norm(video_token)

        mamba_video = video_token.reshape(b,t,l,c)

        return mamba_video[:,-1] 
    
    def forward_JST_multi_t(self,whole_video,index): 
        video_token = whole_video[:,index] + self.time_embed.unsqueeze(dim=2)[:,index]
        b,t,l,c = video_token.shape
        video_token = video_token.reshape(b,t*l,c)
        
        for module in self.layer_3:
            video_token = module(video_token)
        #video_token = self.dual_forward(video_token)
        video_token = self.layer_norm(video_token)

        mamba_video = video_token.reshape(b,t,l,c)

        return mamba_video.mean(dim=1)#mamba_video[:,-1] # #

    def forward_DST(self, whole_video):  ##### stmamba
        mean_token = whole_video.mean(dim=1)
        b,t,l,c = whole_video.shape  #8,n,256,1024
        x_t = rearrange(whole_video,'b t l c -> (b l) t c')
        #x_t = self.dropout(x_t)
        x_t = self.t_mamba(x_t)
        x_t = self.layer_norm(x_t)
        #x_t = rearrange(x_t, '(b l) t c -> b t l c') 
        x_t = x_t.reshape(b,l,t,c).permute(0,2,1,3)

        x_s = whole_video + x_t
        x_s = rearrange(x_s, 'b t l c -> (b t) l c' )∂
        x_t_s = self.dual_forward(x_s)
        x_t_s = x_t_s.reshape(b,t,l,c)
        return x_t_s[:,-1]          #.mean(dim=1)
    
    def forward_T(self, whole_video):
        mean_token = whole_video.mean(dim=1)
        input_features = torch.mean(whole_video,dim=2)
        x = input_features
        reverse_x = x.flip([1])
        for module in self.layer_1:
           x = module(x)
        
        for module in self.reverse_layer_1:
            reverse_x = module(reverse_x)
        #out = (x[:,-1] + 0.2 * reverse_x[:,0])
        #out = (x + 0.2*reverse_x.flip([1]))#.mean(dim=1)
        out = x
        out = self.layer_norm(out)
        return out    #0.2*out.unsqueeze(1) + mean_token
    
    def forward_t_2(self, whole_video):
        mean_token = whole_video.mean(dim=1)
        input_features = torch.mean(whole_video,dim=2)
        x = input_features
        reverse_x = x.flip([1])
        for module in self.layer_2:
           x = module(x)
        
        for module in self.reverse_layer_2:
            reverse_x = module(reverse_x)
        #out = (x[:,-1] + 0.2 * reverse_x[:,0])
        out = (x + 0.2*reverse_x.flip([1]))#.mean(dim=1)
        #out = x
        out = self.layer_norm(out)
        return out    #0.2*out.unsqueeze(1) + mean_token

    
    def forward_st(self, whole_video):  ##### stmamba
        mean_token = whole_video.mean(dim=1)
        b,t,l,c = whole_video.shape  #8,n,256,1024
        video_with_cls = torch.cat((whole_video,self.cls_token.repeat(b,1,1,1,)),dim=2)
        
        x = video_with_cls.reshape(b*t,(l+1),c)
        #x = self.dropout(x)
        # for module in self.layer:
        #     x = module(x)
        x = self.dual_forward(x)
        
        x_s = x.reshape(b,t,(l+1),c)[:,:,-1,]
        x_s_t = self.t_mamba(x_s)
        #print('before',x_s_t)
        x_s_t = self.layer_norm(x_s_t)
        #print('after',x_s_t)
        return x_s_t
        #0.2*x_s_t[:,-1].unsqueeze(1) #+ mean_token
    
    def query_adaptation(self,t1,t2):
        t1_sum = (t1*t1).sum(dim=-1)
        t2_sum = (t2*t2).sum(dim=-1)
        sim_ma = (t1*t2).sum(-1) / torch.sqrt(t1_sum*t2_sum)
        channel_wise_feature = sim_ma.unsqueeze(2) * t1
        return channel_wise_feature

    # def multi_temporal(self, whole_video):
    #     index1 = [4]
    #     index2 = [0,2,4]
    #     feature1 = whole_video[:,index1]
    #     feature2 = whole_video[:,index2]
    #     f3 = self.forward_t(whole_video)
    #     f2 = self.mamba_block(feature2.mean(dim=2))
    #     f2 = self.layer_norm(f2)
        
    #     return f2, f3


    def forward(self, whole_video,image_features):

        mean_token = whole_video.mean(dim=1)
        query = self.fc_q(image_features)
       
        
        f1 = self.forward_T(whole_video)[:,-1].unsqueeze(1)
        f1 = self.query_adaptation(f1, query)
       

        f2 = self.forward_DST(whole_video)
        f2 = self.query_adaptation(f2, query)
    

        f3 = self.forward_JST(whole_video)
        f3 = self.query_adaptation(f3, query)

        #inception = 0.01*(self.v1@f1 + self.v2@f2) + 0.1* self.v3*f3 #在justification时是0.01
        #inception = f1 + f2 + self.v3*f3
    ########################## multi-temporal #############################  
        

        # index2 = [0,2,4,]
        # f1_dt = self.forward_T(whole_video[:,index2])[:,-1].unsqueeze(1)
        # f1_dt = self.channel_wise_add(f1_dt, query)
        # f2_dt = self.forward_DST(whole_video[:,index2])
        # f2_dt = self.channel_wise_add(f2_dt, query)
        # f3_dt = self.forward_JST(whole_video,index2)
        # f3_dt = self.channel_wise_add(f3_dt, query)
        
        ST_frame_feature = 0.1*self.v3*f3  + 0.005 * self.v2 * f2 + 0.01* self.v3 * f3
        
    #########################################################################
        return  image_features +  ST_frame_feature