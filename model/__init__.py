from model.encoder import *
from model.decoder import *
from model.model_utils import *

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LM2Tree(nn.Module):
    def __init__(self,enc_conf,dec_conf,token_len):
        super(LM2Tree,self).__init__()
        self.encoder = ElectraEnc(enc_conf,token_len)
        self.decoder = TreeDecoder(dec_conf)
        self.dec_conf = dec_conf
        
    def evaluate(self,input_batch, input_length, output_lang,num_pos_batch,beam_size=5,beam_search=True):
        
        max_length = self.dec_conf.max_output_length
        seq_mask = torch.ByteTensor(1, input_length).fill_(0).to(device)
        input_batch = torch.LongTensor(input_batch).unsqueeze(0).to(device)
        
        num_mask = torch.ByteTensor(1, len(num_pos_batch) + len(self.dec_conf.generate_nums)+ \
                                    len(self.dec_conf.var_nums)).fill_(0).to(device)
        
        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.dec_conf.hidden_size)]).unsqueeze(0).to(device)
        batch_size = 1
        
        #### encoder ####
        encoder_outputs,problem_output = self.encoder(input_batch)
        encoder_outputs = encoder_outputs.permute(1,0,2).contiguous()
        #################
        
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)] # root embedding B x 1        
        num_size = len(num_pos_batch)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos_batch], batch_size, num_size,
                                                                  self.dec_conf.hidden_size)
        num_start = output_lang.num_start - len(self.dec_conf.var_nums)
        # B x P x N
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        
        if beam_search:
            return self.decoder.beam_out(node_stacks, embeddings_stacks, left_childs, max_length, encoder_outputs, \
                                  all_nums_encoder_outputs, padding_hidden,seq_mask,num_mask,beam_size,num_start)
        else:
            return self.decoder.node_out()
        
    def forward(self,input_batch,input_length, target_batch, target_length, nums_stack_batch,\
                num_size_batch, num_pos_batch, output_lang, batch_first=False):
        
        #### build mask ####
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.ByteTensor(seq_mask).to(device)

        num_mask = []
        max_num_size = max(num_size_batch) + len(self.dec_conf.generate_nums) + len(self.dec_conf.var_nums) # 最大的位置列表数目+常识数字数目+未知数列表
        for i in num_size_batch:
            d = i + len(generate_nums) + len(self.dec_conf.var_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.ByteTensor(num_mask).to(device) # 用于屏蔽无关数字，防止生成错误的Nx
        ####################
        
        input_batch = torch.LongTensor(input_batch).to(device)
        target = torch.LongTensor(target_batch).transpose(0, 1)
        padding_hidden = torch.FloatTensor([0.0 for _ in range(dec_conf.hidden_size)]).unsqueeze(0).to(device)
        batch_size = len(input_length)
        
        #### encoder ####
        encoder_outputs,problem_output = self.encoder(input_batch)
        encoder_outputs = encoder_outputs.permute(1,0,2).contiguous()
        #################
        
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)] # root embedding B x 1
        max_target_length = max(target_length)

        all_node_outputs = []
        all_sa_outputs = []
        # all_leafs = []
        
        unk = output_lang.word2index["UNK"]

        copy_num_len = [len(_) for _ in num_pos_batch]
        num_size = max(copy_num_len)
        # 提取与问题相关的数字embedding
        all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, batch_size, num_size,
                                                                  dec_conf.hidden_size)
        num_start = output_lang.num_start - len(self.dec_conf.var_nums)
        embeddings_stacks = [[] for _ in range(batch_size)] # B x 1  当前的tree state/ subtree embedding / output
        left_childs = [None for _ in range(batch_size)] # B x 1
        
        #### decoder ####
        result , losses = self.decoder(max_target_length,node_stacks,target_length,left_childs,encoder_outputs, \
                              all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask, target,nums_stack_batch,num_start,unk,embeddings_stacks,batch_first)
        #################
        
        return result , losses

def get_model(model_type,enc_conf,dec_conf,token_len):
    model = LM2Tree(enc_conf,dec_conf,token_len)
    
    return model