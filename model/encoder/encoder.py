import torch.nn as nn

class LMEncoder(nn.Module):
    def __init__(self,config):
        super(ElectraEnc,self).__init__()
        #self.electra = ElectraModel(config)
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.pooler = nn.Linear(config.hidden_size,config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self,input_batch):
        elec_out = self.electra(input_batch,return_dict=False)
        last_layers = elec_out[-1]
        pooler_input = last_layers[:, 0]+last_layers[:, -1]
        pooled_output = self.pooler(pooler_input)
        pooled_output = self.activation(pooled_output)
        
        return last_layers, pooled_output