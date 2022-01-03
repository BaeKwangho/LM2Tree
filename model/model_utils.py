import torch.optim as optim
import torch.nn.functional as F

def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size, batch_first=False):
    indices = list()
    if batch_first:
        sen_len = encoder_outputs.size(1)
    else:
        sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)  # 用于记录数字在问题的位置
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]  # 屏蔽多余的数字，即词表中不属于该题目的数字
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices).to(device)
    masked_index = torch.ByteTensor(masked_index) # B x num_size x H
    masked_index = masked_index.view(batch_size, num_size, hidden_size).to(device)
    
    if batch_first:
        all_outputs = encoder_outputs.contiguous() # B x S x H
    else:
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()  # S x B x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H or B x S x H -> (S x B) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0) 