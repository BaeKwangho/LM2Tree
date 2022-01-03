from model.decoder.modules import *

class TreeDecoder(nn.Module):
    def __init__(self,config):
        super(TreeDecoder,self).__init__()
        self.predict = Seq2TreePrediction(hidden_size=config.hidden_size, 
                                          op_nums=config.out_words_cnt - config.copy_nums - 1 - len(config.generate_nums) - len(config.var_nums), 
                                          vocab_size=len(config.generate_nums) + len(config.var_nums))
        self.generate = Seq2TreeNodeGeneration(
            hidden_size=config.hidden_size, op_nums=config.out_words_cnt - config.copy_nums - 1 - len(config.generate_nums),
                        embedding_size=config.embedding_size)
        self.merge = Seq2TreeSubTreeMerge(hidden_size=config.hidden_size, embedding_size=config.embedding_size)
        # the embedding layer is  only for generated number embeddings, operators, and paddings
        self.semantic_alignment = Seq2TreeSemanticAlignment(encoder_hidden_size=config.hidden_size, 
                                                            decoder_hidden_size=config.hidden_size, 
                                                            hidden_size=config.hidden_size)
        
        
    def forward(self,max_target_length,node_stacks,target_length ,left_childs,encoder_outputs, \
                              all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask,\
                                target,nums_stack_batch,num_start,unk,embeddings_stacks,batch_first=False):
        
        all_node_outputs = []
        all_sa_outputs = []
        
        batch_size = len(nums_stack_batch)
        
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
            outputs = torch.cat((op, num_score), 1)
            temp = op
            
            print(num_score.shape, op.shape)
            all_node_outputs.append(outputs)

            target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
            target[t] = target_t
            if USE_CUDA:
                generate_input = generate_input.cuda()
            left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                   node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                # 未知数当数字处理，SEP当操作符处理
                if i < num_start:  # 非数字
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))
                    # print(o[-1].embedding.size())
                    # print(encoder_outputs[idx].size())
                else:  # 数字
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)  # Subtree embedding
                        if batch_first:
                            encoder_mapping, decoder_mapping = self.semantic_alignment(current_num, encoder_outputs[idx])
                        else:
                            temp_encoder_outputs = encoder_outputs.transpose(0,1)
                            encoder_mapping, decoder_mapping = self.semantic_alignment(current_num, temp_encoder_outputs[idx])
                        all_sa_outputs.append((encoder_mapping, decoder_mapping))
                    o.append(TreeEmbedding(current_num, terminal=True))

                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)

                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        target = target.transpose(0, 1).contiguous() # B x S

        if USE_CUDA:
            # all_leafs = all_leafs.cuda()
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()
            new_all_sa_outputs = []
            for sa_pair in all_sa_outputs:
                new_all_sa_outputs.append((sa_pair[0].cuda(),sa_pair[1].cuda()))
            all_sa_outputs = new_all_sa_outputs

        semantic_alignment_loss = nn.MSELoss()
        total_semanti_alognment_loss = 0
        sa_len = len(all_sa_outputs)
        for sa_pair in all_sa_outputs:
            total_semanti_alognment_loss += semantic_alignment_loss(sa_pair[0],sa_pair[1])
        # print(total_semanti_alognment_loss)
        total_semanti_alognment_loss = total_semanti_alognment_loss / sa_len
        
        # op_target = target < num_start
        # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
        temp_loss, losses = masked_cross_entropy_with_logit(all_node_outputs, target, target_length,use_gpu=USE_CUDA)
        loss =  temp_loss + 0.01 * total_semanti_alognment_loss
        # loss = loss_0 + loss_1 
        all_node_outputs = (temp,num_score)
        return loss, (losses,all_node_outputs)
    
    def beam_out(self,node_stacks, embeddings_stacks, left_childs, max_length, encoder_outputs, \
                                  all_nums_encoder_outputs, padding_hidden,seq_mask,num_mask,beam_size,num_start):
        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)

                # leaf = p_leaf[:, 0].unsqueeze(1)
                # repeat_dims = [1] * leaf.dim()
                # repeat_dims[1] = op.size(1)
                # leaf = leaf.repeat(*repeat_dims)
                #
                # non_leaf = p_leaf[:, 1].unsqueeze(1)
                # repeat_dims = [1] * non_leaf.dim()
                # repeat_dims[1] = num_score.size(1)
                # non_leaf = non_leaf.repeat(*repeat_dims)
                #
                # p_leaf = torch.cat((leaf, non_leaf), dim=1)
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score
                topv, topi = out_score.topk(beam_size)

                # is_leaf = int(topi[0])
                # if is_leaf:
                #     topv, topi = op.topk(1)
                #     out_token = int(topi[0])
                # else:
                #     topv, topi = num_score.topk(1)
                #     out_token = int(topi[0]) + num_start
                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)
                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    # var_num当时数字处理，SEP/;当操作符处理
                    if out_token < num_start: # 非数字
                        generate_input = torch.LongTensor([out_token])
                        if USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:  # 数字
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                  current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        return beams[0].out
    
    def node_out(self):
        return False

class TreeConfig(object):
    def __init__(self,
        out_words_cnt,
        generate_nums,
        copy_nums,
        batch_size = 32,
        embedding_size = 128,
        hidden_size = 768,
        n_epochs = 80,
        learning_rate = 1e-4,
        weight_decay = 1e-5,
        beam_size = 5,
        n_layers = 2,
        beam_search = True,
        fold_num = 5,
        random_seed = 1,
        max_output_length = 80,
        var_nums = ['x','y'],
    ):
        self.out_words_cnt = int(out_words_cnt)
        self.generate_nums = generate_nums
        self.copy_nums = copy_nums
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beam_size = beam_size
        self.n_layers = n_layers
        self.beam_search = beam_search
        self.fold_num = fold_num
        self.random_seed = random_seed
        self.var_nums = var_nums
        self.max_output_length = max_output_length