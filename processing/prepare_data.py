import random
import json
import copy
import re
from processing.data_utils import remove_brackets
from processing.lang import OutputLang, InputLang
from processing.data_utils import indexes_from_sentence, pad_seq, check_bracket, get_num_stack
from processing.operation_law import exchange, allocation
from tqdm import tqdm

def evaluate_tmp(model, test_pairs,output_lang):
    acc = 0
    model.eval()
    for batch in tqdm(test_pairs):
        result = model.module.evaluate(batch[0], batch[1], output_lang,batch[5])
        if not batch[2]==result:
            acc+=0
        else:
            acc+=1
    prediction = [output_lang.index2word[i] for i in result]
    model.train()
    return (acc/len(test_pairs)), prediction, batch[2], batch[0]

def prepare_test(pairs_tested,tokenizer,output_lang,tree=False):
    vocabs = tokenizer.get_vocab()
    n_word = vocabs['n']
    um_word = vocabs['##um']
    num_word = vocabs['NUM']
    ###
    def token_handling(input_list):
        flag = False
        pos = []
        for i, token in enumerate(input_list):
            if token == n_word:
                flag = True
            elif flag == True and token == um_word:
                flag = False
                input_list.pop(i-1)
                input_list.pop(i-1)
                input_list.insert(i-1,num_word)
                pos.append(i-1)
            else:
                pass
        return input_list,pos
    
    test_pairs = []
    for pair in tqdm(pairs_tested):
        input_cell = tokenizer.encode(' '.join(pair[0]),add_special_tokens=True)
        input_cell,pos = token_handling(input_cell)
        
        test_pairs.append((input_cell, len(input_cell),
                                  pair[1], pos))
    return test_pairs

def prepare_data(pairs_trained, pairs_tested, trim_min_count, tokenizer, generate_nums, copy_nums, tree=False, use_tfm=False,assigned_tokens={}):
    input_lang = InputLang()
    output_lang = OutputLang()
    train_pairs = []
    test_pairs = []
    vocabs = tokenizer.get_vocab()
    '''
    n_word = vocabs['n']
    um_word = vocabs['##um']
    num_word = vocabs['NUM']
    ###
    def token_handling(input_list):
        flag = False
        pos = []
        for i, token in enumerate(input_list):
            if token == n_word:
                flag = True
            elif flag == True and token == um_word:
                flag = False
                input_list.pop(i-1)
                input_list.pop(i-1)
                input_list.insert(i-1,num_word)
                pos.append(i-1)
            else:
                pass
        return input_list,pos
    '''
    def assigned_replacer(input_list):
        assign_list = []
        for tok in input_list:
            if tok in assigned_tokens.keys():
                assign_list.append(tok)
        
        input_list = tokenizer.encode(' '.join(input_list),add_special_tokens=True)
        n_word = vocabs['▁N']
        um_word = vocabs['UM']
        num_word = assigned_tokens['NUM']
        flag = False
        pos = []
        for i, token in enumerate(input_list):
            if token == n_word:
                flag = True
            elif flag == True and token == um_word:
                flag = False
                input_list.pop(i-1)
                input_list.pop(i-1)
                input_list.insert(i-1,num_word)
                pos.append(i-1)
            elif token == '<unk>' and len(assign_list):
                token_num = assign_list[0]
                assign_list.pop(0)
                input_list.pop(i)
                input_list.insert(i,token_num)
            else:
                pass
                
        return input_list,pos
    

    def token_handling(input_list):
        n_word = vocabs['NUM']
        num_word = vocabs['NUM']
        flag = False
        ''' 
        2054 == 'n', 17321 == '##um',
        41224 == 'NUM'
        '''
        pos = []
        for i, token in enumerate(input_list):
            if token == n_word:
                input_list.pop(i)
                input_list.insert(i,num_word)
                pos.append(i)
            else:
                pass
        return input_list,pos
    
    for pair in pairs_trained:
        if not use_tfm:
            input_lang.add_sen_to_vocab(pair[0])
        output_lang.add_sen_to_vocab(pair[1])
       
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    if not use_tfm:
        input_lang.build_input_lang(trim_min_count)
    
    print('convert to tokenized question pairs')
    for pair in tqdm(pairs_trained):
        num_stack = []  # 用于记录不在输出词典的数字
        for word in pair[1]:
            temp_num = []
            flag_not = True  # 用检查等式是否存在不在字典的元素
            if word not in output_lang.index2word:  # 如果该元素不在输出字典里
                flag_not = False
                for i, j in enumerate(pair[2]): # 遍历nums, 看是否存在
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])  # 生成从0到等式长度的数字

        num_stack.reverse()
        if not use_tfm:
            input_cell = indexes_from_sentence(input_lang, pair[0])
            pos = pair[3]
        else:
            if len(assigned_tokens):
                input_cell,pos = assigned_replacer(pair[0])
            else:
                input_cell = tokenizer.encode(' '.join(pair[0]),add_special_tokens=True)
                input_cell,pos = token_handling(input_cell)
            if not len(pos):
                pos = pair[3]
                
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # print(pair[1])
        # print(output_cell)
        if len(pair) == 4:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
            train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                                pair[2], pos, num_stack))
        else:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
            train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                                pair[2], pos, pair[4], num_stack)) #, pair[5]

    print('Indexed %d words in output' % (output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    
    for pair in tqdm(pairs_tested):
        num_stack = []
        for word in pair[1]:  # out_seq
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word: # 非符号，即word为数字
                flag_not = False
                for i, j in enumerate(pair[2]): # nums
                    if j == word:
                        temp_num.append(i) # 在等式的位置信息

            if not flag_not and len(temp_num) != 0:# 数字在数字列表中
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                # 数字不在数字列表中，则生成数字列表长度的位置信息，
                # 生成时根据解码器的概率选一个， 参见generate_tree_input
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        if not use_tfm:
            input_cell = indexes_from_sentence(input_lang, pair[0])
            pos = pair[3]
        else:
            if len(assigned_tokens):
                input_cell,pos = assigned_replacer(pair[0])
            else:
                input_cell = tokenizer.encode(' '.join(pair[0]),add_special_tokens=True)
                input_cell,pos = token_handling(input_cell)
            if not len(pos):
                pos = pair[3]
        
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        if len(pair) == 4:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                                  pair[2], pos, num_stack))
        else:
            # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
            # 입력, 입력길이, 수식, 수식길이, 수, 수의 위치, 정답, 스택은 모르겠다.
            test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                                pair[2], pos, pair[4], num_stack))

    print('Number of testind data %d' % (len(test_pairs)))
    if not use_tfm:
        return input_lang,output_lang, train_pairs, test_pairs
    else:
        return output_lang, train_pairs, test_pairs

# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []
    ans_flag = False if len(pairs[0]) == 7 else True
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
        if not ans_flag:
            for _, i, _, j, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)
        else:
            for _, i, _, j, _, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)

        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        if not ans_flag:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, num_stack in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
        else:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, ans, num_stack in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                ans_batch.append(ans)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        if ans_flag:
            ans_batches.append(ans_batch)
    if not ans_flag:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches,\
               num_pos_batches, num_size_batches
    else:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
               num_pos_batches, num_size_batches, ans_batches

# prepare the batches
def prepare_test_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    ans_batches = []
    ans_flag = False if len(pairs[0]) == 7 else True
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack)
        # pairs: (input_seq, input_len, eq_segs, eq_len, nums, num_pos, ans, num_stack)
        if not ans_flag:
            for _, i, _, j, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)
        else:
            for _, i, _, j, _, _, _, _ in batch:
                input_length.append(i)
                output_length.append(j)

        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        ans_batch = []
        if not ans_flag:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, num_stack in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
        else:
            for input_seq, input_seq_length, output_seq, output_seq_length, num, num_pos, ans, num_stack in batch:
                num_batch.append(num)
                input_batch.append(pad_seq(input_seq, input_seq_length, input_len_max))
                output_batch.append(pad_seq(output_seq, output_seq_length, output_len_max))
                num_stack_batch.append(num_stack)
                num_pos_batch.append(num_pos)
                num_size_batch.append(len(num_pos))
                ans_batch.append(ans)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        if ans_flag:
            ans_batches.append(ans_batch)
    if not ans_flag:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
               num_pos_batches, num_size_batches
    else:
        return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, \
               num_pos_batches, num_size_batches, ans_batches
