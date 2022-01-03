import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from model import *
from processing import *
from transformers import ElectraModel, ElectraConfig,ElectraTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#from GSM8K.grade_school_math.dataset import *
#위는 영어버전 활성화 시 추가 예정

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="solving math problems in korean with language model to tree decoder"
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        help="모델 타입 정의: 'electra','gpt2','bert' ",
        default='electra'
    )
    
    parser.add_argument(
        "-df",
        "--data_file",
        type=str,
        help="사용할 데이터 파일의 경로 명시 (json 파일 필요)",
        required=True
    )
    
    parser.add_argument(
        "-tp",
        "--tokenizer_path",
        type=str,
        help="huggingface에 등록된 tokenzier 모델 혹은 저장된 로컬 폴더 경로",
        default='monologg/koelectra-base-v3-discriminator'
    )
    return parser.parse_args()

def train():
    args  = parse_arguments()
    # 데이터셋 불러오기
    old_pairs, generate_nums, copy_nums = get_data(args.data_file)
    
    # 여러가지 정의..
    if args.model_type == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained(args.tokenizer_path)
        
        ### 우선은 확대하는 걸로 해보자.
        if args.tokenizer_path=='monologg/koelectra-base-v3-discriminator':
            tokenizer.add_tokens('NUM')
            tokenizer.add_tokens('㎤')
            tokenizer.add_tokens('㎡')
            tokenizer.add_tokens('㎥')
            tokenizer.save_pretrained('./temp_tokenizer')
            tokenizer = ElectraTokenizer.from_pretrained('./temp_tokenizer')
       
    temp_pairs = []
    for p in old_pairs:
        ept = ExpressionTree()
        ept.build_tree_from_infix_expression(p[1])
        #print('exp: ',ept.get_prefix_expression())
        if len(p) == 5:
            temp_pairs.append((p[0], ept.get_prefix_expression(), p[2], p[3],p[4]))
        else:
            temp_pairs.append((p[0], ept.get_prefix_expression(), p[2], p[3], p[4],p[5]))
    pairs = temp_pairs        
    output_lang, train_pairs, test_pairs = prepare_data(pairs, [], 5, tokenizer, generate_nums,copy_nums, tree=True,use_tfm=True)
    
    enc_conf = ElectraConfig.from_pretrained(args.tokenizer_path)
    dec_conf = TreeConfig(output_lang.n_words,generate_nums,copy_nums,var_nums=output_lang.index2var)
    
    
    model = get_model(args.model_type,enc_conf,dec_conf,len(tokenizer))
    
    epochs = 200

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    optimizer = optim.Adam(filtered_parameters, lr=dec_conf.learning_rate, betas=(0.9, 0.999), weight_decay=dec_conf.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=epochs)
    model.to(device)

    use_gpu = True
    n_gpu = torch.cuda.device_count()
    if use_gpu:
        model = torch.nn.DataParallel(model)
    #model.load_state_dict(torch.load('./For_Submission/1028_epoch34_model.bin'))
    acc = 0
    pred = 'none'
    out = 'none'
    question = 'none'
    for epoch in range(epochs):

        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
                            num_stack_batches, num_pos_batches, num_size_batches,ans_batches = prepare_train_batch(train_pairs, dec_conf.batch_size)

        model.train()

        with tqdm(range(len(input_batches))) as batches:
            for ith, batch in enumerate(batches):
                model.zero_grad()
                input_batch, input_length, output_batch, output_length, nums_batch,num_stack_batch, num_pos_batch, num_size_batch, ans_batch = \
                                                    input_batches[batch], input_lengths[batch], output_batches[batch], output_lengths[batch], \
                                                        nums_batches[batch],num_stack_batches[batch], num_pos_batches[batch], \
                                                        num_size_batches[batch], ans_batches[batch]

                loss, losses = model(input_batch, input_length, output_batch, output_length, num_stack_batch,num_size_batch,num_pos_batch, output_lang)
                if use_gpu:
                    loss.sum().backward()
                    cur_loss = (loss.sum()/len(loss)).item()
                    loss_total += cur_loss
                    batches.set_postfix({'loss':(loss_total/(ith+1))})
                else:
                    loss.backward()
                    cur_loss = loss.item()
                    loss_total += cur_loss
                    batches.set_postfix({'loss':(loss_total/(ith+1))})
                optimizer.step()
                scheduler.step()

                '''
                flag = False
                for t,loss_item in enumerate(losses[0]):
                    loss_list= [i for i in loss_item if i.item()>1.0e+5]
                    if len(loss_list):
                        flag = True
                        break
                if flag:
                    break
                '''
        acc = 0
        model.eval()
        with tqdm(test_pairs) as test_batches:
            for test_batch in test_batches:
                result = model.module.evaluate(test_batch[0], test_batch[1], output_lang,test_batch[5])
                if not test_batch[2]==result:
                    acc+=0
                else:
                    acc+=1
            prediction = [output_lang.index2word[i] for i in result]
            acc = acc/len(test_pairs)
            pred = ' '.join(prediction)
            out = ' '.join([output_lang.index2word[i] for i in test_batch[2]])
            question = ' '.join([tokenizer.convert_ids_to_tokens(i) for i in test_batch[0]])
            print(acc,pred,out,question)

        # 최종 result는 파일로, tokenizer와 pth, 로그 기록된 txt를 동봉할 것.




if __name__=='__main__':
    train()