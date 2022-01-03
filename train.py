import os

#from GSM8K.grade_school_math.dataset import *
#위는 영어버전 활성화 시 추가 예정

import os
import argparse
from model import *
from processing import *
from transformers import ElectraModel, ElectraConfig,ElectraTokenizer


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
    
    
    model = get_model(args.model_type,enc_conf,dec_conf)
    
    
    
    # 최종 result는 파일로, tokenizer와 pth, 로그 기록된 txt를 동봉할 것.




if __name__=='__main__':
    train()