import re
import copy
import os
import json
from tqdm import tqdm

def get_data(json_path):
    if not json_path.split('.')[-1]=='json':
        return False
    with open(json_path) as f:
        data_list = json.load(f)
    
    for i,v in data_list.items():
        if 'Tree' in v.keys():
            v['Tree']=converter(v['Tree'],v['Answer'])
            continue
        result = converter(v['Equation'],v['Answer']) if 'Equation' in v.keys() else None
        if not isinstance(result,list):
            v['Error'] = result
        else:
            v['Converted'] = result
    converted_data = []

    for key,question in data_list.items():
        if 'Converted' in question.keys():
            qtype = 'Passage' if 'Passage' in question.keys() else 'Question'
            dtype = question['Type'] if 'Type' in question.keys() else 'None'
            converted_data.append({
                'iIndex':key,
                'dType':dtype,
                'tPassage':'',
                'sQuestion':question[qtype],
                'lEquations':'x = '+prefix_to_infix(question['Converted']),
                'lSolutions':question['Answer'],
            })
        elif 'Tree' in question.keys():
            qtype = 'Passage' if 'Passage' in question.keys() else 'Question'
            converted_data.append({
                'iIndex':key,
                'dType':dtype,
                'tPassage':'',
                'sQuestion':question[qtype],
                'lEquations':'x = '+prefix_to_infix(question['Tree']),
                'lSolutions':question['Answer'],
                'lTrees':question['Tree']
            })
        else:
            qtype = 'Passage' if 'Passage' in question.keys() else 'Question'
            converted_data.append({
                'iIndex':key,
                'dType':dtype,
                'tPassage':'',
                'sQuestion':question[qtype],
                'lErrors':question['Error'],
                'lSolutions':question['Answer'],
            })
    for i in converted_data:
        if check_transfer(i):
            i['Confirm'] = True
        else:
            i['Confirm'] = False
            
    count=0
    eqcount = 0
    types_suc = {}
    types_all = {}
    types_err = {}
    for i in converted_data:
        if not i['dType'] in types_err.keys():
            types_suc[i['dType']] = 0
            types_all[i['dType']] = 0
            types_err[i['dType']] = 0
        types_all[i['dType']]+=1
        if 'lEquations' in i.keys():
            eqcount+=1
        if i['Confirm']:
            count+=1
            types_suc[i['dType']]+=1
        else:
            if len(i['iIndex'])<25:
                pass
                #print(i)
            types_err[i['dType']]+=1
    print(count,eqcount,len(converted_data))
    for_save = []
    for j in converted_data:
        if j['Confirm']:
            for_save.append(j)
    old_pairs, generate_nums, copy_nums = transfer_custom(for_save,'testing generated questions')
    '''
    정답이랑 수식 결과 같은지 비교하는 코드
    '''

    cnt = 0
    final = []
    assert len(old_pairs)==len(for_save)
    for i,test_pair in enumerate(old_pairs):
        eq = infix_to_prefix(''.join(test_pair[1])[2:])
        nums = test_pair[2]
        try:
            result,ans = python_code_generator(eq,nums)
        except Exception as e:
            #print(e)
            pass
        if round(ans)!= test_pair[4]:
            if round(ans,2)!= test_pair[4]:
                if round(ans,3)!= test_pair[4]:
                    if int(ans)!=test_pair[4]:
                        cnt +=1
                        continue
        final.append(for_save[i])
    old_pairs, generate_nums, copy_nums = transfer_custom(final,'generating questions')
    return old_pairs, generate_nums, copy_nums
    
def transfer_custom(data_list,description='Transfer numbers...'): # transfer num into "NUM"
    print(description)
    # 15,000 , 
    pattern = re.compile("\d+,\d\d\d|\d+\.\d+|\d+|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in tqdm(data_list):
        nums = []
        input_seq = []
        seg = d['sQuestion'].strip().split()
        equations = d['lEquations']

        for s in seg:
            pos = re.findall(pattern, s)
            text = copy.deepcopy(s)
            if pos:
                end = 0
                for i,frac in enumerate(pos):
                    temp = text.find(frac)
                    if text[:temp]:
                        input_seq.append(text[:temp])
                    nums.append(text[temp:temp+len(frac)])
                    input_seq.append('NUM')
                    if len(pos)==1 or i==len(pos)-1:
                        input_seq.append(text[temp+len(frac):])
                    text = text[temp+len(frac):] 
            else:
                input_seq.append(text)

        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # print(nums)
        # print(nums_fraction)
        new_nums = []
        for num in nums:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                new_nums.append(str(num))
            else:
                new_nums.append(str(num))

        new_nums_fraction = []
        for num in nums_fraction:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                new_nums_fraction.append(str(num))
            else:
                new_nums_fraction.append(str(num))
        # print(float_nums)
        # print(float_nums_fraction)
        nums = new_nums
        nums_fraction = new_nums_fraction

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    elif nums.count(n) > 1:
                        # 多个的时候默认使用第一个index代替
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                elif nums.count(st_num) > 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        if 'lTrees' in d.keys():
            print(d['lTrees'])
            out_seq = d['lTrees']
        else:
            out_seq = seg_and_tag(equations)
        new_out_seq = []
        for seq in out_seq:
            if seq == ' ' or seq == '':
                continue
            if seq == ';':
                new_out_seq.append('SEP')
                continue
            new_out_seq.append(seq)
        out_seq = new_out_seq
        # print(equations)
        # print(' '.join(out_seq))
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # print(nums, num_pos)
        # if len(nums) == 0:
        #     print(d['iIndex'])
        pairs.append((input_seq, out_seq, nums, num_pos, d['lSolutions']))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

def check_transfer(d): # transfer num into "NUM"
    # 15,000 , 
    pattern = re.compile("\d+,\d\d\d|\d+\.\d+|\d+|\d+\.\d+%?|\d+%?")
    
    if 'lErrors' in d.keys():
        return False

        
    nums = []
    input_seq = []
    seg = d['sQuestion'].strip().split()
    equations = d['lEquations']
    
    for s in seg:
        pos = re.findall(pattern, s)
        text = copy.deepcopy(s)
        if pos:
            end = 0
            for i,frac in enumerate(pos):
                temp = text.find(frac)
                if text[:temp]:
                    input_seq.append(text[:temp])
                nums.append(text[temp:temp+len(frac)])
                input_seq.append('NUM')
                if len(pos)==1 or i==len(pos)-1:
                    input_seq.append(text[temp+len(frac):])
                text = text[temp+len(frac):] 
        else:
            input_seq.append(text)

    nums_fraction = []

    for num in nums:
        if re.search("\d*\(\d+/\d+\)\d*", num):
            nums_fraction.append(num)
    nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
    # print(nums)
    # print(nums_fraction)
    new_nums = []
    for num in nums:
        if ',' in num:
            new_num = []
            for c in num:
                if c == ',':
                    continue
                new_num.append(c)
            num = ''.join(new_num)
            new_nums.append(str(num))
        else:
            new_nums.append(str(num))

    new_nums_fraction = []
    for num in nums_fraction:
        if ',' in num:
            new_num = []
            for c in num:
                if c == ',':
                    continue
                new_num.append(c)
            num = ''.join(new_num)
            new_nums_fraction.append(str(num))
        else:
            new_nums_fraction.append(str(num))
    # print(float_nums)
    # print(float_nums_fraction)
    nums = new_nums
    nums_fraction = new_nums_fraction
    def seg_and_tag(st):  # seg the equation and tag the num
        res = []
        for n in nums_fraction:
            if n in st:
                p_start = st.find(n)
                p_end = p_start + len(n)
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                if nums.count(n) == 1:
                    res.append("N"+str(nums.index(n)))
                elif nums.count(n) > 1:
                    # 多个的时候默认使用第一个index代替
                    res.append("N"+str(nums.index(n)))
                else:
                    res.append(n)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res

        pos_st = re.search("\d+\.\d+%?|\d+%?", st)
        if pos_st:
            p_start = pos_st.start()
            p_end = pos_st.end()
            if p_start > 0:
                res += seg_and_tag(st[:p_start])
            st_num = st[p_start:p_end]
            if nums.count(st_num) == 1:
                res.append("N"+str(nums.index(st_num)))
            elif nums.count(st_num) > 1:
                res.append("N"+str(nums.index(st_num)))
            else:
                res.append(st_num)
            if p_end < len(st):
                res += seg_and_tag(st[p_end:])
            return res
        for ss in st:
            res.append(ss)
        return res
    if 'lTrees' in d.keys():
        print(d['lTrees'])
        out_seq = d['lTrees']
    else:
        out_seq = seg_and_tag(equations)
    new_out_seq = []
    for seq in out_seq:
        if seq == ' ' or seq == '':
            continue
        if seq == ';':
            new_out_seq.append('SEP')
            continue
        new_out_seq.append(seq)
    out_seq = new_out_seq
    # print(equations)
    # print(' '.join(out_seq))

    num_pos = []
    for i, j in enumerate(input_seq):
        if j == "NUM":
            num_pos.append(i)
    assert len(nums) == len(num_pos)
    # print(nums, num_pos)
    # if len(nums) == 0:
    #     print(d['iIndex'])
    
    if re.search('N\d',' '.join(out_seq)):
        return True
    else:
        #print(d['sQuestion'],' '.join(out_seq),sep='\n')
        return False

# 1. 괄호 통일
# 2. ans = 결과 통일

def converter(equation,answer):
    try:
        answer = float(answer)
    except:
        return 'span answer type'
    sb = equation.count('(')
    eb = equation.count(')')
    if '\n' in equation:
        return 'python code type answer'
    if not sb==eb:
        return 'bracket error'
    if equation.count('=')>1:
        return 'too many equal ops'

    # ans = 제거
    equation = re.sub('ans.*?=.*?','',equation).strip()
    
    if re.search(r'-\d.*',equation):
        return 'not support negative number as operand'
    
    def mk_tree(equation):
        no_sup = re.search('floor|factorial|combination|min|max|negate|permutation|pow',equation)
        if no_sup:
            return f'not supported operator, {equation[no_sup.start():no_sup.end()]}',None
        else:
            equation = infix_to_prefix(equation)
            return None,equation
        
    status,equation = mk_tree(equation)
    if status is None:
        return equation
    else:
        return status
    
def check_and_tokenize(expression):
    """Tokenizes the expression string into tokens.
    If it is already a list, then just ruturn it. If it is anything else, then we raise an exception.
        Example usage
        ---------------------------------
        >>> check_and_tokenize("2+3^(3-1)/4")
        ['2', '+', '3', '^', '(', '3', '-', '1', ')', '/', '4']
        >>> check_and_tokenize(['2', '+', '3', '^', '3', '/', '4'])
        ['2', '+', '3', '^', '3', '/', '4']
        >>> check_and_tokenize("420.08*2.3-420.08")
        ['420.08', '*', '2.3', '-', '420.08']
        >>> check_and_tokenize("420.08*<N0>")
        ['420.08', '*', '<N0>']
        >>> check_and_tokenize("rectangle_area(8, 18)")
        ['rectangle_area', '(', '8', ',', '18', ')']
        >>> check_and_tokenize(dict())
        Traceback (most recent call last):
            ...
        ValueError: The input expression cannot be processed.
    """
    if isinstance(expression, list):
        return expression
    elif isinstance(expression, str):
        expression = expression.replace(" ", "")
        tokens = re.split(r"([\[\]()+\-*/^,])", expression)
        tokens = [x for x in tokens if x]  # Get rid of empty tokens
        
        return tokens
    else:
        raise ValueError("The input expression cannot be processed.")


def _infix_to_postfix(expression_tokens):
    """
    The low-level conversion function for `infix_to_postfix()` and `infix_to_prefix()`
    See [1] for details of the algorithm.
        Example usage
        --------------------------------
        >>> _infix_to_postfix(["A", "+", "B"])
        ['A', 'B', '+']
        >>> _infix_to_postfix(["[", "(", "A", "+", "B", ")", "^", "(", "A", "-", "B", ")", "]", "/", "2"])
        ['A', 'B', '+', 'A', 'B', '-', '^', '2', '/']
    """
    op_stack = list()
    postfix_tokens = list()

    op_priority = {
        "(": -1,
        "[": -1,
        ",": 0,  # For custom functions, we need to pop all the ops when encoutering a comma
        "+": 1,
        "-": 1,
        "*": 2,
        "/": 2,
        "^": 3,
    }

    for token in expression_tokens:
        if token in ["(", "["]:
            op_stack.append(token)
        elif token in op_priority:  # +-*/^([
            while (len(op_stack) > 0) and (op_priority[token] < op_priority[op_stack[-1]]):
                postfix_tokens.append(op_stack.pop())
            op_stack.append(token)
        elif token == ")":
            while len(op_stack) > 0:
                last_op = op_stack.pop()
                if last_op == "(":
                    break
                else:
                    postfix_tokens.append(last_op)
        elif token == "]":
            while len(op_stack) > 0:
                last_op = op_stack.pop()
                if last_op == "[":
                    break
                else:
                    postfix_tokens.append(last_op)
        else:  # Numbers or variables
            postfix_tokens.append(token)

    # Place the remaining operators into the result
    while len(op_stack) > 0:
        postfix_tokens.append(op_stack.pop())

    return postfix_tokens

        
def infix_to_prefix(expression):
    """Convert an infix expression to prefix.
    See [2] for details of the algorithm.
    We also need to deal with custom formulas in the expressions like "triangle_area(5, 2) + 32".
    In this case, we just tokenize the formula and remove the commas because it is already in prefix order.  
        Example usage
        --------------------------------
        >>> infix_to_prefix("A+B")
        ['+', 'A', 'B']
        >>> infix_to_prefix("[(A+B)^(A-B)]/2")
        ['/', '^', '+', 'A', 'B', '-', 'A', 'B', '2']
        >>> infix_to_prefix("[cuboid_volume(A, B, C)^(A-B)]/2")
        ['/', '^', 'cuboid_volume', 'A', 'B', 'C', '-', 'A', 'B', '2']
        >>> infix_to_prefix("rectangle_perimeter(8+5*2, 8^3-2)")
        ['rectangle_perimeter', '+', '8', '*', '5', '2', '-', '^', '8', '3', '2']
    """
    custom= {
        'divide':'/',
        'add':'+',
        'multiply':'*',
        'subtract':'-',
        'substract':'-',
    }
    expression_tokens = check_and_tokenize(expression)

    # 1. Reverse the tokens and swap the right and left parentheses
    reversed_expression_tokens = list(reversed(expression_tokens))
    reversed_expression_tokens = [swap_parentheses(x)
                                  for x in reversed_expression_tokens]
    # 2. Do postfix conversion
    reversed_postfix_tokens = _infix_to_postfix(reversed_expression_tokens)
    reversed_postfix_tokens = [
        x for x in reversed_postfix_tokens if x != ","]  # Remove commas

    # 3. Reverse it back
    infix_tokens = list(reversed(reversed_postfix_tokens))

    new_tokens = []
    for i in infix_tokens:
        if i in custom.keys():
            new_tokens.append(custom[i])
        else:
            new_tokens.append(i)
    return new_tokens

def swap_parentheses(token):
    if token == "(":
        return ")"
    elif token == "[":
        return "]"
    elif token == ")":
        return "("
    elif token == "]":
        return "["
    else:
        return token
    
def prefix_to_infix(prefix):
    stack = []
    
    # read prefix in reverse order
    i = len(prefix) - 1
    while i >= 0:
        if not isOperator(prefix[i]):
             
            # symbol is operand
            stack.append(prefix[i])
            i -= 1
        else:
            # symbol is operator
            str = ' '.join(['(',stack.pop(),prefix[i],stack.pop(),')'])
            stack.append(str)
            i -= 1
     
    return stack.pop()
 
def isOperator(c):
    if c == "*" or c == "+" or c == "-" or c == "/" or c == "^" or c == "(" or c == ")":
        return True
    else:
        return False

def python_code_generator(pred, nums):
    if not isinstance(pred,list):
        pred = pred.split()
    infix_result = prefix_to_infix(pred)
    python_code = ''
    for i,num in enumerate(nums):
        infix_result = infix_result.replace('N'+str(i),str(num))
    python_code += f'print({infix_result})'
    answer = eval(infix_result)
    return python_code ,answer