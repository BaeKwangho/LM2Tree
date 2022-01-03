# LM2Tree


## Introduction

언어 모델과 트리 구조의 디코더를 사용하여 한국어 문장제 수학 문제 풀이를 학습 및 평가할 수 있는 모델입니다.


## Dataset Features

사용을 위한 데이터셋 구성은 다음과 같은 형태의 json 파일이어야 합니다.

```
{
    'file1':{
        "Question": "기계 a와 기계 b는 각각 330개의 스프로켓을 제조하는데 사용된다. 기계 b보다 330개의 스프로켓을 생산하는데 10시간이 더 걸린다. 기계 b는 기계 a보다 시간당 10% 더 많은 스프로켓을 생산합니다. 기계 a는 시간당 몇개의 스프로켓을 생산하는가?", 
        "Answer": 3, 
        "Equation": "ans =  divide(330, divide(multiply(multiply(10, 330), divide(add(100, 10), 100)), subtract(multiply(330, divide(add(100, 10), 100)), 330)))", 
        "Type": "5"
    },
    'file2':{
        "Question":"...",
        "Answer":"...",
        "Equation":"ans = ...",
    }
}
```

## Restriction
