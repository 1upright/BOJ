# BOJ D5



## 1) [10818. 최소, 최대](https://www.acmicpc.net/problem/10818)

```python
# N = int(input())
# num_list = list(map(int, input().split()))
# print(min(num_list),max(num_list),end=' ')

N = int(input())
num_list = list(map(int, input().split()))
minnum, maxnum = num_list[0], num_list[0]

for num in num_list:
    if num < minnum:
        minnum = num
    if num > maxnum:
        maxnum = num

print(minnum, maxnum)
```



## 2) [2062. 최댓값](https://www.acmicpc.net/problem/2562)

```python
num_list = []
for i in range(9):
    num_list.append(int(input()))
print(max(num_list))
print(num_list.index(max(num_list)) + 1) #index함수 기억!
```



## 3) [2577. 숫자의 개수](https://www.acmicpc.net/problem/2577)

```python
# 첫 풀이 - 노가다
A = int(input())
B = int(input())
C = int(input())

product = A * B * C
count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
count_6 = 0
count_7 = 0
count_8 = 0
count_9 = 0

for i in str(product):
    if i == '0':
        count_0 += 1
    if i == '1':
        count_1 += 1
    if i == '2':
        count_2 += 1
    if i == '3':
        count_3 += 1
    if i == '4':
        count_4 += 1
    if i == '5':
        count_5 += 1
    if i == '6':
        count_6 += 1
    if i == '7':
        count_7 += 1
    if i == '8':
        count_8 += 1
    if i == '9':
        count_9 += 1

print(count_0)
print(count_1)
print(count_2)
print(count_3)
print(count_4)
print(count_5)
print(count_6)
print(count_7)
print(count_8)
print(count_9)
```



```python
# 인터넷 검색 후
A = int(input())
B = int(input())
C = int(input())

product = list(str(A * B * C))
for i in range(10):
    print(product.count(str(i)))
```



## 4) [3052. 나머지](https://www.acmicpc.net/problem/3052)

```python
num_list = []
for i in range(10):
    num_list.append(int(input()))
remains = []
for num in num_list:
    remains.append(num % 42)
s_remains = sorted(remains)
count = 1
for i in range(1, 10):
    if s_remains[i] != s_remains[i-1]:
        count += 1
print(count)
```



## 5) [1546. 평균](https://www.acmicpc.net/problem/1546)

```python
N = int(input())
subjects = list(map(int, input().split()))
sum_score = 0
for score in subjects:
    sum_score += (score/max(subjects)*100)
print(sum_score/N)
```



## 6) [8958. OX퀴즈](https://www.acmicpc.net/problem/8958)

```python
T = int(input())
for test_case in range(1, T + 1):
    result = str(input())
    score = 0 #연속된 점수를 세기 위함
    sum_score = 0
    for case in result:
        if case == 'O':
            score += 1
        else:
            score = 0 #포인트
        sum_score += score
    print(sum_score)
```



## 7) [4344. 평균은 넘겠지](https://www.acmicpc.net/problem/4344)

```python
T = int(input())
for test_case in range(1, T + 1):
    infos = list(map(int, input().split()))
    N = infos[0]
    infos.remove(infos[0])
    avg = sum(infos) / N
    count = 0
    for score in infos:
        if score > avg:
            count += 1
    avg_rate = count / N * 100
    print(f'{avg_rate:.3f}%')
```

