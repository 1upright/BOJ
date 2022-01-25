# BOJ CLASS 1

## 1) [1008. A/B](https://www.acmicpc.net/problem/1008)

```python
A, B = map(int, input().split())
print(A / B)
```



## 2) [1330. 두 수 비교하기](https://www.acmicpc.net/problem/1330)

```python
A, B = map(int, input().split())
if A > B:
    print('>')
elif A < B:
    print('<')
else:
    print('==')
```



## 3) [2753. 윤년](https://www.acmicpc.net/problem/2753)

```python
year = int(input())
result = 0
if year % 4 == 0:
    result += 1
if year % 100 == 0:
    result -= 1
if year % 400 == 0:
    result += 1
print(result)
```



## 4) [9498. 시험 성적](https://www.acmicpc.net/problem/9498)

```python
score = int(input())
if 90 <= score <= 100:
    print('A')
elif 80 <= score:
    print('B')
elif 70 <= score:
    print('C')
elif 60 <= score:
    print('D')
else:
    print('F')
```



## 5) [2438. 별 찍기 - 1](https://www.acmicpc.net/problem/2438)

```python
N = int(input())
for i in range(1, N+1):
    print('*'*i)
```



## 6) [2439. 별 찍기 - 2](https://www.acmicpc.net/problem/2439)

```python
N = int(input())
for i in range(1, N+1):
    print(' '*(N-i)+'*'*i)
```



## 7) [2739. 구구단](https://www.acmicpc.net/problem/2739)

```python
N = int(input())
for i in range(1, 10):
    print(f'{N} * {i} = {N*i}')
```



## 8) [2741. N 찍기](https://www.acmicpc.net/problem/2741)

```python
N = int(input())
for i in range(1, N+1):
    print(i)
```



## 9) [2742. 기찍 N](https://www.acmicpc.net/problem/2742)

```python
N = int(input())
for i in range(1, N+1):
    print(N-i+1)
```



## 10) [2884. 알람 시계](https://www.acmicpc.net/problem/2884)

```python
H, M = map(int, input().split())
if H >= 1:
    if M >= 45:
        minute = M - 45
        hour = H
    else:
        minute = M + 15
        hour = H - 1

if H == 0:
    if M >= 45:
        minute = M - 45
        hour = 0
    else:
        minute = M + 15
        hour = 23

print(f'{hour} {minute}')
# print('{0} {1}'.format(hour, minute))
# print(hour, minute, end='')
```



## 11) [10818. 최소, 최대](https://www.acmicpc.net/problem/10818)

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



## 12) [10871. X보다 작은 수](https://www.acmicpc.net/problem/10871)

```python
N, X = map(int, input().split())
numbers = list(map(int, input().split()))
for i in range(N):
    if numbers[i] < X:
        print(numbers[i], end=' ')
```



## 13) [10950. A+B - 3](https://www.acmicpc.net/problem/10950)

```python
T = int(input())
for test_case in range(1, T+1):
    A, B = map(int, input().split())
    print(A + B)
```



## 14) [10951. A+B - 4](https://www.acmicpc.net/problem/10951)

```python
while 1:
    try:
        A, B = map(int, input().split())
        print(A + B)

    except:
        break
```



## 15) [10952. A+B - 5](https://www.acmicpc.net/problem/10952)

```python
while 1:
    A, B = map(int, input().split())
    if A == 0 and B == 0:
        break
    else:
        print(A + B)
```



## 16) [1152. 단어의 개수](https://www.acmicpc.net/problem/1152)

```python
input_str = input()
modified_str = input_str.title()
result = 0
for char in modified_str:
    if 65 <= ord(char) <= 90:
         result += 1
print(result) # 미친 짓인듯
```



```python
# 인터넷 검색 후
word = input().split()
print(len(word))
```



## 17) [2062. 최댓값](https://www.acmicpc.net/problem/2562)

```python
num_list = []
for i in range(9):
    num_list.append(int(input()))
print(max(num_list))
print(num_list.index(max(num_list)) + 1) #index함수 기억!
```



## 18) [2577. 숫자의 개수](https://www.acmicpc.net/problem/2577)

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



## 19) [2675. 문자열 반복](https://www.acmicpc.net/problem/2675)

```python
T = int(input())
for test_case in range(1, T + 1):
    R, S = input().split()
    result = ''
    for char in S:
        result += char * int(R)
    print(result)
```



## 20) [2908. 상수](https://www.acmicpc.net/problem/2908)

```python
A, B = input().split()
if int(A[::-1]) > int(B[::-1]):
    print(A[::-1])
else:
    print(B[::-1])
```



## 21) [2920. 음계](https://www.acmicpc.net/problem/2920)

```python
note = list(map(int, input().split()))
if note == [1, 2, 3, 4, 5, 6, 7, 8]:
    print('ascending')
elif note == [8, 7, 6, 5, 4, 3, 2, 1]:
    print('descending')
else:
    print('mixed')
```



## 22) [3052. 나머지](https://www.acmicpc.net/problem/3052)

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



## 23) [8958. OX퀴즈](https://www.acmicpc.net/problem/8958)

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



## 24) [10809. 알파벳 찾기](https://www.acmicpc.net/problem/10809)

```python
S = input()
for i in range(97, 123):
    print(S.find(chr(i)),end=' ')
```



## 25) [11720. 숫자의 합](https://www.acmicpc.net/problem/11720)

```python
N = int(input())
num = input()
result = 0
for i in range(N):
    result += int(num[i])
print(result)
```



## 26) [1157. 단어 공부](https://www.acmicpc.net/problem/1157)

```python
word = input()
u_word = word.upper()
count_list = []
for i in range(65,91):
    count_list.append(u_word.count(chr(i)))
most_used = []
for i in range(26):
    if count_list[i] == max(count_list):
        most_used.append(chr(i + 65))

if len(most_used) == 1:
    print(most_used[0])
else:
    print('?')
```



## 27) [1546. 평균](https://www.acmicpc.net/problem/1546)

```python
N = int(input())
subjects = list(map(int, input().split()))
sum_score = 0
for score in subjects:
    sum_score += (score/max(subjects)*100)
print(sum_score/N)
```
