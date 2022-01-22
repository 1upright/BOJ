# BOJ D2



## 1) [1330. 두 수 비교하기](https://www.acmicpc.net/problem/1330)

```python
A, B = map(int, input().split())
if A > B:
    print('>')
elif A < B:
    print('<')
else:
    print('==')
```



## 2) [9498. 시험 성적](https://www.acmicpc.net/problem/9498)

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



## 4) [14681. 사분면 고르기](https://www.acmicpc.net/problem/14681)

```python
x = int(input())
y = int(input())
if x > 0 and y > 0:
    print('1')
if x > 0 and y < 0:
    print('4')
if x < 0 and y < 0:
    print('3')
if x < 0 and y > 0:
    print('2')
```



## 5) [2884. 알람 시계](https://www.acmicpc.net/problem/2884)

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

