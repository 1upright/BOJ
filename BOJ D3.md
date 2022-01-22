# BOJ D3

## 1) [2739. 구구단](https://www.acmicpc.net/problem/2739)

```python
N = int(input())
for i in range(1, 10):
    print(f'{N} * {i} = {N*i}')
```



## 2) [10950. A+B - 3](https://www.acmicpc.net/problem/10950)

```python
T = int(input())
for test_case in range(1, T+1):
    A, B = map(int, input().split())
    print(A + B)
```



## 3) [8393. 합](https://www.acmicpc.net/problem/8393)

```python
n = int(input())
count = 0
for i in range(1, n+1):
    count += i
print(count)
```



## 4) [15552. 빠른 A+B](https://www.acmicpc.net/problem/15552)

```python
import sys

T = int(input())
for test_case in range(1, T+1):
    A, B = map(int, sys.stdin.readline().split())
    print(A + B)
```



## 5) [2741. N 찍기](https://www.acmicpc.net/problem/2741)

```python
N = int(input())
for i in range(1, N+1):
    print(i)
```



## 6) [2742. 기찍 N](https://www.acmicpc.net/problem/2742)

```python
N = int(input())
for i in range(1, N+1):
    print(N-i+1)
```



## 7) [11021. A+B - 7](https://www.acmicpc.net/problem/11021)

```python
T = int(input())
for test_case in range(1, T+1):
    A, B = map(int, input().split())
    print(f'Case #{test_case}: {A+B}')
```



## 8) [11022. A+B - 8](https://www.acmicpc.net/problem/11022)

```python
T = int(input())
for test_case in range(1, T+1):
    A, B = map(int, input().split())
    print(f'Case #{test_case}: {A} + {B} = {A+B}')
```



## 9) [2438. 별 찍기 - 1](https://www.acmicpc.net/problem/2438)

```python
N = int(input())
for i in range(1, N+1):
    print('*'*i)
```



## 10) [2439. 별 찍기 - 2](https://www.acmicpc.net/problem/2439)

```python
N = int(input())
for i in range(1, N+1):
    print(' '*(N-i)+'*'*i)
```



## 11) [10871. X보다 작은 수](https://www.acmicpc.net/problem/10871)

```python
N, X = map(int, input().split())
numbers = list(map(int, input().split()))
for i in range(N):
    if numbers[i] < X:
        print(numbers[i], end=' ')
```

