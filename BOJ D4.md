# BOJ D4

## 1) [10952. A+B - 5](https://www.acmicpc.net/problem/10952)

```python
while 1:
    A, B = map(int, input().split())
    if A == 0 and B == 0:
        break
    else:
        print(A + B)
```



## 2) [10951. A+B - 4](https://www.acmicpc.net/problem/10951)

```python
while 1:
    try:
        A, B = map(int, input().split())
        print(A + B)

    except:
        break
```



## 3) [1110. 더하기 사이클](https://www.acmicpc.net/problem/1110)

```python
N = int(input())
start_N = N
count = 0
while 1:
    N = (N % 10)*10 + (((N // 10)+(N % 10)) % 10)
    count += 1
    if N == start_N:
        break
print(count)
```

