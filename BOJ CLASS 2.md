# BOJ CLASS 2



## 1) [1085. 직사각형에서 탈출](https://www.acmicpc.net/problem/1085)

```python
x, y, w, h = map(int, input().split())
dis = [x, y, abs(x - w), abs(y - h)]
print(min(dis))
```



## 2) [4153. 직각삼각형](https://www.acmicpc.net/problem/4153)

```python
while 1:
    a, b, c = map(int, input().split())
    if a == 0 and b == 0 and c == 0:
        break
    nums = [a, b, c]
    nums.sort()
    if nums[0] ** 2 + nums[1] ** 2 == nums[2] ** 2:
        print('right')
    else:
        print('wrong')
```



## 3) [10250. ACM 호텔](https://www.acmicpc.net/problem/10250)

```python
T = int(input())
for test_case in range(T):
    H, W, N = map(int, input().split())

    a = N // H + 1
    b = N % H

    if N % H == 0:
        b = H
        a = N // H

    print(100 * b + a)
```



## 4) [2231. 분해합](https://www.acmicpc.net/problem/2231)

```python
N = int(input())
result = []
for i in range(N):
    answer = 0
    sepr_sum = 0
    for_count = 0
    sepr_sum += i
    for_count += i
    while i > 0:
        answer += (i % 10)
        i //= 10
    sepr_sum += answer
    if sepr_sum == N:
        result.append(for_count)
if result:
    print(min(result))
else:
    print(0)
```



## 5) [2292. 벌집](https://www.acmicpc.net/problem/2292)

```python
N = int(input())
count = 0
num = 1
while num < N:
    count += 1
    num += 6 * count
print(count + 1)
```



## 6) [2775. 부녀회장이 될테야](https://www.acmicpc.net/problem/2775)

```python
T = int(input())
for i in range(T):
    k, n = int(input()), int(input())
    k_floor_people = []
    for i in range(n):
        k_floor_people.append(i+1)
    for i in range(1, k + 1):
        for j in range(len(k_floor_people)-1, -1, -1):
            for l in range(j):
                k_floor_people[j] += k_floor_people[l]
    print(k_floor_people[n - 1])
```



## 7) [2789. 블랙잭](https://www.acmicpc.net/problem/2798)

```python
N, M = map(int, input().split())
cards = list(map(int, input().split()))
under_M = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            if cards[i] + cards[j] + cards[k] > M:
                continue
            else:
                under_M.append(cards[i] + cards[j] + cards[k])
print(max(under_M))
```



## 8) [15929. Hashing](https://www.acmicpc.net/problem/15829)

```python
# 50점
L = int(input())
alp_str = input()
hash_val = 0
for i in range(L):
    hash_val += (ord(alp_str[i]) - 96) * (31 ** i)
print(hash_val)

# 100점
L = int(input())
alp_str = input()
hash_val = 0
for i in range(L):
    hash_val += (ord(alp_str[i]) - 96) * (31 ** i)
print(hash_val % 1234567891)
```



## 9) [1259. 팰린드롬수](https://www.acmicpc.net/problem/1259)

```python
while 1:
    num = int(input())
    if num == 0:
        break
    result = 'yes'
    for i in range(len(str(num)) // 2):
        if str(num)[i] != str(num)[-i-1]:
            result = 'no'
            continue
    print(result)
```

