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



## 10) [2839. 설탕 배달](https://www.acmicpc.net/problem/2839)

```python
# 실패 - 범위 날먹
N = int(input())
num_list = []
try:
    for i in range(1, 1001):
        for j in range(1, 1668):
                if 5 * i + 3 * j == N:
                    num_list.append(i + j)
    num_list.sort()
    print(num_list[0])
except IndexError:
    print(-1)
    
# 성공
N = int(input())
count = 0
while N >= 0:
    if N % 5 == 0:
        count += N // 5
        print(count)
        break
    N -= 3
    count += 1
else: # while문이 거짓이 될 경우
    print(-1)
```



## 11) [2869. 달팽이는 올라가고 싶다](acmicpc.net/problem/2869)

```python
# 시간 초과
A, B, V = map(int, input().split())
day_count = 0
height = 0
while height < V:
    day_count += 1
    height += A
    if height >= V:
        print(day_count)
        break
    height -= Bay_count = 0

# 정답
A, B, V = map(int, input().split())
from math import ceil
print(ceil((V - A) / (A - B)) + 1)

```



## 12) [11050. 이항 계수 1](https://www.acmicpc.net/problem/11050)

```python
N, K = map(int, input().split())
from math import factorial
print(int(factorial(N) / factorial(K) / factorial(N - K)))
```



## 13) [1018. 체스판 다시 칠하기](https://www.acmicpc.net/problem/1018)

```python
M, N = map(int, input().split())
arr = [list(input()) for _ in range(M)]
cnts = []
for i in range(M-7):
    for j in range(N-7):
        cnt1 = 0
        cnt2 = 0
        for k in range(i, i+8):
            for l in range(j, j+8):
                if (k+l)%2:
                    if arr[k][l] != 'B':
                        cnt1 += 1
                    if arr[k][l] != 'W':
                        cnt2 += 1
                else:
                    if arr[k][l] != 'W':
                        cnt1 += 1
                    if arr[k][l] != 'B':
                        cnt2 += 1
        cnts.append(cnt1)
        cnts.append(cnt2)

print(min(cnts))
```



## 14) [1181. 단어 정렬](https://www.acmicpc.net/problem/1181)

```python
# 시간 초과
N = int(input())
words = []
for _ in range(N):
    words.append(input())
word_lst = list(set(words))
for i in range(len(word_lst)):
    for j in range(i+1, len(word_lst)):
        if len(word_lst[i]) > len(word_lst[j]):
            word_lst[i], word_lst[j] = word_lst[j], word_lst[i]
        if len(word_lst[i]) == len(word_lst[j]):
            if word_lst[i] > word_lst[j]:
                word_lst[i], word_lst[j] = word_lst[j], word_lst[i]

print(word_lst)


# 다시 풀어봄 - 정답 => 튜플을 정리하는거였음
N = int(input())
words = []
for _ in range(N):
    words.append(input())
word_lst = list(set(words))

sorted_words = []
for word in word_lst:
    sorted_words.append((len(word), word))
sorted_words.sort()

for a, b in sorted_words:
    print(b)
    
# 인터넷에서 본 다른 풀이
n = int(input())

words = []
for _ in range(n):
    word = input()
    if word not in words:
        words.append(word)

words = sorted(words, key = lambda x : (len(x), x))
for word in words:
    print(word)
```



## 15) [1436. 영화감독 숌](https://www.acmicpc.net/problem/1436)

``` python
N = int(input())
key = 666
cnt = 1
while cnt != N:
    key += 1
    if '666' in str(key):
        cnt += 1
print(key)
```



## 16) [2609. 최대공약수와 최소공배수](https://www.acmicpc.net/problem/2609)

```python
# 첫 풀이
a, b = map(int, input().split())
for i in range(1, min(a,b)+1):
    if not a%i and not b%i:
        key = i
print(key)
print(a*b//key)

## 인터넷 도움 받은 유클리드 호제법
a, b = map(int, input().split())
l = [a, b]
while b > 0:
    a, b = b, a % b
print(a)
print(l[0]*l[1]//a)
```



## 17) [2751. 수 정렬하기](https://www.acmicpc.net/problem/2751)

```python
# 굉장히 오래걸림 => pypy3 사용시 어느정도 커버 가능
N = int(input())
nums = []
for _ in range(N):
    nums.append(int(input()))
for i in sorted(nums):
    print(i)
     
# 찾아보니 고급 정렬(병합 정렬, 퀵 정렬, 힙 정렬)을 이용해야 한다고 함 
```



## 18) [7568. 덩치](https://www.acmicpc.net/problem/7568)

```python
# 처음 했던 짓) 정렬하고 등수 동일화시키기! (오답)
# 두 명이 몸무게나 키가 서로 같을 경우를 설명 못해서 틀린듯
N = int(input())
bw = []
h = []
rank = [0]*N

for i in range(N):
    a, b = map(int, input().split())
    bw.append([i,a])
    h.append([i,b])

bw_list = sorted(bw, key = lambda x: x[1], reverse=True)
for i in range(N):
    bw_list[i].append(i+1)


h_list = []
for i in range(N):
    h_list.append(h[bw_list[i][0]])

for i in range(N-1):
    if h_list[i][1] < h_list[i+1][1]:
        bw_list[i+1][2] = bw_list[i][2]

for i in range(N):
    rank[bw_list[i][0]] = bw_list[i][2]

print(*rank)

### 아이디어) 나보다 덩치가 큰 사람의 수 + 1 == 등수일 것!
N = int(input())
ppl = []
rank = [0]*N
for _ in range(N):
    ppl.append(list(map(int, input().split())))
for i in range(N):
    cnt = 1
    for j in range(N):
        if ppl[i][0] < ppl[j][0] and ppl[i][1] < ppl[j][1]:
            cnt += 1
    rank[i] = cnt
print(*rank)
```



## 19) [10814. 나이순 정렬](https://www.acmicpc.net/problem/10814)

```python
# 실행 시간 길긴 함
N = int(input())
member = []
for i in range(N):
    age, name = map(str, input().split())
    member.append([int(age), i, name])
member.sort()
for i in range(N):
    print(f'{member[i][0]} {member[i][2]}')
```



## 20) [10989. 수 정렬하기 3](https://www.acmicpc.net/problem/10989)

```python
#
```



## 21) [11650. 좌표 정렬하기](https://www.acmicpc.net/problem/11650)

```python
# 실행 시간 길긴 함
N = int(input())
spot = []
for i in range(N):
    x, y = map(int, input().split())
    spot.append((x, y))
spot.sort()
for i in range(N):
    print(f'{spot[i][0]} {spot[i][1]}')
```



## 22) [11651. 좌표 정렬하기 2](https://www.acmicpc.net/problem/11651)

```python
# 실행 시간 길긴 함
N = int(input())
spot = []
for i in range(N):
    x, y = map(int, input().split())
    spot.append((y, x))
spot.sort()
for i in range(N):
    print(f'{spot[i][1]} {spot[i][0]}')

# 혹은
N = int(input())
spot = []
for i in range(N):
    x, y = map(int, input().split())
    spot.append((x, y))
spot = sorted(spot, key = lambda x : x[1])
for i in range(N):
    print(f'{spot[i][1]} {spot[i][0]}')
```



## 23) [1920. 수 찾기](https://www.acmicpc.net/problem/1920)

```python
#
```



## 24) [1978. 소수 찾기](https://www.acmicpc.net/problem/1978)

```python
# 소수 구하기를 먼저 했어서 쉽게 넘김
N = int(input())
nums = list(map(int, input().split()))
cnt = 0
for num in nums:
    if num == 1:
        continue
    for i in range(2, int(num**0.5)+1):
        if not num%i:
            break
    else:
        cnt += 1
print(cnt)
```



## 25) [2108. 통계학](https://www.acmicpc.net/problem/2108)

```python
#
```



## 26) [2164. 카드2](https://www.acmicpc.net/problem/2164)

```python
# 역시나 시간 초과
N = int(input())
nums = list(range(1, N+1))
while len(nums) > 1:
    for i in range(len(nums)):
        if not i%2:
            nums[i] = 0
    for num in nums:
        if not num:
            nums.remove(num)
print(nums[0])

# 덱
```



## 27) [4949. 균형잡힌 세상](https://www.acmicpc.net/problem/4949)

```python
def check(s):
    stack = []
    for x in s:
        if x == '[' or x == '(':
            stack.append(x)
        elif x == ')':
            if not stack:
                return 'no'
            if stack[-1] == '[':
                return 'no'
            stack.pop(-1)
        elif x == ']':
            if not stack:
                return 'no'
            if stack[-1] == '(':
                return 'no'
            stack.pop(-1)
    if stack:
        return 'no'
    return 'yes'

while 1:
    sent = input()
    if sent == '.':
        break
    else:
        print(check(sent))
```



## 28) [9012. 괄호](https://www.acmicpc.net/problem/9012)

```python
def c(s):
    st = []
    for x in s:
        if x == '(':
            st.append(x)
        if x == ')':
            if not st:
                return 'NO'
            st.pop(-1)
    if st:
        return 'NO'
    return 'YES'

for _ in range(int(input())):
    print(c(input()))
    
## 인터넷을 통해 찾은 숏코딩
exec(("print(['NO','YES'][not input()"+".replace('()','')"*25+"]);")*int(input()))
```



## 29) [10773. 제로](https://www.acmicpc.net/problem/10773)

```python
# 시간 초과
K = int(input())
s = []
for _ in range(K):
    n = int(input())
    if n:
        s.append(n)
    else:
        s.pop(-1)
print(sum(s))

# 정답
K = int(input())
res = 0
top = -1
s = [0]*K
for _ in range(K):
    n = int(input())
    if n:
        res += n
        top += 1
        s[top] = n
    else:
        res -= s[top]
        top -= 1
print(res)
```



## 30) [10816. 숫자 카드 2](https://www.acmicpc.net/problem/10816)

```python
# 시간 초과
N = int(input())
a = list(map(int, input().split()))
M = int(input())
for b in list(map(int, input().split())):
    print(a.count(b), end=' ')
   
#
```



## 31) [10828. 스택](https://www.acmicpc.net/problem/10828)

```python
# 시간 초과
N = int(input())
s = []
for _ in range(N):
    com = input()
    if com == 'pop':
        if s:
            print(s.pop())
        else:
            print(-1)
    elif com == 'size':
        print(len(s))
    elif com == 'empty':
        print((int(bool(s))-1)*(-1))
    elif com == 'top':
        if s:
            print(s[-1])
        else:
            print(-1)
    else:
        s.append(com[5:])
       
# 이것도 시간초과
N = int(input())
s = [0]*N
top = -1
for _ in range(N):
    c = input()
    if c == 'pop':
        if top == -1:
            print(-1)
        else:
            print(s[top])
            top -= 1
    elif c == 'size':
        print(top+1)
    elif c == 'empty':
        if top == -1:
            print(1)
        else:
            print(0)
    elif c == 'top':
        if top == -1:
            print(-1)
        else:
            print(s[top])
    else:
        top += 1
        s[top] = c[5:]
        
        
# 억까인듯
import sys
N = int(sys.stdin.readline())
s = []
for _ in range(N):
    com = list(sys.stdin.readline().split())
    if com[0] == 'push':
        s.append(com[1])
    if com[0] == 'pop':
        if s:
            print(s.pop())
        else:
            print(-1)
    if com[0] == 'size':
        print(len(s))
    if com[0] == 'empty':
        print((int(bool(s))-1)*(-1))
    if com[0] == 'top':
        if s:
            print(s[-1])
        else:
            print(-1)
```



## 32) [10845. 큐](https://www.acmicpc.net/problem/10845)

```python
import sys
N = int(sys.stdin.readline())
s = []
for _ in range(N):
    com = list(sys.stdin.readline().split())
    if com[0] == 'push':
        s.append(com[1])
    if com[0] == 'pop':
        if s:
            print(s.pop(0))
        else:
            print(-1)
    if com[0] == 'size':
        print(len(s))
    if com[0] == 'empty':
        print((int(bool(s))-1)*(-1))
    if com[0] == 'front':
        if s:
            print(s[0])
        else:
            print(-1)
    if com[0] == 'back':
        if s:
            print(s[-1])
        else:
            print(-1)
```



## 33) [10866. 덱](https://www.acmicpc.net/problem/10866)

```python
#
```



## 34) [11866. 요세푸스 문제 0](https://www.acmicpc.net/problem/11866)

```python
N, K = map(int, input().split())
ppl = list(range(1, N+1))
i = 1
print('<', end='')
while ppl:
    i = (i-1+K) % N
    if not i:
        i += N
    if len(ppl) == 1:
        print(ppl.pop(i-1), end='')
    else:
        print(ppl.pop(i-1), end=', ')
        N -= 1
print('>')
# 원래는 덱 문제인듯?
```



## 35) [1654. 랜선 자르기](https://www.acmicpc.net/problem/1654)

```python
K, N = map(int, input().split())
lines = [int(input()) for _ in range(K)]
start, end = 1, max(lines)

while start <= end:
    mid = (start+end)//2
    cnt = 0
    for line in lines:
        cnt += line//mid
    if cnt >= N:
        start = mid + 1
    else:
        end = mid - 1
print(end)
```



## 36) [1874. 스택 수열](https://www.acmicpc.net/problem/1874)

```python
# 시간초과
n = int(input())
nums = list(range(1, n+1))
s = []
res = ''

for _ in range(n):
    num = int(input())
    if s and s[-1] == num:
        res += '-'
        s.pop()
    elif num in s:
        res = 'NO'
        break
    else:
        while nums[0] != num:
            s.append(nums.pop(0))
            res += '+'
        nums.pop(0)
        res += '+-'

if res == 'NO':
    print(res)
else:
    for x in res:
        print(x)

# 정답
n = int(input())
record = 0
s = []
res = ''
for _ in range(n):
    num = int(input())
    if record < num:
        res += '+'*(num-record) + '-'
        s += list(range(record+1, num))
        record = num
    else:
        if s[-1] == num:
            s.pop()
            res += '-'
        else:
            res = 'NO'
            break
if res == 'NO':
    print(res)
else:
    for x in res:
        print(x)
```



## 37) [1966. 프린터 큐](https://www.acmicpc.net/problem/1966)

```python
def check():
    cnt = 1
    while 1:
        if arr[0] == max(arr):
            if idx[0] == M:
                return cnt
            else:
                arr.pop(0)
                idx.pop(0)
                cnt += 1
        else:
            arr.append(arr.pop(0))
            idx.append(idx.pop(0))

T = int(input())
for tc in range(T):
    N, M = map(int, input().split())
    arr = list(map(int, input().split()))
    idx = list(range(N))
    print(check())
```



## 38) [2805. 나무 자르기](https://www.acmicpc.net/problem/2805)

```python
import sys

N, M = map(int, sys.stdin.readline().split())
trees = list(map(int, sys.stdin.readline().split()))
start = 0
end = max(trees)
while start <= end:
    mid = (start + end) // 2
    cnt = 0
    for x in trees:
        if x > mid:
            cnt += x - mid
    if cnt >= M:
        start = mid + 1
    else:
        end = mid - 1
print(end)
```



## 39) [1929. 소수 구하기](https://www.acmicpc.net/problem/1929)

```python
# 시간 초과 - 최대한 줄인다고 i//2까지만 세도록 했는데도..
M, N = map(int, input().split())
for i in range(M, N+1):
    cnt = 2
    while 1:
        if i % cnt == 0:
            break
        cnt += 1
        if cnt >= i//2:
            print(i)
            break

# 시간 초과 - 돌리는 크기를 i**0.5+1까지로 줄여봤는데도..
M, N = map(int, input().split())
for i in range(M, N+1):
    cnt = 2
    while 1:
        if not i%cnt:
            break
        cnt += 1
        if cnt >= int(i**0.5)+1:
            print(i)
            break

# (오답) for 안의 while-break는 안되고 for안의 for-break는 돌아가나봄
# 근데 M,N에 1이 들어갈 수 있었음
M, N = map(int, input().split())
for i in range(M, N+1):
    for j in range(2, int(i**0.5)+1):
        if not i%j:
            break
    else:
        print(i)
        
# 정답
M, N = map(int, input().split())
for i in range(M, N+1):
    if i == 1:
        continue
    for j in range(2, int(i**0.5)+1):
        if not i%j:
            break
    else:
        print(i)
```



## 40) [18111. 마인크래프트](https://www.acmicpc.net/problem/18111)
```python
# 시간초과
N, M, B = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]
s = 256
e = 0
for i in range(N):
    for j in range(M):
        if arr[i][j] < s:
            s = arr[i][j]
        if arr[i][j] > e:
            e = arr[i][j]

min_time = 256*N*M*2
for x in range(s, e+1):
    time = 0
    block = B
    for i in range(N):
        for j in range(M):
            if arr[i][j] > x:
                time += 2*(arr[i][j]-x)
                block += arr[i][j]-x
            elif arr[i][j] < x:
                time += x-arr[i][j]
                block -= x-arr[i][j]
    if min_time > time and block>=0:
        min_time = time
        min_x = x
print(min_time, min_x)
# 억지
import sys
N, M, B = map(int, sys.stdin.readline().split())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
s = 256
e = 0
for i in range(N):
    for j in range(M):
        if arr[i][j] < s:
            s = arr[i][j]
        if arr[i][j] > e:
            e = arr[i][j]

min_time = 256*N*M*2
for x in range(s, e+1):
    ovr = 0
    undr = 0
    for i in range(N):
        for j in range(M):
            h = arr[i][j] - x
            if h>0:
                ovr += h
            elif h<0:
                undr += h
    if B + undr + ovr < 0:
        continue
    time = ovr*2 - undr
    if min_time >= time:
        min_time = time
        min_x = x
print(min_time, min_x)
```

