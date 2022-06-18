# BOJ CLASS 6

## 1) [2533. 사회망 서비스(SNS)](https://www.acmicpc.net/problem/2533)

```python
import sys; input = sys.stdin.readline; sys.setrecursionlimit(10**9)

def dfs(x):
    visited[x] = 1
    dp[x][0] = 1
    for y in graph[x]:
        if not visited[y]:
            dfs(y)
            dp[x][0] += min(dp[y])
            dp[x][1] += dp[y][0]

N = int(input())
graph = [[] for _ in range(N+1)]
for _ in range(N-1):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

dp = [[0, 0] for _ in range(N+1)]
visited = [0]*(N+1)

dfs(1)
print(min(dp[1]))
```



## 2) [14725. 개미굴](https://www.acmicpc.net/problem/14725)

```python
import sys; input = sys.stdin.readline

N = int(input())
data = [list(input().strip().split())[1:] for _ in range(N)]
data.sort()

floor = '--'
for i in range(len(data[0])):
    print(floor*i+data[0][i])

for i in range(1, N):
    k = 0
    for j in range(len(data[i])):
        if len(data[i-1]) <= j or data[i-1][j] != data[i][j]:
            break
        else:
            k += 1

    for j in range(k, len(data[i])):
        print(floor*j+data[i][j])
```



## 3) [13334. 철로](https://www.acmicpc.net/problem/13334)

```python
import sys; input = sys.stdin.readline
import heapq

N = int(input())
homes = [sorted(list(map(int, input().split()))) for _ in range(N)]
homes.sort(key=lambda x: x[1])
d = int(input())
res = 0
heap = []

for s, e in homes:
    tmp = e-d
    if tmp <= s:
        heapq.heappush(heap, s)
    while heap and heap[0] < tmp:
        heapq.heappop(heap)
    res = max(res, len(heap))

print(res)
```



## 4) [16565. N포커](https://www.acmicpc.net/problem/16565)

```python
from math import factorial

def ncr(n, r):
    return factorial(n) // factorial(r) // factorial(n-r)

N = int(input())

res = 0
for i in range(4, N+1, 4):
    tmp = ncr(13, i//4) * ncr(52-i, N-i)
    res += tmp if i//4%2 else -tmp
    res %= 10007

print(res)

# 더 깔끔하게
from math import comb

N = int(input())
res = 0
for i in range(4, N+1, 4):
    res += comb(13, i//4) * comb(52-i, N-i) * (i//4%2*2-1)
    res %= 10007

print(res)

# 숏코딩
from math import comb as c;N,r=int(input()),0
for i in range(4,N+1,4):r+=c(13,i//4)*c(52-i,N-i)*(i//4%2*2-1)
print(r%10007)
```



## 5) [1019. 책 페이지](https://www.acmicpc.net/problem/1019)

```python
N = int(input())
cnt = [0]*10
val = 1

while N > 0:
    while N % 10 != 9:
        for x in str(N):
            cnt[int(x)] += val
        N -= 1

    tmp = (N+1)//10*val
    for i in range(10):
        cnt[i] += tmp
    cnt[0] -= val
    val *= 10
    N //= 10

print(*cnt)
```



## 6) [2042. 구간 합 구하기](https://www.acmicpc.net/problem/2042)

```python
# 시간 초과
import sys; input = sys.stdin.readline

N, M, K = list(map(int, input().split()))
res = [0]
for i in range(N):
    x = int(input())
    res.append(res[i]+x)

for _ in range(M+K):
    a, b, c = map(int, input().split())
    if a == 1:
        tmp = c-res[b]+res[b-1]
        for i in range(b, N+1):
            res[i] += tmp
    elif a == 2:
        print(res[c]-res[b-1])
        
# 정답 - 인터넷 참고
import sys; input = sys.stdin.readline

def init(start, end, node):
    if start == end:
        tree[node] = l[start-1]
        return tree[node]
    
    mid = (start+end)//2
    tree[node] = init(start, mid, node*2) + init(mid+1, end, node*2+1)
    return tree[node]

def find(start, end, node, left, right):
    if left > end or right < start:
        return 0
    if left <= start and end <= right:
        return tree[node]
    
    mid = (start+end)//2
    return find(start, mid, node*2, left, right)+find(mid+1, end, node*2+1, left, right)

def update(start, end, node, idx, diff):
    if start <= idx <= end:
        tree[node] += diff
        if start != end:
            mid = (start+end)//2
            update(start, mid, node*2, idx, diff)
            update(mid+1, end, node*2+1, idx, diff)

N, M, K = map(int, input().split())
l = list(int(input()) for _ in range(N))
tree = [0]*(N*4)
init(1, N, 1)

for _ in range(M+K):
    a, b, c = map(int, input().split())
    if a == 1:
        tmp = c-l[b-1]
        l[b-1] = c
        update(1, N, 1, b, tmp)
    elif a == 2:
        print(find(1, N, 1, b, c))
```



## 7) [2357. 최솟값과 최댓값](https://www.acmicpc.net/problem/2357)

```python
# 당연히 시간초과
import sys; input = sys.stdin.readline

N, M = map(int, input().split())
nums = [int(input()) for _ in range(N)]

for _ in range(M):
    a, b = map(int, input().split())
    check = nums[a-1:b]
    print(min(check), max(check))
    
# 정답
import sys; input = sys.stdin.readline
from math import ceil, log2

def init_min(start, end, node):
    if start == end:
        tree_min[node] = l[start-1]
        return tree_min[node]

    mid = (start+end)//2
    tree_min[node] = min(init_min(start, mid, node*2), init_min(mid+1, end, node*2+1))
    return tree_min[node]

def init_max(start, end, node):
    if start == end:
        tree_max[node] = l[start-1]
        return tree_max[node]

    mid = (start+end)//2
    tree_max[node] = max(init_max(start, mid, node*2), init_max(mid+1, end, node*2+1))
    return tree_max[node]

def find_min(start, end, node, left, right):
    if left > end or right < start:
        return 1000000001

    if left <= start and end <= right:
        return tree_min[node]

    mid = (start+end)//2
    return min(find_min(start, mid, node*2, left, right), find_min(mid+1, end, node*2+1, left, right))


def find_max(start, end, node, left, right):
    if left > end or right < start:
        return 0

    if left <= start and end <= right:
        return tree_max[node]

    mid = (start+end)//2
    return max(find_max(start, mid, node*2, left, right), find_max(mid+1, end, node*2+1, left, right))

N, M = map(int, input().split())
size = 1<<int(ceil(log2(N)))+1
l = [int(input()) for _ in range(N)]
tree_min = [0]*size
tree_max = [0]*size

init_min(1, N, 1)
init_max(1, N, 1)

for _ in range(M):
    a, b = map(int, input().split())
    print(find_min(1, N, 1, a, b), find_max(1, N, 1, a, b))
```



## 8) [3015. 오아시스 재결합](https://www.acmicpc.net/problem/3015)

```python
import sys; input = sys.stdin.readline

N = int(input())
res = 0
stack = []

for _ in range(N):
    h = int(input())

    while stack and stack[-1][0] < h:
        res += stack.pop()[1]

    if stack and stack[-1][0] == h:
        tmp = stack.pop()[1]
        res += tmp

        if stack:
            res += 1
        stack.append((h, tmp+1))

    else:
        if stack:
            res += 1
        stack.append((h, 1))
print(res)
```



## 9) [11505. 구간 곱 구하기](https://www.acmicpc.net/problem/11505)

```python
import sys;input = sys.stdin.readline

def init(start, end, node):
    if start == end:
        tree[node] = l[start-1]
        return tree[node]

    mid = (start+end)//2
    tree[node] = init(start, mid, node*2)*init(mid+1, end, node*2+1)%MOD
    return tree[node]

def find(start, end, node, left, right):
    if left > end or right < start:
        return 1
    if left <= start and end <= right:
        return tree[node]

    mid = (start+end)//2
    return find(start, mid, node*2, left, right)*find(mid+1, end, node*2+1, left, right)%MOD

def update(start, end, node, idx, diff):
    if start <= idx <= end:
        if start == end:
            tree[node] = diff
            return

        mid = (start+end)//2
        update(start, mid, node*2, idx, diff)
        update(mid+1, end, node*2+1, idx, diff)
        tree[node] = tree[node*2]*tree[node*2+1]%MOD

MOD = 1000000007
N, M, K = map(int, input().split())
l = list(int(input()) for _ in range(N))
tree = [0]*(N*4)
init(1, N, 1)

for _ in range(M + K):
    a, b, c = map(int, input().split())
    if a == 1:
        l[b-1] = c
        update(1, N, 1, b, c)
    elif a == 2:
        print(find(1, N, 1, b, c))
```



## 10) [11689. GCD(n, k) = 1](https://www.acmicpc.net/problem/11689)

```python
# 인터넷 참고 - 오일러 피 함수
N = int(input())
res = N

for i in range(2, int(N**0.5)+1):
    if not N%i:
        while not N%i:
            N //= i
        res *= 1-1/i

if N > 1:
    res *= 1-1/N

print(int(res))
```



## 11) [13977. 이항 계수와 쿼리](https://www.acmicpc.net/problem/13977)

```python
# 인터넷 참고 - 이항 계수
import sys; input = sys.stdin.readline

MOD = 1000000007
l = 4000001
factorial = [1]*l

for i in range(1, l):
    factorial[i] = factorial[i-1]*i%MOD

for _ in range(int(input())):
    N, K = map(int, input().split())
    x, y, z, exp = factorial[N], factorial[K]*factorial[N-K]%MOD, 1, MOD-2
    while exp:
        if exp%2:
            z = y*z%MOD
        y = y*y%MOD
        exp //= 2

    print(x*z%MOD)
```



## 12) [14428. 수열과 쿼리 16](https://www.acmicpc.net/problem/14428)

```python
import sys; input = sys.stdin.readline
from math import ceil, log2

def init(start, end, node):
    if start == end:
        tree[node] = l[start-1]
        return tree[node]

    mid = (start+end)//2
    tree[node] = min(init(start, mid, node*2), init(mid+1, end, node*2+1))
    return tree[node]

def find(start, end, node, left, right):
    if left > end or right < start:
        return [1000000001, 1000000001]

    if left <= start and end <= right:
        return tree[node]

    mid = (start+end)//2
    return min(find(start, mid, node*2, left, right), find(mid+1, end, node*2+1, left, right))

def update(start, end, node, idx, diff):
    if start <= idx <= end:
        if start == end:
            tree[node] = diff
            return
        mid = (start+end)//2
        update(start, mid, node*2, idx, diff)
        update(mid+1, end, node*2+1, idx, diff)

        tree[node] = min(tree[node*2], tree[node*2+1])

N = int(input())
size = 1<<int(ceil(log2(N)))+1
tree = [0]*size

tmp = list(map(int, input().split()))
l = [[tmp[i], i+1] for i in range(N)]
init(1, N, 1)

M = int(input())
for _ in range(M):
    a, b, c = map(int, input().split())
    if a == 1:
        l[b-1][0] = c
        update(1, N, 1, b, l[b-1])
    elif a == 2:
        print(find(1, N, 1, b, c)[1])
```



## 13) [15824. 너 봄에는 캡사이신이 맛있단다](https://www.acmicpc.net/problem/15824)

```python
import sys; input = sys.stdin.readline

def dnc(x):
    if x == 0: return 1
    if x == 1: return 2
    tmp = dnc(x//2)
    if x % 2: return 2*tmp*tmp%MOD
    else: return tmp*tmp%MOD

N = int(input())
vals = sorted(list(map(int, input().split())))
MOD = 1000000007
res = 0

for i in range(N):
    res += vals[i]*(dnc(i)-dnc(N-i-1))

print(res%MOD)
```



## 14) [17371. 이사](https://www.acmicpc.net/problem/17371)

```python
import sys; input = sys.stdin.readline

N = int(input())
arr = [list(map(int, input().split())) for _ in range(N)]
m = 800000001
x, y = 0, 0
print(arr)
for x1, y1 in arr:
    mx = 0
    for x2, y2 in arr:
        tmp = (x1-x2)**2+(y1-y2)**2
        mx = max(mx, tmp)
    if m > mx:
        m = mx
        x, y = x1, y1

print(x, y)
```



## 15) [1086. 박성원](https://www.acmicpc.net/problem/1086)

```python

```



## 16) [1708. 볼록 껍질](https://www.acmicpc.net/problem/1708)

```python

```



## 17) [1761. 정점들의 거리](https://www.acmicpc.net/problem/1761)

```python

```



## 18) [1786. 찾기](https://www.acmicpc.net/problem/1786)

```python

```



## 19) [1948. 임계경로](https://www.acmicpc.net/problem/1948)

```python

```



## 20) [2150. Strongly Connected Component](https://www.acmicpc.net/problem/2150)

```python

```



## 21) [2243. 사탕상자](https://www.acmicpc.net/problem/2243)

```python

```



## 22) [2618. 경찰차](https://www.acmicpc.net/problem/2618)

```python

```



## 23) [5719. 거의 최단 경로](https://www.acmicpc.net/problem/5719)

```python

```



## 24) [6549. 히스토그램에서 가장 큰 직사각형](https://www.acmicpc.net/problem/6549)

```python

```



## 25) [7578. 공장](https://www.acmicpc.net/problem/7578)

```python

```



## 26) [11438. LCA 2](https://www.acmicpc.net/problem/11438)

```python

```



## 27) [13141. Ignition](https://www.acmicpc.net/problem/13141)

```python

```



## 28) [14942. 개미](https://www.acmicpc.net/problem/14942)

```python

```



## 29) [16287. Parcel](https://www.acmicpc.net/problem/16287)

```python

```



## 30) [17401. 일하는 세포](https://www.acmicpc.net/problem/17401)

```python

```



## 31) [1014. 컨닝](https://www.acmicpc.net/problem/1014)

```python

```



## 32) [3176. 도로 네트워크](https://www.acmicpc.net/problem/3176)

```python

```



## 33) [3648. 아이돌](https://www.acmicpc.net/problem/3648)

```python

```



## 34) [3653. 영화 수집](https://www.acmicpc.net/problem/3653)

```python

```



## 35) [3679. 단순 다각형](https://www.acmicpc.net/problem/3679)

```python

```



## 36) [5670. 휴대폰 자판](https://www.acmicpc.net/problem/5670)

```python

```



## 37) [11266. 단절점](https://www.acmicpc.net/problem/11266)

```python

```



## 38) [11280. 2-SAT - 3](https://www.acmicpc.net/problem/11280)

```python

```



## 39) [11400. 단절선](https://www.acmicpc.net/problem/11400)

```python

```



## 40) [13907. 세금](https://www.acmicpc.net/problem/13907)

```python

```



## 41) [20149. 선분 교차 3](https://www.acmicpc.net/problem/20149)

```python

```



## 42) [1006. 습격자 초라기](https://www.acmicpc.net/problem/1006)

```python

```



## 43) [1533. 길의 개수](https://www.acmicpc.net/problem/1533)

```python

```



## 44) [3830. 교수님은 기다리지 않는다](https://www.acmicpc.net/problem/3830)

```python

```



## 45) [4225. 쓰레기 슈트](https://www.acmicpc.net/problem/4225)

```python

```



## 46) [11281. 2-SAT - 4](https://www.acmicpc.net/problem/11281)

```python

```



## 47) [19585. 전설](https://www.acmicpc.net/problem/19585)

```python

```



## 48) [2261. 가장 가까운 두 점](https://www.acmicpc.net/problem/2261)

```python

```