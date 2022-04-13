# BOJ_CLASS_4

## 1)  [2047. 조합](https://www.acmicpc.net/problem/2407)

```python
from math import factorial
n, m = map(int, input().split())
print(factorial(n)//factorial(m)//factorial(n-m))
```



## 2) [15650. N과 M (2)](https://www.acmicpc.net/problem/15650) 

```python
# 치트키
from itertools import combinations
N, M = map(int, input().split())
arr = list(range(1, N+1))
for c in list(combinations(arr, M)):
    print(*c)

# import 없이
def dfs(i, M, tmp):
    if i == M:
        print(*arr)
        return
    else:
        for j in range(tmp, N+1):
            if j not in arr:
                arr[i] = j
                dfs(i+1, M, j)
                arr[i] = 0

N, M = map(int, input().split())
arr = [0]*M
dfs(0, M, 0)
```



## 3) [15652. N과 M (4)](https://www.acmicpc.net/problem/15652)

```python
def dfs(i, M, tmp):
    if i == M:
        print(*arr)
        return
    else:
        for j in range(tmp, N+1):
            arr[i] = j
            dfs(i+1, M, j)
            arr[i] = 0

N, M = map(int, input().split())
arr = [0]*M
dfs(0, M, 1)
```



## 4) [15654. N과 M (5)](https://www.acmicpc.net/problem/15654)

```python
def dfs(i, M):
    if i == M:
        print(*ls)
        return
    else:
        for j in range(N):
            if not visited[j]:
                ls[i] = arr[j]
                visited[j] = 1
                dfs(i+1, M)
                ls[i] = 0
                visited[j] = 0

N, M = map(int, input().split())
arr = list(map(int, input().split()))
arr.sort()
ls = [0]*M
visited = [0]*N
dfs(0, M)
```



## 5) [15657. N과 M (8)](https://www.acmicpc.net/problem/15657) 

```python
def dfs(i, M, tmp):
    if i == M:
        print(*ls)
        return
    else:
        for j in range(tmp, N):
            ls[i] = arr[j]
            dfs(i+1, M, j)
            ls[i] = 0

N, M = map(int, input().split())
arr = list(map(int, input().split()))
arr.sort()
ls = [0]*M
dfs(0, M, 0)
```



## 6) [11053. 가장 긴 증가하는 부분 수열](https://www.acmicpc.net/problem/11053) 

```python
N = int(input())
arr = list(map(int, input().split()))
dp = [1]*N
for i in range(1, N):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j]+1)
print(max(dp))
```



## 7) [11725. 트리의 부모 찾기](https://www.acmicpc.net/problem/11725)

```python
N = int(input())
tree = [[] for _ in range(N+1)]
for _ in range(N-1):
    p, c = map(int, input().split())
    tree[p].append(c)
    tree[c].append(p)

visited = [0]*(N+1)
visited[1] = 1

q = [1]
while q:
    v = q.pop(0)
    for x in tree[v]:
        if not visited[x]:
            q.append(x)
            visited[x] = v

for x in visited[2:N+1]:
    print(x)
```



## 8) [15663. N과 M (9)](https://www.acmicpc.net/problem/15663)

```python
# 시간 초과
def dfs(i, M):
    if i == M:
        tmp = list(map(str, ls))
        if tmp not in res:
            res.append(tmp)
        return
    for x in arr:
        ls.append(x)
        dfs(i+1, M)
        ls.pop()

N, M = map(int, input().split())
arr = list(map(int, input().split()))
arr.sort()
res = []
ls = []

dfs(0, M)
for x in res:
    print(*x)
    
# 정답
import sys

def dfs(i, M):
    if i == M:
        print(*ls)
        return
    tmp = 0
    for j in range(N):
        if not visited[j] and arr[j] != tmp:
            visited[j] = 1
            ls.append(arr[j])
            tmp = arr[j]
            dfs(i+1, M)
            visited[j] = 0
            ls.pop()

N, M = map(int, sys.stdin.readline().split())
arr = list(map(int, sys.stdin.readline().split()))
arr.sort()
ls = []
visited = [0]*N

dfs(0, M)
```



## 9) [15663. N과 M (12)](https://www.acmicpc.net/problem/15666) 

```python
import sys

def dfs(i, M):
    if i == M:
        print(*ls[1:])
        return
    tmp = 0
    for j in range(N):
        if ls[i] <= arr[j] and arr[j] != tmp:
            ls.append(arr[j])
            tmp = arr[j]
            dfs(i+1, M)
            ls.pop()

N, M = map(int, sys.stdin.readline().split())
arr = list(map(int, sys.stdin.readline().split()))
arr.sort()
ls = [0]
dfs(0, M)
```



## 10) [1149. RGB거리](https://www.acmicpc.net/problem/1149) 

```python
# 시간 초과
import sys

def dfs(i, N, s, tmp):
    global res
    if i == N:
        if res > s:
            res = s
        return
    for j in range(3):
        if tmp != j:
            dfs(i+1, N, s+arr[i][j], j)

N = int(sys.stdin.readline())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
res = 10000000
dfs(0, N, 0, -1)
print(res)

# 정답
import sys
N = int(sys.stdin.readline())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
for i in range(1, N):
    arr[i][0] = min(arr[i-1][1], arr[i-1][2]) + arr[i][0]
    arr[i][1] = min(arr[i-1][0], arr[i-1][2]) + arr[i][1]
    arr[i][2] = min(arr[i-1][0], arr[i-1][1]) + arr[i][2]
print(min(arr[N-1]))
```



## 11) [1629. 곱셈](https://www.acmicpc.net/problem/1629)

```python
# 나눗셈 분배 법칙 인터넷 참고
def boo(a, b, c):
    if b == 1:
        return a%c
    tmp = boo(a, b//2, c)
    if b%2:
        return tmp*tmp*a%c
    return tmp*tmp%c

a, b, c = map(int, input().split())
print(boo(a, b, c))
```



## 12) [1932. 정수 삼각형](https://www.acmicpc.net/problem/1932)

```python
N = int(input())
if N == 1:
    print(input())
else:
    arr = [list(map(int, input().split())) for _ in range(N)]
    dp = [[0]*i for i in range(3, N+3)]
    dp[0][1] = arr[0][0]
    for i in range(1, N):
        for j in range(1, i+2):
            dp[i][j] = max(dp[i-1][j-1]+arr[i][j-1], dp[i-1][j]+arr[i][j-1])
    print(max(dp[N-1]))
```



## 13) [1991. 트리 순회](https://www.acmicpc.net/problem/1991)

```python
import sys

def pre_order(p):
    if p != '.':
        print(p, end='')
        pre_order(tree[p][0])
        pre_order(tree[p][1])

def in_order(p):
    if p != '.':
        in_order(tree[p][0])
        print(p, end='')
        in_order(tree[p][1])

def post_order(p):
    if p != '.':
        post_order(tree[p][0])
        post_order(tree[p][1])
        print(p, end='')

N = int(sys.stdin.readline())
tree = {}
for i in range(N):
    p, c1, c2 = sys.stdin.readline().split()
    tree[p] = [c1, c2]
pre_order('A')
print()
in_order('A')
print()
post_order('A')
```



## 14) [9465. 스티커](https://www.acmicpc.net/problem/9465)

```python
import sys

T = int(input())
for _ in range(T):
    N = int(sys.stdin.readline())
    arr = [list(map(int, sys.stdin.readline().split())) for _ in range(2)]
    dp = [[0]*N for _ in range(2)]
    dp[0][0] = arr[0][0]
    dp[1][0] = arr[1][0]
    
    for i in range(1, N):
        if i == 1:
            dp[0][1] = arr[1][0]+arr[0][1]
            dp[1][1] = arr[0][0]+arr[1][1]
        else:
            dp[0][i] = max(dp[1][i-1]+arr[0][i], dp[1][i-2]+arr[0][i])
            dp[1][i] = max(dp[0][i-1]+arr[1][i], dp[0][i-2]+arr[1][i])
    
    print(max(dp[0][N-1], dp[1][N-1]))
```



## 15) [11660. 구간 합 구하기 5](https://www.acmicpc.net/problem/11660) 

```python
# 시간 초과
import sys
N, M = map(int, sys.stdin.readline().split())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
for _ in range(M):
    x1, y1, x2, y2 = map(int, sys.stdin.readline().split())
    cnt = 0
    for i in range(x1-1, x2):
        cnt += sum(arr[i][y1-1:y2])
    print(cnt)

# 정답
import sys
N, M = map(int, sys.stdin.readline().split())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
cnt = [[0]*(N+1) for _ in range(N+1)]

for i in range(N):
    for j in range(N):
        cnt[i+1][j+1] = cnt[i][j+1] + cnt[i+1][j] - cnt[i][j] + arr[i][j]

for _ in range(M):
    x1, y1, x2, y2 = map(int, sys.stdin.readline().split())
    print(cnt[x2][y2] + cnt[x1-1][y1-1] - cnt[x1-1][y2] - cnt[x2][y1-1])
```



## 16) [16953. A → B](https://www.acmicpc.net/problem/16953)

```python
# 메모리 초과
from collections import deque

def bfs(a, b):
    q = deque([a])
    visited[a] = 1
    while q:
        v = q.popleft()
        if v == b:
            return visited[v]
        for x in [v*2, v*10+1]:
            if x<=B and not visited[x]:
                q.append(x)
                visited[x] = visited[v]+1
    return -1

A, B = map(int, input().split())
visited = [0]*(B+1)
print(bfs(A, B))

# 정답
from collections import deque

def bfs(a, b):
    q = deque([(a, 1)])
    while q:
        v, cnt = q.popleft()
        if v == b:
            return cnt
        for x in [v*2, v*10+1]:
            if x<=B:
                q.append((x, cnt+1))
    return -1

A, B = map(int, input().split())
print(bfs(A, B))
```



## 17) [1753. 최단경로](https://www.acmicpc.net/problem/1753)

```python
# 메모리 초과
import sys

def dijkstra(s, N):
    U = [0]*(N+1)
    U[s] = 1
    for i in range(N+1):
        D[i] = arr[s][i]

    for _ in range(N+1):
        minV = INF
        w = 0
        for i in range(N+1):
            if not U[i] and minV > D[i]:
                minV = D[i]
                w = i
        U[w] = 1
        for v in range(N+1):
            if 0<arr[w][v]<INF:
                D[v] = min(D[v], D[w]+arr[w][v])

V, E = map(int, sys.stdin.readline().split())
n = int(sys.stdin.readline())
INF = 999999
arr = [[INF]*V for _ in range(V)]

for i in range(V):
    arr[i][i] = 0

for _ in range(E):
    u, v, w = map(int, sys.stdin.readline().split())
    arr[u-1][v-1] = w

D = [0]*V
dijkstra(n-1, V-1)
for x in D:
    print("INF" if x == INF else x)
    
# 시간 초과
import sys

def dijkstra(s, V):
    U = [0]*(V+1)
    U[s] = 1
    D[s] = 0
    for v, w in adj[s]:
        D[v] = w

    for _ in range(V):
        minV = INF
        t = 0
        for i in range(V+1):
            if U[i] == 0 and minV > D[i]:
                minV = D[i]
                t = i
        U[t] = 1
        for v, w in adj[t]:
            D[v] = min(D[v], D[t]+w)

V, E = map(int, sys.stdin.readline().split())
n = int(sys.stdin.readline())
INF = 999999
adj = [[] for _ in range(V+1)]
for _ in range(E):
    u, v, w = map(int, sys.stdin.readline().split())
    adj[u].append([v, w])
D = [INF]*(V+1)
dijkstra(n, V)
for i in range(1, V+1):
    print('INF' if D[i] == INF else D[i])
    
# 인터넷 참고 정답
import sys
import heapq

def dijkstra(s):
    D[s] = 0
    heapq.heappush(heap, [0, s])
    while heap:
        val, i = heapq.heappop(heap)
        for v, w in adj[i]:
            tmp = w + val
            if D[v] > tmp:
                D[v] = tmp
                heapq.heappush(heap, [tmp, v])

V, E = map(int, sys.stdin.readline().split())
n = int(sys.stdin.readline())
INF = 999999
adj = [[] for _ in range(V+1)]
for _ in range(E):
    u, v, w = map(int, sys.stdin.readline().split())
    adj[u].append([v, w])
D = [INF]*(V+1)
heap = []
dijkstra(n)
for x in D[1:]:
    print('INF' if x == INF else x)
```



## 18) [1916. 최소비용 구하기](https://www.acmicpc.net/problem/1916)

```python
# 시간 초과
import sys, heapq

def dijkstra(s):
    D[s] = 0
    heapq.heappush(heap, [0, s])
    while heap:
        val, i = heapq.heappop(heap)
        for v, w in adj[i]:
            tmp = w + val
            if D[v] > tmp:
                D[v] = tmp
                heapq.heappush(heap, [tmp, v])

V = int(sys.stdin.readline())
E = int(sys.stdin.readline())
adj = [[] for _ in range(V+1)]

for _ in range(E):
    u, v, w = map(int, sys.stdin.readline().split())
    adj[u].append([v, w])
s, e = map(int, sys.stdin.readline().split())

INF = 100000000
D = [INF]*(V+1)
heap = []
dijkstra(s)
print(D[e])

# 정답
import sys, heapq

def dijkstra(s, e):
    D[s] = 0
    heapq.heappush(heap, [0, s])
    while heap:
        val, i = heapq.heappop(heap)
        if i == e:
            return D[e]
        for v, w in adj[i]:
            tmp = w + val
            if D[v] > tmp:
                D[v] = tmp
                heapq.heappush(heap, [tmp, v])

V = int(sys.stdin.readline())
E = int(sys.stdin.readline())
adj = [[] for _ in range(V+1)]

for _ in range(E):
    u, v, w = map(int, sys.stdin.readline().split())
    adj[u].append([v, w])
s, e = map(int, sys.stdin.readline().split())

INF = 100000000
D = [INF]*(V+1)
heap = []
print(dijkstra(s, e))
```



## 19) [5639. 이진 검색 트리](https://www.acmicpc.net/problem/5639)

```python

```



## 20) [9251. LCS](https://www.acmicpc.net/problem/9251)

```python

```



## 21) [9663. N-Queen](https://www.acmicpc.net/problem/9663)

```python

```



## 22) [12851. 숨바꼭질 2](https://www.acmicpc.net/problem/12851)

```python

```



## 23) [12865. 평범한 배낭](https://www.acmicpc.net/problem/12865)

```python

```



## 24) [13549. 숨바꼭질 3](https://www.acmicpc.net/problem/13549)

```python

```



## 25) [14502. 연구소](https://www.acmicpc.net/problem/14502)

```python

```



## 26) [15686. 치킨 배달](https://www.acmicpc.net/problem/15686)

```python

```



## 27) [17070. 파이프 옮기기 1](https://www.acmicpc.net/problem/17070)

```python

```



## 28) [1043. 거짓말](https://www.acmicpc.net/problem/1043)

```python

```



## 29) [1504. 특정한 최단 경로](https://www.acmicpc.net/problem/1504)

```python

```



## 30) [1967. 트리의 지름](https://www.acmicpc.net/problem/1967)

```python

```



## 31) [2096. 내려가기](https://www.acmicpc.net/problem/2096)

```python

```



## 32) [2206. 벽 부수고 이동하기](https://www.acmicpc.net/problem/2206)

```python

```



## 33) [2448. 별 찍기 - 11](https://www.acmicpc.net/problem/2448)

```python

```



## 34) [2638. 치즈](https://www.acmicpc.net/problem/2638)

```python

```



## 35) [9935. 문자열 폭발](https://www.acmicpc.net/problem/9935)

```python

```



## 36) [10830. 행렬 제곱](https://www.acmicpc.net/problem/10830)

```python

```



## 37) [11404. 플로이드](https://www.acmicpc.net/problem/11404)

```python

```



## 38) [13172. Σ](https://www.acmicpc.net/problem/13172)

```python

```



## 39) [14938. 서강그라운드](https://www.acmicpc.net/problem/14938)

```python

```



## 40) [17144. 미세먼지 안녕!](https://www.acmicpc.net/problem/17144)

```python

```



## 41) [1167. 트리의 지름](https://www.acmicpc.net/problem/1167)

```python

```



## 42) [1238. 파티](https://www.acmicpc.net/problem/1238)

```python

```



## 43) [1865. 웜홀](https://www.acmicpc.net/problem/1865)

```python

```



## 44) [1918. 후위 표기식](https://www.acmicpc.net/problem/1918)

```python

```



## 45) [11054. 가장 긴 바이토닉 부분 수열](https://www.acmicpc.net/problem/11054)

```python

```



## 46) [11779. 최소비용 구하기 2](https://www.acmicpc.net/problem/11779)

```python

```



## 47) [2263. 트리의 순회](https://www.acmicpc.net/problem/2263)

```python

```



## 48) [11444. 피보나치 수 6](https://www.acmicpc.net/problem/11444)

```python

```

