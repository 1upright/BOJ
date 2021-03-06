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
# 인터넷 참고
import sys
sys.setrecursionlimit(10**9)

def post_order(s, e):
    if s >= e:
        return
    root = pre_order[s]
    idx = s+1

    for i in range(s+1, e):
        if pre_order[i] > root:
            idx = i
            break

    post_order(s+1, idx)
    post_order(idx, e)
    print(root)
    return

pre_order = []
while 1:
    try:
        pre_order.append(int(sys.stdin.readline()))
    except:
        break

post_order(0, len(pre_order))
```



## 20) [9251. LCS](https://www.acmicpc.net/problem/9251)

```python
import sys
s1 = list(sys.stdin.readline().rstrip())
s2 = list(sys.stdin.readline().rstrip())
N, M = len(s1), len(s2)
arr = [[0]*(M+1) for _ in range(N+1)]

for i in range(1, N+1):
    for j in range(1, M+1):
        if s1[i-1] == s2[j-1]:
            arr[i][j] = arr[i-1][j-1] + 1
        else:
            arr[i][j] = max(arr[i-1][j], arr[i][j-1])

print(arr[N][M])
```



## 21) [9663. N-Queen](https://www.acmicpc.net/problem/9663)

```python
# 시간 초과
def foo(i, n):
    global res
    if i == n:
        res += 1
        return
    for j in range(n):
        v[i] = j
        if bar(i):
            foo(i+1, N)

def bar(x):
    for i in range(x):
        if v[i] == v[x] or abs(i-x) == abs(v[i]-v[x]):
            return 0
    return 1

N = int(input())
res = 0
v = [0]*N
foo(0, N)
print(res)

# pypy로 돌려야 성공
def dfs(i, n):
    global res
    if i == n:
        res += 1
        return

    for j in range(n):
        if v1[j]==v2[i+j]==v3[i-j]==0:
            v1[j] = v2[i+j] = v3[i-j] = 1
            dfs(i+1, n)
            v1[j] = v2[i+j] = v3[i-j] = 0

N = int(input())
res = 0
v1, v2, v3 = [0]*30, [0]*30, [0]*30
dfs(0, N)
print(res)
```



## 22) [12851. 숨바꼭질 2](https://www.acmicpc.net/problem/12851)

```python
from collections import deque

N, K = map(int, input().split())
q = deque([N])
visited = [0]*100001
visited[N] = 1
cnt = 0

while q:
    v = q.popleft()
    if v==K:
        cnt += 1
    for x in [v+1, v-1, v*2]:
        if 0<=x<=100000 and (not visited[x] or visited[x]==visited[v]+1):
            visited[x] = visited[v]+1
            q.append(x)

print(visited[K]-1)
print(cnt)
```



## 23) [12865. 평범한 배낭](https://www.acmicpc.net/problem/12865)

```python
import sys
input = sys.stdin.readline

N, K = map(int, input().split())
dp = [[0]*(K+1) for _ in range(N+1)]
bag = [[0, 0]] + [list(map(int, input().split())) for _ in range(N)]

for i in range(1, N+1):
    for j in range(1, K+1):
        w, v = bag[i][0], bag[i][1]
        if j < w:
            dp[i][j] = dp[i-1][j]
        else:
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-w]+v)

print(dp[N][K])
```



## 24) [13549. 숨바꼭질 3](https://www.acmicpc.net/problem/13549)

```python
# v*2를 먼저 봐줘야함
from collections import deque

N, K = map(int, input().split())
visited = [0]*100001
visited[N] = 1
q = deque([N])
while q:
    v = q.popleft()
    if v == K:
        print(visited[v]-1)
        break
    
    tmp = v*2
    if tmp <= 100000 and not visited[tmp]:
        visited[tmp] = visited[v]
        q.appendleft(tmp)
    
    for x in [v+1, v-1]:
        if 0<=x<=100000 and not visited[x]:
            visited[x] = visited[v]+1
            q.append(x)
```



## 25) [14502. 연구소](https://www.acmicpc.net/problem/14502)

```python
import sys
input = sys.stdin.readline
from itertools import combinations
from copy import deepcopy
from collections import deque

def infection(i, j):
    q = deque([(i, j)])
    while q:
        si, sj = q.popleft()
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = si+di, sj+dj
            if 0<=ni<N and 0<=nj<M and not tmp[ni][nj]:
                tmp[ni][nj] = 2
                q.append((ni, nj))


N, M = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]

virus = []
safe = []
for i in range(N):
    for j in range(M):
        if not arr[i][j]:
            safe.append((i, j))
        if arr[i][j] == 2:
            virus.append((i, j))

res = 0
for com in list(combinations(safe, 3)):
    tmp = deepcopy(arr)
    for i, j in com:
        tmp[i][j] = 1
    for i, j in virus:
        infection(i, j)
    cnt = 0
    for i in range(N):
        for j in range(M):
            if not tmp[i][j]:
                cnt += 1
    if res < cnt:
        res = cnt
print(res)
```



## 26) [15686. 치킨 배달](https://www.acmicpc.net/problem/15686)

```python
# 시간 초과
import sys
input = sys.stdin.readline
from itertools import combinations
from copy import deepcopy
from collections import deque

def chk_dis(i, j):
    global cnt
    visited = [[0]*N for _ in range(N)]
    visited[i][j] = 1
    q = deque([(i, j)])
    while q:
        si, sj = q.popleft()
        if tmp[si][sj] == 2:
            cnt += visited[si][sj]-1
            return
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = si+di, sj+dj
            if 0<=ni<N and 0<=nj<N and not visited[ni][nj]:
                visited[ni][nj] = visited[si][sj] + 1
                q.append((ni, nj))

N, M = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]

chk = []
for i in range(N):
    for j in range(N):
        if arr[i][j] == 2:
            chk.append((i, j))

res = 99999
for com in combinations(chk, len(chk)-M):
    tmp = deepcopy(arr)
    for i, j in com:
        tmp[i][j] = 0
    cnt = 0
    for i in range(N):
        for j in range(N):
            if tmp[i][j] == 1:
                chk_dis(i, j)
    if res > cnt:
        res = cnt
print(res)

# 정답
import sys
input = sys.stdin.readline
from itertools import combinations

N, M = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]

home = []
chk = []
for i in range(N):
    for j in range(N):
        if arr[i][j] == 2:
            chk.append((i, j))
        if arr[i][j] == 1:
            home.append((i, j))

res = 999999
for com in combinations(chk, M):
    cnt = 0
    for h in home:
        cnt += min([abs(h[0]-c[0])+abs(h[1]-c[1]) for c in com])
    if res > cnt:
        res = cnt
print(res)
```



## 27) [17070. 파이프 옮기기 1](https://www.acmicpc.net/problem/17070)

```python
import sys
input = sys.stdin.readline

def dfs(i, j, k):
    global cnt
    if i == N-1 and j == N-1:
        cnt += 1
        return

    if k == 1 or k == 3:
        if j+1<N and not arr[i][j+1]:
            dfs(i, j+1, 1)
    if k == 2 or k == 3:
        if i+1<N and not arr[i+1][j]:
            dfs(i+1, j, 2)
    if i+1<N and j+1<N and not arr[i][j+1] and not arr[i+1][j] and not arr[i+1][j+1]:
        dfs(i+1, j+1, 3)

N = int(input())
arr = [list(map(int, input().split())) for _ in range(N)]
cnt = 0
dfs(0, 1, 1)
print(cnt)
```



## 28) [1043. 거짓말](https://www.acmicpc.net/problem/1043)

```python
import sys
input = sys.stdin.readline

N, M = map(int, input().split())
truth = list(map(int, input().split()))[1:]
visited = [0]*(N+1)
for t in truth:
    visited[t] = 1

party = []
for _ in range(M):
    p = list(map(int, input().split()))[1:]
    party.append(p)

res = [0]*M
while truth:
    v = truth.pop(0)

    check = set()
    for i in range(M):
        if v in party[i]:
            check |= set(party[i])
            res[i] = 1

    for c in check:
        if not visited[c]:
            visited[c] = 1
            truth.append(c)

print(res.count(0))
```



## 29) [1504. 특정한 최단 경로](https://www.acmicpc.net/problem/1504)

```python
# 시간 초과
import sys
input = sys.stdin.readline

def dfs(now, s, lst):
    global res
    if now == N and v1 in lst and v2 in lst:
        if res > s:
            res = s
        return
    for x in range(N+1):
        if graph[now][x] and not visited[x]:
            visited[x] = 1
            dfs(x, s+graph[now][x], lst+[x])
            visited[x] = 0

N, E = map(int, input().split())

graph = [[0]*(N+1) for _ in range(N+1)]
for _ in range(E):
    a, b, c = map(int, input().split())
    graph[a][b] = c
    graph[b][a] = c

v1, v2 = map(int, input().split())

visited = [0]*(N+1)
visited[1] = 1
res = 9999999
dfs(1, 0, [1])
print(res if res<9999999 else -1)

# 시간 초과 2
import sys
input = sys.stdin.readline

N, E = map(int, input().split())
INF = 9999999
graph = [[INF]*N for _ in range(N)]
for i in range(N):
    graph[i][i] = 0
for _ in range(E):
    a, b, c = map(int, input().split())
    graph[a-1][b-1] = c
    graph[b-1][a-1] = c

v1, v2 = map(int, input().split())

for k in range(N):
    for i in range(N):
        for j in range(N):
            if graph[i][j] > graph[i][k] + graph[k][j] and j != k:
                graph[i][j] = graph[i][k] + graph[k][j]

print(min(graph[0][v1-1]+graph[v1-1][v2-1]+graph[v2-1][N-1], graph[0][v2-1]+graph[v2-1][v1-1]+graph[v1-1][N-1]))

# 정답
import sys, heapq
input = sys.stdin.readline

def dijkstra(s, e):
    D = [INF]*(N+1)
    heap = []
    D[s] = 0
    heapq.heappush(heap, [0, s])
    while heap:
        val, i = heapq.heappop(heap)
        if i == e:
            return D[e]
        for v, w in adj[i]:
            tmp = w+val
            if D[v] > tmp:
                D[v] = tmp
                heapq.heappush(heap, [tmp, v])
    return INF
    
N, E = map(int, input().split())
adj = [[] for _ in range(N+1)]

for _ in range(E):
    a, b, c = map(int, input().split())
    adj[a].append([b, c])
    adj[b].append([a, c])
v1, v2 = map(int, input().split())

INF = 9999999
res = min(dijkstra(1, v1)+dijkstra(v1, v2)+dijkstra(v2, N), dijkstra(1, v2)+dijkstra(v2, v1)+dijkstra(v1, N))
print(res if res<INF else -1)
```



## 30) [1967. 트리의 지름](https://www.acmicpc.net/problem/1967)

```python
# 당연히 시간 초과
import sys
input = sys.stdin.readline

def dfs(now, s):
    global res
    for v, w in adj[now]:
        if not visited[v]:
            break
    else:
        if res < s:
            res = s
            return
    for v, w in adj[now]:
        if not visited[v]:
            visited[v] = 1
            dfs(v, s+w)
            visited[v] = 0

N = int(input())
adj = [[]*(N+1) for _ in range(N+1)]
while 1:
    try:
        u, v, w = map(int, input().split())
        adj[u].append([v, w])
        adj[v].append([u, w])
    except:
        break

res = 0
for i in range(1, N+1):
    visited = [0]*(N+1)
    visited[i] = 1
    dfs(i, 0)
print(res)

# 정답
# 인터넷 참고 - 임의의 한 점에서 가장 멀리 있는 점은 트리 지름의 한 쪽 끝
import sys
input = sys.stdin.readline
sys.setrecursionlimit(10**9)

def dfs(now, s):
    global res, ans
    for v, w in adj[now]:
        if not visited[v]:
            break
    else:
        if res < s:
            res = s
            ans = now
            return
    for v, w in adj[now]:
        if not visited[v]:
            visited[v] = 1
            dfs(v, s+w)
            visited[v] = 0

N = int(input())
adj = [[]*(N+1) for _ in range(N+1)]
while 1:
    try:
        u, v, w = map(int, input().split())
        adj[u].append([v, w])
        adj[v].append([u, w])
    except:
        break

res = ans = 0
visited = [0]*(N+1)
visited[1] = 1
dfs(1, 0)
res = 0
visited = [0]*(N+1)
visited[ans] = 1
dfs(ans, 0)
print(res)
```



## 31) [2096. 내려가기](https://www.acmicpc.net/problem/2096)

```python
# 메모리 초과
import sys
input = sys.stdin.readline

N = int(input())
arr = [list(map(int, input().split())) for _ in range(N)]
min_dp = [[999999]*N for _ in range(N)]
max_dp = [[0]*N for _ in range(N)]
min_dp[0] = max_dp[0] = arr[0]
for i in range(1, N):
    min_dp[i][0] = min(min_dp[i-1][0], min_dp[i-1][1]) + arr[i][0]
    min_dp[i][1] = min(min_dp[i-1]) + arr[i][1]
    min_dp[i][2] = min(min_dp[i-1][1], min_dp[i-1][2]) + arr[i][2]
    max_dp[i][0] = max(max_dp[i-1][0], max_dp[i-1][1]) + arr[i][0]
    max_dp[i][1] = max(max_dp[i-1]) + arr[i][1]
    max_dp[i][2] = max(max_dp[i-1][1], max_dp[i-1][2]) + arr[i][2]

print(max(max_dp[N-1]), min(min_dp[N-1]))

# 정답
import sys
input = sys.stdin.readline
from copy import deepcopy

N = int(input())

dp = [[0]*3, [0]*3]
tmp = [[0]*3, [0]*3]

for i in range(N):
    a, b, c = map(int, input().split())
    tmp[0][0] = a + max(dp[0][0], dp[0][1])
    tmp[0][1] = b + max(dp[0])
    tmp[0][2] = c + max(dp[0][1], dp[0][2])
    tmp[1][0] = a + min(dp[1][0], dp[1][1])
    tmp[1][1] = b + min(dp[1])
    tmp[1][2] = c + min(dp[1][1], dp[1][2])
    dp = deepcopy(tmp)

print(max(dp[0]), min(dp[1]))

# 숏코딩
a=b=c=d=e=f=0;m,M=min,max
for _ in range(int(input())):x,y,z=map(int,input().split());a,b,c,d,e,f=M(a,b)+x,M(a,b,c)+y,M(b,c)+z,m(d,e)+x,m(d,e,f)+y,m(e,f)+z
print(M(a,b,c),m(d,e,f))
```



## 32) [2206. 벽 부수고 이동하기](https://www.acmicpc.net/problem/2206)

```python
# 시간 초과
import sys
input = sys.stdin.readline
from collections import deque

def bfs():
    global res
    visited = [[0]*M for _ in range(N)]
    visited[0][0] = 1
    q = deque([(0, 0)])
    while q:
        x, y = q.popleft()
        if x == N-1 and y == M-1:
            if res > visited[x][y]:
                res = visited[x][y]
            return
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<N and 0<=ny<M and not arr[nx][ny] and not visited[nx][ny]:
                visited[nx][ny] = visited[x][y] + 1
                q.append((nx, ny))

N, M = map(int, input().split())
arr = [list(map(int, input().strip())) for _ in range(N)]

res = 999999
for i in range(N):
    for j in range(M):
        if arr[i][j]:
            arr[i][j] = 0
            bfs()
            arr[i][j] = 1
    
print(res if res<999999 else -1)

# 정답
import sys
input = sys.stdin.readline
from collections import deque

def bfs():
    q = deque([(0, 0, 0)])
    while q:
        x, y, z = q.popleft()
        if x == N-1 and y == M-1:
            return visited[x][y][z]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<N and 0<=ny<M:
                if not arr[nx][ny] and not visited[nx][ny][z]:
                    visited[nx][ny][z] = visited[x][y][z] + 1
                    q.append((nx, ny, z))
                elif arr[nx][ny] == 1 and not z:
                    visited[nx][ny][1] = visited[x][y][z] + 1
                    q.append((nx, ny, 1))
    return -1

N, M = map(int, input().split())
arr = [list(map(int, input().strip())) for _ in range(N)]
visited = [[[0]*2 for _ in range(M)] for _ in range(N)]
visited[0][0][0] = 1
print(bfs())
```



## 33) [2448. 별 찍기 - 11](https://www.acmicpc.net/problem/2448)

```python
def dnc(i, j, x):
    if x == 3:
        star[i][j] = '*'
        star[i+1][j-1] = star[i+1][j+1] = '*'
        star[i+2][j-2:j+3] = ['*']*5
        return

    y = x//2
    dnc(i, j, y)
    dnc(i+y, j-y, y)
    dnc(i+y, j+y, y)

N = int(input())
star = [[' ']*(N*2-1) for _ in range(N)]
dnc(0, N-1, N)
for s in star:
    print(''.join(s))
```



## 34) [2638. 치즈](https://www.acmicpc.net/problem/2638)

```python
import sys
input = sys.stdin.readline
from collections import deque

def bfs():
    q = deque([(0, 0)])
    visited = [[0]*M for _ in range(N)]
    visited[0][0] = 1
    while q:
        i, j = q.popleft()
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i+di, j+dj
            if 0<=ni<N and 0<=nj<M and not visited[ni][nj]:
                if arr[ni][nj]:
                    arr[ni][nj] += 1
                else:
                    visited[ni][nj] = 1
                    q.append((ni, nj))

N, M = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]
cnt = 0
while 1:
    bfs()
    flag = 0
    for i in range(N):
        for j in range(M):
            if arr[i][j] >= 3:
                arr[i][j] = 0
                flag = 1
            if arr[i][j] == 2:
                arr[i][j] = 1
    if flag:
        cnt += 1
    else:
        break

print(cnt)
```



## 35) [9935. 문자열 폭발](https://www.acmicpc.net/problem/9935)

```python
# 시간 초과
import sys
input = sys.stdin.readline

target = input().strip()
bomb = input().strip()
s = []
N = len(bomb)

for x in target:
    s.append(x)
    if x == bomb[-1] and s[-N:] == list(bomb):
        s = s[:-N]

print(''.join(s) if s else 'FRULA')

# 정답
import sys
input = sys.stdin.readline

target = input().strip()
bomb = input().strip()
s = []
N = len(bomb)

for x in target:
    s.append(x)
    if x == bomb[-1] and s[-N:] == list(bomb):
        for _ in range(N):
            s.pop()

print(''.join(s) if s else 'FRULA')

# 좀 더 빠른 방법
import sys
input = sys.stdin.readline

target = input().strip()
bomb = input().strip()
s = []
N = len(bomb)

for x in target:
    s.append(x)
    if x == bomb[-1] and s[-N:] == list(bomb):
        del s[-N:]

print(''.join(s) if s else 'FRULA')
```



## 36) [10830. 행렬 제곱](https://www.acmicpc.net/problem/10830)

```python
# 시간 초과
import sys
input = sys.stdin.readline
from copy import deepcopy

N, B = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(N)]

res = deepcopy(A)
for _ in range(B-1):
    tmp = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                tmp[i][j] += res[i][k]*A[k][j]
    res = deepcopy(tmp)

for i in range(N):
    for j in range(N):
        res[i][j] %= 1000

for r in res:
    print(*r)
    
# 정답
import sys
input = sys.stdin.readline

def mul(X, Y):
    Z = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                Z[i][j] += X[i][k]*Y[k][j]
            Z[i][j] %= 1000
    return Z

def dnc(A, B):
    if B == 1:
        return A

    tmp = dnc(A, B//2)
    if B%2:
        return mul(mul(tmp, tmp), A)
    else:
        return mul(tmp, tmp)

N, B = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(N)]

res = dnc(A, B)
for i in range(N):
    for j in range(N):
        res[i][j] %= 1000

for r in res:
    print(*r)
```



## 37) [11404. 플로이드](https://www.acmicpc.net/problem/11404)

```python
import sys
input = sys.stdin.readline

N = int(input())
M = int(input())
INF = 10000000

arr = [[INF]*N for _ in range(N)]
for i in range(N):
    arr[i][i] = 0

for _ in range(M):
    u, v, w = map(int, input().split())
    arr[u-1][v-1] = min(arr[u-1][v-1], w)

for k in range(N):
    for i in range(N):
        for j in range(N):
            if arr[i][j] > arr[i][k] + arr[k][j]:
                arr[i][j] = arr[i][k] + arr[k][j]

for i in range(N):
    for j in range(N):
        if arr[i][j] == INF:
            arr[i][j] = 0

for a in arr:
    print(*a)
```



## 38) [13172. Σ](https://www.acmicpc.net/problem/13172)

```python
# 인터넷 참고
import sys
input = sys.stdin.readline

def mul(x, n):
    if n == 1:
        return x%key
    if n%2:
        return x*mul(x, n-1)%key
    tmp = mul(x, n//2)
    return tmp**2%key

key = 1000000007
N = int(input())
s = 0
for _ in range(N):
    a, b = map(int, input().split())
    s += b*mul(a, key-2)%key
    s %= key
print(s)
```



## 39) [14938. 서강그라운드](https://www.acmicpc.net/problem/14938)

```python
import sys
input = sys.stdin.readline

n, m, r = map(int, input().split())
item = list(map(int, input().split()))
arr = [[1500]*n for _ in range(n)]
for i in range(n):
    arr[i][i] = 0

for _ in range(r):
    u, v, w = map(int, input().split())
    arr[u-1][v-1] = w
    arr[v-1][u-1] = w

for k in range(n):
    for i in range(n):
        for j in range(n):
            if arr[i][j] > arr[i][k] + arr[k][j]:
                arr[i][j] = arr[i][k] + arr[k][j]

res = 0
for i in range(n):
    cnt = 0
    for j in range(n):
        if arr[i][j] <= m:
            cnt += item[j]
    if res < cnt:
        res = cnt

print(res)
```



## 40) [17144. 미세먼지 안녕!](https://www.acmicpc.net/problem/17144)

```python
# pypy만

import sys
input = sys.stdin.readline
from copy import deepcopy

R, C, T = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(R)]
for i in range(R):
    if arr[i][0] == -1:
        up, down = i, i+1
        break

for _ in range(T):
    tmp = [[0]*C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            if arr[i][j] == -1:
                tmp[i][j] = -1

            if arr[i][j] > 0:
                x = arr[i][j]
                cnt = 0
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i+di, j+dj
                    if 0<=ni<R and 0<=nj<C and arr[ni][nj] != -1:
                        tmp[ni][nj] += x//5
                        cnt += 1
                tmp[i][j] += x - x//5*cnt

    di = [0, -1, 0, 1]
    dj = [1, 0, -1, 0]
    i, j = up, 1
    d = flag = 0
    while 1:
        if i == up and j == 0:
            break
        ni, nj = i+di[d], j+dj[d]
        if ni<0 or ni>=R or nj<0 or nj>=C:
            d += 1
            continue
        tmp[i][j], flag = flag, tmp[i][j]
        i, j = ni, nj

    di = [0, 1, 0, -1]
    dj = [1, 0, -1, 0]
    i, j = down, 1
    d = flag = 0
    while 1:
        if i == down and j == 0:
            break
        ni, nj = i+di[d], j+dj[d]
        if ni<0 or ni>=R or nj<0 or nj>=C:
            d += 1
            continue
        tmp[i][j], flag = flag, tmp[i][j]
        i, j = ni, nj

    arr = deepcopy(tmp)

cnt = 2
for i in range(R):
    cnt += sum(arr[i])
print(cnt)
```



## 41) [1167. 트리의 지름](https://www.acmicpc.net/problem/1167)

```python
import sys
input = sys.stdin.readline
from collections import deque

def bfs(x):
    visited = [-1]*(V+1)
    q = deque([x])
    visited[x] = 0
    res = [0, 0]

    while q:
        t = q.popleft()
        for v, w in adj[t]:
            if visited[v] == -1:
                visited[v] = visited[t] + w
                q.append(v)
                if res[0] < visited[v]:
                    res[0] = visited[v]
                    res[1] = v
    return res

V = int(input())
adj = [[] for _ in range(V+1)]
for _ in range(V):
    data = list(map(int, input().split()))
    u = data[0]
    for i in range(len(data)//2-1):
        v = data[i*2+1]
        w = data[i*2+2]
        adj[u].append((v, w))

dis, node = bfs(1)
print(bfs(node)[0])
```



## 42) [1238. 파티](https://www.acmicpc.net/problem/1238)

```python
import sys
input = sys.stdin.readline
import heapq

def dijkstra(s, N, adj):
    D = [INF]*(N+1)
    D[s] = 0
    heap = []
    heapq.heappush(heap, (0, s))
    while heap:
        val, i = heapq.heappop(heap)
        for v, w in adj[i]:
            tmp = w + val
            if D[v] > tmp:
                D[v] = tmp
                heapq.heappush(heap, (tmp, v))
    return D

N, M, X = map(int, input().split())
INF = 999999
adj1 = [[] for _ in range(N+1)]
adj2 = [[] for _ in range(N+1)]

for _ in range(M):
    x, y, c = map(int, input().split())
    adj1[x].append((y, c))
    adj2[y].append((x, c))

D1 = dijkstra(X, N, adj1)
D2 = dijkstra(X, N, adj2)

res = 0
for i in range(1, N+1):
    tmp = D1[i] + D2[i]
    if res < tmp:
        res = tmp

print(res)
```



## 43) [1865. 웜홀](https://www.acmicpc.net/problem/1865)

```python
# 오답 - 왜 틀린지 모르겠음
import sys
input = sys.stdin.readline

def check():
    for i in range(N):
        for j in range(N):
            if arr[i][j] + arr[j][i] < 0:
                return 'YES'
    return 'NO'

for _ in range(int(input())):
    N, M, W = map(int, input().split())
    arr = [[9999999]*N for _ in range(N)]
    for __ in range(M):
        S, E, T = map(int, input().split())
        arr[S-1][E-1] = T
        arr[E-1][S-1] = T
    for __ in range(W):
        S, E, T = map(int, input().split())
        arr[S-1][E-1] = -T
    for i in range(N):
        arr[i][i] = 0

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if arr[i][j] > arr[i][k] + arr[k][j]:
                    arr[i][j] = arr[i][k] + arr[k][j]

    print(check())
   
# 정답 - 인터넷 검색 : 벨만 포드 알고리즘
import sys
input = sys.stdin.readline

def bellman_ford(start):
    dist = [99999999]*(N+1)
    dist[start] = 0
    for _ in range(1, N+1):
        for s, e, t in data:
            if dist[e] > dist[s] + t:
                dist[e] = dist[s] + t

    for s, e, t in data:
        if dist[e] > dist[s] + t:
            return 'YES'
    return 'NO'

for _ in range(int(input())):
    N, M, W = map(int, input().split())
    data = []

    for __ in range(M):
        S, E, T = map(int, input().split())
        data.append((S, E, T))
        data.append((E, S, T))

    for __ in range(W):
        S, E, T = map(int, input().split())
        data.append((S, E, -T))

    print(bellman_ford(1))
```



## 44) [1918. 후위 표기식](https://www.acmicpc.net/problem/1918)

```python
exp = input()
s = []
res = ''
icp = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 3}
isp = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0}

for x in exp:
    if x == ')':
        while isp[s[-1]]:
            res += s.pop()
        s.pop()
    elif x in ('+', '-', '*', '/', '('):
        while s and isp[s[-1]] >= icp[x]:
            res += s.pop()
        s.append(x)
    else:
        res += x
while s:
    res += s.pop()
    
print(res)
```



## 45) [11054. 가장 긴 바이토닉 부분 수열](https://www.acmicpc.net/problem/11054)

```python
import sys
input = sys.stdin.readline

N = int(input())
arr = list(map(int, input().split()))
dp1 = [1]*N
dp2 = [1]*N

for i in range(1, N):
    for j in range(i):
        if arr[j] < arr[i]:
            dp1[i] = max(dp1[i], dp1[j]+1)

for i in range(N-2, -1, -1):
    for j in range(N-1, i, -1):
        if arr[j] < arr[i]:
            dp2[i] = max(dp2[i], dp2[j]+1)

res = 0
for i in range(N):
    res = max(res, dp1[i]+dp2[i])
print(res-1)
```



## 46) [11779. 최소비용 구하기 2](https://www.acmicpc.net/problem/11779)

```python
# 처음에 res를 str으로 받고 출력할때 *list(res) 했더니 의문의 오답이었음
import sys
input = sys.stdin.readline
from heapq import *

def dijkstra(s, e):
    D = [INF]*(N+1)
    D[s] = 0
    heap = []
    heappush(heap, (0, s, [s]))
    while heap:
        val, i, res = heappop(heap)
        if i == e:
            return D[e], res
        for v, w in adj[i]:
            tmp = val + w
            if D[v] > tmp:
                D[v] = tmp
                heappush(heap, (tmp, v, res + [v]))

N = int(input())
M = int(input())
adj = [[] for _ in range(N+1)]
for _ in range(M):
    u, v, w = map(int, input().split())
    adj[u].append((v, w))
s, e = map(int, input().split())
INF = 99999999

x, r = dijkstra(s, e)
print(x)
print(len(r))
print(*r)
```



## 47) [2263. 트리의 순회](https://www.acmicpc.net/problem/2263)

```python
# 인터넷 참고
import sys
input = sys.stdin.readline
sys.setrecursionlimit(10**6)

def make_pre_order(iS, iE, pS, pE):
    if iS > iE or pS > pE:
        return
    tmp = post_order[pE]
    pre_order.append(tmp)
    make_pre_order(iS, index[tmp] - 1, pS, index[tmp] + pS - iS - 1)
    make_pre_order(index[tmp] + 1, iE, index[tmp] + pE - iE, pE - 1)

n = int(input())
in_order = list(map(int, input().split()))
post_order = list(map(int, input().split()))
pre_order = []
index = [0]*(n+1)
for i in range(n):
    index[in_order[i]] = i

make_pre_order(0, n-1, 0, n-1)
print(*pre_order)
```



## 48) [11444. 피보나치 수 6](https://www.acmicpc.net/problem/11444)

> https://www.acmicpc.net/blog/view/28

```python
# 시간 초과
def fibo(x):
    if x == 1:
        return 1
    if x == 2:
        return 1
    if x % 2:
        a, b = fibo(x//2), fibo(x//2+1)
        return (a**2+b**2)%key
    a, b = fibo(x//2-1), fibo(x//2)
    return (2*a*b+b**2)%key

n = int(input())
key = 1000000007
print(fibo(n))

# 정답
def mul(A, B):
    Z = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            e = 0
            for k in range(2):
                e += A[i][k] * B[k][j]
            Z[i][j] = e % key
    return Z

def foo(mat, x):
    if x == 1:
        return mat
    tmp = foo(mat, x//2)
    tmp2 = mul(tmp, tmp)
    if x % 2:
        return mul(tmp2, mat)
    return tmp2

n = int(input())
key = 1000000007
mat = [[1, 1], [1, 0]]
print(foo(mat, n)[0][1])
```

