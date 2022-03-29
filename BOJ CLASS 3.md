# BOJ CLASS 3



## 1) [11723. 집합](https://www.acmicpc.net/problem/11723)

```python
# 시간 초과
import sys
M = int(sys.stdin.readline())
S = []
for _ in range(M):
    act = list(sys.stdin.readline().split())
    if act[0] == 'add':
        S.append(act[1])
    elif act[0] == 'remove':
        S.remove(act[1])
    elif act[0] == 'check':
        if act[1] in S:
            print(1)
        else:
            print(0)
    elif act[0] == 'toggle':
        if act[1] in S:
            S.remove(act[1])
        else:
            S.append(act[1])
    elif act[0] == 'all':
        S = list(str(i) for i in range(1, 21))
    elif act[0] == 'empty':
        S = []

# 정답
import sys
M = int(sys.stdin.readline())
S = [0]*21
for _ in range(M):
    act = list(sys.stdin.readline().split())
    if act[0] == 'add':
        S[int(act[1])] = 1
    elif act[0] == 'remove':
        S[int(act[1])] = 0
    elif act[0] == 'check':
        if S[int(act[1])]:
            print(1)
        else:
            print(0)
    elif act[0] == 'toggle':
        if S[int(act[1])]:
            S[int(act[1])] = 0
        else:
            S[int(act[1])] = 1
    elif act[0] == 'all':
        S = [1]*21
    elif act[0] == 'empty':
        S = [0]*21
```



## 2) [1620. 나는야 포켓몬 마스터 이다솜](https://www.acmicpc.net/problem/1620)

```python
# 시간 초과
import sys
N, M = map(int, sys.stdin.readline().split())
pok = []
for _ in range(N):
    pok.append(input())
for _ in range(M):
    com = input()
    if com in [str(i) for i in range(N)]:
        print(pok[int(com)-1])
    else:
        print(pok.index(com)+1)
# 정답
import sys
N, M = map(int, sys.stdin.readline().split())
pok = {}
for i in range(N):
    p = input()
    pok[p] = i+1
    pok[str(i+1)] = p
for _ in range(M):
    print(pok[input()])
```



## 3) [1676. 팩토리얼 0의 개수](https://www.acmicpc.net/problem/1676)

```python
n=int(input())
print(n//5+n//25+n//125)
```



## 4) [1764. 듣보잡](https://www.acmicpc.net/problem/1764)

```python
N, M = map(int, input().split())
s = set()
for _ in range(N):
    s.add(input())
cnt = 0
result = set()
for _ in range(M):
    name = input()
    if name in s:
        result.add(name)
        cnt += 1
print(cnt)
for i in sorted(result):
    print(i)
```



## 5) [17219. 비밀번호 찾기](https://www.acmicpc.net/problem/17219)

```python
N, M = map(int, input().split())
note = {}
for _ in range(N):
    site, pw = input().split()
    note[site] = pw
for _ in range(M):
    print(note[input()])
```



## 6) [17626. Four Squares](https://www.acmicpc.net/problem/17626)

```python
# 인터넷 참고 - dp
n = int(input())
dp = [0, 1]

for i in range(2, n+1):
    min_val = 50000
    j = 1
    
    while i>=j**2:
        min_val = min(min_val, dp[i-j**2])
        j += 1
    dp.append(min_val+1)
    
print(dp[n])
```



## 7) [1003. 피보나치 함수](https://www.acmicpc.net/problem/1003)

```python
def fibo(n):
    global memo
    if n>=2 and not memo[n]:
        memo[n] = fibo(n-1) + fibo(n-2)
    return memo[n]

T = int(input())
for _ in range(T):
    N = int(input())
    if N == 0:
        print(1, 0)
    elif N == 1:
        print(0, 1)
    else:
        memo = [0]*(N+1)
        memo[1] = 1
        print(fibo(N-1), fibo(N))
```



## 8) [1463. 1로만들기](https://www.acmicpc.net/problem/1463)

```python
N = int(input())
dp = [0]*(N+1)

for i in range(2, N+1):
    min_val = dp[i-1]
    if not i%3 and dp[i//3] < min_val:
        min_val = dp[i//3]
    if not i%2 and dp[i//2] < min_val:
        min_val = dp[i//2]
    dp[i] = min_val+1

print(dp[N])
```



## 9) [2579. 계단 오르기](https://www.acmicpc.net/problem/2579)

```python
n = int(input())
step = [0]
for _ in range(n):
    step.append(int(input()))
dp = [0]*(n+1)

if n == 1:
    print(step[1])
    
else:
    dp[1] = step[1]
    dp[2] = step[1] + step[2]
    for i in range(3, n+1):
        dp[i] = max(dp[i-2]+step[i], dp[i-3]+step[i-1]+step[i])
    print(dp[n])
```



## 10) [2606. 바이러스](https://www.acmicpc.net/problem/2606)

```python
def virus():
    global cnt
    q = []
    for i in range(N+1):
        if arr[1][i]:
            q.append(i)
            visited[i] = 1
            cnt += 1
    while q:
        i = q.pop(0)
        for j in range(N+1):
            if arr[i][j] and not visited[j]:
                q.append(j)
                visited[j] = 1
                cnt += 1

N = int(input())
M = int(input())
arr = [[0]*(N+1) for _ in range(N+1)]
visited = [0]*(N+1)
visited[1] = 1
for _ in range(M):
    a, b = map(int, input().split())
    arr[a][b] = arr[b][a] = 1
cnt = 0
virus()
print(cnt)
```



## 11) [2630. 색종이 만들기](https://www.acmicpc.net/problem/2630)

```python
N = int(input())
arr = [list(map(int, input().split())) for _ in range(N)]
cnt_W = 0
cnt_B = 0
if arr == [[0]*N for _ in range(N)]:
    cnt_W += 1
elif arr == [[1]*N for _ in range(N)]:
    cnt_B += 1
else:
    q = []
    N //= 2
    q.append((0, 0, N))
    q.append((0, N, N))
    q.append((N, 0, N))
    q.append((N, N, N))
    while q:
        i, j, n = q.pop()
        cnt = 0
        for x in range(i, i+n):
            for y in range(j, j+n):
                if arr[x][y]:
                    cnt += 1
        if cnt == 0:
            cnt_W += 1
        elif cnt == n**2:
            cnt_B += 1
        else:
            n //= 2
            q.append((i, j, n))
            q.append((i+n, j, n))
            q.append((i, j+n, n))
            q.append((i+n, j+n, n))
print(cnt_W)
print(cnt_B)
```



## 12) [9095. 1, 2, 3 더하기](https://www.acmicpc.net/problem/9095)

```python
T = int(input())
for _ in range(T):
    n = int(input())
    m = [0, 1, 2, 4]
    if n > 3:
        for i in range(4, n+1):
            m.append(m[i-1]+m[i-2]+m[i-3])
    print(m[n])
```



## 13) [9375. 패션왕 신해빈](https://www.acmicpc.net/problem/9375)

```python
T = int(input())
for t in range(T):
    N = int(input())
    part = []
    clothes = []
    
    for _ in range(N):
        a, b = input().split()
        if b in part:
            clothes[part.index(b)].append(a)
        else:
            part.append(b)
            clothes.append([a])
    
    res = 1
    for i in range(len(clothes)):
        res *= len(clothes[i])+1
    print(res-1)
```



## 14) [9461. 파도반 수열](https://www.acmicpc.net/problem/9461)

```python
T = int(input())
for t in range(T):
    N = int(input())
    pado = [0, 1, 1, 1, 2]
    if N < 5:
        print(pado[N])
    else:
        for i in range(5, N+1):
            pado.append(pado[i-1]+pado[i-5])
        print(pado[N])
```



## 15) [11047. 동전 0](https://www.acmicpc.net/problem/11047)

```python
N, K = map(int, input().split())
s = []
for _ in range(N):
    s.append(int(input()))
cnt = 0
while s:
    x = s.pop()
    cnt += K//x
    K -= K//x*x
print(cnt)
```



## 16) [11399. ATM](https://www.acmicpc.net/problem/11399)

```python
N = int(input())
l = sorted(list(map(int, input().split())))
cnt = 0
for i in range(N):
    cnt += l[i]*(N-i)
print(cnt)
```



## 17) [11659. 구간 합 구하기 4](https://www.acmicpc.net/problem/11659)

```python
import sys
N, M = map(int, sys.stdin.readline().split())
a = list(map(int, sys.stdin.readline().split()))
sums = [0]
for i in range(N):
    sums.append(sums[i]+a[i])
for _ in range(M):
    i, j = map(int, sys.stdin.readline().split())
    print(sums[j]-sums[i-1])
```



## 18) [11726. 2xn 타일링](https://www.acmicpc.net/problem/11726)

```python
N = int(input())
s = [1]*(N+1)
if N > 1:
    for i in range(2,N+1):
        s[i] = s[i-1] + s[i-2]
print(s[N]%10007)
```



## 19) [11727. 2xn 타일링 2](https://www.acmicpc.net/problem/11727)

```python
N = int(input())
s = [1]*(N+1)
if N > 1:
    for i in range(2,N+1):
        s[i] = s[i-1] + 2*s[i-2]
print(s[N]%10007)
```



## 20) [1012. 유기농 배추](https://www.acmicpc.net/problem/1012)

```python
T = int(input())
for tc in range(1, T+1):
    M, N, K = map(int, input().split())
    cabb = []
    cnt = 0
    for _ in range(K):
        i, j = map(int, input().split())
        cabb.append([i, j])
    while cabb:
        q = [cabb.pop(0)]
        while q:
            i, j = q.pop(0)
            for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                ni, nj = i+di, j+dj
                if 0<=ni<M and 0<=nj<N and [ni, nj] in cabb:
                    q.append(cabb.pop(cabb.index([ni, nj])))
        cnt += 1
    print(cnt)
```



## 21) [1260. DFS와 BFS](https://www.acmicpc.net/problem/1260)

```python
def dfs(N, V):
    visited[V] = 1
    print(V, end=' ')
    for x in tree[V]:
        if not visited[x]:
            visited[x] = 1
            dfs(N, x)

def bfs(N, V):
    q = []
    visited[V] = 1
    print(V, end=' ')
    for x in tree[V]:
        q.append(x)
    while q:
        i = q.pop(0)
        if not visited[i]:
            visited[i] = 1
            print(i, end=' ')
            for x in tree[i]:
                q.append(x)

N, M, V = map(int, input().split())
tree = [[] for _ in range(N+1)]
for _ in range(M):
    i, j = map(int, input().split())
    tree[i].append(j)
    tree[j].append(i)
for x in tree:
    x.sort()

visited = [0]*(N+1)
dfs(N, V)
print()
visited = [0]*(N+1)
bfs(N, V)
```



## 22) [1541. 잃어버린 괄호](https://www.acmicpc.net/problem/1541)

```python
exp = input().split('-')
for i in range(len(exp)):
    if '+' in exp[i]:
        nums = exp[i].split('+')
        s = 0
        for x in nums:
            s += int(x)
        exp[i] = s
res = int(exp.pop(0))
while exp:
    res -= int(exp.pop(0))
print(res)
```



## 23) [1780. 종이의 개수](https://www.acmicpc.net/problem/1780)

```python
def dfs(x, y, n):
    num = arr[x][y]
    for i in range(x, x+n):
        for j in range(y, y+n):
            if arr[i][j] != num:
                for k in range(3):
                    for l in range(3):
                        dfs(n//3*k+x, n//3*l+y, n//3)
                return
    cnt[num+1] += 1
    return

N = int(input())
arr = [list(map(int, input().split())) for _ in range(N)]
v = [[0]*N for _ in range(N)]
cnt = [0]*3
dfs(0, 0, N)
for x in cnt:
    print(x)
```



## 24) [1927. 최소 힙](https://www.acmicpc.net/problem/1927)

```python
# 인터넷 참고
import sys, heapq

N = int(sys.stdin.readline())
heap = []
for _ in range(N):
    x = int(sys.stdin.readline())
    if not x:
        if heap:
            print(heapq.heappop(heap))
        else:
            print(0)
    else:
        heapq.heappush(heap, x)
```



## 25) [1931. 회의실 배정](https://www.acmicpc.net/problem/1931)

```python
# 시간 초과 - 틀리기도 했을듯?
N = int(input())
conf = []
for _ in range(N):
    conf.append(list(map(int,input().split())))
res = 0
for i, j in conf:
    v = [[i, j]]
    for p, q in conf:
        for r, s in v:
            if p<s and q>r:
                break
        else:
            v.append([p, q])
    if res < len(v):
        res = len(v)
print(res)

# 인터넷 참고
N = int(input())
conf = []
for _ in range(N):
    conf.append(list(map(int,input().split())))
conf.sort(key = lambda x: (x[1],x[0]))

e = cnt = 0
for i,j in conf:
    if i >= e:
        e = j
        cnt += 1
print(cnt)
```



## 26) [5525. IOIOI](https://www.acmicpc.net/problem/5525)

```python
# 시간 초과
N, M = int(input()), int(input())
S = input()
v = [0]*(M-2)
for i in range(M-2):
    if S[i:i+3] == 'IOI':
        v[i] = 1
cnt = 0
for i in range(M-2-N):
    if v[i]:
        c = 1
        while c < N:
            if v[i+2*c]:
                c += 1
            else:
                break
        else:
            cnt += 1
print(cnt)

# 정답
N, M = int(input()), int(input())
S = input()
i = cnt = res = 0
while i<M-2:
    if S[i:i+3] == 'IOI':
        i += 2
        cnt += 1
        if cnt == N:
            res += 1
            cnt -= 1
    else:
        i += 1
        cnt = 0
print(res)
```



## 27) [11279. 최대 힙](https://www.acmicpc.net/problem/11279)

```python
import sys, heapq

N = int(sys.stdin.readline())
heap = []
for _ in range(N):
    x = int(sys.stdin.readline())
    if not x:
        if heap:
            print(-heapq.heappop(heap))
        else:
            print(0)
    else:
        heapq.heappush(heap, -x)
```



## 28) [11724. 연결  요소의 개수](https://www.acmicpc.net/problem/11724)

```python
# 시도 1(BFS) - 시간 초과
import sys

N, M = map(int, sys.stdin.readline().split())
tree = [[] for _ in range(N+1)]
for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    tree[u].append(v)
    tree[v].append(u)
visited = [0]*(N+1)
cnt = 0
for i in range(N+1):
    if tree[i]:
        q = [i]
        while q:
            x = q.pop(0)
            visited[x] = 1
            for _ in range(len(tree[x])):
                y = tree[x].pop(0)
                if not visited[y]:
                    q.append(y)
        cnt += 1
print(cnt)

# 시도 2(DFS) - 오답
import sys

def dfs(v):
    visited[v] = 1
    for e in tree[v]:
        if not visited[e]:
            dfs(e)

N, M = map(int, sys.stdin.readline().split())
tree = [[] for _ in range(N+1)]
for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    tree[u].append(v)
    tree[v].append(u)

visited = [0]*(N+1)
cnt = 0
for i in range(N):
    if not visited[i+1]:
        cnt += 1
        dfs(i+1)
print(cnt)

# 정답 - 시도1 수정
import sys

N, M = map(int, sys.stdin.readline().split())
tree = [[] for _ in range(N+1)]
visited = [0]*(N+1)
cnt = 0

for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    tree[u].append(v)
    tree[v].append(u)

for i in range(1, N+1):
    if not visited[i]:
        q = [i]
        while q:
            x = q.pop(0)
            for y in tree[x]:
                if not visited[y]:
                    q.append(y)
                    visited[y]=1
        cnt += 1
print(cnt)

# 정답 - 시도2 수정(인터넷 참고) - 좀 억까인듯
import sys

sys.setrecursionlimit(10000) # 달라진 점

def dfs(v):
    visited[v] = 1
    for e in tree[v]:
        if not visited[e]:
            dfs(e)

N, M = map(int, sys.stdin.readline().split())
tree = [[] for _ in range(N+1)]
visited = [0]*(N+1)
cnt = 0

for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    tree[u].append(v)
    tree[v].append(u)

for i in range(N):
    if not visited[i+1]:
        cnt += 1
        dfs(i+1)
print(cnt)
```



## 29) [18870. 좌표 압축](https://www.acmicpc.net/problem/18870)

```python
# 당연히 시간 초과
N = int(input())
nums = list(map(int, input().split()))
cnts = []
for i in nums:
    cnt = 0
    v = []
    for j in nums:
        if i > j and j not in v:
            cnt += 1
            v.append(j)
    cnts.append(cnt)
print(*cnts)

# 정답
import sys
N = int(sys.stdin.readline())
nums = list(map(int, sys.stdin.readline().split()))
s_nums = sorted(list(set(nums)))
dic = {}
for i in range(len(s_nums)):
    dic[s_nums[i]] = i
for x in nums:
    print(dic[x], end=' ')
```



## 30) [1074. Z](https://www.acmicpc.net/problem/1074)

```python
import sys

def zfbin(v):
    s = ''
    for i in range(N-1, -1, -1):
        s += '1' if v&(1<<i) else '0'
    return s

N, r, c = map(int, sys.stdin.readline().split())
rs, cs = zfbin(r), zfbin(c)
res = 0

for k in range(N):
    i, j = int(rs[k]), int(cs[k])
    val = 2**((N-k-1)*2)
    if not i and j:
        res += val
    elif i and not j:
        res += val*2
    elif i and j:
        res += val*3

print(res)
```



## 31) [1389. 케빈 베이컨의 6단계 법칙](https://www.acmicpc.net/problem/1389)

```python
import sys

def dfs(v):
    nxt = []
    for x in tree[v]:
        if not visited[x]:
            visited[x] = visited[v] + 1
            nxt.append(x)
    for x in nxt:
        dfs(x)
    return

N, M = map(int, sys.stdin.readline().split())
tree = [[] for _ in range(N+1)]
res = 10000

for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    tree[u].append(v)
    tree[v].append(u)

for i in range(1, N+1):
    visited = [0]*(N+1)
    visited[i] = 1
    dfs(i)
    bacon = sum(visited)
    if res > bacon:
        res = bacon
        ans = i

print(ans)
```



## 32) [1697. 숨바꼭질](https://www.acmicpc.net/problem/1697)

```python
def bfs(v):
    q = [v]
    while q:
        x = q.pop(0)
        for i in [x-1, x+1, x*2]:
            if i == K:
                return time[x]
            if 0<=i<=100000 and not time[i]:
                time[i] = time[x]+1
                q.append(i)

N, K = map(int, input().split())
time = [0]*100001
time[N] = 1
if N == K:
    print(0)
else:
    print(bfs(N))
```



## 33) [1992. 쿼드트리](https://www.acmicpc.net/problem/1992)

```python
def dfs(i, j, n):
    c = arr[i][j]
    for a in range(i, i+n):
        for b in range(j, j+n):
            if arr[a][b] != c:
                c = -1
                break

    if c == 0 or c == 1:
        print(c, end='')

    elif c == -1:
        print('(', end='')
        n //= 2
        dfs(i, j, n)
        dfs(i, j+n, n)
        dfs(i+n, j, n)
        dfs(i+n, j+n, n)
        print(')', end='')

N = int(input())
arr = [list(map(int, input())) for _ in range(N)]
dfs(0, 0, N)
```



## 34) [2178. 미로 탐색](https://www.acmicpc.net/problem/2178)

```python
def bfs(i, j):
    q = [(i, j)]
    while q:
        si, sj = q.pop(0)
        for di, dj in [(1,0), (-1,0), (0, 1), (0, -1)]:
            ni, nj = si+di, sj+dj
            if 0<=ni<N and 0<=nj<M and arr[ni][nj] and not visited[ni][nj]:
                visited[ni][nj] = visited[si][sj] + 1
                q.append((ni, nj))

N, M = map(int, input().split())
arr = [list(map(int, input())) for _ in range(N)]
visited =[[0]*M for _ in range(N)]
visited[0][0] = 1
bfs(0, 0)
print(visited[N-1][M-1])
```



## 35) [2667. 단지번호붙이기](https://www.acmicpc.net/problem/2667)

```python
def bfs(i, j):
    cnt = 0
    q = [(i, j)]
    visited[i][j] = 1
    while q:
        si, sj = q.pop(0)
        cnt += 1
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = si+di, sj+dj
            if 0<=ni<N and 0<=nj<N and arr[ni][nj] and not visited[ni][nj]:
                q.append((ni, nj))
                visited[ni][nj] = 1
    cnt_list.append(cnt)

N = int(input())
arr = [list(map(int, input())) for _ in range(N)]
visited = [[0]*N for _ in range(N)]
cnt_list = []
res = 0
for i in range(N):
    for j in range(N):
        if arr[i][j] and not visited[i][j]:
            res += 1
            bfs(i, j)

print(res)
for x in sorted(cnt_list):
    print(x)
```



## 36) [6064. 카잉 달력](https://www.acmicpc.net/problem/6064)

```python
for _ in range(int(input())):
    M, N, x, y = map(int, input().split())
    while x <= M*N:
        if (x-y)%N == 0:
            print(x)
            break
        x += M
    else:
        print(-1)
```



## 37) [11286. 절댓값 힙](https://www.acmicpc.net/problem/11286)

```python
import sys, heapq

N = int(sys.stdin.readline())
heap = []
for _ in range(N):
    x = int(sys.stdin.readline())
    if not x:
        if heap:
            print(heapq.heappop(heap)[1])
        else:
            print(0)
    else:
        heapq.heappush(heap, [abs(x), x])
```



## 38) [11403. 경로 찾기](https://www.acmicpc.net/problem/11403)

```python
# 인터넷 참고
N = int(input())
arr = [list(map(int, input().split())) for _ in range(N)]
for k in range(N):
    for i in range(N):
        for j in range(N):
            if arr[i][k] and arr[k][j]:
                arr[i][j] = 1
for i in range(N):
    print(*arr[i])
```



## 39) [1107. 리모컨](https://www.acmicpc.net/problem/1107)

```python
# 하루종일 한듯
def dfs(i, l, num):
    global cdd
    if i == l:
        if abs(int(cdd)-int(N))+len(cdd) > abs(int(num)-int(N))+len(str(int(num))):
            cdd = num
    else:
        for j in yes:
            dfs(i+1, l, num + str(j))

N = input().strip()
M = int(input())
if M:
    no = list(map(int, input().split()))
    yes = list(set(list(range(10))) - set(no))
    L = len(N)

    if M == 10:
        print(abs(int(N)-100))
    else:
        cdd = str(yes[0])*L
        if int(N[0]) > yes[9-M]:
            if 1 in yes:
                cdd = '1' + str(yes[0])*L
        dfs(0, L, '')

        cdd2 = str(yes[0])*L
        if L > 1:
            cdd2 = str(yes[9-M])*(L-1)

        print(min(abs(int(N)-100), abs(int(cdd)-int(N))+len(str(int(cdd))), abs(int(cdd2)-int(N))+len(cdd2)))
else:
    print(min(len(N), abs(100-int(N))))
```

- 문제점 찾는 곳 : https://www.acmicpc.net/board/view/31853
- 테스트케이스 다 되는데 틀려서 pypy로 해보니 정답;



## 40) [5430. AC](https://www.acmicpc.net/problem/5430)

```python
def AC():
    state = 1
    for x in p:
        if x == 'R':
            state *= -1
        else:
            if not arr:
                print('error')
                return
            else:
                if state == 1:
                    arr.pop(0)
                else:
                    arr.pop(-1)
    if state == -1:
        arr.reverse()
    print('['+','.join(arr)+']')

T = int(input())
for tc in range(T):
    p = input()
    n = int(input())
    a = input()
    if a == '[]':
        arr = []
    else:
        arr = list(a[1:len(a)-1].split(','))
    AC()
```



## 41) [7576. 토마토](https://www.acmicpc.net/problem/7576)

```python
# 시간 초과
def bfs(N, M):
    q = []
    for i in range(N):
        for j in range(M):
            if arr[i][j] == 1:
                q.append((i, j))

    while q:
        si, sj = q.pop(0)
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = si+di, sj+dj
            if 0<=ni<N and 0<=nj<M and not arr[ni][nj]:
                arr[ni][nj] = arr[si][sj] + 1
                q.append((ni, nj))

def check():
    res = 0
    for i in range(N):
        for j in range(M):
            if not arr[i][j]:
                return -1
            if res < arr[i][j]:
                res = arr[i][j]
    return res-1

M, N = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]
bfs(N, M)
print(check())

# 해결 방법 1 - import deque
import sys
from collections import deque

def bfs(N, M):
    q = deque([])
    for i in range(N):
        for j in range(M):
            if arr[i][j] == 1:
                q.append((i, j))

    while q:
        si, sj = q.popleft()
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = si+di, sj+dj
            if 0<=ni<N and 0<=nj<M and not arr[ni][nj]:
                arr[ni][nj] = arr[si][sj] + 1
                q.append((ni, nj))

def check():
    res = 0
    for i in range(N):
        for j in range(M):
            if not arr[i][j]:
                return -1
            if res < arr[i][j]:
                res = arr[i][j]
    return res-1

M, N = map(int, sys.stdin.readline().split())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
bfs(N, M)
print(check())

# 해결 방법 2 - front, rear(1번 방법이 조금 더 빠른듯)
import sys

def bfs(N, M):
    q = [0]*(N*M)
    front = rear = -1
    for i in range(N):
        for j in range(M):
            if arr[i][j] == 1:
                rear += 1
                q[rear] = (i, j)

    while front != rear:
        front += 1
        si, sj = q[front]
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = si+di, sj+dj
            if 0<=ni<N and 0<=nj<M and not arr[ni][nj]:
                arr[ni][nj] = arr[si][sj] + 1
                rear += 1
                q[rear] = (ni, nj)

def check():
    res = 0
    for i in range(N):
        for j in range(M):
            if not arr[i][j]:
                return -1
            if res < arr[i][j]:
                res = arr[i][j]
    return res-1

M, N = map(int, sys.stdin.readline().split())
arr = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
bfs(N, M)
print(check())
```



## 42) [7569. 토마토](https://www.acmicpc.net/problem/7569)

```python
import sys
from collections import deque

def bfs(H, N, M):
    q = deque([])
    for k in range(H):
        for i in range(N):
            for j in range(M):
                if arr[k][i][j] == 1:
                    q.append((i, j, k))

    while q:
        si, sj, sk = q.popleft()
        for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            ni, nj, nk = si+di, sj+dj, sk+dk
            if 0<=ni<N and 0<=nj<M and 0<=nk<H and not arr[nk][ni][nj]:
                arr[nk][ni][nj] = arr[sk][si][sj] + 1
                q. append((ni, nj, nk))

def check():
    res = 0
    for k in range(H):
        for i in range(N):
            for j in range(M):
                if not arr[k][i][j]:
                    return -1
                if res < arr[k][i][j]:
                    res = arr[k][i][j]
    return res-1

M, N, H = map(int, sys.stdin.readline().split())
arr = [[list(map(int, sys.stdin.readline().split())) for _ in range(N)] for __ in range(H)]
bfs(H, N, M)
print(check())
```



## 43) [7662. 이중 우선순위 큐](https://www.acmicpc.net/problem/7662)

```python

```



## 44) [10026. 적록색약](https://www.acmicpc.net/problem/10026)

```python

```



## 45) [14500. 테트로미노](https://www.acmicpc.net/problem/14500)

```python

```



## 46) [16928. 뱀과 사다리 게임](https://www.acmicpc.net/problem/16928)

```python

```



## 47) [9019. DSLR](https://www.acmicpc.net/problem/9019)

```python

```



## 48) [16236. 아기 상어](https://www.acmicpc.net/problem/16236)

```python

```

