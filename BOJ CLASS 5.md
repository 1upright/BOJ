# BOJ CLASS 5

## 1) [12852. 1로 만들기 2](https://www.acmicpc.net/problem/12852)

```python
# pypy에서만 정답
from collections import deque

X = int(input())
q = deque([[X]])
while q:
    li = q.popleft()
    v = li[-1]
    if v == 1:
        res = li
        break
    if not v%3:
        q.append(li + [v//3])
    if not v%2:
        q.append(li + [v//2])
    q.append(li + [v-1])

print(len(res)-1)
print(*res)

# python에서도 정답
X = int(input())
dp = [[] for _ in range(X+1)]
dp[1].append(1)

for i in range(2, X+1):
    dp[i] = dp[i-1] + [i]
    if not i%2 and len(dp[i]) > len(dp[i//2])+1:
        dp[i] = dp[i//2] + [i]
    if not i%3 and len(dp[i]) > len(dp[i//3])+1:
        dp[i] = dp[i//3] + [i]

tmp = dp[X]
print(len(tmp)-1)
print(*tmp[::-1])
```



## 2) [2166. 다각형의 면적](https://www.acmicpc.net/problem/2166)

```python
import sys
input = sys.stdin.readline

N = int(input())
data = [tuple(map(int, input().split())) for _ in range(N)]
res = 0
for i in range(N):
    res += data[i][0]*data[i-1][1]
    res -= data[i][1]*data[i-1][0]
print(round(abs(res)/2,1))
```



## 3) [2467. 용액](https://www.acmicpc.net/problem/2467)

```python
# 시간 초과
import sys
input = sys.stdin.readline

N = int(input())
sols = list(map(int, input().split()))
tmp = 2000000000
x = y = 0
for i in range(N-1):
    for j in range(i+1, N):
        a, b = sols[i], sols[j]
        if tmp > abs(a+b):
            tmp = abs(a+b)
            x, y = a, b
print(x, y)

# 정답
import sys
input = sys.stdin.readline

N = int(input())
sols = list(map(int, input().split()))
res = 2000000000
l, r = 0, N-1
while l < r:
    tmp = sols[l] + sols[r]

    if res > abs(tmp):
        res = abs(tmp)
        x, y = l, r

    if tmp > 0:
        r -= 1
    elif tmp < 0:
        l += 1
    else:
        break

print(sols[x], sols[y])
```



## 4) [1197. 최소 스패닝 트리](https://www.acmicpc.net/problem/1197)

```python
# 시간 초과(배운 Kruskal)
import sys
input = sys.stdin.readline

def find_set(x):
    while x!=rep[x]:
        x = rep[x]
    return x

def union(x, y):
    rep[find_set(y)] = find_set(x)

V, E = map(int, input().split())
edge = []
for _ in range(E):
    A, B, C = map(int, input().split())
    edge.append((C, B, A))
edge.sort()
rep = list(range(V+1))
N = V + 1
cnt = 0
res = 0
for w, v, u in edge:
    if find_set(v) != find_set(u):
        cnt += 1
        union(u, v)
        res += w
        if cnt == N-1:
            break
print(res)

# 정답 - 인터넷 참고
import sys
input = sys.stdin.readline

def union(a, b):
    a = find(a)
    b = find(b)
    if b < a:
        rep[a] = b
    else:
        rep[b] = a

def find(a):
    if a == rep[a]:
        return a
    rep[a] = find(rep[a])
    return rep[a]

V, E = map(int, input().split())
edge = []
for _ in range(E):
    A, B, C = map(int, input().split())
    edge.append((C, B, A))
edge.sort()
rep = list(range(V+1))

res = 0
for w, v, u in edge:
    if find(u) != find(v):
        union(u, v)
        res += w
print(res)
```



## 5) [1647. 도시 분할 계획](https://www.acmicpc.net/problem/1647)

```python
import sys
input = sys.stdin.readline

def find(a):
    if a == rep[a]:
        return a
    rep[a] = find(rep[a])
    return rep[a]

def union(a, b):
    a = find(a)
    b = find(b)
    if a > b:
        rep[a] = b
    else:
        rep[b] = a

N, M = map(int, input().split())
edge = []
for _ in range(M):
    A, B, C = map(int, input().split())
    edge.append((C, B, A))
edge.sort()

rep = list(range(N+1))
res = cnt = 0
for w, u, v in edge:
    if find(u) != find(v):
        union(u, v)
        res += w
        cnt += 1
    if cnt == N-2:
        break

print(res)
```



## 6) [1806. 부분합](https://www.acmicpc.net/problem/1806)

```python
# 시간 초과
import sys
input = sys.stdin.readline

N, S = map(int, input().split())
seq = list(map(int, input().split()))

res = 100001
for i in range(N):
    tmp = cnt = 0
    for j in range(i, N):
        tmp += seq[j]
        cnt += 1
        if tmp > S:
            if res > cnt:
                res = cnt
            break

print(0 if res == 100001 else res)

# 정답
import sys
input = sys.stdin.readline

N, S = map(int, input().split())
seq = list(map(int, input().split()))
sum_seq = [0]*(N+1)
for i in range(1, N+1):
    sum_seq[i] = sum_seq[i-1] + seq[i-1]

s, e = 0, 1
res = 100001
while s < N:
    if sum_seq[e] - sum_seq[s] >= S:
        if res > e - s:
            res = e - s
        s += 1

    else:
        if e < N:
            e += 1
        else:
            s += 1

print(0 if res == 100001 else res)
```



## 7) [1987. 알파벳](https://www.acmicpc.net/problem/1987)

```python
# 시간 초과
import sys
input = sys.stdin.readline

def dfs(i, j, s):
    global res
    if res < s:
        res = s
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ni, nj = i+di, j+dj
        if 0<=ni<R and 0<=nj<C and not visited[dic[arr[ni][nj]]]:
            visited[dic[arr[ni][nj]]] = 1
            dfs(ni, nj, s+1)
            visited[dic[arr[ni][nj]]] = 0

R, C = map(int, input().split())
arr = [list(input().strip()) for _ in range(R)]

dic = {}
for i in range(26):
    dic[chr(i+65)] = i
visited = [0]*26
visited[dic[arr[0][0]]] = 1

res = 0
dfs(0, 0, 1)
print(res)

# pypy에서만
import sys
input = sys.stdin.readline

def dfs(i, j, s):
    global res
    if res < s:
        res = s
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ni, nj = i+di, j+dj
        if 0<=ni<R and 0<=nj<C and not visited[arr[ni][nj]]:
            visited[arr[ni][nj]] = 1
            dfs(ni, nj, s+1)
            visited[arr[ni][nj]] = 0

R, C = map(int, input().split())
arr = [list(map(lambda x : ord(x)-65 , input().strip())) for _ in range(R)]
visited = [0]*26
visited[arr[0][0]] = 1

res = 0
dfs(0, 0, 1)
print(res)

# 정답 - set을 이용하여 중복 제거
# append로 한다면 중복이 수없이 많이 일어나는 경우가 생김
import sys
input = sys.stdin.readline

R, C = map(int, input().split())
arr = [list(input().strip()) for _ in range(R)]
res = 0
q = set([(0, 0, arr[0][0])])
while q:
    i, j, s = q.pop()
    for di, dj in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        ni, nj = i+di, j+dj
        if 0<=ni<R and 0<=nj<C and arr[ni][nj] not in s:
            q.add((ni, nj, s+arr[ni][nj]))
            res = max(res, len(s))
print(res+1)
```



## 8) [2239. 스도쿠](https://www.acmicpc.net/problem/2239)

```python
# 시간 초과
import sys
input = sys.stdin.readline

arr = [list(map(int, input().strip())) for _ in range(9)]
tmp = set(range(1, 10))

cnt = 0
for i in range(9):
    for j in range(9):
        if arr[i][j]:
            cnt += 1

while cnt < 81:
    for i in range(9):
        for j in range(9):
            if not arr[i][j]:
                si, sj = i//3*3, j//3*3
                check = set(arr[i])
                for x in range(9):
                    check.add(arr[x][j])
                for x in range(3):
                    for y in range(3):
                        check.add(arr[si+x][sj+y])
                if len(tmp - check) == 1:
                    arr[i][j] = list(tmp - check)[0]
                    cnt += 1

for i in range(9):
    for j in range(9):
        print(arr[i][j], end='')
    print()
    
# pypy에서만 정답
import sys
input = sys.stdin.readline

def c1(i, k):
    for j in range(9):
        if arr[i][j] == k:
            return 0
    return 1

def c2(j, k):
    for i in range(9):
        if arr[i][j] == k:
            return 0
    return 1

def c3(i, j, k):
    ni, nj = i//3*3, j//3*3
    for di in range(3):
        for dj in range(3):
            if arr[ni+di][nj+dj] == k:
                return 0
    return 1

def dfs(s):
    if s == len(target):
        for i in range(9):
            for j in range(9):
                print(arr[i][j], end='')
            print()
        exit()

    i, j = target[s]
    for k in range(1, 10):
        if c1(i, k) and c2(j, k) and c3(i, j, k):
            arr[i][j] = k
            dfs(s+1)
            arr[i][j] = 0

arr = [list(map(int, input().strip())) for _ in range(9)]
tmp = set(range(1, 10))

target = []
for i in range(9):
    for j in range(9):
        if not arr[i][j]:
            target.append((i, j))
dfs(0)
```



## 9) [2473. 세 용액](https://www.acmicpc.net/problem/2473)

```python
# pypy에서만 정답
import sys
input = sys.stdin.readline

N = int(input())
sols = list(map(int, input().split()))
sols.sort()
res = 3000000001

for l in range(N-2):
    m, r = l+1, N-1
    while m < r:
        tmp = sols[l]+sols[m]+sols[r]

        if res > abs(tmp):
            res = abs(tmp)
            ans = [sols[l], sols[m], sols[r]]

        if tmp > 0:
            r -= 1
        elif tmp < 0:
            m += 1
        else:
            print(*ans)
            exit()

print(*ans)
```



## 10) [4386. 별자리 만들기](https://www.acmicpc.net/problem/4386)

```python
import sys
input = sys.stdin.readline

def union(a, b):
    a = find(a)
    b = find(b)
    if b < a:
        rep[a] = b
    else:
        rep[b] = a

def find(a):
    if a == rep[a]:
        return a
    rep[a] = find(rep[a])
    return rep[a]

N = int(input())
star = [list(map(float, input().split())) for _ in range(N)]

edge = []
for i in range(N-1):
    for j in range(i+1, N):
        dis = ((star[i][0]-star[j][0])**2+(star[i][1]-star[j][1])**2)**0.5
        edge.append((dis, j, i))
edge.sort()

rep = list(range(N))
res = 0
for w, u, v in edge:
    if find(u) != find(v):
        union(u, v)
        res += w
print(f'{res:.2f}')
```



## 11) [9252. LCS 2](https://www.acmicpc.net/problem/9252)

```python
import sys
s1 = list(sys.stdin.readline().rstrip())
s2 = list(sys.stdin.readline().rstrip())
N, M = len(s1), len(s2)
arr = [['']*(M+1) for _ in range(N+1)]

for i in range(1, N+1):
    for j in range(1, M+1):
        if s1[i-1] == s2[j-1]:
            arr[i][j] = arr[i-1][j-1] + s1[i-1]
        else:
            if len(arr[i-1][j]) > len(arr[i][j-1]):
                arr[i][j] = arr[i-1][j]
            else:
                arr[i][j] = arr[i][j-1]

tmp = arr[N][M]
print(len(tmp))
print(tmp)
```



## 12) [17404. RGB거리 2](https://www.acmicpc.net/problem/17404)

```python
import sys
input = sys.stdin.readline

N = int(input())
arr = [list(map(int, input().split())) for _ in range(N)]
res = 1000000

for i in range(3):
    dp = [[1000]*3 for _ in range(N)]
    dp[0][i] = arr[0][i]
    for j in range(1, N):
        dp[j][0] = min(dp[j-1][1], dp[j-1][2]) + arr[j][0]
        dp[j][1] = min(dp[j-1][0], dp[j-1][2]) + arr[j][1]
        dp[j][2] = min(dp[j-1][0], dp[j-1][1]) + arr[j][2]
    for j in range(3):
        if i != j and res > dp[N-1][j]:
            res = dp[N-1][j]

print(res)
```



## 13) [20040. 사이클 게임](https://www.acmicpc.net/problem/20040)

```python
import sys
input = sys.stdin.readline

def find(x):
    if x == rep[x]:
        return x
    rep[x] = find(rep[x])
    return rep[x]

N, M = map(int, input().split())
rep = list(range(N))
for i in range(M):
    x, y = map(int, input().split())
    x, y = find(x), find(y)
    if x == y:
        print(i+1)
        exit()
    else:
        if x > y:
            rep[x] = y
        else:
            rep[y] = x
print(0)
```



## 14) [1005. ACM Craft](https://www.acmicpc.net/problem/1005)

```python
# 인터넷 참고 - 위상 정렬
# https://freedeveloper.tistory.com/390
import sys
input = sys.stdin.readline
from collections import deque

T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    time = [0] + list(map(int, input().split()))

    graph = [[] for __ in range(N+1)]
    indegree = [0]*(N+1)
    for __ in range(K):
        x, y = map(int, input().split())
        graph[x].append(y)
        indegree[y] += 1

    q = deque()
    dp = [0]*(N+1)
    for i in range(1, N+1):
        if not indegree[i]:
            q.append(i)
            dp[i] = time[i]

    while q:
        v = q.popleft()
        for i in graph[v]:
            indegree[i] -= 1
            dp[i] = max(dp[v]+time[i], dp[i])
            if not indegree[i]:
                q.append(i)

    W = int(input())
    print(dp[W])
```



## 15) [1644. 소수의 연속합](https://www.acmicpc.net/problem/1644)

```python
# pypy에서만 정답
N = int(input())
prime = [0]
for i in range(1, N+1):
    if i == 1:
        continue
    for j in range(2, int(i**0.5)+1):
        if not i%j:
            break
    else:
        prime.append(i)

M = len(prime)
for i in range(M-1):
    prime[i+1] += prime[i]

s, e, cnt = 0, 1, 0
while e < M:
    tmp = prime[e] - prime[s]
    if tmp > N:
        s += 1
        e = s+1
    elif tmp < N:
        e += 1
    else:
        cnt += 1
        s += 1
        e = s+1
print(cnt)

# python에서도 정답 - 소수 구하는거 인터넷 참고
N = int(input())
prime = [0]
memo = [0, 0] + [1]*(N-1)
for i in range(2, N+1):
    if memo[i]:
        prime.append(i)
        for j in range(2*i, N+1, i):
            memo[j] = 0

M = len(prime)
for i in range(M-1):
    prime[i+1] += prime[i]

s, e, cnt = 0, 1, 0
while e < M:
    tmp = prime[e] - prime[s]
    if tmp > N:
        s += 1
        e = s+1
    elif tmp < N:
        e += 1
    else:
        cnt += 1
        s += 1
        e = s+1
print(cnt)
```



## 16) [2143. 두 배열의 합](https://www.acmicpc.net/problem/2143)

```python
# 시간 초과
import sys
input = sys.stdin.readline

T = int(input())
n = int(input())
A = [0] + list(map(int, input().split()))
m = int(input())
B = [0] + list(map(int, input().split()))
for i in range(n):
    A[i+1] += A[i]
for i in range(m):
    B[i+1] += B[i]

new_A = []
i, j = 0, 1
while j < n+1:
    new_A.append(A[j]-A[i])
    if j < n:
        j += 1
    else:
        i += 1
        j = i+1
new_A.sort()

new_B = []
i, j = 0, 1
while j < m+1:
    new_B.append(B[j]-B[i])
    if j < m:
        j += 1
    else:
        i += 1
        j = i+1
new_B.sort()

a = len(new_A)
b = len(new_B)
print(new_A)
print(new_B)

cnt = 0
i = j = 0
while new_A[i]+new_B[j]<=T and i<a and j<b:
    if new_A[i] + new_B[j] == T:
        cnt += 1
    elif new_A[i] + new_B[j] > T:
        i += 1
        j = i
    else:
        j += 1
print(cnt)

# 정답 - 인터넷 참고
import sys
input = sys.stdin.readline

T = int(input())

n = int(input())
A = list(map(int, input().split()))
check = {}

for i in range(n):
    tmp = A[i]
    if tmp not in check:
        check[tmp] = 1
    else:
        check[tmp] += 1
    for j in range(i+1, n):
        tmp += A[j]
        if tmp not in check:
            check[tmp] = 1
        else:
            check[tmp] += 1

m = int(input())
B = list(map(int, input().split()))

res = 0
for i in range(m):
    tmp = B[i]
    if T-tmp in check:
        res += check[T-tmp]
    for j in range(i+1, m):
        tmp += B[j]
        if T-tmp in check:
            res += check[T-tmp]

print(res)

# defaultdict를 쓰는 방법도 있다고 한다
import sys
input = sys.stdin.readline
from _collections import defaultdict

T = int(input())
check = defaultdict(int)

n = int(input())
A = list(map(int, input().split()))
for i in range(n):
    for j in range(i, n):
        check[sum(A[i:j+1])] += 1

m = int(input())
B = list(map(int, input().split()))
res = 0
for i in range(m):
    for j in range(i, m):
        res += check[T-sum(B[i:j+1])]

print(res)
```



## 17) [2252. 줄 세우기](https://www.acmicpc.net/problem/2252)

```python
import sys
input = sys.stdin.readline
from collections import deque

N, M = map(int, input().split())
indegree = [0]*(N+1)
graph = [[] for _ in range(N+1)]

for _ in range(M):
    x, y = map(int, input().split())
    indegree[y] += 1
    graph[x].append(y)

q = deque()
for i in range(1, N+1):
    if not indegree[i]:
        q.append(i)

res = []
while q:
    x = q.popleft()
    res.append(x)
    for y in graph[x]:
        indegree[y] -= 1
        if not indegree[y]:
            q.append(y)

print(*res)
```



## 18) [2342. Dance Dance Revolution](https://www.acmicpc.net/problem/2342) 

```python
# 인터넷 참고
import sys
input = sys.stdin.readline
sys.setrecursionlimit(10**6)

def move(a, b):
    if a == b: return 1
    elif b == 0:
        if a == 0:
            return 0
        return 2
    elif (a-b)%2: return 3
    else: return 4

step = list(map(int, input().split()))
step.pop()
N = len(step)
dp = [[[400001]*5 for _ in range(5)] for _ in range(N+1)]
dp[-1][0][0] = 0

for i in range(N):
    for l in range(5):
        for k in range(5):
            dp[i][l][step[i]] = min(dp[i][l][step[i]], dp[i-1][l][k] + move(step[i], k))

    for r in range(5):
        for k in range(5):
            dp[i][step[i]][r] = min(dp[i][step[i]][r], dp[i-1][k][r] + move(step[i], k))

res = 400001
for l in range(5):
    for r in range(5):
        if res > dp[N-1][l][r]:
            res = dp[N-1][l][r]
print(res)
```



## 19) [7579. 앱](https://www.acmicpc.net/problem/7579)

```python
# 인터넷 참고
import sys
input = sys.stdin.readline

N, M = map(int, input().split())
memory = [0] + list(map(int, input().split()))
cost = [0] + list(map(int, input().split()))
K = sum(cost)
dp = [[0]*(K+1) for _ in range(N+1)]

res = K
for i in range(1, N+1):
    for j in range(1, K+1):
        if cost[i] > j:
            dp[i][j] = dp[i-1][j]
        else:
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-cost[i]]+memory[i])
        if dp[i][j] >= M:
            res = min(res, j)
print(res if M else 0)
```



## 20) [9466. 텀 프로젝트](https://www.acmicpc.net/problem/9466)

```python
# 인터넷 참고
import sys
input = sys.stdin.readline

def dfs(v):
    global res
    visited[v] = 1
    tmp.append(v)
    x = choice[v]

    if visited[x]:
        if x in tmp:
            res += tmp[tmp.index(x):]
            return
    else:
        dfs(x)

for _ in range(int(input())):
    N = int(input())
    choice = [0] + list(map(int, input().split()))
    visited = [0]*(N+1)
    res = []
    for i in range(1, N+1):
        if not visited[i]:
            tmp = []
            dfs(i)
    print(N-len(res))
```



## 21) [10942. 팰린드롬?](https://www.acmicpc.net/problem/10942)

```python
# 시간 초과
import sys
input = sys.stdin.readline

N = int(input())
nums = [0] + list(map(int, input().split()))
M = int(input())

for _ in range(M):
    i, j = map(int, input().split())
    while i < j:
        if nums[i] != nums[j]:
            print(0)
            break
        else:
            i += 1
            j -= 1
    else:
        print(1)
        
# 정답
import sys
input = sys.stdin.readline

N = int(input())
nums = list(map(int, input().split()))
dp = [[0]*N for _ in range(N)]

for i in range(N):
    dp[i][i] = 1

for i in range(N-1):
    if nums[i] == nums[i+1]:
        dp[i][i+1] = 1

for k in range(2, N):
    for i in range(N-k):
        if nums[i] == nums[i+k] and dp[i+1][i+k-1]:
            dp[i][i+k] = 1

M = int(input())
for _ in range(M):
    x, y = map(int, input().split())
    print(dp[x-1][y-1])
```



## 22) [11049. 행렬 곱셈 순서](https://www.acmicpc.net/problem/11049)

> [나무위키](https://namu.wiki/w/%EC%97%B0%EC%87%84%20%ED%96%89%EB%A0%AC%20%EA%B3%B1%EC%85%88%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98) 참고 실화냐

```python
# pypy에서만 정답

import sys
input = sys.stdin.readline

N = int(input())
val = list(map(int, input().split()))
for _ in range(N-1):
    x, y = map(int, input().split())
    val.append(y)

dp = [[0]*N for _ in range(N)]
for d in range(1, N):
    for i in range(N-d):
        j = i+d
        dp[i][j] = 4294967296
        for k in range(i, j):
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + val[i]*val[k+1]*val[j+1])

print(dp[0][N-1])
```



## 23) [1007. 벡터 매칭](https://www.acmicpc.net/problem/1007)

```python
import sys
input = sys.stdin.readline
from itertools import combinations

for _ in range(int(input())):
    N = int(input())
    tot_x = tot_y = 0
    coor = []
    for __ in range(N):
        x, y = map(int, input().split())
        tot_x += x
        tot_y += y
        coor.append((x, y))

    res = 1000000
    combi = list(combinations(coor, N//2))
    for com in combi[:len(combi)//2]:
        tmp_x = tmp_y = 0
        for x, y in com:
            tmp_x += x
            tmp_y += y
        tmp = ((tot_x-tmp_x*2)**2 + (tot_y-tmp_y*2)**2)**0.5
        if res > tmp:
            res = tmp
    print(res)
```



## 24) [1202. 보석 도둑](https://www.acmicpc.net/problem/1202)

```python
# 인터넷 참고
import sys
input = sys.stdin.readline
from heapq import *

N, K = map(int, input().split())

jewel = []
for _ in range(N):
    heappush(jewel, list(map(int, input().split())))

bags = sorted(list(int(input()) for _ in range(K)))

tmp = []
res = 0
for b in bags:
    while jewel and b >= jewel[0][0]:
        heappush(tmp, -heappop(jewel)[1])
    if tmp:
        res -= heappop(tmp)
    elif not jewel:
        break
print(res)
```



## 25) [1766. 문제집](https://www.acmicpc.net/problem/1766)

> #17 + heap

```python
# 인터넷 참고

import sys
input = sys.stdin.readline
from heapq import *

N, M = map(int, input().split())
indegree = [0]*(N+1)
graph = [[] for _ in range(N+1)]

for _ in range(M):
    x, y = map(int, input().split())
    indegree[y] += 1
    graph[x].append(y)

heap = []
for i in range(1, N+1):
    if not indegree[i]:
        heappush(heap, i)

while heap:
    x = heappop(heap)
    print(x, end=' ')
    for y in graph[x]:
        indegree[y] -= 1
        if not indegree[y]:
            heappush(heap, y)
```



## 26) [2623. 음악프로그램](https://www.acmicpc.net/problem/2623)

```python
import sys
from collections import deque

input = sys.stdin.readline
N, M = map(int, input().split())
indegree = [0]*(N+1)
graph = [[] for _ in range(N+1)]

for _ in range(M):
    order = list(map(int, input().split()))
    for i in range(1, order[0]):
        indegree[order[i+1]] += 1
        graph[order[i]].append(order[i+1])

q = deque()
for i in range(1, N+1):
    if not indegree[i]:
        q.append(i)

res = []
while q:
    x = q.popleft()
    res.append(x)
    for y in graph[x]:
        indegree[y] -= 1
        if not indegree[y]:
            q.append(y)

if len(res) == N:
    for x in res:
        print(x)
else:
    print(0)
```



## 27) [9527. 1의 개수 세기](https://www.acmicpc.net/problem/9527)

```python
# 인터넷 참고
def solve(x):
    cnt, k = 0, 1
    while k <= x:
        k *= 2
        cnt += (x+1)//k*(k//2) + max(0, (x+1)%k-k//2)
    return cnt

A, B = map(int, input().split())
print(solve(B)-solve(A-1))
```



## 28) [10775. 공항](https://www.acmicpc.net/problem/10775)

```python
# 시간 초과
import sys
input = sys.stdin.readline
G = int(input())
P = int(input())
visited = [0]*(G+1)
cnt = 0
for _ in range(P):
    g = int(input())
    for i in range(g, 0, -1):
        if not visited[i]:
            visited[i] = 1
            cnt += 1
            break
    else:
        print(cnt)
        exit()
      
# 정답
import sys
input = sys.stdin.readline

def find(x):
    if x == rep[x]:
        return x
    rep[x] = find(rep[x])
    return rep[x]

G = int(input())
P = int(input())
rep = list(range(G+1))
cnt = 0
for _ in range(P):
    g = int(input())
    tmp = find(g)
    if not tmp:
        print(cnt)
        exit()
    rep[tmp] = rep[tmp-1]
    cnt += 1
else:
    print(cnt)
```



## 29) [12015. 가장 긴 증가하는 부분 수열 2](https://www.acmicpc.net/problem/12015)

```python
# 1(dp)처럼 하면 시간 초과
# 정답 - 인터넷 참고(이분 탐색)

import sys
input = sys.stdin.readline

N = int(input())
arr = list(map(int, input().split()))
res = [0]

for x in arr:
    if res[-1] < x:
        res.append(x)
    else:
        l = 0
        r = len(res)
        while l < r:
            mid = (l+r)//2
            if res[mid] < x:
                l = mid + 1
            else:
                r = mid
        res[r] = x

print(len(res)-1)

# 모듈 사용
import sys
from bisect import bisect_left as bs

input = sys.stdin.readline

N = int(input())
arr = list(map(int, input().split()))
res = [0]

for x in arr:
    if res[-1] < x:
        res.append(x)
    else:
        res[bs(res, x)] = x

print(len(res)-1)
```



## 30) [12100. 2048 (Easy)](https://www.acmicpc.net/problem/12100)

```python
import sys
input = sys.stdin.readline
from copy import deepcopy

def move(arr, x):
    if x == 0: # 상
        for j in range(N):
            s = 0
            for i in range(1, N):
                if arr[i][j]:
                    arr[i][j], tmp = 0, arr[i][j]
                    if arr[s][j] == 0:
                        arr[s][j] = tmp
                    elif arr[s][j] == tmp:
                        arr[s][j] *= 2
                        s += 1
                    else:
                        s += 1
                        arr[s][j] = tmp
    elif x == 1: # 하
        for j in range(N):
            s = N-1
            for i in range(N-2, -1, -1):
                if arr[i][j]:
                    arr[i][j], tmp = 0, arr[i][j]
                    if arr[s][j] == 0:
                        arr[s][j] = tmp
                    elif arr[s][j] == tmp:
                        arr[s][j] *= 2
                        s -= 1
                    else:
                        s -= 1
                        arr[s][j] = tmp

    elif x == 2: # 좌
        for i in range(N):
            s = 0
            for j in range(1, N):
                if arr[i][j]:
                    arr[i][j], tmp = 0, arr[i][j]
                    if arr[i][s] == 0:
                        arr[i][s] = tmp
                    elif arr[i][s] == tmp:
                        arr[i][s] *= 2
                        s += 1
                    else:
                        s += 1
                        arr[i][s] = tmp

    elif x == 3: # 우
        for i in range(N):
            s = N-1
            for j in range(N-2, -1, -1):
                if arr[i][j]:
                    arr[i][j], tmp = 0, arr[i][j]
                    if arr[i][s] == 0:
                        arr[i][s] = tmp
                    elif arr[i][s] == tmp:
                        arr[i][s] *= 2
                        s -= 1
                    else:
                        s -= 1
                        arr[i][s] = tmp
    return arr

def dfs(board, idx):
    global res
    if idx == 5:
        res = max(max(map(max, board)), res)
        return

    for i in range(4):
        dfs(move(deepcopy(board), i), idx+1)

N = int(input())
board = [list(map(int, input().split())) for _ in range(N)]
res = 0
dfs(board, 0)
print(res)
```



## 31) [16724. 피리 부는 사나이](https://www.acmicpc.net/problem/16724)

```python
import sys

def dfs(i, j, flag):
    global res
    if visited[i][j]:
        if visited[i][j] == flag:
            res += 1
        return

    visited[i][j] = flag
    if arr[i][j] == 'U':
        dfs(i-1, j, flag)
    elif arr[i][j] == 'D':
        dfs(i+1, j, flag)
    elif arr[i][j] == 'L':
        dfs(i, j-1, flag)
    elif arr[i][j] == 'R':
        dfs(i, j+1, flag)

input = sys.stdin.readline
N, M = map(int, input().split())
arr = [list(input().strip()) for _ in range(N)]
visited = [[0]*M for _ in range(N)]

flag = res = 0
for i in range(N):
    for j in range(M):
        flag += 1
        dfs(i, j, flag)
print(res)
```



## 32) [16946. 벽 부수고 이동하기 4](https://www.acmicpc.net/problem/16946)

```python
import sys
input = sys.stdin.readline
from collections import deque

N, M = map(int, input().split())
arr = [list(map(int, input().strip())) for _ in range(N)]
visited = [[0]*M for _ in range(N)]
for i in range(N):
    for j in range(M):
        if not arr[i][j] and not visited[i][j]:
            visited[i][j] = 1
            q = deque([(i, j)])
            cnt = 1
            tmp = []

            while q:
                si, sj = q.popleft()
                for di, dj in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
                    ni, nj = si+di, sj+dj
                    if 0<=ni<N and 0<=nj<M and not visited[ni][nj]:
                        visited[ni][nj] = 1
                        if arr[ni][nj]:
                            tmp.append((ni, nj))
                        else:
                            q.append((ni, nj))
                            cnt += 1

            for mi, mj in tmp:
                visited[mi][mj] = 0
                arr[mi][mj] += cnt

for i in range(N):
    for j in range(M):
        print(arr[i][j]%10, end="")
    print()
```



## 33) [17143. 낚시왕](https://www.acmicpc.net/problem/17143)

```python

```



## 34) [17387. 선분 교차 2](https://www.acmicpc.net/problem/17387)

```python

```



## 35) [1208. 부분수열의 합 2](https://www.acmicpc.net/problem/1208)

```python

```



## 36) [1509. 팰린드롬 분할](https://www.acmicpc.net/problem/1509)

```python

```



## 37) [1562. 계단 수](https://www.acmicpc.net/problem/1562) 

```python

```



## 38) [1799. 비숍](https://www.acmicpc.net/problem/1799)

```python

```



## 39) [2098. 외판원 순회](https://www.acmicpc.net/problem/2098) 

```python

```



## 40) [2887. 행성 터널](https://www.acmicpc.net/problem/2887)

```python

```



## 41) [9328. 열쇠](https://www.acmicpc.net/problem/9328)

```python

```



## 42) [12850. 본대 산책2](https://www.acmicpc.net/problem/12850)

```python

```



## 43) [13460. 구슬 탈출 2](https://www.acmicpc.net/problem/13460)

```python

```



## 44) [2162. 선분 그룹](https://www.acmicpc.net/problem/2162)

```python

```



## 45) [2568. 전깃줄 - 2](https://www.acmicpc.net/problem/2568)

```python

```



## 46) [14003. 가장 긴 증가하는 부분 수열 5](https://www.acmicpc.net/problem/14003)

```python
# 인터넷 참고 1
import sys
from bisect import bisect_left as bs

input = sys.stdin.readline
N = int(input())
arr = list(map(int, input().split()))
res = [-1000000001]
data = [(-1, -1000000001)]

for x in arr:
    if res[-1] < x:
        res.append(x)
        data.append((len(res)-1, x))
    else:
        tmp = bs(res, x)
        res[tmp] = x
        data.append((tmp, x))

M = len(res)-1
print(M)

ans = []
for i in range(N, -1, -1):
    if data[i][0] == M:
        ans.append(data[i][1])
        M -= 1
print(*reversed(ans))

# 인터넷 참고 2
import sys
input = sys.stdin.readline
from bisect import bisect_left as bs

N = int(input())
arr = [0]+list(map(int, input().split()))
dp = [0]*(N+1)
res = [-1000000001]

for i in range(1, N+1):
    if arr[i] > res[-1]:
        res.append(arr[i])
        dp[i] = len(res)-1
    else:
        tmp = bs(res, arr[i])
        dp[i] = tmp
        res[tmp] = arr[i]

M = max(dp)
print(M)

ans = []
for i in range(N, 0, -1):
    if dp[i] == M:
        ans.append(arr[i])
        M -= 1
print(*reversed(ans))
```



## 47) [14939. 불 끄기](https://www.acmicpc.net/problem/14939)

```python

```



## 48) [16566. 카드 게임](https://www.acmicpc.net/problem/16566)

```python

```