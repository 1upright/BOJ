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

```



## 8) [2239. 스도쿠](https://www.acmicpc.net/problem/2239)

```python

```



## 9) [2473. 세 용액](https://www.acmicpc.net/problem/2473)

```python

```



## 10) [4386. 별자리 만들기](https://www.acmicpc.net/problem/4386)

```python

```



## 11) [9252. LCS 2](https://www.acmicpc.net/problem/9252)

```python

```



## 12) [17404. RGB거리 2](https://www.acmicpc.net/problem/17404)

```python

```



## 13) [20040. 사이클 게임](https://www.acmicpc.net/problem/20040)

```python

```



## 14) [1005. ACM Craft](https://www.acmicpc.net/problem/1005)

```python

```



## 15) [1644. 소수의 연속합](https://www.acmicpc.net/problem/1644)

```python

```



## 16) [2143. 두 배열의 합](https://www.acmicpc.net/problem/2143)

```python

```



## 17) [2252. 줄 세우기](https://www.acmicpc.net/problem/2252)

```python

```



## 18) [2342. Dance Dance Revolution](https://www.acmicpc.net/problem/2342) 

```python

```



## 19) [7579. 앱](https://www.acmicpc.net/problem/7579)

```python

```



## 20) [9466. 텀 프로젝트](https://www.acmicpc.net/problem/9466)

```python

```



## 21) [10942. 팰린드롬?](https://www.acmicpc.net/problem/10942)

```python

```



## 22) [11049. 행렬 곱셈 순서](https://www.acmicpc.net/problem/11049)

```python

```



## 23) [1007. 벡터 매칭](https://www.acmicpc.net/problem/1007)

```python

```



## 24) [1202. 보석 도둑](https://www.acmicpc.net/problem/1202)

```python

```



## 25) [1766. 문제집](https://www.acmicpc.net/problem/1766)

```python

```



## 26) [2623. 음악프로그램](https://www.acmicpc.net/problem/2623)

```python

```



## 27) [9527. 1의 개수 세기](https://www.acmicpc.net/problem/9527)

```python

```



## 28) [10775. 공항](https://www.acmicpc.net/problem/10775)

```python

```



## 29) [12015. 가장 긴 증가하는 부분 수열 2](https://www.acmicpc.net/problem/12015)

```python

```



## 30) [12100. 2048 (Easy)](https://www.acmicpc.net/problem/12100)

```python

```



## 31) [16724. 피리 부는 사나이](https://www.acmicpc.net/problem/16724)

```python

```



## 32) [16946. 벽 부수고 이동하기 4](https://www.acmicpc.net/problem/16946)

```python

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

```



## 47) [14939. 불 끄기](https://www.acmicpc.net/problem/14939)

```python

```



## 48) [16566. 카드 게임](https://www.acmicpc.net/problem/16566)

```python

```