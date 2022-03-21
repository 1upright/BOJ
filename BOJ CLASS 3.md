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

```



## 14) [9461. 파도반 수열](https://www.acmicpc.net/problem/9461)

```python

```



## 15) [11047. 동전 0](https://www.acmicpc.net/problem/11047)

```python

```



## 16) [11399. ATM](https://www.acmicpc.net/problem/11399)

```python

```



## 17) [11659. 구간 합 구하기 4](https://www.acmicpc.net/problem/11659)

```python

```



## 18) [11726. 2xn 타일링](https://www.acmicpc.net/problem/11726)

```python

```



## 19) [11727. 2xn 타일링 2](https://www.acmicpc.net/problem/11727)

```python

```



## 20) [1012. 유기농 배추](https://www.acmicpc.net/problem/1012)

```python

```



## 21) [1260. DFS와 BFS](https://www.acmicpc.net/problem/1260)

```python

```



## 22) [1541. 잃어버린 괄호](https://www.acmicpc.net/problem/1541)

```python

```



## 23) [1780. 종이의 개수](https://www.acmicpc.net/problem/1780)

```python

```



## 24) [1927. 최소 힙](https://www.acmicpc.net/problem/1927)

```python

```



## 25) [1931. 회의실 배정](https://www.acmicpc.net/problem/1931)

```python

```



## 26) [5525. IOIOI](https://www.acmicpc.net/problem/5525)

```python

```



## 27) [11279. 최대 힙](https://www.acmicpc.net/problem/11279)

```python

```



## 28) [11724. 연결  요소의 개수](https://www.acmicpc.net/problem/11724)

```python

```



## 29) [18870. 좌표 압축](https://www.acmicpc.net/problem/18870)

```python

```



## 30) [1074. Z](https://www.acmicpc.net/problem/1074)

```python

```



## 31) [1389. 케빈 베이컨의 6단계 법칙](https://www.acmicpc.net/problem/1389)

```python

```



## 32) [1697. 숨바꼭질](https://www.acmicpc.net/problem/1697)

```python

```



## 33) [1992. 쿼드트리](https://www.acmicpc.net/problem/1992)

```python

```



## 34) [2178. 미로 탐색](https://www.acmicpc.net/problem/2178)

```python

```



## 35) [2667. 단지번호붙이기](https://www.acmicpc.net/problem/2667)

```python

```



## 36) [6064. 카잉 달력](https://www.acmicpc.net/problem/6064)

```python

```



## 37) [11286. 절댓값 힙](https://www.acmicpc.net/problem/11286)

```python

```



## 38) [11403. 경로 찾기](https://www.acmicpc.net/problem/11403)

```python

```



## 39) [1107. 리모컨](https://www.acmicpc.net/problem/1107)

```python

```



## 40) [5430. AC](https://www.acmicpc.net/problem/5430)

```python

```



## 41) [7576. 토마토](https://www.acmicpc.net/problem/7576)

```python

```



## 42) [7569. 토마토](https://www.acmicpc.net/problem/7569)

```python

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

