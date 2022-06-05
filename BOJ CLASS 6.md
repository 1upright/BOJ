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



## 2) [13334. 철로](https://www.acmicpc.net/problem/13334)

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



## 3) [14725. 개미굴](https://www.acmicpc.net/problem/14725)

```python

```



## 4) [16565. N포커](https://www.acmicpc.net/problem/16565)

```python

```



## 5) [1019. 책 페이지](https://www.acmicpc.net/problem/1019)

```python

```



## 6) [2042. 구간 합 구하기](https://www.acmicpc.net/problem/2042)

```python

```



## 7) [2357. 최솟값과 최댓값](https://www.acmicpc.net/problem/2357)

```python

```



## 8) [3015. 오아시스 재결합](https://www.acmicpc.net/problem/3015)

```python

```



## 9) [11505. 구간 곱 구하기](https://www.acmicpc.net/problem/11505)

```python

```



## 10) [11689. GCD(n, k) = 1](https://www.acmicpc.net/problem/11689)

```python

```



## 11) [13977. 이항 계수와 쿼리](https://www.acmicpc.net/problem/13977)

```python

```



## 12) [14428. 수열과 쿼리 16](https://www.acmicpc.net/problem/14428)

```python

```



## 13) [15824. 너 봄에는 캡사이신이 맛있단다](https://www.acmicpc.net/problem/15824)

```python

```



## 14) [17371. 이사](https://www.acmicpc.net/problem/17371)

```python

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