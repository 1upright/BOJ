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

