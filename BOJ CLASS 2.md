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

