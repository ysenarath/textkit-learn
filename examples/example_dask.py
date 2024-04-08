import random
import time

import dask


@dask.delayed
def inc(x):
    time.sleep(random.random())
    return x + 1


@dask.delayed
def dec(x):
    time.sleep(random.random())
    return x - 1


@dask.delayed
def add(x, y):
    time.sleep(random.random())
    return x + y


x = inc(1)
y = dec(2)
z = add(x, y)

z.visualize(rankdir="LR")

res = z.compute()

print(res)

zs = []

for i in range(256):
    x = inc(i)
    y = dec(x)
    z = add(x, y)
    zs.append(z)


zs = dask.persist(*zs)

print(zs)
