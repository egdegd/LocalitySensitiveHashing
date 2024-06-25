import math
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


class LshBlackBox:
    def __init__(self, dataset: List, k=2, l=2, w=2, filename="lsh.txt", seed=1485):
        np.random.seed(seed)
        assert len(dataset) != 0
        self.d = len(dataset[0])
        self.w = w
        self.k = k
        self.l = l
        self.lsh = [[self.elementary_hash() for _ in range(k)] for _ in range(l)]
        self.hashes = [dict() for _ in range(l)]
        for point in dataset:
            for i, hash_fun in enumerate(self.lsh):
                self.hashes[i][self.compute_hash(hash_fun, point)] = point
        self.save_to_file(filename)

    def elementary_hash(self):
        a = np.random.randn(self.d)
        b = np.random.uniform(0, self.w)
        return [float(x) for x in a], float(b)

    def compute_hash(self, hash_fun: List[Tuple[List[float], float]], point: List[float]):
        ret_hash = []
        for one_elementary_hash in hash_fun:
            ret_hash.append(math.floor((np.dot(one_elementary_hash[0], point) + one_elementary_hash[1]) / self.w))
        return tuple(ret_hash)

    def query(self, point):
        for i, hash_fun in enumerate(self.lsh):
            new_hash = self.compute_hash(hash_fun, point)
            if new_hash in self.hashes[i]:
                return self.hashes[i][new_hash]
        return None

    def get_full_hash_table(self, point):
        full_hash_table = []
        for i, hash_fun in enumerate(self.lsh):
            full_hash_table.append(self.compute_hash(hash_fun, point))
        return full_hash_table

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            for hash_vector in self.lsh:
                f.write(str(hash_vector) + '\n')


def rand_search(lsh, R, c):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in tqdm(range(100000)):
        if i % 2 == 0:
            p = np.random.randn(lsh.d) / (lsh.d ** 0.5)
        else:
            p = np.random.randn(lsh.d) / (lsh.d ** 0.5) * 2
        norm = np.linalg.norm(p)
        if R < norm < c * R:
            continue
        # print(norm)
        ans = lsh.query(p)
        if norm <= R:
            if ans is not None:
                tp += 1
            else:
                fn += 1
        elif norm >= c * R:
            if ans is not None:
                fp += 1
            else:
                tn += 1
    return tp, tn, fp, fn


class ObjectiveFunction:
    def __init__(self, delta: float, lsh: LshBlackBox):
        self.delta = delta
        self.lsh = lsh

    def value(self, x: List[float], precision: int):
        collisions = 0
        for i in range(precision):
            y = np.random.randn(self.lsh.d) / (self.lsh.d ** 0.5) * self.delta
            ans = self.lsh.query(x + y)
            # print(np.linalg.norm(x), np.linalg.norm(y), np.linalg.norm(x + y))
            if ans is not None:
                collisions += 1
        return collisions / precision

    def grad(self, x, precision, h):
        grad = []
        for direction in range(self.lsh.d):
            new_x = x.copy()
            new_x[direction] += h
            diff = self.value(new_x, precision) - self.value(x, precision)
            grad.append(diff / h)
            # print(diff, np.linalg.norm(new_x) - np.linalg.norm(x))
        return grad


def make_onw_grad_step(x, function, precision, step=0.01):
    grad1 = function.grad(x, precision, step)
    x1 = x.copy() - step * np.array(grad1)
    return x1


def make_one_return_step(x, R=1):
    if np.linalg.norm(x) <= R:
        return x
    else:
        return x / (np.linalg.norm(x) / R)


def grad_search(lsh, function, prec):
    x0 = np.random.randn(lsh.d) / (lsh.d ** 0.5)
    x0 = x0 / (np.linalg.norm(x0))
    value = function.value(x0, precision=prec)

    print(np.linalg.norm(x0), value)

    while value > 0.5:
        x1 = make_onw_grad_step(x0, function, prec)
        print(np.linalg.norm(x1), function.value(x1, precision=prec))

        x0 = make_one_return_step(x1)
        value = function.value(x0, precision=prec)
        print(np.linalg.norm(x0), value)


dimension = 10
prec = 100000
dataset = [tuple([0 for _ in range(dimension)])]
seed = np.random.randint(10000)
# seed = 8592
print(seed)
lsh = LshBlackBox(dataset, k=20, l=200, w=4, seed=seed)
function = ObjectiveFunction(0.01, lsh)

grad_search(lsh, function, prec)

# tp, tn, fp, fn = rand_search(lsh, 1, 2)
#
# print(tp, tn, fp, fn)
# print((fp + fn) / (tp + tn))

