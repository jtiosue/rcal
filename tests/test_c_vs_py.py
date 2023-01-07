from rcal import c_generate_matrix, py_generate_matrix, py_generate_matrix_slow
import numpy as np
import random


def assertclose(data, lam):
    P, R = set(), set()
    for x in data:
        P.add(x[1])
        R.add(x[0])

    rating_delta = max(data.values()) - min(data.values())

    indices = {}
    for i, r in enumerate(R):
        indices[('a', r)] = i
        indices[('b', r)] = i + len(R)
    for i, p in enumerate(P):
        indices[('alpha', p)] = i + 2 * len(R)

    params = data, indices, rating_delta, lam

    c = c_generate_matrix(*params)
    p = py_generate_matrix(*params)
    ps = py_generate_matrix_slow(*params)

    np.testing.assert_allclose(c[0], p[0], atol=1e-12)
    np.testing.assert_allclose(c[1], p[1], atol=1e-12)
    np.testing.assert_allclose(c[0], ps[0], atol=1e-12)
    np.testing.assert_allclose(c[1], ps[1], atol=1e-12)


def test_simple_cases():

    data = {
        ('r1', 'p1', 0): 2,
        ('r1', 'p1', 1): 4,
        ('r1', 'p1', 2): 5,
        ('r2', 'p1', 0): 3,
        ('r2', 'p1', 1): 4,
        ('r2', 'p1', 2): 5,

        ('r1', 'p2', 0): 0,
        ('r1', 'p2', 1): 0,
        ('r1', 'p2', 2): 0,
        ('r2', 'p2', 0): 1,
        ('r2', 'p2', 1): 0,
        ('r2', 'p2', 2): 0
    }

    assertclose(data, 1e-3)
    assertclose(data, 1)

def test_simple_cases_1():

    data = {
        ('r1', 'p0', 0): 1,
        ('r1', 'p1', 1): 3,
        ('r1', 'p2', 2): 3,

        ('r2', 'p2', 0): 3,
        ('r2', 'p0', 1): 3,
        ('r2', 'p1', 2): 4,

        ('r3', 'p1', 0): 2,
        ('r3', 'p2', 1): 2,
        ('r3', 'p0', 2): 3,

        ('r1', 'p3', 0): 1,
        ('r2', 'p3', 1): 1,
        ('r3', 'p3', 2): 1
    }

    assertclose(data, 1e-3)
    assertclose(data, 1)


def test_single_day():

    # when there's only a single day of data, alpha's should just be set to 0
    data = {
        ('r1', 'p1', 0): 1,
        ('r2', 'p1', 0): 2,
        ('r1', 'p2', 0): 3,
        ('r2', 'p2', 0): 2
    }

    assertclose(data, 1e-3)
    assertclose(data, 1)


def test_large_case():

    data = {}
    for r in range(10):
        for p in range(100):
            for d in range(8):
                data[(r, p, d)] = random.random()

    assertclose(data, 1e-3)
    assertclose(data, 1)
