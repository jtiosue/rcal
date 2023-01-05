from rcal import calibrate_parameters
import random
import numpy as np


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
    
    cp = calibrate_parameters(data)
    cp.rescale_parameters(data)
    calibrated_data = cp.calibrate_data(data)
    assert all((0 <= v <= 1 for v in calibrated_data.values()))
    cp.improvement_rates()

    cp.rescale_parameters(data, (0, 4))
    calibrated_data = cp.calibrate_data(data)
    assert all((0 <= v <= 4 for v in calibrated_data.values()))

    for ((r, p, d), y) in data.items():
        assert np.allclose(
            calibrated_data[(r, p, d)], 
            cp.sigma(r, y)
        )

    cp.rescale_parameters(data, (0, 4), True)
    calibrated_data = cp.calibrate_data(data, True)
    assert all((0 <= v <= 4 for v in calibrated_data.values()))

    for ((r, p, d), y) in data.items():
        assert np.allclose(
            calibrated_data[(r, p, d)], 
            cp.sigma(r, y) + cp.f(p, d)
        )

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

    cp = calibrate_parameters(data, rating_delta=4)
    cp.rescale_parameters(data)
    cp.improvement_rates()
    cp.reviewer_scales()
    cp.reviewer_offsets()


def test_single_day():

    # when there's only a single day of data, alpha's should just be set to 0
    data = {
        ('r1', 'p1', 0): 1,
        ('r2', 'p1', 0): 2,
        ('r1', 'p2', 0): 3,
        ('r2', 'p2', 0): 2
    }

    cp = calibrate_parameters(data)
    cp.rescale_parameters(data)
    assert np.allclose(
        list(cp.improvement_rates().values()),
        [0.] * len(cp.P)
    )


def test_large_case():

    # once c code is written, need to make this even larger.

    data = {}
    for r in range(10):
        for p in range(100):
            for d in range(8):
                data[(r, p, d)] = random.random()
    
    cp = calibrate_parameters(data)

    cp.rescale_parameters(data).calibrate_data(data)

    # for this type of random data, we expect all the calibrated
    # ratings to be roughly around  .5
