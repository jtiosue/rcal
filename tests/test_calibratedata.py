from rcal import CalibrateData
import random


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
    
    cd = CalibrateData(data)
    cd.calibrate().rescale()
    cd.average_daily_calibrated_ratings()
    cd.improvement_rates()

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
        ('r2',  'p3', 1): 1,
        ('r3', 'p3', 2): 1
    }

    cd = CalibrateData(data, rating_delta=4)
    cd.calibrate().rescale()
    cd.uncalibrated_ratings()
    cd.calibrated_ratings()
    cd.average_daily_calibrated_ratings()
    cd.improvement_rates()
    cd.average_daily_calibrated_ratings_with_improvement()


def test_large_case():

    # once c code is written, need to make this even larger.

    data = {}
    for r in range(10):
        for p in range(100):
            for d in range(8):
                data[(r, p, d)] = random.random()
    
    cd = CalibrateData(data)
    assert cd.D == set(range(4))

    cd.calibrate().rescale()

    # for this type of random data, we expect all the calibrated
    # ratings to be roughly around  .5
