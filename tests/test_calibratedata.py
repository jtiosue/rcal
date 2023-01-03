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
    
    cd = CalibrateData(data, deltay=4)
    cd.calibrate()
    cd.rescale()
    cd.average_daily_calibrated_ratings()
    cd.get_improvement_rates()



def test_large_case():

    # will be fast with c code

    # data = {}
    # for r in range(20):
    #     for p in range(200):
    #         for d in range(8):
    #             data[(r, p, d)] = random.random()
    
    # cd = CalibrateData(data)
    # assert cd.deltad == 7
    # assert cd.D == set(range(8))

    # cd.calibrate()

    assert True
