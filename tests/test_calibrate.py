# Copyright 2023 Joseph T. Iosue

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contains tests for the calibrate_parameters function.
"""

from rcal import calibrate_parameters, RescaleException, CalibrationParameters
import random
import numpy as np


def test_errors():

    data = {
        ('r1', 'p1', 0): 1.
    }
    with np.testing.assert_raises(np.linalg.LinAlgError):
        calibrate_parameters(data)

    cp = CalibrationParameters(
        {('a', 'r1'): 1, ('b', 'r1'): 1, ('alpha', 'p1'): 0}
    )
    with np.testing.assert_raises(RescaleException):
        cp.rescale_parameters(data)


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
    cp1 = calibrate_parameters(data)
    assert cp == cp1
    cp2 = cp.copy()
    cp2.parameters[('a', 'r1')] = cp.parameters[('a', 'r1')] - 1
    assert cp != cp2

    cp.rescale_parameters(data)
    calibrated_data = cp.calibrate_data(data)
    assert all((0 <= round(v, 7) <= 1 for v in calibrated_data.values()))
    cp.improvement_rates()

    cp.rescale_parameters(data, (0., 4.))
    calibrated_data = cp.calibrate_data(data)
    assert all((0 <= round(v, 7) <= 4 for v in calibrated_data.values()))

    for ((r, p, d), y) in data.items():
        assert np.allclose(
            calibrated_data[(r, p, d)], 
            cp.calibrate_rating(r, y)
        )

    cp.rescale_parameters(data, (0., 4.), True)
    calibrated_data = cp.calibrate_data(data, True)
    assert all((0 <= round(v, 7) <= 4 for v in calibrated_data.values()))

    for ((r, p, d), y) in data.items():
        assert np.allclose(
            calibrated_data[(r, p, d)], 
            cp.calibrate_rating(r, y) + cp.improvement_function(p, d)
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

    assert np.allclose(cp.uncalibrate_rating('r1', cp.calibrate_rating('r1', 1)), 1)

    cp.improvement_function('p1', 1)


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

    data = {}
    for r in range(10):
        for p in range(100):
            for d in range(8):
                data[(r, p, d)] = random.random()
    
    cp = calibrate_parameters(data)

    cp.rescale_parameters(data).calibrate_data(data)

    # for this type of random data, we expect all the calibrated
    # ratings to be roughly around  .5


def test_verylarge_case():

    data = {}
    for r in range(20):
        for p in range(200):
            for d in range(8):
                data[(r, p, d)] = random.random()
    # 32000 reviews

    # one my laptop, this takes 2 seconds with the c code,
    # 35 with py_generate_matrix, and 55 seconds with
    # py_generate_matrix_slow
    cp = calibrate_parameters(data)
    cp.rescale_parameters(data).calibrate_data(data)


def test_ignore_outliers():

    train_data = {
        ('r1', 'p1', 0): 4,
        ('r2', 'p1', 0): 1,
        ('r1', 'p2', 0): 5,
        ('r2', 'p2', 0): 2,
        ('r2', 'p2', 1): 1,
        ('r2', 'p1', 2): 2,

        ('r2', 'p3', 2): 2,
        ('r2', 'p3', 3): 1,
        ('r2', 'p3', 4): 1,
        ('r1', 'p3', 2): 5,
        ('r1', 'p3', 3): 4
    }
    data = train_data.copy()
    data[('r2', 'p1', 1)] = 5
    cp = calibrate_parameters(train_data, rating_delta=4)
    cp.rescale_parameters(data, ignore_outliers=2.5)
    calibrated_data = cp.calibrate_data(data)
    assert calibrated_data[('r2', 'p1', 1)] > 1

    with np.testing.assert_raises(RescaleException):
        cp.rescale_parameters(data, ignore_outliers=0)
