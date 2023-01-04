rcal
====

.. image:: https://github.com/jtiosue/rcal/workflows/build/badge.svg?branch=main
    :target: https://github.com/jtiosue/rcal/actions/workflows/build.yml
    :alt: GitHub Actions CI

Calibrating reviews from multiple reviewers over the course of multiple days. See the `report <https://github.com/jtiosue/rcal/blob/main/report/review_calibration.pdf>`_ for details.


**README contents**

.. contents::
    :local:
    :backlinks: top


Installation
------------

To install:

.. code:: shell

    git clone https://github.com/jtiosue/rcal.git
    cd rcal
    pip install -e .

Or you can try:

.. code:: shell

    pip install git+https://github.com/jtiosue/rcal




Example usage
-------------

.. code:: python
    
    from rcal import CalibrateData

    data = {
        ('r1', 'p0', 0): 1,  # reviewer 1 gives person 0 a 1 star rating on day 0
        ('r1', 'p1', 1): 3,  # reviewer 1 gives person 1 a 3 star rating on day 1
        ('r1', 'p2', 2): 3,  # reviewer 1 gives person 2 a 3 star rating on day 2

        ('r2', 'p2', 0): 3,  # reviewer 2 gives person 2 a 3 star rating on day 0
        ('r2', 'p0', 1): 3,  # reviewer 2 gives person 0 a 3 star rating on day 1
        ('r2', 'p1', 2): 4,  # reviewer 2 gives person 1 a 4 star rating on day 2

        ('r3', 'p1', 0): 2,  # reviewer 3 gives person 1 a 2 star rating on day 0
        ('r3', 'p2', 1): 2,  # reviewer 3 gives person 2 a 2 star rating on day 1
        ('r3', 'p0', 2): 3,  # reviewer 3 gives person 0 a 3 star rating on day 2

        ('r1', 'p3', 0): 1,  # reviewer 1 gives person 3 a 1 star rating on day 0
        ('r2', 'p3', 1): 1,  # reviewer 2 gives person 3 a 1 star rating on day 1
        ('r3', 'p3', 2): 1   # reviewer 3 gives person 3 a 1 star rating on day 2
    }

    cd = CalibrateData(data, rating_delta=4)  # rating_delta is the max score (5 stars) minus the min score (1 star) 
    cd.calibrate().rescale(0, 1)  # calibrate and rescale data to between 0 and 1
    print(cd.calibrated_ratings())
    print(cd.average_daily_calibrated_ratings())
    print(cd.improvement_rates())
    print(cd.average_daily_calibrated_ratings_with_improvement())



To do
-----

- Implement the C code so that generating the calibration matrix is feasible for large numbers of reviews.
- Clean up code
- Add docstrings
- Make it so you can learn the parameters on a subset of the data. Add tests to see how well this performs on the rest of the data.
