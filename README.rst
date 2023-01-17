rcal
====

.. image:: https://github.com/jtiosue/rcal/workflows/test/badge.svg?branch=master
    :target: https://github.com/jtiosue/rcal/actions/workflows/test.yml?query=branch%3Amaster
    :alt: GitHub Actions CI


Calibrating reviews from multiple reviewers over the course of multiple days. See the `report <https://github.com/jtiosue/rcal/blob/main/report/review_calibration.pdf>`_ for details.



Installation
------------

To install:

.. code:: shell

    pip install git+https://github.com/jtiosue/rcal

Or:

.. code:: shell

    git clone https://github.com/jtiosue/rcal.git
    cd rcal
    pip install -e .





Example usage
-------------

See the `notebook examples <https://github.com/jtiosue/rcal/tree/main/examples>`_ for a detailed example. Here we do a quick *Hello World* example.

.. code:: python
    
    from rcal import calibrate_parameters

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

    # rating_delta is the max score (5 stars) minus the min score (1 star)
    cp = calibrate_parameters(data, rating_delta=4)
    
    # rescale the parameters so that the calibrated reviews are between 0 and 1
    cp.rescale_parameters(data, (0, 1))

    # get the calibrated data with these parameters
    print(cp.calibrate_data(data))
    
    # get the improvement rates
    print(cp.improvement_rates())
