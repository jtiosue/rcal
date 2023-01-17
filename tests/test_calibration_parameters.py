from rcal import CalibrationParameters
import random


def test_setters():

    P = set(range(4))
    R = set(range(3))
    parameters = {('alpha', p): random.randint(0, 10) for p in P}
    parameters.update({('beta', p): random.randint(0, 10) for p in P})
    parameters.update({('a', r): random.randint(0, 10) for r in R})
    parameters.update({('b', r): random.randint(0, 10) for r in R})
    cp = CalibrationParameters(parameters, True)
    cp1 = CalibrationParameters()
    for p in P:
        cp1.set_improvement_rate(p, parameters[('alpha', p)])
        cp1.set_person_offset(p, parameters[('beta', p)])
    for r in R:
        cp1.set_reviewer_scale(r, parameters[('a', r)])
        cp1.set_reviewer_offset(r, parameters[('b', r)])
    cp2 = CalibrationParameters()
    cp2.set_improvement_rates({p: parameters[('alpha', p)] for p in P})
    cp2.set_person_offsets({p: parameters[('beta', p)] for p in P})
    cp2.set_reviewer_scales({r: parameters[('a', r)] for r in R})
    cp2.set_reviewer_offsets({r: parameters[('b', r)] for r in R})

    assert cp == cp1
    assert cp == cp2
    assert cp.R == R
    assert cp.P == P
    assert cp1.R == R
    assert cp1.P == P
    assert cp2.R == R
    assert cp2.P == P
