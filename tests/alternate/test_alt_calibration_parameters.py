from rcal.alternate import CalibrationParameters
import random


def test_setters():

    P = set(range(4))
    R = set(range(3))
    parameters = {('beta', p): random.randint(0, 10) for p in P}
    parameters.update({('gamma', p): random.randint(0, 10) for p in P})
    parameters.update({('c', r): random.randint(0, 10) for r in R})
    parameters.update({('e', r): random.randint(0, 10) for r in R})

    cp = CalibrationParameters(parameters, True)
    cp1 = CalibrationParameters()
    for p in P:
        cp1.set_improvement_rate(p, parameters[('beta', p)])
        cp1.set_person_offset(p, parameters['gamma', p])
    for r in R:
        cp1.set_reviewer_scale(r, parameters[('e', r)])
        cp1.set_reviewer_offset(r, parameters[('c', r)])
    cp2 = CalibrationParameters()
    cp2.set_improvement_rates({p: parameters[('beta', p)] for p in P})
    cp2.set_person_offsets({p: parameters[('gamma', p)] for p in P})
    cp2.set_reviewer_scales({r: parameters[('e', r)] for r in R})
    cp2.set_reviewer_offsets({r: parameters[('c', r)] for r in R})

    assert cp == cp1
    assert cp == cp2
    assert cp.R == R
    assert cp.P == P
    assert cp1.R == R
    assert cp1.P == P
    assert cp2.R == R
    assert cp2.P == P
