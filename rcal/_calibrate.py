# from ._c_code import c_generate_matrix as generate_matrix
import numpy as np


# maybe change this to a warning?
class RescaleException(Exception):
    """
    Raised when there is an error when trying to rescale
    parameters.
    """
    threshold = 1e-10


def generate_matrix(data, indices, rating_delta, lam):
    """

    data is a dictionary. The keys are tuples (r,p,d) corresponding to
    a reviewer r, a person p, and the day of the rating d. The values
    are the corresponding rating of the person on that day by the reviewer.

    indices is a dictionary mapping parameters to unique integer indices. keys are
    ('a', r), ('b', r), and ('alpha', p), for reviewers r and people p.

    rating_delta is the range of the allowed ratings. E.g. for 1 through 5 stars,
    rating_delta = 5-1 = 4.

    lam is the multiplier lambda from the paper. We guess that lamda = 1 is the best.

    This function returns a tuple (calibrated_data, parameters)
    where calibrated_data is the same shape dictionary as data but with the ratings
    calibrated. 
    parameters is a dictionary with the relevant parameters. For example, 
    parameters[('alpha', p)] gives the improvement rate of person p.

    """

    # P = set(x[1] for x in data)
    R = set(x[0] for x in data)

    P = {}
    for (r, p, d) in data:
        P.setdefault(p, set()).add((r, d))

    # N = len(set(
    #     (p1, r1, d1, r2, d2)
    #     for (r1, p1, d1) in data
    #     for (r2, p2, d2) in data
    #     if p1 == p2
    # ))
    N = sum([len(P[p])**2 for p in P])

    A = np.zeros((len(indices), len(indices)))
    c = np.zeros(len(indices))

    # derivatives wrt ar
    for r in R:

        A[
            indices[('a', r)],
            indices[('a', r)]
        ] += (2 * lam / len(R)) * rating_delta**2
        c[
            indices[('a', r)]
        ] += (2 * lam / len(R)) * rating_delta

    
    for p in P:
        for (r1, d1) in P[p]:
            for (r2, d2) in P[p]:

                # derivatives wrt alphap
                A[
                    indices[('alpha', p)],
                    indices[('a', r1)]
                ] += (2 / N) * (d2 - d1) * data[(r1, p, d1)]
                A[
                    indices[('alpha', p)],
                    indices[('a', r2)]
                ] -= (2 / N) * (d2 - d1) * data[(r2, p, d2)]
                A[
                    indices[('alpha', p)],
                    indices[('b', r1)]
                ] += (2 / N) * (d2 - d1)
                A[
                    indices[('alpha', p)],
                    indices[('b', r2)]
                ] -= (2 / N) * (d2 - d1)
                A[
                    indices[('alpha', p)],
                    indices[('alpha', p)]
                ] += (2 / N) * (d2 - d1)**2


                # derivative wrt ar
                A[
                    indices[('a', r1)],
                    indices[('a', r1)]
                ] += (2 / N) * data[(r1, p, d1)] * data[(r1, p, d1)]
                A[
                    indices[('a', r2)],
                    indices[('a', r1)]
                ] -= (2 / N) * data[(r2, p, d2)] * data[(r1, p, d1)]
                A[
                    indices[('a', r1)],
                    indices[('a', r2)]
                ] -= (2 / N) * data[(r1, p, d1)] * data[(r2, p, d2)]
                A[
                    indices[('a', r2)],
                    indices[('a', r2)]
                ] += (2 / N) * data[(r2, p, d2)] * data[(r2, p, d2)]

                A[
                    indices[('a', r1)],
                    indices[('b', r1)]
                ] += (2 / N) * data[(r1, p, d1)]
                A[
                    indices[('a', r2)],
                    indices[('b', r1)]
                ] -= (2 / N) * data[(r2, p, d2)]
                A[
                    indices[('a', r1)],
                    indices[('b', r2)]
                ] -= (2 / N) * data[(r1, p, d1)]
                A[
                    indices[('a', r2)],
                    indices[('b', r2)]
                ] += (2 / N) * data[(r2, p, d2)]

                A[
                    indices[('a', r1)],
                    indices[('alpha', p)]
                ] += (2 / N) * data[(r1, p, d1)] * (d2 - d1)
                A[
                    indices[('a', r2)],
                    indices[('alpha', p)]
                ] -= (2 / N) * data[(r2, p, d2)] * (d2 - d1)


                # derivatives wrt br
                    
                A[
                    indices[('b', r1)],
                    indices[('a', r1)]
                ] += (2 / N) * data[(r1, p, d1)]
                A[
                    indices[('b', r2)],
                    indices[('a', r1)]
                ] -= (2 / N) * data[(r1, p, d1)]

                A[
                    indices[('b', r1)],
                    indices[('a', r2)]
                ] -= (2 / N) * data[(r2, p, d2)]
                A[
                    indices[('b', r2)],
                    indices[('a', r2)]
                ] += (2 / N) * data[(r2, p, d2)]

                A[
                    indices[('b', r1)],
                    indices[('b', r1)]
                ] += (2 / N)
                A[
                    indices[('b', r2)],
                    indices[('b', r1)]
                ] -= (2 / N)

                A[
                    indices[('b', r1)],
                    indices[('b', r2)]
                ] -= (2 / N)
                A[
                    indices[('b', r2)],
                    indices[('b', r2)]
                ] += (2 / N)

                A[
                    indices[('b', r1)],
                    indices[('alpha', p)]
                ] += (2 / N) * (d2 - d1)
                A[
                    indices[('b', r2)],
                    indices[('alpha', p)]
                ] -= (2 / N) * (d2 - d1)


    # get rid of singular behavior of A wrt b
    # remove equation for b0 and replace with the condition hat
    # the first b is zero
    for i in indices.values():
        A[len(R), i] = 0.
    A[len(R), len(R)] = 1.

    # get rid of singular behavior of A wrt to alpha
    # if a person only has one day of rating
    for p in P:
        if len(set(d for (_, d) in P[p])) == 1:
            row = indices[('alpha', p)]
            for col in indices.values():
                A[row, col] = 0.
            A[row, row] = 1.

    return A, c


def calibrate_parameters(data, rating_delta=None, lam=1e-3):

    P, R = set(), set()
    for x in data:
        P.add(x[1])
        R.add(x[0])

    if rating_delta is None:
        rating_delta = max(data.values()) - min(data.values())

    indices = {}
    for i, r in enumerate(R):
        indices[('a', r)] = i
        indices[('b', r)] = i + len(R)
    for i, p in enumerate(P):
        indices[('alpha', p)] = i + 2 * len(R)

    z = np.linalg.solve(*generate_matrix(data, indices, rating_delta, lam))

    parameters = {}
    for r in R:
        parameters[('a', r)] = z[indices[('a', r)]]
        parameters[('b', r)] = z[indices[('b', r)]]
    for p in P:
        parameters[('alpha', p)] = z[indices[('alpha', p)]]

    return CalibrationParameters(parameters)



class CalibrationParameters:

    def __init__(self, parameters, copy=False):

        if copy:
            self.parameters = parameters.copy()
        else:
            self.parameters = parameters

        self.P, self.R = set(), set()

        for (t, i) in self.parameters:
            if t in ('a', 'b'):
                self.R.add(i)
            elif t == 'alpha':
                self.P.add(i)

    def calibrate_data(self, data, with_improvement=False):
        """
        Uses the internal parameters to calibrate ``data``.
        """

        calibrated_data = {}
        w = float(with_improvement)

        for ((r, p, d), y) in data.items():
            if isinstance(y, list):
                calibrated_data[(r, p, d)] = [
                    self.parameters[('a', r)] * yy + self.parameters[('b', r)]
                    - w * self.parameters[('alpha', p)] * d
                    for yy in y
                ]
            else:
                calibrated_data[(r, p, d)] = (
                    self.parameters[('a', r)] * y + self.parameters[('b', r)]
                    - w * self.parameters[('alpha', p)] * d
                )
        
        return calibrated_data

    def rescale_parameters(self, data, bounds=(0., 1.), with_improvement=False):
        """
        Rescales the internal parameters based on the input data.
        """

        data = self.calibrate_data(data, with_improvement)

        vals = set()
        for v in data.values():
            vals.update(v) if isinstance(v, list) else vals.add(v)
        y1 = max(vals)
        y0 = min(vals)

        # maybe change this to a warning unless exactly zero?
        if abs(y1 - y0) < RescaleException.threshold:
            raise RescaleException("Calibrated reviews are too close together to rescale")

        lower, upper = bounds
        ran = upper - lower

        # for key, rating in data.items():
        #     data[key] = ran * (rating - y0) / (y1 - y0) + lower
        for r in self.R:
            self.parameters[('a', r)] *= ran / (y1 - y0)
            self.parameters[('b', r)] = ran * (self.parameters[('b', r)] - y0) / (y1 - y0) + lower
        for p in self.P:
            self.parameters[('alpha', p)] *= ran / (y1 - y0)

        return self

    def sigma(self, r, y):

        return self.parameters[('a', r)] * y + self.parameters[('b', r)]

    def f(self, p, d):

        return - self.parameters[('alpha', p)] * d

    def improvement_rate(self, p):
        return self.parameters[('alpha', p)]

    def reviewer_scale(self, r):
        return self.parameters[('a', r)]

    def reviewer_offset(self, r):
        return self.parameters[('b', r)]

    def improvement_rates(self):
        return {p: self.parameters[('alpha', p)] for p in self.P}

    def reviewer_scales(self):
        return {r: self.parameters[('a', r)] for r in self.R}

    def reviewer_offsets(self):
        return {r: self.parameters[('b', r)] for r in self.R}



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    data = {
        ('r0', 'p0', 0): 1,
        ('r0', 'p1', 1): 3,
        ('r0', 'p2', 2): 3,

        ('r1', 'p2', 0): 3,
        ('r1', 'p0', 1): 3,
        ('r1', 'p1', 2): 4,

        ('r2', 'p1', 0): 2,
        ('r2', 'p2', 1): 2,
        ('r2', 'p0', 2): 3,

        ('r0', 'p3', 0): 1,
        ('r1', 'p3', 1): 1,
        ('r2', 'p3', 2): 1

        # ('r0', 'p4', 0): 5,
        # ('r1', 'p4', 1): 5,
        # ('r2', 'p4', 2): 5
    }

    cp = calibrate_parameters(data, rating_delta=4)
    calibrated_data = cp.rescale_parameters(data, (1, 5)).calibrate_data(data)
    print({k: round(v, 2) for k, v in calibrated_data.items()})
    print({k: round(v, 2) for k, v in cp.improvement_rates().items()})
    # print(cd.average_daily_calibrated_ratings())
    plt.figure()
    # plt.title("First data")
    plt.xlabel('raw rating')
    plt.ylabel('calibrated rating')
    ys = np.arange(1, 5, .01)
    plt.plot(ys, cp.sigma('r0', ys), label='r0')
    plt.plot(ys, cp.sigma('r1', ys), label='r1')
    plt.plot(ys, cp.sigma('r2', ys), label='r2')
    plt.legend()
    plt.show()

