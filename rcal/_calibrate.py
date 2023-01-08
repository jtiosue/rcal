from rcal._c_generate_matrix import c_generate_matrix
import numpy as np


# maybe change this to a warning?
class RescaleException(Exception):
    """
    Raised when there is an error when trying to rescale
    parameters.
    """
    threshold = 1e-10


def calibrate_parameters(data, rating_delta=None, lam=1e-3):

    if not data:
        return CalibrationParameters({})

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

    z = np.linalg.solve(*c_generate_matrix(data, indices, rating_delta, lam))

    return CalibrationParameters({k: z[v] for k, v in indices.items()})



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

    def rescale_parameters(self, data, bounds=(0., 1.), with_improvement=False, ignore_outliers=float("inf")):
        """
        Rescales the internal parameters based on the input data.
        ignore_outliners dictates whether outliers should be ignored when doing the rescaling.
        If ignore_outliers=float("inf") (this is default), then outliers will never be ignored.
        If ignore_outliers=f for some float f, then all calibrated scores that are not within f
        standard deviations of the mean of all calibrated scores will be treated as outliers and ignored.
        If outliers are ignored then there will be some data that is not within the bounds.
        """

        data = self.calibrate_data(data, with_improvement)

        vals = []
        for v in data.values():
            vals.extend(v) if isinstance(v, list) else vals.append(v)
        vals.sort()
        y0, y1 = vals[0], vals[-1]

        # maybe change this to a warning unless exactly zero?
        if abs(y1 - y0) < RescaleException.threshold:
            raise RescaleException("Calibrated reviews are too close together to rescale")

        # remove outliers
        if ignore_outliers <= 0:
            raise RescaleException("ignore_outliers must be positive")
        elif ignore_outliers != float("inf"):
            mean_rating, std_rating = np.mean(vals), np.std(vals)
            if std_rating < RescaleException.threshold:
                raise RescaleException("Standard deviation is too small to ignore outliers")
            i = 0
            while (mean_rating - y0) / std_rating > ignore_outliers:
                i += 1
                y0 = vals[i]
            i = len(vals) - 1
            while (y1 - mean_rating) / std_rating > ignore_outliers:
                i -= 1
                y1 = vals[i]

            # check again
            if abs(y1 - y0) < RescaleException.threshold:
                raise RescaleException("Calibrated reviews are too close together to rescale")

        # calibrate

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

    def __str__(self):
        return "CalibrationParameters(%s)" % self.parameters



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

