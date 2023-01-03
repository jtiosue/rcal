# from ._c_code import c_generate_matrix as generate_matrix

#### OLD #####

delta = lambda x, y: float(x == y)

def generate_matrix(data, indices, deltay=1, final_day=0, lam=1):
    """

    data is a dictionary. The keys are tuples (r,p,d) corresponding to
    a reviewer r, a person p, and the day of the rating d. The values
    are the corresponding rating of the person on that day by the reviewer.

    deltay is the range of the allowed ratings. E.g. for 1 through 5 stars,
    deltay = 5-1 = 4.

    final_day is the last day that ratings are being taken.

    lam is the multiplier lambda from the paper. We guess that lamda = 1 is the best.

    This function returns a tuple (calibrated_data, parameters)
    where calibrated_data is the same shape dictionary as data but with the ratings
    calibrated. 
    parameters is a dictionary with the relevant parameters. For example, 
    parameters[('alpha', p)] gives the improvement rate of person p.

    """

    P = set(x[1] for x in data)
    R = set(x[0] for x in data)
    # D = set(x[2] for x in data)

    N = len(set(
        (p1, r1, d1, r2, d2)
        for (r1, p1, d1) in data
        for (r2, p2, d2) in data
        if p1 == p2
    ))

    # indices = {}
    # for i, r in enumerate(R):
    #     indices[('a', r)] = i
    #     indices[('b', r)] = i + len(R)
    # for i, p in enumerate(P):
    #     indices[('alpha', p)] = i + 2 * len(R)

    A = np.zeros((len(indices), len(indices)))
    c = np.zeros(len(indices))

    # derivatives wrt ar
    for r in R:

        A[
            indices[('a', r)],
            indices[('a', r)]
        ] += (2 * lam / len(R)) * deltay**2
        c[
            indices[('a', r)]
        ] += (2 * lam / len(R)) * deltay
        
        for (r1, p, d1) in data:
            for (r2, p2, d2) in data:
                if p2 == p:
                    A[
                        indices[('a', r)],
                        indices[('a', r1)]
                    ] += (2 / N) * (
                        delta(r, r1) * data[(r1, p, d1)] - delta(r, r2) * data[(r2, p, d2)]
                    ) * data[(r1, p, d1)]
                    A[
                        indices[('a', r)],
                        indices[('a', r2)]
                    ] -= (2 / N) * (
                        delta(r, r1) * data[(r1, p, d1)] - delta(r, r2) * data[(r2, p, d2)]
                    ) * data[(r2, p, d2)]
                    A[
                        indices[('a', r)],
                        indices[('b', r1)]
                    ] += (2 / N) * (
                        delta(r, r1) * data[(r1, p, d1)] - delta(r, r2) * data[(r2, p, d2)]
                    )
                    A[
                        indices[('a', r)],
                        indices[('b', r2)]
                    ] -= (2 / N) * (
                        delta(r, r1) * data[(r1, p, d1)] - delta(r, r2) * data[(r2, p, d2)]
                    )
                    A[
                        indices[('a', r)],
                        indices[('alpha', p)]
                    ] += (2 / N) * (
                        delta(r, r1) * data[(r1, p, d1)] - delta(r, r2) * data[(r2, p, d2)]
                    ) * (d2 - d1)


    # derivatives wrt br
    for r in R:

        for (r1, p, d1) in data:
            for (r2, p2, d2) in data:
                if p2 == p:
                    A[
                        indices[('b', r)],
                        indices[('a', r1)]
                    ] += (2 / N) * (delta(r, r1) - delta(r, r2)) * data[(r1, p, d1)]
                    A[
                        indices[('b', r)],
                        indices[('a', r2)]
                    ] -= (2 / N) * (delta(r, r1) - delta(r, r2)) * data[(r2, p, d2)]
                    A[
                        indices[('b', r)],
                        indices[('b', r1)]
                    ] += (2 / N) * (delta(r, r1) - delta(r, r2))
                    A[
                        indices[('b', r)],
                        indices[('b', r2)]
                    ] -= (2 / N) * (delta(r, r1) - delta(r, r2))
                    A[
                        indices[('b', r)],
                        indices[('alpha', p)]
                    ] += (2 / N) * (delta(r, r1) - delta(r, r2)) * (d2 - d1)


    # derivatives wrt alphap
    for (r1, p, d1) in data:
        for (r2, p2, d2) in data:
            if p2 == p:
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

    # get rid of singular behavior of A
    # remove equation for b0 and replace with the condition hat
    # the first b is zero
    for i in indices.values():
        A[len(R), i] = 0.
    A[len(R), len(R)] = 1.

    return A, c


###########



import numpy as np


class CalibrateData:
    
    def __init__(self, data, deltay=None, final_day=None):
        self.P, self.R, self.D = set(), set(), set()
        self.data, self.calibrated_data = {}, {}

        for x, y in data.items():
            self.data[x] = y
            self.calibrated_data[x] = y
            self.P.add(x[1])
            self.R.add(x[0])
            self.D.add(x[2])

        self.deltay = deltay if deltay is not None else max(self.data.values()) - min(self.data.values())
        self.final_day = final_day if final_day is not None else max(self.D) - min(self.D)

        self.indices, self.parameters = {}, {}
        for i, r in enumerate(self.R):
            self.indices[('a', r)] = i
            self.indices[('b', r)] = i + len(self.R)
            self.parameters[('a', r)] = 1.
            self.parameters[('b', r)] = 0.
        for i, p in enumerate(self.P):
            self.indices[('alpha', p)] = i + 2 * len(self.R)
            self.parameters[('alpha', p)] = 0.


    def calibrate(self, lam=1.):
        z = np.linalg.solve(*generate_matrix(self.data, self.indices, self.deltay, self.final_day, lam))

        self.parameters = {}
        for r in self.R:
            self.parameters[('a', r)] = z[self.indices[('a', r)]]
            self.parameters[('b', r)] = z[self.indices[('b', r)]]
        for p in self.P:
            self.parameters[('alpha', p)] = z[self.indices[('alpha', p)]]

        self.calibrated_data = {
            (r, p, d): (
                self.parameters[('a', r)] * rating + self.parameters[('b', r)]
            )
            for ((r, p, d), rating) in self.data.items()
        }

    def rescale(self, lower=0., upper=1.):

        y1 = max(self.calibrated_data.values())
        y0 = min(self.calibrated_data.values())

        if abs(y0 - y1) < 1e-5:
            raise Exception("Calibrated reviews are too close together to rescale")

        ran = upper - lower

        for key, rating in self.calibrated_data.items():
            self.calibrated_data[key] = ran * (rating - y0) / (y1 - y0) + lower
        for r in self.R:
            self.parameters[('a', r)] *= ran / (y1 - y0)
            self.parameters[('b', r)] = ran * (self.parameters[('b', r)] - y0) / (y1 - y0) + lower
        for p in self.P:
            self.parameters[('alpha', p)] *= ran / (y1 - y0)

    def get_improvement_rate(self, p):
        return self.parameters[('alpha', p)]

    def get_reviewer_scale(self, r):
        return self.parameters[('a', r)]

    def get_reviewer_offset(self, r):
        return self.parameters[('b', r)]

    def get_calibrated_rating(self, r, p, d):
        return self.calibrated_data[(r, p, d)]

    def get_uncalibrated_rating(self, r, p, d):
        return self.data[(r, p, d)]

    def get_improvement_rates(self):
        return {p: self.parameters[('alpha', p)] for p in self.P}

    def get_reviewer_scales(self):
        return {r: self.parameters[('a', r)] for r in self.R}

    def get_reviewer_offsets(self):
        return {r: self.parameters[('b', r)] for r in self.R}

    def get_calibrated_ratings(self):
        return self.calibrated_data.copy()

    def get_uncalibrated_ratings(self):
        return self.data.copy()

    def all_reviews(self):
        yield from self.data.keys()

    def all_reviewers(self):
        yield from self.R

    def all_persons(self):
        yield from self.P

    def all_days(self):
        yield from self.D

    def average_daily_calibrated_ratings(self):
        ratings = {}
        for ((_, p, d), y) in self.calibrated_data.items():
            ratings.setdefault(p, {}).setdefault(d, []).append(y)
        for p in ratings.keys():
            for d in ratings[p].keys():
                ratings[p][d] = sum(ratings[p][d]) / len(ratings[p][d])
        return ratings

    def average_daily_uncalibrated_ratings(self):
        ratings = {}
        for ((_, p, d), y) in self.data.items():
            ratings.setdefault(p, {}).setdefault(d, []).append(y)
        for p in ratings.keys():
            for d in ratings[p].keys():
                ratings[p][d] = sum(ratings[p][d]) / len(ratings[p][d])
        return ratings

    def average_calibrated_ratings(self):
        ratings = {}
        for ((_, p, __), y) in self.calibrated_data.items():
            ratings.setdefault(p, []).append(y)
        for p in ratings.keys():
            ratings[p] = sum(ratings[p]) / len(ratings[p])
        return ratings

    def average_uncalibrated_ratings(self):
        ratings = {}
        for ((_, p, __), y) in self.data.items():
            ratings.setdefault(p, []).append(y)
        for p in ratings.keys():
            ratings[p] = sum(ratings[p]) / len(ratings[p])
        return ratings

    def average_reviewer_daily_calibrated_ratings(self):
        ratings = {}
        for ((r, _, d), y) in self.calibrated_data.items():
            ratings.setdefault(r, {}).setdefault(d, []).append(y)
        for r in ratings.keys():
            for d in ratings[r].keys():
                ratings[r][d] = sum(ratings[r][d]) / len(ratings[r][d])
        return ratings

    def average_reviewer_daily_uncalibrated_ratings(self):
        ratings = {}
        for ((r, _, d), y) in self.data.items():
            ratings.setdefault(r, {}).setdefault(d, []).append(y)
        for r in ratings.keys():
            for d in ratings[r].keys():
                ratings[r][d] = sum(ratings[r][d]) / len(ratings[r][d])
        return ratings

    def average_reviewer_calibrated_ratings(self):
        ratings = {}
        for ((r, _, __), y) in self.calibrated_data.items():
            ratings.setdefault(r, []).append(y)
        for r in ratings.keys():
            ratings[r] = sum(ratings[r]) / len(ratings[r])
        return ratings

    def average_reviewer_uncalibrated_ratings(self):
        ratings = {}
        for ((r, _, __), y) in self.data.items():
            ratings.setdefault(r, []).append(y)
        for r in ratings.keys():
            ratings[r] = sum(ratings[r]) / len(ratings[r])
        return ratings
