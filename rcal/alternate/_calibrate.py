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

"""_calibrate.py

Contains the main functionality of rcal.alternate.

"""

import numpy as np
from rcal import RcalWarning


def calibrate_parameters(data, rating_delta=None, lam=1e-5):
    """calibrate_parameters.

    Calibrate the reviewer and persons parameters from the data
    via the alternate method.

    Parameteres
    -----------
    data : dict.
        data is a dictionary. The keys are tuples (r, p, d) corresponding to
        a reviewer r, a person p, and the day of the rating d. The values
        are the corresponding rating of the person on that day by the reviewer.
        In this case, r and p should be either ints or strings, and d should be
        floats.
    rating_delta : float (default None).
        rating_delta is the range of the allowed ratings. E.g. 
        for 1 through 5 stars, rating_delta = 5-1 = 4.
        If rating_delta is None, then it will be automatically computed.
    lam : float (default .00001).
        lam is the multiplier from the report. By default, this should be very small
        but nonzero.

    Returns
    -------
    cp : rcal.alternate.CalibrationParameters object.
        Contains the information for the parameters calibrated from the data.

    Example
    -------
    >>> data = {
    >>>     ('r1', 'p1', 0): 2,
    >>>     ('r1', 'p1', 1): 4,
    >>>     ('r1', 'p1', 2): 5,
    >>>     ('r2', 'p1', 0): 3,
    >>>     ('r2', 'p1', 1): 4,
    >>>     ('r2', 'p1', 2): 5,
    >>>     ('r1', 'p2', 0): 0,
    >>>     ('r1', 'p2', 1): 0,
    >>>     ('r1', 'p2', 2): 0,
    >>>     ('r2', 'p2', 0): 1,
    >>>     ('r2', 'p2', 1): 0,
    >>>     ('r2', 'p2', 2): 0
    >>> }
    >>> cp = calibrate_parameters(data)
    >>> calibrated_data = cp.calibrate_data(data)
    >>> print([cp.performance_function(d) for d in range(3)])

    """

    if not data:
        return CalibrationParameters({})

    if rating_delta is None:
        rating_delta = max(data.values()) - min(data.values())

    N = len(data)

    P, R = {}, {}
    for (r, p, d) in data:
        P.setdefault(p, set()).add((r, d))
        R.setdefault(r, set()).add((p, d))

    indices = {}
    for i, r in enumerate(R):
        indices[('e', r)] = i
        indices[('c', r)] = i + len(R)
    for i, p in enumerate(P):
        indices[('beta', p)] = i + 2*len(R)
        indices[('gamma', p)] = i + 2*len(R) + len(P)

    # create A and c matrices
    A = np.zeros((len(indices), len(indices)))
    c = np.zeros(len(indices))


    # derivative wrt r stuff
    for r in R:
        er = indices[('e', r)]
        cr = indices[('c', r)]

        # # derivatives wrt er in the lambda term
        A[er,er] += (2 * lam / len(R)) * rating_delta**2
        c[er] += (2 * lam / len(R)) * rating_delta

        for (p, d) in R[r]:
            betap = indices[('beta', p)]
            gammap = indices[('gamma', p)]
            rating = data[(r, p, d)]

            # derivatives wrt er in the main term
            A[er,er] += (2. / N) * rating**2
            A[er,cr] += (2. / N) * rating
            A[er,betap] -= (2. / N) * rating * d
            A[er,gammap] -= (2. / N) * rating

            # derivatives wrt cr in the main term
            A[cr,er] += (2. / N) * rating
            A[cr,cr] += (2. / N)
            A[cr,betap] -= (2. / N) * d
            A[cr,gammap] -= (2. / N)


    # derivative wrt p stuff
    for p in P:
        betap = indices[('beta', p)]
        gammap = indices[('gamma', p)]

        for (r, d) in P[p]:
            er = indices[('e', r)]
            cr = indices[('c', r)]
            rating = data[(r, p, d)]

            # derivative wrt beta
            A[betap,er] -= (2. / N) * d * rating
            A[betap,cr] -= (2. / N) * d
            A[betap,betap] += (2. / N) * d**2
            A[betap,gammap] += (2. / N) * d

            # derivative wrt gamma
            A[gammap,er] -= (2. / N) * rating
            A[gammap,cr] -= (2. / N)
            A[gammap,betap] += (2. / N) * d
            A[gammap,gammap] += (2. / N)


    # get rid of singular behavior of A wrt c
    # remove equation for c0 and replace with the condition that
    # the first c is zero
    for i in indices.values():
        A[len(R), i] = 0.
    A[len(R), len(R)] = 1.
    c[len(R)] = 0.

    # get rid of singular behavior of A wrt gamma
    # remove equation for gamma0 and replace with the condition that
    # the first gamma is zero
    # for i in indices.values():
    #     A[2*len(R) + len(P), i] = 0.
    # A[2*len(R) + len(P), 2*len(R) + len(P)] = 1.
    # c[2*len(R) + len(P)] = 0.

    # get rid of singular behavior of A wrt to beta
    # if a person only has one day of rating
    for p in P:
        if len(set(d for (_, d) in P[p])) == 1:
            row = indices[('beta', p)]
            for col in indices.values():
                A[row, col] = 0.
            A[row, row] = 1.
            c[row] = 0.

    try:
        z = np.linalg.solve(A, c).tolist()
        params = {k: z[v] for k, v in indices.items()}
    except np.linalg.LinAlgError as e:
        RcalWarning.warn(str(e))
        params = {}
        for r in R:
            params[('e', r)] = 1.
            params[('c', r)] = 0.
        for p in P:
            params[('beta', p)] = 0.
            params[('gamma', p)] = float(np.mean([data[(r, p, d)] for (r, d) in P[p]]))

    return CalibrationParameters(params)



class CalibrationParameters:
    """CalibrationParameters.

    Class to manage the calibration parameters 
    beta_p, gamma_p, c_r, and e_r. See the report for more details.

    """

    def __init__(self, parameters=None, copy=False):
        """__init__.

        Parameters
        ----------
        parameters : dict (default None).
            Dictionary mapping the parameters ('c', r), ('e', r), ('beta', p) and ('gamma', p).
            to their respective floats, where r and p denote reviewers and persons.
            If parameters is set to None (default), then it will be set to the empty dict.
        copy : bool (default False).
            Whether or not to internally copy the parameters dictionary.

        """
        if parameters is None:
            self.parameters = {}
        elif copy:
            self.parameters = parameters.copy()
        else:
            self.parameters = parameters

        self.P, self.R = set(), set()

        for (t, i) in self.parameters:
            if t in ('c', 'e'):
                self.R.add(i)
            elif t in ('beta', 'gamma'):
                self.P.add(i)

    def calibrate_data(self, data, clip_endpoints=(-float('inf'), float('inf'))):
        """calibrate_data.

        Uses the internal parameters to calibrate ``data``.

        Parameters
        ----------
        data : dict.
            data is a dictionary. The keys are tuples (r, p, d) corresponding to
            a reviewer r, a person p, and the day of the rating d. The values
            are the corresponding rating of the person on that day by the reviewer.
            In this case, r and p should be either ints or strings, and d should be
            floats.
        clip_endpionts : tuple of two floats (default (-inf, inf)).
            Any calibrated data that is > clip_endpoints[1] will be set to clip_endpoints[1].
            Any calibrated data that is < clip_endpoints[0[ will be set to clip_endpoints[0].
            By default, clip_endpoints = (-float('inf'), float('inf')) so that no clipping occurs.

        Returns
        --------
        calibrated_data : dict.
            Same shape as data.

        """
        clip = lambda x: min(clip_endpoints[1], max(clip_endpoints[0], x))

        calibrated_data = {}

        for t, y in data.items():
            if isinstance(y, list):
                calibrated_data[t] = [
                    clip(self.parameters[('e', t[0])] * yy + self.parameters[('c', t[0])])
                    for yy in y
                ]
            else:
                calibrated_data[t] = clip(self.parameters[('e', t[0])] * y + self.parameters[('c', t[0])])
        
        return calibrated_data

    def rescale_parameters(self, data, bounds=(0., 1.), ignore_outliers=float("inf")):
        """rescale_parameters.

        Rescales the internal parameters based on the input data.

        Parameters
        ----------
        data : dict.
            data is a dictionary. The keys are tuples (r, p, d) corresponding to
            a reviewer r, a person p, and the day of the rating d. The values
            are the corresponding rating of the person on that day by the reviewer.
            In this case, r and p should be either ints or strings, and d should be
            floats.
        bounds : tuple of floats (default (0, 1)).
            lower and upper bounds to scale the calibrated data to within.
        ignore_outliers : float (default inf).
            ignore_outliners dictates whether outliers should be ignored when doing the rescaling.
            If ignore_outliers=float("inf") (this is default), then outliers will never be ignored.
            If ignore_outliers=f for some float f, then all calibrated scores that are not within f
            standard deviations of the mean of all calibrated scores will be treated as outliers and ignored.
            If outliers are ignored then there will be some data that is not within the bounds.

        Returns
        -------
        self

        """

        data = self.calibrate_data(data)

        vals = []
        for v in data.values():
            vals.extend(v) if isinstance(v, list) else vals.append(v)
        vals.sort()

        if not vals:
            RcalWarning.warn("Cannot rescale based on no data")
            return self

        y0, y1 = vals[0], vals[-1]

        if abs(y1 - y0) < 1e-15:
            RcalWarning.warn("Calibrated reviews are too close together to rescale")
            return self

        # remove outliers
        if ignore_outliers <= 0:
            raise RuntimeError("ignore_outliers must be positive")
        elif ignore_outliers != float("inf"):
            mean_rating, std_rating = np.mean(vals), np.std(vals)
            if std_rating < 1e-15:
                RcalWarning.warn("Standard deviation is too small to ignore outliers")
                return self
            i = 0
            while (mean_rating - y0) / std_rating > ignore_outliers:
                i += 1
                y0 = vals[i]
            i = len(vals) - 1
            while (y1 - mean_rating) / std_rating > ignore_outliers:
                i -= 1
                y1 = vals[i]

            # check again
            if abs(y1 - y0) < 1e-15:
                RcalWarning.warn("Calibrated reviews are too close together to rescale")
                return self

        # rescale

        lower, upper = bounds
        ran = upper - lower

        for r in self.R:
            self.parameters[('e', r)] *= ran / (y1 - y0)
            self.parameters[('c', r)] = lower + ran * (self.parameters[('c', r)] - y0) / (y1 - y0)

        for p in self.P:
            self.parameters[('beta', p)] *= ran / (y1 - y0)
            self.parameters[('gamma', p)] = lower + ran * (self.parameters[('gamma', p)] - y0) / (y1 - y0)

        return self

    def calibrate_rating(self, r, y, clip_endpoints=(-float('inf'), float('inf'))):
        """calibrate_rating.

        Computes xi_r(y). See the report for more details.
        Given a reviewer r and a rating y that they gave, xi_r(y)
        is their calibrated rating.

        Parameters
        ----------
        r : str or int.
            Reviewer.
        y : float.
            Rating
        clip_endpionts : tuple of two floats (default (-inf, inf)).
            If xi_r(y) > clip_endpoints[1], then this function returns clip_endpoints[1].
            If xi_r(y) < clip_endpoints[0], then this function returns clip_endpoints[0].
            By default, clip_endpoints = (-float('inf'), float('inf')) so that no clipping occurs.

        Returns
        -------
        xi_r(y) : float.
            See the report for details.

        """
        return min(
            clip_endpoints[1],
            max(
                clip_endpoints[0],
                self.parameters[('e', r)] * y + self.parameters[('c', r)]
            )
        )

    def uncalibrate_rating(self, r, y):
        """uncalibrate_rating.

        Computes xi_r^{-1}(y). See the report for more details.
        Given a reviewer r and a rating y that they gave, xi_r(y)
        is their calibrated rating. Hence, given a calibrated rating y
        and a reviewer r, xi_r^{-1](y) is what reviewer r rated.

        Parameters
        ----------
        r : str or int.
            Reviewer.
        y : float.
            Rating

        Returns
        -------
        xi_r^{-1}(y) : float.
            See the report for details.

        """
        return (y - self.parameters[('c', r)]) / self.parameters[('e', r)]

    def performance_function(self, p, d):
        """performance_function.

        Computes g_p(d). See the report for more details.
        Given a person p and a day d, g_p(d) is the performance
        function g_p(d) = beta_p d + gamma_p.

        Parameters
        ----------
        p : str or int.
            Person.
        d : float.
            Day.

        Returns
        -------
        g_p(d) : float.
            See the report for details.

        """
        return self.parameters[('beta', p)] * d + self.parameters[('gamma', p)]

    def improvement_rate(self, p):
        """improvement_rate.

        Parameters
        ----------
        p : str or int.
            Person.

        Returns
        -------
        beta_p : float.
            See the report for more details.

        """
        return self.parameters[('beta', p)]

    def set_improvement_rate(self, p, beta):
        """set_improvement_rate.

        Set the internal parameter of person p's improvement rate to be beta.

        Parameters
        ----------
        p : str or int.
            Person.
        beta : float.
            Improvement rate to set.

        """
        self.parameters[('beta', p)] = beta
        self.P.add(p)

    def person_offset(self, p):
        """person_offset.

        Parameters
        ----------
        p : str or int.
            Person.

        Returns
        -------
        gamma_p : float.
            See the report for more details.

        """
        return self.parameters[('gamma', p)]

    def set_person_offset(self, p, gamma):
        """set_person_offset.

        Set the internal parameter of person p's offset to be gamma.

        Parameters
        ----------
        p : str or int.
            Person.
        gamma : float.
            Offset to set.

        """
        self.parameters[('gamma', p)] = gamma
        self.P.add(p)

    def reviewer_offset(self, r):
        """reviewer_offset.

        Parameters
        ----------
        r : str or int.
            Reviewer.

        Returns
        -------
        c_r : float.
            See the report for more details.

        """
        return self.parameters[('c', r)]

    def set_reviewer_offset(self, r, c):
        """set_reviewer_offset.

        Set the internal parameter of reviewer r's offset to be c.

        Parameters
        ----------
        r : str or int.
            Reviewer.
        c : float.
            Reviewer offset to set.

        """
        self.parameters[('c', r)] = c
        self.R.add(r)

    def reviewer_scale(self, r):
        """reviewer_scale.

        Parameters
        ----------
        r : str or int.
            Reviewer.

        Returns
        -------
        e_r : float.
            See the report for more details.

        """
        return self.parameters[('e', r)]

    def set_reviewer_scale(self, r, e):
        """set_reviewer_scale.

        Set the internal parameter of reviewer r's scale to be e.

        Parameters
        ----------
        r : str or int.
            Reviewer.
        e : float.
            Reviewer scale to set.

        """
        self.parameters[('e', r)] = e
        self.R.add(r)

    def improvement_rates(self):
        """improvement_rates.

        Returns
        -------
        rates : dict.
            Dictionary mapping persons p to their respective beta_p.
            See the report for more details.

        """
        return {p: self.parameters[('beta', p)] for p in self.P}

    def set_improvement_rates(self, betas):
        """set_improvement_rates.

        Set the internal parameter of each person p's improvement rate to be betas[p].

        Parameters
        ----------
        betas : dict.
            betas[p] is a float.

        """
        for p, beta in betas.items():
            self.parameters[('beta', p)] = beta
            self.P.add(p)

    def reviewer_offsets(self):
        """reviewer_offsets.

        Returns
        -------
        rates : dict.
            Dictionary mapping reviewers r to their respective c_r.
            See the report for more details.

        """
        return {r: self.parameters[('c', r)] for r in self.R}

    def set_reviewer_offsets(self, cs):
        """set_reviewer_offsets.

        Set the internal parameter of reviewer r's offset to be cs[r].

        Parameters
        ----------
        cs : dict.
             cs[r] is a float.

        """
        for r, c in cs.items():
            self.parameters[('c', r)] = c
            self.R.add(r)

    def person_offsets(self):
        """person_offsets.

        Returns
        -------
        gammas : dict.
            gammas[p] is a float.

        """
        return {p: self.parameters[('gamma', p)] for p in self.P}

    def set_person_offsets(self, gammas):
        """set_person_offsets.

        Set the internal parameter of person p's offset to be gammas[p].

        Parameters
        ----------
        gamma : dict.
            Offsets to set.

        """
        for p, gamma in gammas.items():
            self.parameters[('gamma', p)] = gamma
            self.P.add(p)

    def reviewer_scales(self):
        """reviewer_scales.

        Returns
        -------
        es : dict.
            es[r] is a float.

        """
        return {r: self.parameters[('e', r)] for r in self.R}

    def set_reviewer_scales(self, es):
        """set_reviewer_scales.

        Set the internal parameter of reviewer r's scale to be es[r].

        Parameters
        ----------
        es : dict.
            es[r] is a float

        """
        for r, e in es.items():
            self.parameters[('e', r)] = e
            self.R.add(r)

    def __str__(self):
        """__str__.
        
        Returns
        -------
        String representation of this object.

        """
        return "alternate.CalibrationParameters(%s)" % self.parameters

    def copy(self):
        """copy.

        Returns a copy of self.

        """
        return CalibrationParameters(self.parameters, True)

    def __eq__(self, other):
        """__eq__.

        Defines equality of parameters.

        """
        return self.parameters == other.parameters
