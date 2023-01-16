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

Contains the main functionality of rcal.

"""

from rcal._c_generate_matrix import c_generate_matrix
import numpy as np
from rcal import RcalWarning


def calibrate_parameters(data, rating_delta=None, lam=1e-5):
    """calibrate_parameters.

    Calibrate the reviewer and persons parameters from the data.

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
    cp : rcal.CalibrationParameters object.
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
    >>> cp.rescale_parameters(data)
    >>> calibrated_data = cp.calibrate_data(data)

    """

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

    try:
        z = np.linalg.solve(
            *c_generate_matrix(data, indices, rating_delta, lam)
        ).tolist()
        params = {k: z[v] for k, v in indices.items()}
    except np.linalg.LinAlgError as e:
        RcalWarning.warn("np.linalg.LinAlgError: " + str(e))
        params = {('alpha', p): 0. for p in P}
        for r in R:
            params[('a', r)] = 1.
            params[('b', r)] = 0.

    return CalibrationParameters(params)


class CalibrationParameters:
    """CalibrationParameters.

    Class to manage the calibration parameters 
    alpha_p, a_r, and b_r. See the report for more details.

    """

    def __init__(self, parameters=None, copy=False):
        """__init__.

        Parameters
        ----------
        parameters : dict (default None).
            Dictionary mapping the parameters ('a', r), ('b', r) and ('alpha', p).
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
            if t in ('a', 'b'):
                self.R.add(i)
            elif t == 'alpha':
                self.P.add(i)

    def calibrate_data(self, data, with_improvement=False, clip_endpoints=(-float('inf'), float('inf'))):
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
        with_improvement : bool (default False).
            Whether to calibrate with just sigma (with_improvement = False) or
            with sigma and f (with_improvement = True).
            See the report for more details.
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
                    clip(
                        self.parameters[('a', t[0])] * yy + self.parameters[('b', t[0])]
                        - (self.parameters[('alpha', t[1])] * t[2] if with_improvement else 0.)
                    )
                    for yy in y
                ]
            else:
                calibrated_data[t] = clip(
                    self.parameters[('a', t[0])] * y + self.parameters[('b', t[0])]
                    - (self.parameters[('alpha', t[1])] * t[2] if with_improvement else 0.)
                )
        
        return calibrated_data

    def rescale_parameters(self, data, bounds=(0., 1.), with_improvement=False, ignore_outliers=float("inf")):
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
        with_improvement : bool (default False).
            Whether to rescale with just sigma (with_improvement = False) or
            with sigma and f (with_improvement = True).
            See the report for more details.
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

        data = self.calibrate_data(data, with_improvement)

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

        # for key, rating in data.items():
        #     data[key] = ran * (rating - y0) / (y1 - y0) + lower
        for r in self.R:
            self.parameters[('a', r)] *= ran / (y1 - y0)
            self.parameters[('b', r)] = ran * (self.parameters[('b', r)] - y0) / (y1 - y0) + lower
        for p in self.P:
            self.parameters[('alpha', p)] *= ran / (y1 - y0)

        return self

    def calibrate_rating(self, r, y, clip_endpoints=(-float('inf'), float('inf'))):
        """calibrate_rating.

        Computes sigma_r(y). See the report for more details.
        Given a reviewer r and a rating y that they gave, sigma_r(y)
        is their calibrated rating.

        Parameters
        ----------
        r : str or int.
            Reviewer.
        y : float.
            Rating
        clip_endpionts : tuple of two floats (default (-inf, inf)).
            If sigma_r(y) > clip_endpoints[1], then this function returns clip_endpoints[1].
            If sigma_r(y) < clip_endpoints[0], then this function returns clip_endpoints[0].
            By default, clip_endpoints = (-float('inf'), float('inf')) so that no clipping occurs.

        Returns
        -------
        sigma_r(y) : float.
            See the report for details.

        """
        return min(
            clip_endpoints[1],
            max(
                clip_endpoints[0],
                self.parameters[('a', r)] * y + self.parameters[('b', r)]
            )
        )

    def uncalibrate_rating(self, r, y):
        """uncalibrate_rating.

        Computes sigma_r^{-1}(y). See the report for more details.
        Given a reviewer r and a rating y that they gave, sigma_r(y)
        is their calibrated rating. Hence, given a calibrated rating y
        and a reviewer r, sigma_r^{-1}(y) is what reviewer r rated.

        Parameters
        ----------
        r : str or int.
            Reviewer.
        y : float.
            Rating

        Returns
        -------
        sigma_r^{-1}(y) : float.
            See the report for details.

        """
        return (y - self.parameters[('b', r)]) / self.parameters[('a', r)]

    def improvement_function(self, p, d, final_day=0):
        """improvement_function.

        Computes f_p(d). See the report for more details.
        Given a person p and a day d, f_p(d) is the improvement
        function f_p(d) = - alpha_p d, hwere alpha_p is the improvement rate
        of person p.

        Parameters
        ----------
        p : str or int.
            Person.
        d : float.
            Day.
        final_day : float (default 0).
            Day to extend improvement to. Namely, if y is a calibrated
            review for person p on day d, then 
            y + improvement_function(p, d, final_day) will be that review
            projected to the final_day.

        Returns
        -------
        f_p(d) : float.
            See the report for details.

        """
        return self.parameters[('alpha', p)] * (final_day - d)

    def improvement_rate(self, p):
        """improvement_rate.

        Parameters
        ----------
        p : str or int.
            Person.

        Returns
        -------
        alpha_p : float.
            See the report for more details.

        """
        return self.parameters[('alpha', p)]

    def set_improvement_rate(self, p, alpha):
        """set_improvement_rate.

        Set the internal parameter of person p's improvement rate to be alpha.

        Parameters
        ----------
        p : str or int.
            Person.
        alpha : float.
            Improvement rate to set.

        """
        self.parameters[('alpha', p)] = alpha
        self.P.add(p)

    def reviewer_scale(self, r):
        """reviewer_scale.

        Parameters
        ----------
        r : str or int.
            Reviewer.

        Returns
        -------
        a_r : float.
            See the report for more details.

        """
        return self.parameters[('a', r)]

    def set_reviewer_scale(self, r, a):
        """set_reviewer_scale.

        Set the internal parameter of reviewer r's scale to be a.

        Parameters
        ----------
        r : str or int.
            Reviewer.
        a : float.
            Reviewer scale to set.

        """
        self.parameters[('a', r)] = a
        self.R.add(r)

    def reviewer_offset(self, r):
        """reviewer_offset.

        Parameters
        ----------
        r : str or int.
            Reviewer.

        Returns
        -------
        b_r : float.
            See the report for more details.

        """
        return self.parameters[('b', r)]

    def set_reviewer_offset(self, r, b):
        """set_reviewer_offset.

        Set the internal parameter of reviewer r's offset to be b.

        Parameters
        ----------
        r : str or int.
            Reviewer.
        b : float.
            Reviewer offset to set.

        """
        self.parameters[('b', r)] = b
        self.R.add(r)

    def improvement_rates(self):
        """improvement_rates.

        Returns
        -------
        rates : dict.
            Dictionary mapping persons p to their respective alpha_p.
            See the report for more details.

        """
        return {p: self.parameters[('alpha', p)] for p in self.P}

    def set_improvement_rates(self, alphas):
        """set_improvement_rates.

        Set the internal parameter of each person p's improvement rate to be alpha[p].

        Parameters
        ----------
        alphas : dict.
            alphas[p] is a float.

        """
        for p, alpha in alphas.items():
            self.parameters[('alpha', p)] = alpha
            self.P.add(p)

    def reviewer_scales(self):
        """reviewer_scales.

        Returns
        -------
        rates : dict.
            Dictionary mapping reviewers r to their respective a_r.
            See the report for more details.

        """
        return {r: self.parameters[('a', r)] for r in self.R}

    def set_reviewer_scales(self, a):
        """set_reviewer_scales.

        Set the internal parameter of reviewer r's scale to be a[r].

        Parameters
        ----------
        a : dict.
             a[r] is a float.

        """
        for r, aa in a.items():
            self.parameters[('a', r)] = aa
            self.R.add(r)

    def reviewer_offsets(self):
        """reviewer_offsets.

        Returns
        -------
        rates : dict.
            Dictionary mapping reviewers r to their respective b_r.
            See the report for more details.

        """
        return {r: self.parameters[('b', r)] for r in self.R}

    def set_reviewer_offsets(self, bs):
        """set_reviewer_offsets.

        Set the internal parameter of reviewer r's offset to be b[r].

        Parameters
        ----------
        bs : dict.
             bs[r] is a float.

        """
        for r, b in bs.items():
            self.parameters[('b', r)] = b
            self.R.add(r)

    def __str__(self):
        """__str__.
        
        Returns
        -------
        String representation of this object.

        """
        return "CalibrationParameters(%s)" % self.parameters

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
