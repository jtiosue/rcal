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

"""_calibration_parameters.py

Contains the CalibrationParameters class.

"""


from . import RcalException
import numpy as np


class CalibrationParameters:
    """CalibrationParameters.

    Class to manage the calibration parameters 
    alpha_p, beta_p, b_r, and a_r. See the report for more details.

    """

    def __init__(self, parameters=None, copy=False):
        """__init__.

        Parameters
        ----------
        parameters : dict (default None).
            Dictionary mapping the parameters ('b', r), ('a', r), ('alpha', p) and ('beta', p).
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
            if t in ('b', 'a'):
                self.R.add(i)
            elif t in ('alpha', 'beta'):
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
            with sigma and the improvement rate alpha (with_improvement = True).
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
            with sigma and the improvement rate alpha (with_improvement = True).
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
            raise RcalException("Cannot rescale based on no data")

        y0, y1 = vals[0], vals[-1]

        if abs(y1 - y0) < 1e-15:
            raise RcalException("Calibrated reviews are too close together to rescale")

        # remove outliers
        if ignore_outliers <= 0:
            raise RcalException("ignore_outliers must be positive")
        elif ignore_outliers != float("inf"):
            mean_rating, std_rating = np.mean(vals), np.std(vals)
            if std_rating < 1e-15:
                raise RcalException("Standard deviation is too small to ignore outliers")
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
                raise RcalException("Calibrated reviews are too close together to rescale")

        # rescale

        lower, upper = bounds
        ran = upper - lower

        for r in self.R:
            self.parameters[('a', r)] *= ran / (y1 - y0)
            self.parameters[('b', r)] = lower + ran * (self.parameters[('b', r)] - y0) / (y1 - y0)

        for p in self.P:
            self.parameters[('alpha', p)] *= ran / (y1 - y0)
            self.parameters[('beta', p)] = lower + ran * (self.parameters[('beta', p)] - y0) / (y1 - y0)

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
        and a reviewer r, sigma_r^{-1](y) is what reviewer r rated.

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

        Computes alpha_p final_day - f_p(d). See the report for more details.
        Given a person p and a day d, f_p(d) is the improvement
        function f_p(d) = alpha_p d, where alpha_p is
        the improvement rate of person p. Thus, alpha_p final_day - f_p(d)
        is intuitively the amount that person p improves from day d to day
        final_day.

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
        alpha_p final_day - f_p(d) : float.
            See the report for details.

        """
        return self.parameters[('alpha', p)] * (final_day - d)

    def performance_function(self, p, d, clip_endpoints=(-float('inf'), float('inf'))):
        """performance_function.

        Computes g_p(d). See the report for more details.
        Given a person p and a day d, g_p(d) is the performance
        function g_p(d) = alpha_p d + beta_p.

        Parameters
        ----------
        p : str or int.
            Person.
        d : float.
            Day.
        clip_endpionts : tuple of two floats (default (-inf, inf)).
            If g_p(d) > clip_endpoints[1], then this function returns clip_endpoints[1].
            If g_p(d) < clip_endpoints[0], then this function returns clip_endpoints[0].
            By default, clip_endpoints = (-float('inf'), float('inf')) so that no clipping occurs.

        Returns
        -------
        g_p(d) : float.
            See the report for details.

        """
        return min(
            clip_endpoints[1],
            max(
                clip_endpoints[0],
                self.parameters[('alpha', p)] * d + self.parameters[('beta', p)]
            )
        )

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

    def person_offset(self, p):
        """person_offset.

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

    def set_person_offset(self, p, beta):
        """set_person_offset.

        Set the internal parameter of person p's offset to be beta.

        Parameters
        ----------
        p : str or int.
            Person.
        beta : float.
            Offset to set.

        """
        self.parameters[('beta', p)] = beta
        self.P.add(p)

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

        Set the internal parameter of each person p's improvement rate to be alphas[p].

        Parameters
        ----------
        alphas : dict.
            alphas[p] is a float.

        """
        for p, alpha in alphas.items():
            self.parameters[('alpha', p)] = alpha
            self.P.add(p)

    def reviewer_offsets(self):
        """reviewer_offsets.

        Returns
        -------
        bs : dict.
            Dictionary mapping reviewers r to their respective b_r.
            See the report for more details.

        """
        return {r: self.parameters[('b', r)] for r in self.R}

    def set_reviewer_offsets(self, bs):
        """set_reviewer_offsets.

        Set the internal parameter of reviewer r's offset to be bs[r].

        Parameters
        ----------
        bs : dict.
             bs[r] is a float.

        """
        for r, b in bs.items():
            self.parameters[('b', r)] = b
            self.R.add(r)

    def person_offsets(self):
        """person_offsets.

        Returns
        -------
        betas : dict.
            betas[p] is a float.

        """
        return {p: self.parameters[('beta', p)] for p in self.P}

    def set_person_offsets(self, betas):
        """set_person_offsets.

        Set the internal parameter of person p's offset to be betas[p].

        Parameters
        ----------
        beta : dict.
            Offsets to set.

        """
        for p, beta in betas.items():
            self.parameters[('beta', p)] = beta
            self.P.add(p)

    def reviewer_scales(self):
        """reviewer_scales.

        Returns
        -------
        aas : dict.
            aas[r] is a float.

        """
        return {r: self.parameters[('a', r)] for r in self.R}

    def set_reviewer_scales(self, aas):
        """set_reviewer_scales.

        Set the internal parameter of reviewer r's scale to be aas[r].

        Parameters
        ----------
        aas : dict.
            aas[r] is a float

        """
        for r, a in aas.items():
            self.parameters[('a', r)] = a
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
