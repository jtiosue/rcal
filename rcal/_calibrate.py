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

Contains the main functionality of rcal; calibrate_parameters.

"""

import numpy as np
from . import RcalWarning, CalibrationParameters


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
        rating_delta is the range of the allowed ratings. a.g. 
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

    allP, allR, P, R, D = set(), set(), {}, {}, {}
    for (r, p, d) in data:
        allP.add(p)
        allR.add(r)
        P.setdefault(r, set()).add(p)
        R.setdefault(p, set()).add(r)
        D.setdefault((r, p), set()).add(d)

    indices = {}
    for i, r in enumerate(allR):
        indices[('a', r)] = i
        indices[('b', r)] = i + len(allR)
    for i, p in enumerate(allP):
        indices[('alpha', p)] = i + 2*len(allR)
        indices[('beta', p)] = i + 2*len(allR) + len(allP)

    # create A and c matrices
    A = np.zeros((len(indices), len(indices)))
    c = np.zeros(len(indices))


    ### METHOD 1 FOR CREATING A AND C

    # # derivative wrt r stuff
    # for r in allR:
    #     ar = indices[('a', r)]
    #     br = indices[('b', r)]

    #     # # derivatives wrt ar in the lambda term
    #     A[ar,ar] += (2 * lam / len(allR)) * rating_delta**2
    #     c[ar] += (2 * lam / len(allR)) * rating_delta

    #     for p in P[r]:
    #         N = len(allP) * len(R[p]) * len(D[(r, p)])
    #         alphap = indices[('alpha', p)]
    #         betap = indices[('beta', p)]
    #         for d in D[(r, p)]:
    #             rating = data[(r, p, d)]

    #             # derivatives wrt ar in the main term
    #             A[ar,ar] += (2. / N) * rating**2
    #             A[ar,br] += (2. / N) * rating
    #             A[ar,alphap] -= (2. / N) * rating * d
    #             A[ar,betap] -= (2. / N) * rating

    #             # derivatives wrt br in the main term
    #             A[br,ar] += (2. / N) * rating
    #             A[br,br] += (2. / N)
    #             A[br,alphap] -= (2. / N) * d
    #             A[br,betap] -= (2. / N)

    # # derivative wrt p stuff
    # for p in allP:
    #     alphap = indices[('alpha', p)]
    #     betap = indices[('beta', p)]

    #     for r in R[p]:
    #         N = len(allP) * len(R[p]) * len(D[(r, p)])
    #         ar = indices[('a', r)]
    #         br = indices[('b', r)]
    #         for d in D[(r, p)]:
    #             rating = data[(r, p, d)]

    #             # derivative wrt alpha
    #             A[alphap,ar] -= (2. / N) * d * rating
    #             A[alphap,br] -= (2. / N) * d
    #             A[alphap,alphap] += (2. / N) * d**2
    #             A[alphap,betap] += (2. / N) * d

    #             # derivative wrt beta
    #             A[betap,ar] -= (2. / N) * rating
    #             A[betap,br] -= (2. / N)
    #             A[betap,alphap] += (2. / N) * d
    #             A[betap,betap] += (2. / N)



    ### METHOD 2 FOR CREATING A AND C

    # for r in allR:
    #     ar = indices[('a', r)]

    #     # # derivatives wrt ar in the lambda term
    #     A[ar,ar] += (2 * lam / len(allR)) * rating_delta**2
    #     c[ar] += (2 * lam / len(allR)) * rating_delta

    # for ((r, p, d), y) in data.items():
    #     N = len(allP) * len(R[p]) * len(D[(r, p)])
    #     alphap = indices[('alpha', p)]
    #     betap = indices[('beta', p)]
    #     ar = indices[('a', r)]
    #     br = indices[('b', r)]

    #     # derivatives wrt ar in the main term
    #     A[ar,ar] += (2. / N) * y**2
    #     A[ar,br] += (2. / N) * y
    #     A[ar,alphap] -= (2. / N) * y * d
    #     A[ar,betap] -= (2. / N) * y

    #     # derivatives wrt br in the main term
    #     A[br,ar] += (2. / N) * y
    #     A[br,br] += (2. / N)
    #     A[br,alphap] -= (2. / N) * d
    #     A[br,betap] -= (2. / N)

    #     # derivative wrt alpha
    #     A[alphap,ar] -= (2. / N) * d * y
    #     A[alphap,br] -= (2. / N) * d
    #     A[alphap,alphap] += (2. / N) * d**2
    #     A[alphap,betap] += (2. / N) * d

    #     # derivative wrt beta
    #     A[betap,ar] -= (2. / N) * y
    #     A[betap,br] -= (2. / N)
    #     A[betap,alphap] += (2. / N) * d
    #     A[betap,betap] += (2. / N)


    ### METHOD 3 FOR CREATING A AND C

    # derivative wrt r stuff
    for r in allR:
        ar = indices[('a', r)]
        br = indices[('b', r)]

        # # derivatives wrt ar in the lambda term
        A[ar,ar] += (2. * lam / len(allR)) * rating_delta**2
        c[ar] += (2. * lam / len(allR)) * rating_delta

        for p in P[r]:
            alphap = indices[('alpha', p)]
            betap = indices[('beta', p)]
            
            N = len(allP) * len(R[p]) * len(D[(r, p)])

            for d in D[(r, p)]:
                rating = data[(r, p, d)]

                # derivatives wrt ar in the main term
                A[ar,ar] += (2. / N) * rating**2
                A[ar,br] += (2. / N) * rating
                A[ar,alphap] -= (2. / N) * rating * d
                A[ar,betap] -= (2. / N) * rating

                # derivatives wrt br in the main term
                A[br,ar] += (2. / N) * rating
                A[br,br] += (2. / N)
                A[br,alphap] -= (2. / N) * d
                A[br,betap] -= (2. / N)

                # derivative wrt alpha
                A[alphap,ar] -= (2. / N) * d * rating
                A[alphap,br] -= (2. / N) * d
                A[alphap,alphap] += (2. / N) * d**2
                A[alphap,betap] += (2. / N) * d

                # derivative wrt beta
                A[betap,ar] -= (2. / N) * rating
                A[betap,br] -= (2. / N)
                A[betap,alphap] += (2. / N) * d
                A[betap,betap] += (2. / N)

    ### END DIFFERENT METHODS FOR CREATING A AND C


    # get rid of singular behavior of A wrt b
    # remove equation for b0 and replace with the condition that
    # the first b is zero
    for i in indices.values():
        A[len(allR), i] = 0.
    A[len(allR), len(allR)] = 1.
    c[len(allR)] = 0.

    # get rid of singular behavior of A wrt to alpha
    # if a person only has one day of rating
    for p in allP:
        if len(set(d for r in R[p] for d in D[(r, p)])) == 1:
            row = indices[('alpha', p)]
            for col in indices.values():
                A[row, col] = 0.
            A[row, row] = 1.
            c[row] = 0.

    # if a reviewer has only one review, then set their slope to 0.
    # for r in allR:
    #     if len(P[r]) == 1:
    for r in filter(lambda x: len(P[x]) == 1, allR):
        row = indices[('a', r)]
        for col in indices.values():
            A[row, col] = 0.
        A[row, row] = 1.
        c[row] = 0.

    # solve the system
    try:
        z = np.linalg.solve(A, c).tolist()
        params = {k: z[v] for k, v in indices.items()}
    except np.linalg.LinAlgError as e:
        RcalWarning.warn(str(e))
        params = {}
        for r in allR:
            params[('a', r)] = 1.
            params[('b', r)] = 0.
        for p in allP:
            params[('alpha', p)] = 0.
            params[('beta', p)] = float(np.mean([data[(r, p, d)] for r in R[p] for d in D[(r, p)]]))

    return CalibrationParameters(params)
