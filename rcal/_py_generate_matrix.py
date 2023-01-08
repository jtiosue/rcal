import numpy as np


def py_generate_matrix_slow(data, indices, rating_delta, lam):
    """

    data is a dictionary. The keys are tuples (r,p,d) corresponding to
    a reviewer r, a person p, and the day of the rating d. The values
    are the corresponding rating of the person on that day by the reviewer.

    indices is a dictionary mapping parameters to unique integer indices. keys are
    ('a', r), ('b', r), and ('alpha', p), for reviewers r and people p.
    It must be that indices[('a', r)] takes values 0 through num_reviewers - 1,
    indices[('b', r)] takes values num_reviewers through 2 num_reviewers - 1,
    and indices[('alpha', p)] takes values 2 num_reviewers and above.

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
            # row is already all zero
            # for col in indices.values():
            #     A[row, col] = 0.
            A[row, row] = 1.

    return A, c




def py_generate_matrix(data, indices, rating_delta, lam):
    """

    data is a dictionary. The keys are tuples (r,p,d) corresponding to
    a reviewer r, a person p, and the day of the rating d. The values
    are the corresponding rating of the person on that day by the reviewer.

    indices is a dictionary mapping parameters to unique integer indices. keys are
    ('a', r), ('b', r), and ('alpha', p), for reviewers r and people p.
    It must be that indices[('a', r)] takes values 0 through num_reviewers - 1,
    indices[('b', r)] takes values num_reviewers through 2 num_reviewers - 1,
    and indices[('alpha', p)] takes values 2 num_reviewers and above.

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

        ar = indices[('a', r)]

        A[ar, ar] += (2 * lam / len(R)) * rating_delta**2
        c[ar] += (2 * lam / len(R)) * rating_delta

    
    for p in P:
        alphap = indices[('alpha', p)]
        for (r1, d1) in P[p]:
            ar1 = indices[('a', r1)]
            br1 = indices[('b', r1)]
            rating1 = data[(r1, p, d1)]
            for (r2, d2) in P[p]:
                ar2 = indices[('a', r2)]
                br2 = indices[('b', r2)]
                rating2 = data[(r2, p, d2)]

                # derivatives wrt alphap
                A[alphap,ar1] += (2 / N) * (d2 - d1) * rating1
                A[alphap,ar2] -= (2 / N) * (d2 - d1) * rating2
                A[alphap,br1] += (2 / N) * (d2 - d1)
                A[alphap,br2] -= (2 / N) * (d2 - d1)
                A[alphap,alphap] += (2 / N) * (d2 - d1)**2


                # derivative wrt ar
                A[ar1,ar1] += (2 / N) * rating1 * rating1
                A[ar2,ar1] -= (2 / N) * rating2 * rating1
                A[ar1,ar2] -= (2 / N) * rating1 * rating2
                A[ar2,ar2] += (2 / N) * rating2 * rating2

                A[ar1,br1] += (2 / N) * rating1
                A[ar2,br1] -= (2 / N) * rating2
                A[ar1,br2] -= (2 / N) * rating1
                A[ar2,br2] += (2 / N) * rating2

                A[ar1,alphap] += (2 / N) * rating1 * (d2 - d1)
                A[ar2,alphap] -= (2 / N) * rating2 * (d2 - d1)


                # derivatives wrt br
                    
                A[br1,ar1] += (2 / N) * rating1
                A[br2,ar1] -= (2 / N) * rating1

                A[br1,ar2] -= (2 / N) * rating2
                A[br2,ar2] += (2 / N) * rating2

                A[br1,br1] += (2 / N)
                A[br2,br1] -= (2 / N)

                A[br1,br2] -= (2 / N)
                A[br2,br2] += (2 / N)

                A[br1,alphap] += (2 / N) * (d2 - d1)
                A[br2,alphap] -= (2 / N) * (d2 - d1)


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
            # row is already all zero
            # for col in indices.values():
            #     A[row, col] = 0.
            A[row, row] = 1.

    return A, c

