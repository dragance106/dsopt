"""
Implementation of Crombecq's MIPT sampling.
"""
import numpy as np


def inter_dist(candidates, sampled):
    """
    Computes intersite distance from each candidate point to the set of already sampled points.
    For the candidate point c, its intersite distance to sampled points is
    min_j sqrt(sum_k (c_k - sampled_{j,k})^2)

    :param candidates: candidates for the new sample point
    :param sampled:    already selected sample points
    :return:           sequence of intersite distances of candidate points
    """
    ind = np.zeros(len(candidates))
    for i, c in enumerate(candidates):
        ind[i] = np.amin(np.sqrt(np.sum((c-sampled)**2, axis=1)))
    return ind


def proj_dist(candidates, sampled):
    """
    Computes projected distance from each candidate point to the set of already sampled points.
    For the candidate point c, its projected distance to sampled points is
    min_j min_k |c_k - sampled_{j,k}|

    :param candidates: candidates for the new sample point
    :param sampled:    already selected sample points
    :return:           sequence of projected distances of candidate points
    """
    prd = np.zeros(len(candidates))
    for i, c in enumerate(candidates):
        prd[i] = np.amin(np.abs(c-sampled))
    return prd


def mipt(n, dim, rng, alpha=0.5, k=100, negligible=1e-6):
    """
    Implementation of the Crombecq's mc-intersite-proj-th sampling scheme,
    in which the new candidate points are generated only within the allowed intervals,
    obtained after subtracting from [0,1]^dim
    the hypercubes covering the minimum projected distance around the already selected sample points.

    :param n:     number of sample points to be generated
    :param dim:   dimension of the hypercube (and the sample points)
    :param rng:   random number generator to be used for generating new candidate points
    :param alpha: the tolerance parameter for the minimum projected distance:
                  any candidate points with projected distance smaller than alpha/n is discarded
    :param k:     the number of candidate points to be generated in the i-th iteration
                  (after i-1 points have already been generated) will be equal to k*i
    :param negligible:   the value considered negligible when mutually comparing
                         boundaries of different intervals

    :return:      the sequence of n sample points from [0,1]^dim.
    """
    # placeholder for the sampled points
    sample = np.zeros((n, dim))

    # the first point is just randomly generated
    sample[0] = rng.random((dim,))

    for s in range(1, n):
        # minimum allowed projected distance
        dmin = alpha/(s+1)

        # placeholder for the candidates
        candidates = np.zeros((k*s, dim))

        # for each coordinate x
        for x in range(dim):
            # determine the union of disjoint intervals le?a?er removing from [0,1]
            # the intervals [sample[j,x]-dmin, sample[j,x]+dmin] for j=0,...,i-1
            start_intervals = [(0,1)]

            for j in range(s):
                # subtract [sample[j,x]-dmin, sample[j,x]+dmin] from each interval in intervals
                l2 = sample[j,x] - dmin
                u2 = sample[j,x] + dmin

                end_intervals = []
                for (l1, u1) in start_intervals:
                    if u2<l1+negligible:
                        end_intervals.append((l1,u1))
                    elif u1<l2+negligible:
                        end_intervals.append((l1,u1))
                    elif l2<l1+negligible and l1<u2+negligible and u2<u1+negligible:
                        end_intervals.append((u2,u1))
                    elif l1<l2+negligible and l2<u1+negligible and u1<u2+negligible:
                        end_intervals.append((l1,l2))
                    elif l1<l2+negligible and u2<u1+negligible:
                        end_intervals.append((l1,l2))
                        end_intervals.append((u2,u1))
                    else:
                        pass

                # now substitute end_intervals for start_intervals, and repeat
                start_intervals = end_intervals

            # after this loop finishes we have the requested union of allowed intervals,
            # so we want to generate k*i random values within them
            # to serve as the x-th coordinate for the set of candidates
            cum_length = np.zeros((len(start_intervals),))

            (l, u) = start_intervals[0]
            cum_length[0] = u-l

            # if len(start_intervals)>1:
            for i in range(1, len(start_intervals)):
                (l, u) = start_intervals[i]
                cum_length[i] = cum_length[i-1] + u-l

            total_length = cum_length[len(start_intervals)-1]

            # generate k*s random values within [0,1] and rescale them to total_length
            coords = total_length * rng.random((k*s,))

            # distribute them appropriately to the allowed intervals
            for j in range(k*s):
                i = 0
                for i in range(len(start_intervals)):
                    if coords[j] < cum_length[i] + 1e-8:   # just so that we do not miss total_length
                        break
                if i == 0:
                    coords[j] = start_intervals[i][0] + coords[j]
                else:
                    coords[j] = start_intervals[i][0] + (coords[j]-cum_length[i-1])

            # assign final coordinates to the set of candidates
            candidates[:, x] = coords

        # candidates with proper projected distance from the existing sample points are now selected,
        # so proceed to compute their intersite distance to the existing sample points
        # and add the best candidate to the sample
        ind = inter_dist(candidates, sample[:s])
        sample[s] = candidates[np.argmax(ind)]

    # n points have been now sampled
    return sample
