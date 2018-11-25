import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
from math import factorial as f

def find_mean(data_lst):
    '''Write a function that takes a list of numbers as input and returns
    the mean of that input list.

    In: List of integers
    Out: The mean as an integer or float

    Ex: find_mean([1,4,3,6,3,7,6,5,8,12])
    >>> 5.5'''

    # Two ways
    np_mean = np.mean(np.array(data_lst))
    py_mean = sum(data_lst) / len(data_lst)

    return py_mean

def find_median(data_lst):

    '''Write a function that takes a list of numbers as input and returns
    the median of that input list. Account for both even and odd length lists.

    In: List of integers
    Out: The median as an integer or float

    Ex: find_median([1,4,3,6,3,7,6,5,8,12,9])
    >>> 6'''

    # With NumPy
    np_med = np.median(np.array(data_lst))

    # Without NumPy
    sort_lst = sorted(data_lst)
    if len(sort_lst) % 2 == 0:
        mid1 = sort_lst[:len(sort_lst)//2][-1]
        mid2 = sort_lst[len(sort_lst)//2:][0]
        median = (mid1 + mid2) / 2
    else:
        median = sort_lst[len(sort_lst)//2]

    return median

def find_mode(data_lst):

    '''Write a function that takes a list of numbers as input and returns
    the mode of that input list. Assume there is only one mode.

    In: List of integers
    Out: The mode as an integer or float

    Ex: find_mode([1,4,3,6,3,7,5,8,12,11])
    >>> 3'''

    # With ScipyStats
    stats_mode = stats.mode(np.array(data_lst))[0][0]

    # The hard way
    counts_dict = {}
    for each in data_lst:
        if each in counts_dict.keys():
            counts_dict[each] += 1
        else:
            counts_dict[each] = 1

    py_mode = max(counts_dict, key=counts_dict.get)

    return py_mode


def find_variance(data_lst, population=True):

    '''Write a function that takes a list of numbers as input and returns
    the variance of that input list. Notice the population=True argument:
    write your code so that it calculates population variance if that argument
    is True, and sample variance if it is False.

    In: List of integers
    Out: The variance as an integer or float

    Ex: find_variance([1,4,3,6,3,7,5,8,12,11,0,34])
    >>>74.4722222222'''

    # With NumPy (this defaults to population variance)
    np_var = np.var(np.array(data_lst))

    # The hard way
    mn = find_mean(data_lst)

    dist = []
    for each in data_lst:
        dist.append(each - mn)

    squares = []
    for each in dist:
        squares.append(each**2)

    sums = sum(squares)

    if population:
        v = sums / len(data_lst)
    else:
        v = sums / len(data_lst) - 1
    return v


def find_stand_dev(data_lst):

    '''Write a function that takes a list of numbers as input and returns
    the standard deviation of that input list. Use your variance function.

    In: List of integers
    Out: The standard deviation as an integer or float

    Ex: find_stand_dev([1,4,3,6,3,7,5,8,12,11,0,34])
    >>>8.629728977333079'''

    # Two ways
    np_std = np.sqrt(find_variance(data_lst))
    py_std = find_variance(data_lst)**(1/2)

    # Alicia, np and regular are a few digits off from each other
    # in the tens of thousands decimal place, maybe we should add round for
    # any in which the students might use either np or regular?
    return round(py_std, 3)

def bernoulli_pmf(p, n, k):

    '''Given the probability of success, p and the possible outcomes, k
    (1 or 0), and the number of trials, n (1), calculate the PMF, the E(X), and the Var(X) for
    a binomial distribution.

    In: p: probability of success, k: possible outcomes, n: num trials; all ints or floats
    Out: PMF, E(X), Var(X), all integers or floats

    Ex: bernoulli_pmf(0.4, 1, 1)
    >>>0.4, 0.4, 0.24'''

    p_s = p
    p_f = (1-p)
    ex = p
    varx = p_s * p_f

    pmf = (p_s**k) * (p_f**(n-k))

    return pmf, ex, varx

def binomial_pmf(p, n, k):

    '''Given the probability of success, the total number of trials, and
    the number of successful trials, calculate the PMF, E(X) and
    Var(X) for the binomial distribution.

    In: p: probability of success, k: possible outcomes; both ints or floats
    Out: PMF, E(X), Var(X), all ints or floats.

    Ex: binomial_pmf(0.6, 300, 24)
    >>>8.207452839437304e+58, 13.600000000000001, 8.16'''

    combs = (f(n)) / ((f(k))*(f(n-k)))
    bern_pmf = bernoulli_pmf(p, n, k)[0]
    pmf = combs * bern_pmf

    ex = n*p
    var = (n*p)*(1-p)

    return pmf, ex, var


def poisson_pmf(rate, k):

    '''Given the rate, and k, the number of successes, write a function that
    calculates the PMF, E(X) and Var(X) of the Poisson distribution.

    In: The rate, k: number of successes; both ints or floats
    Out: PMF, E(X), Var(X); all ints or floats

    Ex: poisson_pmf(5, 0)
    >>>0.006737946999085469, 5, 5'''

    e = math.e

    numer = (rate**k) * (e**-rate)
    denom = f(k)

    pmf = numer / denom
    ex = rate
    varx = rate

    return pmf, ex, varx


if __name__ == '__main__':

    even_lst = [1,4,3,6,3,7,5,8,12,11,0,34]
    odd_lst = [1,4,3,6,3,7,6,5,8,12,9]

    mn = find_mean(even_lst)
    med = find_median(even_lst)
    mod = find_mode(even_lst)
    var = find_variance(even_lst)
    std = find_stand_dev(even_lst)

    bern_pmf, bern_ex, bern_varx = bernoulli_pmf(0.4, 1, 1)
    binom_pmf, binom_ex, binom_varx = binomial_pmf(0.4, 34, 5)
    poisson_pmf, poisson_ex, poisson_varx = poisson_pmf(5, 0)




#
