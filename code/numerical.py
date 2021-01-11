import numpy as np
from scipy.stats import norm
import time
from scipy import integrate
from scipy.special import roots_hermite as herm

"""
    This module solves the optimization problem shown in the paper using binary search on 
    each of the three parameters used in the problem. We exploit the fact that the constraints
    are monotic in each function parameter. However, the integrals involved need to be
    evaluated numerically. Here we use the Gauss-Hermite approximation formula to calculate
    the numerical integrals efficiently.
"""

xs, ws = herm(1000)
xs = np.array([np.sqrt(2)*x for x in xs])
ws = np.array(ws)/np.sqrt(np.pi)

def meta_calc_fast_p(r,a,b):
    x = -(np.exp(r*xs) - a*xs)/b
    return lambda c: np.dot(ws,norm.cdf(x + (c/b)))

def calc_fast_mx(r,a,b,c):
    return np.dot(ws,xs*norm.cdf(-(np.exp(r*xs) - (a*xs + c))/b))

def calc_fast_my(r,a,b,c):
    return np.dot(ws,norm.pdf(-(np.exp(r*xs) - (a*xs + c))/b))

def calc_fast_pr(r,a,b,c):
    return np.dot(ws,np.exp(-(r**2 - (2*xs*r))/2)*norm.cdf(-(np.exp(r*xs) - (a*xs + c))/b))

def calc(r, a, b, c):
    def calc_limit(x):
        return (np.exp(r*x) - (a*x + c))/b
    
    p = integrate.quad(lambda x: norm.pdf(x) * norm.cdf(- calc_limit(x)), -np.inf, np.inf)
    mx = integrate.quad(lambda x: x * norm.pdf(x) * norm.cdf(- calc_limit(x)), -np.inf, np.inf)
    my = integrate.quad(lambda x: norm.pdf(x) * norm.pdf(calc_limit(x)), -np.inf, np.inf)
    return p[0], mx[0], my[0]

def fast_solve_p(prob, r, a, b, c=1.0, tol=0.001):
    low = 0.0
    high = None
    calc_fast_p =  meta_calc_fast_p(r,a,b)
    while ((high is None) or (((high - low)/high) > tol)):
        p = calc_fast_p(c)
        if p < prob:
            low = c
            if high is None:
                c = 2*c
            else:
                c = (high + low)/2
        else:
            high = c
            c = (high + low)/2
    return low

def fast_solve_my(prob, my, r, a, b=1.0, tol=0.001):
    low = 0.0
    high = None
    c = 1.0
    while ((high is None) or (((high - low)/high) > tol)):
        c = fast_solve_p(prob, r, a, b, c=c, tol=tol)
        m = calc_fast_my(r,a,b,c)
        if m < my:
            low = b
            if high is None:
                b = 2*b
            else:
                b = (high + low)/2
        else:
            high = b
            b = (high + low)/2
    return low, fast_solve_p(prob, r, a, low)

def fast_solve_mx(prob, my, mx, r, a=1.0, tol=0.001):
    low = 0.0 
    high = None
    b = 1.0
    while ((high is None) or ((high - low) > 0.1*tol)):
        b, c = fast_solve_my(prob, my, r, a, b=b, tol=tol)
        m = calc_fast_mx(r,a,b,c)
        if m < mx:
            low = a
            if high is None:
                a = 2*a
            else:
                a = (high + low)/2
        else:
            high = a
            a = (high + low)/2
    if low > 0:
        a = low
    elif high < 0:
        a = high
    else:
        a = 0.0
    return a, fast_solve_my(prob, my, r, a)

def binary_search(eval_func, high, low, tol):
    x = (high + low)/2
    while ((high - low) > tol):
        if eval_func(x) < 0:
            low = x
        else:
            high = x
        x = (high + low)/2
    return low, high

def calculate_radius(pABar: int, meanBar: int, tol=0.00001) -> float:
    low = norm.ppf(pABar)
    if meanBar == 0.0:
        high = norm.ppf((1+pABar)/2)
    else:
        high = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*abs(meanBar)))

    if meanBar >= norm.pdf(low):
        return low
    elif meanBar <= -norm.pdf(low):
        return np.inf
    elif meanBar == 0:
        z1 = high
        z2 = -high
    elif meanBar > 0:
        def eval_func(x):
            y = -np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(norm.pdf(x) - meanBar)))
            return norm.cdf(x) - (norm.cdf(y) + pABar) 
        l, h = binary_search(eval_func, high, low, tol)
        z1 = l
        z2 = -np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(norm.pdf(z1) - meanBar)))  
    else:
        high, low = -low, -high
        def eval_func(x):
            y = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(norm.pdf(x) + meanBar)))
            return pABar - (norm.cdf(y) - norm.cdf(x)) 
        l, h = binary_search(eval_func, high, low, tol)
        z2 = h
        z1 = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(norm.pdf(z2) + meanBar)))
    high = z1
    low = norm.ppf(pABar)
    r = (high+low)/2
    diff = (norm.cdf(z1 - r) - norm.cdf(z2 - r)) - 0.5
    while ((high - low) > tol) :
        if diff > 0:
            low = r
        else:
            high = r
        r = (high+low)/2
        diff = (norm.cdf(z1 - r) - norm.cdf(z2 - r)) - 0.5
    return low


def numerical_radius(prob, my, mx, eps=1e-4, a=1.0, tol=0.001):

    if my <= 1e-7:
        return calculate_radius(prob, abs(mx))
    prob, my = prob*(1-eps), my*(1-eps)
    mx = mx - eps*abs(mx)

    low = norm.ppf(prob)
    high = calculate_radius(prob, - np.sqrt(my**2 + mx**2))
    r = (high + low)/2
    while (((high - low)/high) > tol):
        r = (high + low)/2
        a, (b, c) = fast_solve_mx(prob, my, mx, r, a=a, tol=tol)
        pr = calc_fast_pr(r,a,b,c)
        if pr > 0.5:
            low = r        
        else:
            high = r
    a, (b, c) = fast_solve_mx(prob, my, mx, low, a=a, tol=tol)
    return low
