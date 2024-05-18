import numpy as np
from scipy.integrate import quad, dblquad
from tqdm import tqdm
from scipy.stats import uniform, pearson3

def K(t, s, A_a, A_b, D_a, D_b, A_a_b, distr):
    '''
    Calculate covariance function of process {Y_t, t \in [0,1]}
    INPUT:
        t - first coordinate
        s - second coordinate 
        A_a, A_b, D_a, D_b, A_a_b - coefficients of covariance functions
        distr - distribution (from scipy)
    '''
    return (np.minimum(t, s) - t*s +
            distr.pdf(distr.ppf(s)) * distr.pdf(distr.ppf(t)) *
            np.nan_to_num(
                A_a(t) + distr.ppf(s) * A_b(t) + A_a(s) + D_a +
                A_a_b * distr.ppf(s) + distr.ppf(t) * A_b(s) +
                A_a_b * distr.ppf(t) + D_b * distr.ppf(t) * distr.ppf(s),
                nan=0)
            )

def calculate_D(h, distr):
    '''
    Calculate coefficient D for function h and distribution distr 
    '''
    return (quad(lambda y: h(y)**2 * distr.pdf(y), - np.inf, np.inf)[0] 
            - quad(lambda y: h(y) * distr.pdf(y), - np.inf, np.inf)[0]**2)

def calculate_A_a_b(h, distr):
    '''
    Calculate coefficient A_a_b for vector-function h = [h_1, h_2]
    and distribution distr 
    '''
    if len(h) != 2:
        return 0
    return (quad(lambda y: h[0](y)*h[1](y) * distr.pdf(y), - np.inf, np.inf)[0] - 
            quad(lambda y: h[0](y) * distr.pdf(y), - np.inf, np.inf)[0] * 
            quad(lambda y: h[1](y) * distr.pdf(y), - np.inf, np.inf)[0])

def first_moment(K, Anderson_Darling=False, epsabs=1e-8):
    '''
    Calculate first moment of limit distribution.
    INPUT: 
        K(t,s) - covariance function (should have 2 arguments)
        Anderson_Darling - find moment of limit distribution of
                           Anderson-Darling statistics (True)
                           or Cramer-von Mises statistics (False)
        epsabs - absolute error of integration
    OUTPUT:
        first_moment - first moment of limit distribution of 
                       intresting statistics 
                       (Anderson-Darling or Cramer-von Mises)
        error - real error of integration
    '''
    if Anderson_Darling: weight = lambda x: 1/(x*(1-x))
    else: weight = lambda x: 1
    def function_to_integrate(t, K):
        return K(t,t)*weight(t)
    x1, x2 = 0, 1
    first_moment_int = quad(lambda x: function_to_integrate(x, K), 
                            x1, x2, epsabs=epsabs)
    return first_moment_int

def second_moment(K, Anderson_Darling=False, epsabs=1e-8):
    '''
    Calculate second moment of limit distribution.
    INPUT: 
        K(t,s) - covariance function (should have 2 arguments)
        Anderson_Darling - find moment of limit distribution of
                           Anderson-Darling statistics (True)
                           or Cramer-von Mises statistics (False)
        epsabs - absolute error of integration
    OUTPUT:
        second_moment - second moment of limit distribution of 
                       intresting statistics 
                       (Anderson-Darling or Cramer-von Mises)
        error - real error of integration
    '''
    if Anderson_Darling: weight = lambda x,y: 1/(x*(1-x)*y*(1-y))
    else: weight = lambda x, y: 1
    def function_to_integrate(t, s, K):
        return (2*K(t,s)*K(t,s) + K(t,t)*K(s,s))*weight(t,s)
    x1, x2 = 0, 1
    y1, y2 = lambda x: 0, lambda x: 1
    second_moment_int = dblquad(lambda x,y: function_to_integrate(x, y, K), 
                            x1, x2, y1, y2, epsabs=epsabs)
    return second_moment_int


def third_moment(K, Anderson_Darling=False, N=1_000_000, M=100):
    '''
    Monte-Carlo inegration to find third moment of limit distribution.
    INPUT: 
        K(t,s) - covariance function (should have 2 arguments)
        Anderson_Darling - find moment of limit distribution of
                           Anderson-Darling statistics (True)
                           or Cramer-von Mises statistics (False)
        N, int - number of observations in one generated sample
        M, int - numer of samples
    OUTPUT:
        second_moment - second moment of limit distribution of 
                       intresting statistics 
                       (Anderson-Darling or Cramer-von Mises)
                       with O(1/sqrt(N*M)) accuracy
    '''
    if Anderson_Darling: weight = lambda x, y,z: 1/(x*(1-x)*y*(1-y)*z*(1-z))
    else: weight = lambda x, y, z: 1
    def function_to_integrate(t, s, r, K):
        return (K(t,t)*(K(s,s)*K(r,r) +2*K(s,r)**2) +
                2*K(t,s)*(K(t,s)*K(r,r) + 2* K(t,r)*K(s,r)) +
                2*K(t,r)*(K(t,r)*K(s,s) + 2* K(t,s)*K(s,r)))*weight(t,s,r)
    S = 0
    for i in tqdm(range(M)):
        S += sum(function_to_integrate(t=uniform.rvs(size=N),
                                       s=uniform.rvs(size=N),
                                       r=uniform.rvs(size=N),
                                       K=K))
    return S/(N*M)

def Pearson_approx(EZ1, EZ2, EZ3):
    '''
    Get Pearson Type III approximation distribution by moments
    INPUT: 
        EZ1, EZ2, EZ3 - first three moments.
    OUTPUT:
        Pearson_approximation - scipy pearson type 3 distribution CDF 
    '''
    mu_3 = EZ3 - 3*EZ2*EZ1 + 3*EZ1**3 - (EZ1)**3
    sigma = np.sqrt(EZ2 - EZ1**2)
    Cs = mu_3 / sigma**3
    Pearson_approximation = lambda x : pearson3.cdf(x, skew = Cs, loc = EZ1, scale = sigma)
    return Pearson_approximation