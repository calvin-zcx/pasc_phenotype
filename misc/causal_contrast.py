import scipy.stats as st
import warnings
from lifelines.statistics import *
from lifelines.statistics import _chisq_test_p_value
# from lifelines import utils
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.utils import (
    # group_survival_table_from_events,
    # string_rjustify,
    # format_p_value,
    # format_floats,
    interpolate_at_times_and_return_pandas,
    # _expected_value_of_survival_up_to_t,
    # _expected_value_of_survival_squared_up_to_t,
)
import numpy as np


def st_survival_difference_at_fixed_point(point_in_time, fitterA, fitterB, **result_kwargs):
    """
        my takes, 2024-2-28: Revised this to support cumulative incidence, and confidence interval

    Often analysts want to compare the survival-ness of groups at specific times, rather than comparing the entire survival curves against each other.
    For example, analysts may be interested in 5-year survival. Statistically comparing the naive Kaplan-Meier points at a specific time
    actually has reduced power (see [1]). By transforming the survival function, we can recover more power. This function uses
    the log(-log(·)) transformation.

    Parameters
    ----------
    point_in_time: float,
        the point in time to analyze the survival curves at.

    fitterA:
        A lifelines univariate model fitted to the data. This can be a ``KaplanMeierFitter``, ``WeibullFitter``, etc.

    fitterB:
        the second lifelines model to compare against.

    result_kwargs:
        add keywords and meta-data to the experiment summary

    Returns
    -------

    StatisticalResult
      a StatisticalResult object with properties ``p_value``, ``summary``, ``test_statistic``, ``print_summary``

    Examples
    --------
    .. code:: python

        T1 = [1, 4, 10, 12, 12, 3, 5.4]
        E1 = [1, 0, 1,  0,  1,  1, 1]
        kmf1 = KaplanMeierFitter().fit(T1, E1)

        T2 = [4, 5, 7, 11, 14, 20, 8, 8]
        E2 = [1, 1, 1, 1,  1,  1,  1, 1]
        kmf2 = KaplanMeierFitter().fit(T2, E2)

        from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
        results = survival_difference_at_fixed_point_in_time_test(12.0, kmf1, kmf2)

        results.print_summary()
        print(results.p_value)        # 0.77
        print(results.test_statistic) # 0.09

    Notes
    -----
    1. Other transformations are possible, but Klein et al. [1] showed that the log(-log(·)) transform has the most desirable
    statistical properties.

    2. The API of this function changed in v0.25.3. This new API allows for right, left and interval censoring models to be tested.


    References
    -----------

    [1] Klein, J. P., Logan, B. , Harhoff, M. and Andersen, P. K. (2007), Analyzing survival curves at a fixed point in time. Statist. Med., 26: 4505-4519. doi:10.1002/sim.2864

    """
    if type(fitterB) != type(fitterA):
        warnings.warn(
            "This test compares survival functions, but your fitters are estimating the survival functions differently. This means that this test is also testing the different ways to estimate the survival function and will be unreliable.",
            UserWarning,
        )

    log = np.log
    clog = lambda s: log(-log(s))

    sA_t = fitterA.predict(point_in_time)
    sB_t = fitterB.predict(point_in_time)

    from lifelines.fitters import NonParametricUnivariateFitter, ParametricUnivariateFitter

    if isinstance(fitterA, AalenJohansenFitter):
        sigma_sqA = interpolate_at_times_and_return_pandas(fitterA.variance_, point_in_time)
        sA_t = 1 - sA_t
    elif isinstance(fitterA, NonParametricUnivariateFitter):
        sigma_sqA = interpolate_at_times_and_return_pandas(fitterA._cumulative_sq_, point_in_time)
    elif isinstance(fitterA, ParametricUnivariateFitter):
        sigma_sqA = fitterA._compute_variance_of_transform(fitterA._survival_function, [point_in_time]).squeeze()

    if isinstance(fitterB, AalenJohansenFitter):
        sigma_sqB = interpolate_at_times_and_return_pandas(fitterB.variance_, point_in_time)
        sB_t = 1 - sB_t
    elif isinstance(fitterB, NonParametricUnivariateFitter):
        sigma_sqB = interpolate_at_times_and_return_pandas(fitterB._cumulative_sq_, point_in_time)
    elif isinstance(fitterB, ParametricUnivariateFitter):
        sigma_sqB = fitterB._compute_variance_of_transform(fitterB._survival_function, [point_in_time]).squeeze()

    X = (clog(sA_t) - clog(sB_t)) ** 2 / (sigma_sqA / log(sA_t) ** 2 + sigma_sqB / log(sB_t) ** 2)
    p_value = _chisq_test_p_value(X, 1)

    # http://fmwww.bc.edu/repec/bocode/s/stsurvdiff.ado
    # https://discourse.datamethods.org/t/kaplan-meier-se-for-absolute-difference-in-time-point-survival/5035/14

    diff_of_mean = fitterA.predict(point_in_time) - fitterB.predict(point_in_time)
    std_diff_of_mean = np.sqrt(sigma_sqA + sigma_sqB)
    z = st.norm.ppf(0.975)  # = 1.959963984540054, when \alpha = 0.05, two sided 1.0 - 0.05/2
    diff_of_mean_lower = diff_of_mean - z * std_diff_of_mean
    diff_of_mean_upper = diff_of_mean + z * std_diff_of_mean

    results = StatisticalResult(
        p_value,
        X,
        null_distribution="chi squared",
        degrees_of_freedom=1,
        point_in_time=point_in_time,
        test_name="survival_difference_at_fixed_point_in_time_test",
        fitterA=fitterA,
        fitterB=fitterB,
        **result_kwargs
    )

    results.diff_of_mean = diff_of_mean
    results.std_diff_of_mean = std_diff_of_mean
    results.diff_of_mean_lower = diff_of_mean_lower
    results.diff_of_mean_upper = diff_of_mean_upper

    # 1. add difference
    # 2. add CI of differences
    return results