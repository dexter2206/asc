import numpy
import asc


class TimeSeries(numpy.ndarray):

    r"""
    Class for representing time series. It has fields for representing both
    time series data and time axis, as well as methods for basic manipulation
    on time series. Inherits from numpy.ndarray and can be created from
    virtually anything that  ndarray can be initialized from.

    EXAMPLES:

    Create TimeSeries instance from list of integers:

        sage: ts = TimeSeries([1,2,3,4])
        sage: ts
        [1,2,3,4]

    Create TimeSeries instance from generator:

        sage: ts = TimeSeries(xrange(10))
        sage: ts
        [0,1,2,3,4,5,6,7,8,9]
    """

    def __new__(cls, input_array, time_data=None):
        r""""
        Returns new instance of TimeSeries.

        :param cls: class used to initialize TimeSeries (usually TimeSeries
            itself).

        :type cls: type.

        :param input_array: array containing TimeSeries data. Can be any object
            from which numpy can instantiate ndarray.

        :type input_array: array-like.

        :param time_data: array of labels of time axis in this time series or
            None. Default is None.

        :type time_data: array-like or None.

        :return: new instance of TimeSeries.
        :rtype: TimeSeries.

        NOTES:

            This method is part of numpy.ndarray subclassing.

        .. SEEALSO:

            :url: http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        """
        obj = numpy.asarray(input_array).view(cls)
        if time_data is not None:
            obj.__time_data = numpy.array(time_data)
        return obj

    def __array_finalize__(self, obj):
        r""""
        Finalize creation of numpy.ndarray subclass.

        :param obj: object to finish initialization with.

        :type obj: subclass of ndarray or NoneType.

        NOTES:

            This method is part of numpy.ndarray subclassing.

        .. SEEALSO:

            :url: http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        """
        if obj is None:
            return
        self.__time_data = getattr(obj, '__time_data', None)

    def __array_wrap__(self, out_arr, context=None):
        r"""
        Wraps the result of ufunc as an instance of TimeSeries.

        :param out_arr: output of the ufunc.

        :type out_arr: subclass of numpy.ndarray.

        :param context: context of the ufunc. Refer to numpy documentation
            for additional details.

        :type context: context

        :return: out_arr wrapped as TimeSeries.
        :rtype: TimeSeries.
        """
        return numpy.ndarray.__array_wrap__(self, out_arr, context)

    @property
    def time_data(self):
        r"""
        Return self's list of time axis labels, or None if no information about
        time axis is available.

        :return: list containing ticks on the time axis associated with this
            time series.

        :rtype: list''.
        """
        if self.__time_data is not None:
            return list(self.__time_data)
        else:
            return None

    @staticmethod
    def fromCSV(filePath, dataIndex=0, timeIndex=None):
        r"""
        Load time series from comma separated values file.

        :param fileName: path to the csv file. Can be either absolute or
            relative.

        :type fileName: string.

        :param dataIndex: index of column containing time series data in the
            csv file. Should be nonnegative. Default is `0`.

        :type dataIndex: integer.

        :param timeIndex: index of column containing time labels corresponding
            to data from dataIndex column. Should be nonnegative integer or
            none if there is no such data in csv file. Default is None.

        :type timeIndex: integer or None.

        :return: TimeSeries object containing data extracted from given
            csv file.
        :rtype: TimeSeries.
        """
        # check if column index of actual time series data is given, else
        # assume it is 0

        if dataIndex is None:
            dataIndex = 0

        lines = numpy.genfromtxt(filePath, delimiter=",")

        data = lines[:, dataIndex]

        if timeIndex is not None:
            time = lines[:, timeIndex]
        else:
            time = None

        return TimeSeries(data, time)

    def auto_cov(self, lag=0):
        r"""
        Return estimation of autocovariance of self with given lag.

        :param lag: lag of the covariance, should be nonnegative.

        :type lag: integer.

        :return: estimation of autocovariance of self with given lag.

        :rtype: float.

        EXAMPLES:

        Compute covariance os sample timme series with given lag::

            sage: ts = TimeSeries([7.0, 11.0, 12.5, -8.3, 4.95, \
                    3.13, 1.14, -5.3])
            sage: ts.covariance(lag=3)
            -4.252985

        Check that the series covariance with lag = 0 is equal to its variance:

            sage: ts = TimeSeries([random() for k in xrange(100)])
            sage: ts.var() = ts.covariance(0)
            True

        NOTES:

            This method assumes time series is stationary.

            (Biased) estimator of autocovariance of stationary time
            series sample $\\{X_t\\}_{t=0}^{n-1}$ of length $n$ and sample
            mean $m$ with lag $d$ is defined as

            .. MATH::

                \gamma(d) = \sum_{k=0}^{n-d-1} (X_{k}-m)(X_{k+d}-m).

        """

        sample_size = len(self)

        # covariance is defined only for nonnegative lags -
        # check if lag is OK

        if lag < 0:
            raise ValueError("Covariance lag should be nonnegative")

        mean = self.mean()

        # prepare vector of normalized data
        v = self - mean

        # we first consider case lag = 0, because indexing cannot be written
        # in uniform way: lag[:-0] doesn't mean nothing

        if lag == 0:
            ret = v.dot(v)
        else:
            ret = v[:-lag].dot(v[lag:])

        return ret / sample_size

    def auto_corr(self, lag=1):
        r"""
        Return autocorrelation of this time series with given lag.

        :param lag: lag of autocorrelation, should be positive integer.

        :type lag: integer.

        :return: biased estimation of self autocorrelation.

        :rtype: float.

        NOTES:

            This method assumes time series is stationary.

            (Biased) estimator of autocorrelation of stationary time
            series sample $\\{X_t\\}_{t=0}^{n-1}$ of length $n$ and sample
            mean $m$ with lag $d > 0$ is defined as

            .. MATH::

                \rho(d) = \gamma(d) / \gamma(0)

            where `\gamma` is series autocovariance function.
        """
        return self.auto_cov(lag) / self.auto_cov(0)

    def get_acv_series(self):
        r"""
        Return series of estimates of this series' autocovariance with all
        possible lags.

        :return: time series containing estimations of autocovariance of this
            time series'with lags ranging from 0 to len(self)-1.

        :rtype: TimeSeries.

        .. SEEALSO:

            :meth: .auto_corr
            :meth: .auto_cov
            :meth: .get_acv_series
        """
        return TimeSeries([self.auto_cov(k) for k in xrange(len(self))])

    def get_acr_series(self):
        r"""
        Return series of estimates of this series' autocorrelation with all
        possible lags.

        :return: time series containing estimations of autocorrelation of this
            time series with lags ranging from 1 to len(self)-1.

        :rtype: TimeSeries.

        NOTES:

            this method assumes time series is stationary.

        .. SEEALSO:

            :meth: .auto_corr
            :meth: .auto_cov
            :meth: .get_acv_series
        """
        return TimeSeries([self.auto_corr(k + 1)
                           for k in xrange(len(self) - 1)])

    def get_differenced(self, order=1, lag=1):
        r"""
        Compute difference of this time series with given order and lag.

        :param order: order of difference, this corresponds to the exponent
            of lag operator in differencing formula. Should be positive,
            default is 1.

        :type order: integer.

        :param lag: lag of differencing. See Notes for details. Should be
            positive, default is 1.

        :type lag: integer.

        :return: differenced time series.

        :rtype: TimeSeries.

        NOTES:

            Differencing operator of lag `d` is defined as

        .. MATH:

            \nabla_d(X_t) = X_{t}-X_{t-d}

            while differencing operator of lag d and order n is simply defined
            as \nabla_d^n.

            If original time sries is of length `k` then differenced time
            series is of length `k-nd`.
        """

        # lag and order must be positive

        if order <= 0:
            raise ValueError("Differencing order must be positive.")

        if lag <= 0:
            raise ValueError("Lag must be positive.")

        # every differencing of lag n reduces length of time series by n
        # it follows easily that minimal length of the series to perform
        # differencing with given parameters is order * lag + 1

        if order * lag >= len(self):
            raise ValueError("Time series is not long enough to perform "
                             "differencing with given parameters. (series "
                             "length: %s, minimal length: %s)"
                             % (len(self), order * lag + 1))

        diffs = self.copy()

        for k in xrange(order):

            # following subtraction follows directly from
            # the definition of differencing

            diffs = diffs[lag:] - diffs[:-lag]

        return TimeSeries(diffs)

    def partial_autocorr(self, lag=0):

        lag = numpy.absolute(lag)

        if lag == 0:
            raise ValueError("Lag in partial autocorrelation must be nonzero.")

        cors = numpy.zeros((lag, lag))

        for i in xrange(lag):
            for j in xrange(lag - i):
                cors[i, i + j] = cors[i + j, i] = self.auto_corr(j)

        ro = numpy.array([self.auto_corr(k + 1) for k in xrange(lag)])

        return numpy.linalg.solve(cors, ro)[lag - 1]

    def get_pacf_series(self):

        return TimeSeries(
            [self.partial_autocorr(k + 1) for k in xrange(len(self))])

    def get_histogram(self, bins=10, normed=True, range=None):
        return asc.core.Histogram(self, bins, normed, range)
