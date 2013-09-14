import numpy
import csv


class TimeSeries(numpy.ndarray):
    r"""
    Class for representing time series. It has fields for representing both
    time series data and time axis, as well as methods for basic manipulation
    on time series.
    """
    def __new__(cls, input_array, time_data=None):
        obj = numpy.asarray(input_array).view(cls)
        if time_data is not None:
            obj.__time_data = numpy.array(time_data)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.__time_data = getattr(obj, '__time_data', None)

    def __array_wrap__(self, out_arr, context=None):
        return numpy.ndarray.__array_wrap__(self, out_arr, context)

    @property
    def time_data(self):
        if self.__time_data is not None:
            return self.__time_data.tolist()
        else:
            return None

    @staticmethod
    def fromCSV(fileName, dataIndex=0, timeIndex=None):

        # check if column index of actual time series data is given, else
        # assume it is 0

        if dataIndex is None:
            dataIndex = 0

        # create csv reader

        reader = csv.reader(open(fileName, "rU"))

        # read all rows

        rows = [row for row in reader]

        # read time series data into list

        data = [float(row[dataIndex]) for row in rows]

        # if column index of time data is given, read it aswell

        if timeIndex is not None:
            time = [float(row[timeIndex]) for row in rows]
        else:
            time = None

        return TimeSeries(data, time)

    def covariance(self, lag=0):
        r"""
        Return estimation of autocovariance of self with given lag.

        INPUT:

            - ``lag`` -- a nonnegative integer.

        OUTPUT:

            estimation of autocovariance of self with given lag.

        EXAMPLES:

        Compute covariance os sample timme series with given lag::

            sage: ts = TimeSeries([7.0, 11.0, 12.5, -8.3, 4.95, \
                    3.13, 1.14, -5.3])
            sage: ts.covariance(lag=3)
            -4.252985

        Check that the series covariance with lag = 0 is equal to its variance::

            sage: ts = TimeSeries([random() for k in xrange(100)])
            sage: ts.var() = ts.covariance(0)
            True


        ALGORITHM:

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
            return v.dot(v) / sample_size

        return v[:-lag].dot(v[lag:]) / (sample_size - lag)

    def auto_corr(self, lag=1):

        return self.covariance(lag) / self.covariance(0)

    def get_acv_series(self):

        return TimeSeries([self.covariance(k) for k in xrange(len(self))])

    def get_acr_series(self):

        return TimeSeries([self.auto_corr(k + 1)
                           for k in xrange(len(self) - 1)])

    def get_differenced(self, order, lag=1):

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
