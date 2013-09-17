import numpy
from asc.core.time_series import TimeSeries


class firFilter(object):
    r""""
    Class for representing finite impluse response (FIR) linear filter.
    """

    def __init__(self, ir):
        r"""
        Initializes instance of FIR class.

        :param ir: array with filter coefficients.

        :type ir: array-like.
        """
        self.__ir = numpy.array(ir)

    @property
    def ir_coefficients(self):
        r"""
        Get the filter impulse response coefficients.

        :return: filter inmpulse reponse coefficients.

        :rtype: numpy.ndarray.
        """
        return self.__ir

    def filter_signal(self, signal):
        r"""
        Filter given signal with this filter.

        :param signal: signal to be filtered.

        :type signal: array-like.

        :return: filtered signal.

        :rtype: array-like, dependent on type of signal. If signal is subclass
            of numpy.ndarray then returned array is of the same type, otherwise
            it is numpy.ndarray.
	:raise: ValueError if signal is empty
        """

        if issubclass(type(signal), numpy.ndarray):
            _signal = signal
        else:
            _signal = numpy.array(signal)

        sample_size = len(_signal)

        if sample_size == 0:
            raise ValueError("Signal cannot be empty.")

        filter_size = len(self.ir_coefficients)

        if filter_size > sample_size:
            raise ValueError("Filter coefficients array cannot be longer "
                             "than signal")

        conv = numpy.zeros(sample_size - filter_size + 1)

        conv[:] = [_signal[k:filter_size + k].dot(self.ir_coefficients[::-1])
                   for k in xrange(sample_size - filter_size + 1)]

        return conv

    def filter(self, time_series):
        r"""
        Filter given given time series with this filter.

        :param time_series: time series to be filtered.

        :type signal: array-like.

        :return: filtered series.

        :rtype: TimeSeries.

        NOTES:
            Unlike ``filter_signal`` this method takes care also of time series
            time data, meaning that filtered time series has its list of time
            labels correctly trimmed. This frees caller from having to manually
            calculate how many time labels should be skipped.
        """
        conv = self.filter_signal(time_series)

        if time_series.time_data is not None:

            # difference of length of original time series
            # and length of its convolution
            diff = len(time_series) - len(conv)

            # compute new time data by skipping sufficient amount
            # of original time data

            time_data = time_series.time_data[diff / 2:diff / 2 + len(conv)]

        else:
            time_data = None

        return TimeSeries(conv, time_data)
