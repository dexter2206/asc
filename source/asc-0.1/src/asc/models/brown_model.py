import numpy
from abstract_model import Model
from asc.core.time_series import TimeSeries


class BrownModel(Model):
    r"""
    Class representing Brown's exponential smoothing model.

    NOTES:

        Brown's model is described by moving average `\hat{m_t}` \ 	for
        `t=1,\dots, n`, which we can count with recursion:

        .. MATH::
            \hat{m_t} = a X_t + (1-a)\hat{m_{t-1}}
            \hat{m_1} = X_1
        for any `a \in [0,1].`
        Thus for `t \ge 2`
        .. MATH:
        \hat{m}_t = \sum\limits_{j=0}^{t-2} a (1 - a)^j x_{t-j} + (1

- a)^{t-1} X_1

        Paramet `a` we choose for trial and error method.

    """

    obligatory_parameters = ("alpha", )

    def __init__(self, data, alpha=0.3):
        r"""
        Initialize new instance of BronwModel class with given

parameters and
        data.

        :param data: data to constructed smoothened model.

        :type data: TimeSeries.

        :param alpha: the smoothing parameter of the model.

        :type alpha: float.
        """
        self._Model_forecast_offset = 1
        super(BrownModel, self).__init__({"alpha": alpha}, data)

    @property
    def alpha(self):
        r"""
        Get smoothing parameter of this model.

        :return: smoothing parameter of the model.

        :rtype: float.
        """
        return self.__alpha

    @alpha.setter
    def alpha(self, value):
        r"""
        Set new value of this model's smoothing parameter.

        :param value: new value of smoothing parameter. ``value``
            should lie in the interval [0,1].

        :type value: float.

        :raise: ValueError if value of alpha is greater than 1 or
            lesser than 0.
        """
        if 0 <= value <= 1:
            self.__alpha = value
        else:
            raise ValueError("alpha must be a number from the interval [0,1].")

    @property
    def estimated_series(self):
        r"""
        Get series estimated from this model using data from which it

was
        constructed.

        :return: sequence of estimated values, i.e. smoothened time

series
            given as the data parameter.

        :rtype: TimeSeries.
        """
        return self.__estimated_series

    @property
    def forecast_offset(self):
        r"""
        Get forecast offset of this model.

        :return: forecast offset of this model. Forecast offset is

the time
            after which model starts to estimate consequtive values

in
            initial data. For Brown's model this is always 1.

        :rtype: integer.

        NOTES:

            This method is included primarily to maintain

compatibility with
            abstract model framework.
        """
        return 1

    def get_parameter(self, param):
        r"""
        Get value of given parameter in this model.

        :param param: name of the parameter. The only valid value for
            BrownModel is "alpha".

        :type param: string.

        :return: value of parameter ``param``.

        :rtype: float.

        :raise: ValueError if ``param`` is anything different than

"apha".

        NOTES:

            This method is included primarily to maintain

compatibility with
            abstract model framework.
        """
        if param == "alpha":
            return self.alpha
        raise ValueError("Unknown parameter %s." % (param))

    def set_parameter(self, param, value):
        r"""
        Set value of a given parameter in this model.

        :param param: parameter for which value should be set. The

only
            valid value for Brown's model is "alpha".

        :type param: float.

        :param value: new value for the parameter. For parameter

alpha it
            should be float in range from the interval [0,1].

        :type value: float.

        :raise: ValueError if ``param`` is anything different than

"alpha"
            or if ``value`` doesn't lie in the interval [0,1].

        NOTES:
            This method is included primarily to maintain

compatibility with
            abstract model framework.
        """
        if param == "alpha":
            self.alpha = value
        else:
            raise ValueError("Unknown parameter %s." % (param))

    def recalculate_model(self):
        r"""
        Recalculate model. This method is used to calculate smoothed

values
        from model's empirical data.
        """
        sample_size = len(self.data)

        m_t = numpy.zeros(sample_size - 1)
        m_t[0] = self.data[0]

        for k in range(sample_size - 2):
            m_t[k + 1] = self.alpha * self.data[k + 1] + \
                        (1 - self.alpha) * m_t[k]

        self.components = {}
        self.components["smoothened"] = self.__estimated_series = \
            TimeSeries(m_t)
        self.components["residues"] = self.goodness_info.errors
