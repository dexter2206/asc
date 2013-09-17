import numpy
from fir import firFilter


class MA(firFilter):
    r"""
    Class representing movin average smoothing filter.
    """

    def __init__(self, step=3):
        r"""
        Initialize new instance of this class with given step.

        :param step: step of moving average. Should be odd and greater than 1.

        :type step: integer.
        """

        self.step = step

    @property
    def step(self):
        r"""
        Get step of this moving average filter.

        :return: step.

        :rtype: integer.
        """
        return len(self)

    @step.setter
    def step(self, value):
        r"""
        Set step of this moving average filter.

        :param value: new value of step. Should be odd and greater than 1.

        :type value: integer.
        """
        # same check as in the constructor

        if not MA.is_step_valid(value):
            raise ValueError("Step must be an odd integer greater "
                             "or equal to 3.")
        self._firFilter__ir = numpy.ones((value)) / value

    @staticmethod
    def is_step_valid(step):
        r"""
        Check if given value of step is valid.

        :param step: step value to be checked.

        :type step: integer.

        :return: true if ``step`` is valid step for moving average.

        :rtype: boolean.
        """
        # check if step is step is an odd positive integer

        return (step % 2 == 1 and step >= 3)
