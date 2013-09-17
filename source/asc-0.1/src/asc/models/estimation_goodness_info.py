import numpy
from asc.utils.indexer_property import IndexerProperty


class EstimationGoodnessInfo(object):
    r"""
    Class providing informations about quality of estimation given empirical
    and estimated data.
    """
    def __init__(self, emp, est):
        r"""
        Initializes new instance of EtimationGoodnesInfo with given data.

        :param emp: empirical data that were estimated.

        :type est: ``numpy.ndarray`` or any subclass of it.

        :param est: estimated values of emp.

        :type est: ``numpy.ndarray`` or any subclass of it.

        :raise: ``ValueError`` if length of estimated and empirical data are
            different.
        """
        if len(est) != len(emp):
            raise ValueError("Cannot create error information with"
                " estimated and actual data of different length.")

        self.__estimation = est
        self.__empirical = emp
        self.__statistics = IndexerProperty(
                                self.get_statistic,
                                None,
                                self.available_statistics)
        self.__errors = est - emp
        self.__se = self.errors.sum()
        self.__sse = (self.errors ** 2).sum()
        self.__mse = self.sse / len(est)
        self.__sae = numpy.abs(self.errors).sum()

    @property
    def estimation(self):
        r"""
        Get estimated values associated with this object.

        :return: sequence estimated values for empirical time series.

        :rtype: ``numpy.ndarray`` or some subclass of it, the same as type of
            ``est`` parameter passed to ``__init__ method``.
        """
        return self.__estimation

    @property
    def empirical(self):
        r"""
        Get empirical data associated with this object.

        :return: sequence with empirical data.

        :rtype: ``numpy.ndarray`` or some subclass of it, the same as type of
            ``emp`` parameter passed to ``__init__ method``.
        """
        return self.__empirical

    @property
    def errors(self):
        r"""
        Get error, i.e. the difference between empirical and estimated data.

        :return: sequence of differences between empirical and estimated data.

        :rtype: ``numpy.ndarray`` or some subclass of it, the same as type of
            the difference ``emp - est``of parameters passed to ``__init__``
            method.
        """
        return self.__errors

    @property
    def se(self):
        r"""
        Get sum of errors of this estimation.

        :return: sum of errors.

        :rtype: float.
        """
        return self.__se

    @property
    def sse(self):
        r"""
        Get sum of squared erros of this estimation.

        :return: sum of squared errros.

        :rtype: float.
        """
        return self.__sse

    @property
    def mse(self):
        r"""
        Get mean square error of this estimation.

        :return: mean square error.

        :rtype: float.
        """
        return self.__mse

    @property
    def sae(self):
        r"""
        Get sum of absolute errors of this estimation.

        :return: sum of absolute errors.

        :rtype: float.
        """
        return self.__sae

    def get_statistic(self, key):
        r"""
        Get statistics of this estimation with given ``key`` (name).

        :param key: name of the statistic.

        :type key: string.

        :return: statistic associated with ``key``.

        :rtype: depending on ``key`` and associated statistic.

        :raise: ``ValueError`` if ``key`` is unknown. Valid keys can be
            obtained through ``available_statistics`` property.

        NOTES:

            This method is intended to be overriden in derived classes.
            Those classes should check for new keys and default to returning
            their superclass ``get_statistic`` when the key is not
            class-specific.
        """
        if key == "errors":
            return self.errors
        elif key == "se":
            return self.se
        elif key == "sse":
            return self.sse
        elif key == "mse":
            return self.mse
        elif key == "sae":
            return self.sae
        else:
            raise ValueError("Unknown statistic: %s" % (key))

    @property
    def available_statistics(self):
        r"""
        Get enumeration of all statistics available from this object.

        :return: tuple of all statistics. Those are valid keys for
            ``get_statistics`` method and ``statistics`` property.

        :rtype: tuple of strings.
        """
        return ("se", "sse", "mse", "sae")

    @property
    def statistics(self):
        r"""
        Indexer property providing access to all statistics stored in this
        object.

        :return: indexer property which indexes all statistics provided by
            this model.

        :rtype: IndexerProperty.
        """
        return self.__statistics

    def _latex_(self):
        r"""
        Get represantation of this object as nicely formatted LaTeX array.

        :return: represantation of this object as LaTeX array.

        :rtype: string

        NOTES:

            This method is intended for use with SAGE system.

        """
        s = """\\begin{array}{|c|c|}
            \\hline
            \\mbox{Statistics} & \\mbox{Value} \\\\
            \\hline

            """

        for stat in self.statistics.keys:
            s += "\\mbox{%s} & %s\\\\ \\hline\n" % (
                            stat,
                            self.statistics[stat])

        s += "\\end{array}"
        return s
