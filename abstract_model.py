from estimation_goodness_info import EstimationGoodnessInfo
from indexer_property import IndexerProperty


class Model(object):
    r"""
    Class representing abstract statistical model of time series. Its
    intended use is to be baseclass for specific models like exponential
    smoothing models, or ARMA models.
    """
    def __init__(self, parameters, data):
        r"""
        Initialize new instance of model with given data and parameters.

        :param parameters: dict-like object that should contain all parameters
            needed to initialize model.

        :type parameters: dictionary-like.

        :param data: data to initialize this Model instance. Its meaning may
            vary between specific models. For eaxample in exponential models
            this would be the data that will be smoothened, for ARMA models
            the ``data`` would be checked for being of ARMA type with
            given parameters.

        :type data: TimeSeries.

        :raise: ValueError if not all obligatory parameters are present in
            ``parameters`` dictionary.
        """
        # we first check if all necessary parameters are present

        for param in self.obligatory_parameters:

            if param not in parameters:
                raise ValueError("Obligatory parameter: %s missing." % (param))

            self.__parameters = IndexerProperty(
                        lambda param: self.get_parameter(param),
                        lambda param, value: self.set_parameter(param, value),
                        None)

            for k, v in parameters.iteritems():
                self.parameters[k] = v

            self.__data = data
            self.recalculate_model()

    @property
    def parameters(self):
        r"""
        Get parameters of this model.

        :return: Indexer property providing access to this model's parameters.

        :rtype: ``IndexerProperty``.
        """
        return self.__parameters

    @property
    def data(self):
        r"""
        Get data used to create this model.

        :return: Data used to create this model.

        :rtype: TimeSeries.
        """
        return self.__data

    @property
    def goodness_info(self):
        r"""
        Get information about goodness of estimation by this model.

        :return: object providing information about statistical goodness of
            fitting in this model.

        :rtype: EstimationGoodnessInfo.
        """
        return EstimationGoodnessInfo(
            self.data[self.forecast_offset:],
            self.estimated_series)
