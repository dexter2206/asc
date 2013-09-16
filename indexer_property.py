class IndexerProperty(object):
    r"""
    Class for representing properties that can be indexed (like dictionary)
    with source of data being user defined functions or methods.
    """
    def __init__(self, get_fun, set_fun, keys_fun):
        r"""
        Initializes new instance of IndexerProperty.

        :param get_fun: function that will be called to get items with given
            key. If None then property is considered write-only.

        :type get_fun: function, method, lambda, or any object callable
            with signature`get_fun(key)`` or None.

        :param set_fun: function that will be called to set items with given
            key and value. If None then property is considered read-only.

        :type set_fun: function, method, lambda or any object callable with
            signature ``set_fun(key,value)`` or None.

        :param keys_fun: function that will be called to get enumeration of
            possible keys or None, if property doesn't have predefined set of
            keys.'

        :type keys_fun: function, method, lambda or any object callable with
            signature ``keys_fun()``.
        """
        self.__get_fun = get_fun
        self.__set_fun = set_fun
        self.__keys_fun = keys_fun

    def __getitem__(self, key):
        r"""
        Get item associated with the given key.

        :param key: key associated with value that should be returned.

        :type key: dependent on get_fun passed to the __init__ method.

        :return: value associated with ``key``.

        :rtype: dependent on get_fun passed to the __init__ method.

        :raise: Exception if property is writeonly.
        """
        if self.__get_fun is None:
            raise Exception("Property is writeonly.")
        else:
            return self.__get_fun(key)

    def __setitem__(self, key, value):
        r"""
        Set value of item associated with given key.

        :param key: key for which value should be set.

        :type key: dependent on set_fun passed to the __init__ method.

        :param value: value to be set.

        :type value: dependent on set_fun passed to the __init__method.

        :raise: Exception if property is read-only.
        """
        if self.__set_fun is None:
            raise Exception("Property is readonly.")
        else:
            self.__set_fun(key, value)

    @property
    def keys(self):
        r"""
        Get enumeration of all possible keys for this property.

        :return: enumeration of possible keys for this property.

        :rtype: dependent on keys_fun passed to __init__ method.

        :raise: Exception if keys_fun hasn't been set.
        """
        if self.__keys_fun is None:
            raise Exception("Enumeration of possible keys is not available")
        else:
            return self.__keys_fun()