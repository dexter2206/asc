import numpy


class FFT(numpy.ndarray):

    def __new__(cls, input_array, time_data=None):
        obj = numpy.fft.fft(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array_wrap__(self, out_arr, context=None):
        return numpy.ndarray.__array_wrap__(self, out_arr, context)

    @property
    def amplitude_spectrum(self):
        return numpy.abs(self)

    @property
    def power_spectrum(self):
        return self.amplitude_spectrum ** 2

    @property
    def phase_spectrum(self):
        return numpy.angle(self)

    @property
    def inverse(self):
        return numpy.fft.ifft(self)

    @staticmethod
    def invert(input_array):
        return numpy.fft.ifft(input_array)
