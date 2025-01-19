from nesy_veri.examples.mnist_addition.comparisons.marabou_nesy.custom_marabou.custom_marabou_network import CustomMarabouNetworkONNX

def custom_read_onnx(filename, inputNames=None, outputNames=None):
    """Constructs a MarabouNetworkONNX object from an ONNX file

    Args:
        filename (str): Path to the ONNX file
        inputNames (list of str, optional): List of node names corresponding to inputs
        outputNames (list of str, optional): List of node names corresponding to outputs

    Returns:
        :class:`~maraboupy.MarabouNetworkONNX.MarabouNetworkONNX`
    """
    return CustomMarabouNetworkONNX(filename, inputNames, outputNames)