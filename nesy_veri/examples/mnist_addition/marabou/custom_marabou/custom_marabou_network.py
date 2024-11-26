from maraboupy.MarabouNetworkONNX import MarabouNetworkONNX
from maraboupy.MarabouNetwork import MarabouNetwork
import onnx
from nesy_veri.examples.mnist_addition.marabou.custom_marabou.custom_onnx_parser import CustomONNXParser 

class CustomMarabouNetworkONNX(MarabouNetworkONNX):
    """Constructs a MarabouNetworkONNX object from an ONNX file

    Args:
        filename (str): Path to the ONNX file
        inputNames: (list of str, optional): List of node names corresponding to inputs
        outputNames: (list of str, optional): List of node names corresponding to outputs

    Returns:
        :class:`~maraboupy.Marabou.marabouNetworkONNX.marabouNetworkONNX`
    """
    def __init__(self, filename, inputNames=None, outputNames=None):
        # Directly call the parent class (MarabouNetwork) __init__
        super(MarabouNetwork, self).__init__()

        # Now call your custom readONNX method
        self.readONNX(filename, inputNames, outputNames)

    def readONNX(self, filename, inputNames=None, outputNames=None, preserveExistingConstraints=False):
        if not preserveExistingConstraints:
            self.clear()

        self.filename = filename
        self.graph = onnx.load(filename).graph

        # Setup input node names
        if inputNames is not None:
            # Check that input are in the graph
            for name in inputNames:
                if not len([nde for nde in self.graph.node if name in nde.input]):
                    raise RuntimeError("Input %s not found in graph!" % name)
            self.inputNames = inputNames
        else:
            # Get default inputs if no names are provided
            assert len(self.graph.input) >= 1
            initNames = [node.name for node in self.graph.initializer]
            self.inputNames = [inp.name for inp in self.graph.input if inp.name not in initNames]

        # Setup output node names
        if outputNames is not None:
            if isinstance(outputNames, str):
                outputNames = [outputNames]

            # Check that outputs are in the graph
            for name in outputNames:
                if not len([nde for nde in self.graph.node if name in nde.output]):
                    raise RuntimeError("Output %s not found in graph!" % name)
            self.outputNames = outputNames
        else:
            # Get all outputs if no names are provided
            assert len(self.graph.output) >= 1
            initNames = [node.name for node in self.graph.initializer]
            self.outputNames = [out.name for out in self.graph.output if out.name not in initNames]

        CustomONNXParser.parse(self, self.graph, self.inputNames, self.outputNames)