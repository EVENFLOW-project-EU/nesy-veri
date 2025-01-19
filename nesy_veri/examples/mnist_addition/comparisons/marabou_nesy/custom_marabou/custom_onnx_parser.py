import numpy as np
from typing import List
from maraboupy import MarabouUtils
from onnx.helper import get_attribute_value
from onnx.reference.ops._op_list import Gather, Split_18 # type: ignore
from maraboupy.parsers.InputQueryBuilder import InputQueryBuilder
from maraboupy.parsers.ONNXParser import ONNXParser, getBroadcastShape


class CustomONNXParser(ONNXParser):

    @staticmethod
    def parse(query:InputQueryBuilder, graph, inputNames:List[str], outputNames:List[str]):
        """
        Parses the provided ONNX graph into constraints which are stored in the query argument.

        Args:
            query: the query to which the constraints are added.
            graph: the graph of the ONNX file to parse.
            inputNames: list of node names corresponding to inputs
            outputNames: list of node names corresponding to outputs

        Returns:
            :class:`~maraboupy.Marabou.marabouNetworkONNX.marabouNetworkONNX`
        """
        parser = CustomONNXParser(query, graph, inputNames, outputNames)
        parser.parseGraph()


    def __init__(self, query:InputQueryBuilder, graph, inputNames, outputNames):
        """
        Should not be called directly. Use `ONNXParser.parse` instead.

        :meta private:
        """
        # super().__init__()
        self.query = query
        self.graph = graph
        self.inputNames = inputNames
        self.outputNames = outputNames

        self.madeGraphEquations = []
        self.varMap = dict()
        self.constantMap = dict()
        self.shapeMap = dict()


    def parseGraph(self):
        """Read an ONNX file and create a MarabouNetworkONNX object

        :meta private:
        """

        # Process the shapes and values of the graph while making Marabou equations and constraints
        self.foundnInputFlags = 0
        self.processGraph()

        # If the given inputNames/outputNames specify only a portion of the network, then we will have
        # shape information saved not relevant to the portion of the network. Remove extra shapes.
        self.cleanShapes()

        for outputName in self.outputNames:
            if outputName in self.constantMap:
                raise RuntimeError("Output variable %s is a constant, not the output of equations!" % outputName)
        self.query.outputVars.extend([self.varMap[outputName] for outputName in self.outputNames])

    def processGraph(self):
        """Processes the ONNX graph to produce Marabou equations

        :meta private:
        """
        # Add shapes for the graph's inputs
        for node in self.graph.input:
            self.shapeMap[node.name] = list([dim.dim_value if dim.dim_value > 0 else 1 for dim in node.type.tensor_type.shape.dim])

            # If we find one of the specified inputs, create new variables
            if node.name in self.inputNames:
                self.madeGraphEquations += [node.name]
                self.foundnInputFlags += 1
                self.makeNewVariables(node.name)
                self.query.inputVars += [np.array(self.varMap[node.name])]

        # Add shapes for constants
        for node in self.graph.initializer:
            self.shapeMap[node.name] = list(node.dims)
            self.madeGraphEquations += [node.name]

        # Recursively create remaining shapes and equations as needed
        for outputName in self.outputNames:
            self.makeGraphEquations(outputName, True)

    def makeGraphEquations(self, nodeName, makeEquations):
        """Recursively populates self.shapeMap, self.varMap, and self.constantMap while adding equations and constraints

        Args:
            nodeName (str): Name of node for making the shape
            makeEquations (bool): Create Marabou equations for this node if True

        :meta private:
        """
        if nodeName in self.madeGraphEquations:
            return

        if nodeName in self.inputNames:
            self.foundnInputFlags += 1
            # If an inputName is an intermediate layer of the network, we don't need to create Marabou
            # equations for its inputs. However, we still need to call makeMarabouEquations in order to
            # compute shapes. We just need to set the makeEquations flag to false
            makeEquations = False
        self.madeGraphEquations += [nodeName]

        # Recursively call makeGraphEquations, then call makeMarabouEquations
        # This ensures that shapes and values of a node's inputs have been computed first
        for inNodeName in self.getInputNodes(nodeName):
            self.makeGraphEquations(inNodeName, makeEquations)

        # By this point, all input variables need to have been found
        if self.foundnInputFlags != len(self.inputNames):
            err_msg = "These input variables could not be found: %s"%(", ".join([inVar for inVar in self.inputNames if inVar not in self.varMap]))
            raise RuntimeError(err_msg)

        # Compute node's shape and create Marabou equations as needed
        self.makeMarabouEquations(nodeName, makeEquations)

        # Create new variables when we find one of the inputs
        if nodeName in self.inputNames:
            self.makeNewVariables(nodeName)
            self.query.inputVars += [np.array(self.varMap[nodeName])]

    def makeMarabouEquations(self, nodeName, makeEquations):
        """Compute the shape and values of a node assuming the input shapes and values have been computed already.

        Args:
            nodeName (str): Name of node for which we want to compute the output shape
            makeEquations (bool): Create Marabou equations for this node if True

        :meta private:
        """
        node = self.getNode(nodeName)

        if node.op_type == 'Mul':
            self.mulEquations(node, makeEquations)
        elif node.op_type == 'Sub':
            self.subEquations(node, makeEquations)
        elif node.op_type == 'Gather':
            self.gather(node)
        else:
            super().makeMarabouEquations(nodeName, makeEquations)

    def gather(self, node):
        """Function representing Gather
        Args:
            node (node): ONNX node representing gather operation
        :meta private:
        """
        nodeName = node.output[0]
        inputName = node.input[0]
        if node.input[1] not in self.constantMap:
            raise RuntimeError("Indices of Gather is not a constant.")
        indices = self.constantMap[node.input[1]]

        axis = None
        for attr in node.attribute:
            if attr.name == "axis":
                axis = get_attribute_value(attr)

        if inputName in self.varMap:
            output_data = Gather.eval(self.varMap[inputName], indices, axis=axis)
            self.shapeMap[nodeName] = output_data.shape
            self.varMap[nodeName] = output_data
        else:
            output_data = Gather.eval(self.constantMap[inputName], indices, axis=axis)
            self.shapeMap[nodeName] = output_data.shape
            self.constantMap[nodeName] = output_data

    def mulEquations(self, node, makeEquations):
        nodeName = node.output[0]

        # Get the inputs
        inputName1, inputName2 = node.input
        shape1 = self.shapeMap[inputName1]
        shape2 = self.shapeMap[inputName2]

        # Get the broadcasted shape
        outShape = getBroadcastShape(shape1, shape2)
        self.shapeMap[nodeName] = outShape
        if not makeEquations:
            return

        # Decide which inputs are variables and which are constants
        firstInputConstant = False; secondInputConstant = False
        if inputName1 in self.constantMap:
            firstInputConstant = True
            input1 = self.constantMap[inputName1]
        else:
            input1 = self.varMap[inputName1]

        if inputName2 in self.constantMap:
            secondInputConstant = True
            input2 = self.constantMap[inputName2]
        else:
            input2 = self.varMap[inputName2]

        # Broadcast inputs to ensure the shapes match
        input1 = np.broadcast_to(input1, outShape)
        input2 = np.broadcast_to(input2, outShape)

        # The shape after broadcasting must match
        assert input1.shape == input2.shape

        # If both inputs to add are constant, then the output is constant too
        # No new variables are needed, we just need to store the output in constantMap
        if firstInputConstant and secondInputConstant:
            self.constantMap[nodeName] = input1 * input2
            return

        # If both inputs are variables, then we need a new variable to represent
        # the sum of the two variables
        elif not firstInputConstant and not secondInputConstant:
            outputVariables = self.makeNewVariables(nodeName)
            input1 = input1.reshape(-1)
            input2 = input2.reshape(-1)
            outputVariables = outputVariables.reshape(-1)
            for i in range(len(input1)):
                self.query.addBilinear(input1[i], input2[i], outputVariables[i])
            return

        # Otherwise, we are multiplying constants with variables.
        if firstInputConstant:
            constInput = input1
            varInput = input2
        else:
            constInput = input2
            varInput = input1
        constInput = constInput.reshape(-1)
        varInput = varInput.reshape(-1)

        outputVariables = self.makeNewVariables(nodeName).reshape(-1)
        for i in range(len(outputVariables)):
            e = MarabouUtils.Equation()
            e.addAddend(constInput[i], varInput[i])
            e.addAddend(-1, outputVariables[i])
            e.setScalar(0.0)
            self.query.addEquation(e)

    def makeNewVariables(self, nodeName):
        """Assuming the node's shape is known, return a set of new variables in the same shape

        Args:
            nodeName (str): Name of node

        Returns:
            (numpy array): Array of variable numbers

        :meta private:
        """
        assert nodeName not in self.varMap
        shape = self.shapeMap[nodeName]
        size = np.prod(shape)
        v = np.array([self.query.getNewVariable() for _ in range(int(size))]).reshape(shape)
        self.varMap[nodeName] = v
        assert all([np.equal(np.mod(i, 1), 0) for i in v.reshape(-1)]) # check if integers
        return v

    def subEquations(self, node, makeEquations):
        """Function to generate equations corresponding to subtraction

        Args:
            node (node): ONNX node representing the Sub operation
            makeEquations (bool): True if we need to create new variables and add new Relus

        :meta private:
        """
        nodeName = node.output[0]

        # Get the inputs
        inputName1, inputName2 = node.input
        shape1 = self.shapeMap[inputName1]
        shape2 = self.shapeMap[inputName2]

        # Get the broadcasted shape
        outShape = getBroadcastShape(shape1, shape2)
        self.shapeMap[nodeName] = outShape
        if not makeEquations:
            return

        # Decide which inputs are variables and which are constants
        firstInputConstant = False; secondInputConstant = False
        if inputName1 in self.constantMap:
            firstInputConstant = True
            input1 = self.constantMap[inputName1]
        else:
            input1 = self.varMap[inputName1]

        if inputName2 in self.constantMap:
            secondInputConstant = True
            input2 = self.constantMap[inputName2]
        else:
            input2 = self.varMap[inputName2]

        # Broadcast inputs to ensure the shapes match
        input1 = np.broadcast_to(input1, outShape)
        input2 = np.broadcast_to(input2, outShape)

        # The shape after broadcasting must match
        assert input1.shape == input2.shape

        # If both inputs to add are constant, then the output is constant too
        # No new variables are needed, we just need to store the output in constantMap
        if firstInputConstant and secondInputConstant:
            self.constantMap[nodeName] = input1 - input2
            return

        # If both inputs are variables, then we need a new variable to represent
        # the sum of the two variables
        elif not firstInputConstant and not secondInputConstant:
            outputVariables = self.makeNewVariables(nodeName)
            input1 = input1.reshape(-1)
            input2 = input2.reshape(-1)
            outputVariables = outputVariables.reshape(-1)
            for i in range(len(input1)):
                e = MarabouUtils.Equation()
                e.addAddend(1, input1[i])
                e.addAddend(-1, input2[i])
                e.addAddend(-1, outputVariables[i])
                e.setScalar(0.0)
                self.query.addEquation(e)
            return

        # Otherwise, we are subtracting constants and variables.
        if firstInputConstant:
            constInput = input1
            varInput = input2
        else:
            constInput = input2
            varInput = input1
        constInput = constInput.reshape(-1)
        varInput = varInput.reshape(-1)

        outputVariables = self.makeNewVariables(nodeName).reshape(-1)
        for i in range(len(outputVariables)):
            e = MarabouUtils.Equation()
            e.addAddend(1, varInput[i])
            e.addAddend(1, outputVariables[i])
            e.setScalar(constInput[i])
            self.query.addEquation(e)