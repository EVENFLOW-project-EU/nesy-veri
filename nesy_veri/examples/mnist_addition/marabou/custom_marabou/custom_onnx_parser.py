from maraboupy import MarabouUtils
from maraboupy.parsers.ONNXParser import ONNXParser

from maraboupy.parsers.InputQueryBuilder import InputQueryBuilder
import numpy as np
from typing import List
from onnx.helper import get_attribute_value
from onnx.reference.ops._op_list import Gather, Split_18 # type: ignore


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
        elif node.op_type == 'Split':
            self.splitEquations(node, nodeName, makeEquations)
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

    def splitEquations(self, node, nodeName, makeEquations):
        """Function to generate equations corresponding to split

        Args:
            node (node): ONNX node representing the Split operation
            nodeName (str): Name of target node
            makeEquations (bool): True if we need to create new variables and write Marabou equations

        :meta private:
        """
        # Get attributes
        axis = None
        split = None
        for attr in node.attribute:
            if attr.name == "axis":
                axis = get_attribute_value(attr)
            if attr.name == "split":
                split = get_attribute_value(attr)

        inputName = node.input[0]
        inputVars = Split_18.eval(self.varMap[inputName], split=split, axis=axis)

        assert len(inputVars) == len(node.output)

        # Set a shape of target output
        for i in range(len(node.output)):
            if node.output[i] == nodeName:
                if inputName == self.graph.input[0].name:
                    self.shapeMap[node.output[i]] = (
                        np.expand_dims(inputVars[i], axis=0)
                    ).shape
                else:
                    self.shapeMap[node.output[i]] = inputVars[i].shape
                break

        if not makeEquations:
            return

        # Get variables and add quations
        for i in range(len(node.output)):
            if node.output[i] == nodeName:
                reshapedInputVars = inputVars[i].reshape(-1)
                outputVars = self.makeNewVariables(node.output[i]).reshape(-1)
                for j in range(len(reshapedInputVars)):
                    # Add equation
                    e = MarabouUtils.Equation()
                    e.addAddend(1, reshapedInputVars[j])
                    e.addAddend(-1, outputVars[j])
                    e.setScalar(0)
                    self.query.addEquation(e)
                break

    def mulEquations(self, node, makeEquations):
        nodeName = node.output[0]

        # Get the inputs
        inputName1, inputName2 = node.input
        shape1 = self.shapeMap[inputName1]
        # shape2 = self.shapeMap[inputName2] # comment out since this is never used.

        # Get the broadcasted shape
        outShape = shape1
        self.shapeMap[nodeName] = outShape
        if not makeEquations:
            return

        if inputName1 in self.constantMap:
            input1 = self.constantMap[inputName1]
        else:
            input1 = self.varMap[inputName1]

        if inputName2 in self.constantMap:
            input2 = self.constantMap[inputName2]
        else:
            input2 = self.varMap[inputName2]

        outputVariables = self.makeNewVariables(nodeName)
        input1 = input1.reshape(-1)
        outputVariables = outputVariables.reshape(-1)

        for i in range(len(input1)):
            e = MarabouUtils.Equation()
            e.addAddend(input2, input1[i])
            e.addAddend(-1, outputVariables[i])
            e.setScalar(0.0)
            self.query.addEquation(e)
        return

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

    # # TODO: maybe not needed? Try without first
    # def concatEquations(self, node):
    #     """Function to generate equations corresponding to concat

    #     Args:
    #         node (node): ONNX node representing the Concat operation

    #     :meta private:
    #     """
    #     nodeName = node.output[0]

    #     # Get attributes
    #     axis = None
    #     for attr in node.attribute:
    #         if attr.name == "axis":
    #             axis = get_attribute_value(attr)

    #     allVars = all(input in self.varMap for input in node.input)
    #     allConstants = all(input in self.constantMap for input in node.input)
    #     if allVars:
    #         # Set maps of shape and var
    #         inputVars = list([self.varMap[input] for input in node.input])
    #         outputVars = np.concatenate(inputVars, axis)
    #         self.shapeMap[nodeName] = outputVars.shape
    #         self.varMap[nodeName] = outputVars
    #     elif allConstants:
    #         # Set maps of shape and constants
    #         inputs = list([self.constantMap[input] for input in node.input])
    #         outputs = np.concatenate(inputs, axis)
    #         self.shapeMap[nodeName] = outputs.shape
    #         self.constantMap[nodeName] = outputs
    #     else:
    #         raise RuntimeError(
    #             "Concat inputs need to be all variables or all constants."
    #         )

    # # TODO: maybe not needed? Try without first
    # def subEquations(self, node, makeEquations):
    #     """Function to generate equations corresponding to subtraction

    #     Args:
    #         node (node): ONNX node representing the Sub operation
    #         makeEquations (bool): True if we need to create new variables and add new Relus

    #     :meta private:
    #     """
    #     nodeName = node.output[0]
    #     inputName1, inputName2 = node.input[1], node.input[0]
    #     assert inputName1 in self.shapeMap and inputName2 in self.shapeMap
    #     assert self.shapeMap[inputName1] == self.shapeMap[inputName2]
    #     self.shapeMap[nodeName] = self.shapeMap[inputName1]

    #     if not makeEquations:
    #         return

    #     assert inputName1 in self.varMap and inputName2 in self.constantMap

    #     # Get variables
    #     inputVars = self.varMap[inputName1].reshape(-1)
    #     outputVars = self.makeNewVariables(nodeName).reshape(-1)
    #     constants = self.constantMap[inputName2].reshape(-1)
    #     assert len(inputVars) == len(outputVars) == len(constants)

    #     # Generate equations
    #     for i in range(len(inputVars)):
    #         e = MarabouUtils.Equation()
    #         e.addAddend(1, inputVars[i])
    #         e.addAddend(-1, outputVars[i])
    #         e.setScalar(-constants[i])
    #         self.query.addEquation(e)
