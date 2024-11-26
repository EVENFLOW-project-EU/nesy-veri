## Using Marabou as a backend
The main concept here is that we save our network + circuit graphs in an .onnx format and then read these into Marabou.

If Marabou succeeds in creating a `MarabouNetwork` from an ONNX graph, it is then able to perform all verification-related computations on that network. Therefore, the bulk of the work in this directory is towards allowing Marabou to correctly read and parse our ONNX graphs.

### Subclassing for using custom functions withing ONNXParser
- Normally, one would call `Marabou.read_onnx` to create a `MarabouNetwork` from an ONNX file. This could then be used by Marabou for all the desired verification-related computations. 
- The `read_onnx` function initializes the `MarabouNetworkONNX` class, which in turn initializes the `ONNXParser` class and calls most of its member functions. 
- However, some of these do not do what we want, namely the functions `makeMarabouEquations()`, `makeNewVariables()`, `splitEquations()`, and `mulEquations()`, and so we have written our own implementations for them. Additionally, we require a `gather()` function for the ONNX Gather operator which shows up when indexing a tensor. For this function, which does not exist in the main Marabou repository, we copied the code from a pull request. 
- In order to use our versions of the modified functions of `ONNXParser`, we had to use subclassing for the entirety of the call stack
until these functions are called. Therefore, in the `custom_marabou` directory we have implemented:
   1. A `custom_read_onnx` function, which is identical to the original `read_onnx` but instead calls
   2. A `CustomMarabouNetworkONNX` class, which is *almost* identical to the original `MarabouNetworkONNX` but instead calls
   3. A `CustomONNXParser` class. As aforementioned, here we have written our own versions of some functions. Besides these, we also had to overwrite the functions that call our custom functions. These are `parse()`, `__init__()`, `parseGraph()`, `processGraph()`, and `makeGraphEquations()`.


### Other minor comments:
- **Exporting ONNX graphs:**
   I'm using the old way for exporting (`torch.onnx.export`) instead of the newer way (`torch.onnx.dymano_export`). This is
   because `dynamo_export` creates some newer (?) ONNX operators which are not recognised by the Marabou `ONNXParser`.
- **Gather:**
   Indexing a tensor shows up in ONNX as a Gather operator. The Gather operator is not implemented in the main Marabou 
   repository. I did find code from a pull request, and have added it to the `ONNXParser` file. But it hadn't passed all 
   checks so maybe take it with a grain of salt. When used, it breaks in the Sub operators, where the self.shapeMap of 
   the two inputs is not the same. I fixed this by changing the Mul operator code.
   One can avoid Gather operators, and instead have Split/Squeeze operators, which are supported in Marabou. This can 
   be done by replacing indexing with other operations (GPT work).

- **Split:**
   Marabou's `ONNXParser` thinks that when you `torch.split` you go from (2, 1, 28, 28) to (1, 28, 28) which is not true, 
   you go to (1, 1, 28, 28). The previous one fucked everything because the input to the convolutional layers was not 
   4D anymore. So, I changed the code in `splitEquations` to include an additional dimension. However, this is wrong for 
   other split operations so currently I only do it when we're talking about splitting the input? This feels kind of 
   wrong looking back. But it works for now and Marabou breaks in the first example so not doing more work for now.

- **Sub:**
   Sub operation for some reason has inputs swapped. The constant is first and the input is second which is opposite 
   to what Marabou thinks so throws an AssertionError. However, because I corrected `eval_sdd` to return 1 for negative 
   literals of categorical variables the Sub operators are no longer in the ONNX graph. So change not needed for now.

- **Mul:**
   Mul operation for some reason thinks one input should be a constant. I changed it to be as it is in addEquations, 
   where we check whether they are constants or variables and act accordingly. Maybe there should also be another 
   check, because if both inputs are constants the output should also be declared as a constant and not as a variable. 
   Also indexing input1 but not input2 in the for loop which is probably delusional.

- **makeNewVariable:**
   At some point size is 1.0, probably because the self.shapeMap() of the outputs of that split node are empty as in 
   `()`. Maybe that's wrong? This is also a problem if I use the Gather PR operator. Changed this to `int(size)` ??

- Switched sdd input to be 1D because concatenate the split freaked out (1, 20) input so len(input)=1 which is not 
   equal to len(outputs)=20.