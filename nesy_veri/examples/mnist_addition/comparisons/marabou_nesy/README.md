## Using Marabou as a backend for NeSy verification

### The Big Idea
The main concept here is:

1. Save our network + circuit graphs in an .onnx format.
2. Read the ONNX graph into Marabou.  
3. Construct whatever verification query you want and ask Marabou to solve it.

### The Final Verdict
It doesn't work.

We did a decent amount of work and "succeeded" in performing all three bullet points from above. The rationale was that after managing to read the NeSy system into Marabou as an ONNX graph we would be able to ask the simple verification queries we wanted. We were wrong. This is because Marabou actually does not have support for solving queries in networks which include some operators (such as Softmax), even though it allows you to read ONNX graphs containing them, such as ours. Regardless, in the "marabou_verification.py" file we do bullet points (1) and (3). These are easy. Bullet point (2) was harder. Here is the work we did in case it is useful for anyone. 

### Our Work for (2)
Our main aim was to empower Marabou to create a `MarabouNetwork` from an ONNX graph. Therefore, the bulk of the work in this directory is towards allowing Marabou to correctly read and parse our ONNX graphs.

#### Subclassing for using custom functions withing ONNXParser
- Normally, one would call `Marabou.read_onnx` to create a `MarabouNetwork` from an ONNX file. This could then be used by Marabou for all the desired verification-related computations. 
- The `read_onnx` function initializes the `MarabouNetworkONNX` class, which in turn initializes the `ONNXParser` class and calls most of its member functions. 
- However, some of these do not do what we want, namely the functions `makeMarabouEquations()`, `makeNewVariables()`, `splitEquations()`, and `mulEquations()`, and so we have written our own implementations for them. Additionally, we require a `gather()` function for the ONNX Gather operator which shows up when indexing a tensor. For this function, which does not exist in the main Marabou repository, we copied the code from a pull request. 
- In order to use our versions of the modified functions of `ONNXParser`, we had to use subclassing for the entirety of the call stack
until these functions are called. Therefore, in the `custom_marabou` directory we have implemented:
   1. A `custom_read_onnx` function, which is identical to the original `read_onnx` but instead calls
   2. A `CustomMarabouNetworkONNX` class, which is *almost* identical to the original `MarabouNetworkONNX` but instead calls
   3. A `CustomONNXParser` class. As aforementioned, here we have written our own versions of some functions. Besides these, we also had to overwrite the functions that call our custom functions. These are `parse()`, `__init__()`, `parseGraph()`, `processGraph()`, and `makeGraphEquations()`.


#### Other minor comments:
- **Exporting ONNX graphs:**
   I'm using the old way for exporting (`torch.onnx.export`) instead of the newer way (`torch.onnx.dymano_export`). This is
   because `dynamo_export` creates some newer (?) ONNX operators which are not recognised by the Marabou `ONNXParser`.
- **Gather:**
   Indexing a tensor shows up in ONNX as a Gather operator. The Gather operator is not implemented in the main Marabou 
   repository. I did find code from a pull request, and have added it to the `ONNXParser` file. But it hadn't passed all 
   checks so maybe take it with a grain of salt. When used, it breaks in the Sub operators, where the self.shapeMap of 
   the two inputs is not the same. I fixed this by changing the Mul operator code.
   One can avoid Gather operators, and instead have Split/Squeeze operators, which are supported in Marabou. This can 
   be done by replacing indexing with other operations (ChatGPT can do this).

- **Sub:**
   Sub operation for some reason asserts that input1 is a variable and input2 is a constant. For example, x-5 is allowed, but 1-y and x-y are not.
   We upgraded this to check what each input is and create the corresponding equation. In our graphs, we need 1-x for probability complements.

- **Mul:**
   Mul operation again for some reason asserts that input1 is a variable and input2 is a constant. For example, y times 5 is allowed, but 5 times y and x times y are not. I
   Again, we upgraded this to check what each input is and create the corresponding equations.

- **makeNewVariable:**
   At some point size is 1.0, so I just changed it to `int(size)`. Maybe this is wrong.
