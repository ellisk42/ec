# ARC DSL
The first step in solving the ARC challenge is writing the DSL for it.
## DSL Description
Each example is a pair of input and output grids of type `tgridin` and `tgridout` respectively. Then, the program synthesis task is to synthesize a program `tgridin -> tgridout` that outputs the expected output grid for a given input grid.
### Basic Types
#### tgridin
The type of the input grid. Grids are represented as sets of points of the form `((y,x),c)` corresponding to the point position and color.
#### tgridout
The type of the output grid. Grids are represented as sets of points of the form `((y,x),c)` corresponding to the point position and color.
#### tcolor
The color type. The 10 different colors that appear in ARC grids with the addition of a special color `invisible`.
#### tcolorpair
A `tcolor` pair.
#### tint
The integer type. The DSL integers range from `0` to `9`.
#### tboolean
The boolean type. Either `True` or `False`.
#### tlogical
The type for logical operators. The logical operators in this DSL are `land`, `lor` and `lxor`.
#### tdirection
The direction type. One of the 8 cardinal directions: `N`,`S`,`W`,`E`,`NW`,`NE`,`SW`,`SE`.
#### tblock
The most fundamental type in this DSL. It is a list of points of the form `((y,x),c)`. Every `tblock` also stores the original `tgridin` that it is a part of.
#### ttile
A special block type that consists of a single point. Each `ttile` also has a parent block depending on the method that created it.
#### tsplitblocks
A pair of two special blocks that are rectangle and have the same dimensions. 
#### ttbs
Short for template block scene. Represents a tuple of a list of blocks and a template block (that is somehow different from each of the blocks in the list of blocks).

There are also `tblocks`, `ttiles` which are lists of `tblock` and `ttile`.

## Results
To view these results clone the repository and open ec/arc-data/apps/testing_interface.html with your browser. Once you can see the main ARC test interface click on the "Choose File" button to "Load EC Console output". Then select the ecResults.json file found in ec/experimentOutputs/arc/.../ecResults.json.

#### Full training corpus Enumeration with Bigram Model (slurm-16731642.out, slurm-16817187.out)
ec/experimentOutputs/arc/2020-05-10T14:49:21.186479

Iterations: 6,
Top-down enumeration timeout: 8 hours,
Bottom-up enumeration timeout: 20 minutes

After 8 Hour unconditional enumeration (w/o bigram model): 31/400 <br />
At the end of iteration 1: (top-dow): 71/400 <br />
At the end of iteration 2: 73/400 <br />
At the end of iteration 3: 74/400 <br />
At the end of iteration 4: 74/400 <br />
At the end of iteration 5: 74/400 <br />


#### Batch Enumeration with Bigram Model (slurm-16769822.out)
experimentOutputs/arc/2020-05-01T19:00:26.769291

Iterations: 20,
(At iteration 0, top-down enumerated for all tasks for 8 hours to jump start)
Batch size: 30,
Top-down enumeration timeout: 1 hour,
Bottom-up enumeration timeout: 1 hour
## Next Steps
### DSL
The DSL code can be found in `solvers/arc.ml`.

##### Add new types
1. trectangle
2. tlaser
##### Generalize existing functions
There is a significant trade off between keeping the search space small and writing general enough primitives. The DSL was constructed with both considerations in mind but with an emphasis on primitives that would result in short programs (mainly as a proof of concept for program induction for ARC). However, to learn a richer DSL, and solver more complex novel tasks, it may be useful to make the existing primitives more general and compositional. 

### Enumeration
##### Search with special color constraints
We can take advantage of the fact that if a color appears in all input examples but no output examples or vice versa, the program we are searching for must include the primitive for that color (**Task 104: 4612dd53.json**). We can enforce this constraing while searching to make the space smaller.
##### Intra task compression
There are tasks that repeat subprograms and so instead of searching for the same subprogram twice it may be useful to upweight already computed subprograms.
### Recognition Neural Network
##### Current work
I started developing the recognition neural network but didn't get to try it in the full system. The code for this can be found in `ARCNN` in `dreamcoder/domains/arc/main.py`. 

The baseline `featuresOfTask` is just a concatenation of a length 900 vector of the flattened input grid (where grids smaller than 30x30 are padded with zeros) with a length 900 vector of the flattened input grid. Since different tasks have different number of examples we only consider one example for each task. Colors are not one hot encoded, only one example is used, the input and output grids do not have a qualitatively different representation and grid dimensions are "lost". This was the simplest task representation meant to establish a baseline and almost certainly not the best one.

The `customFeaturesOfTask` method returns a length 25 vector representation encoding the following features:
* Input special colors: Index i is 1 if the color is a special input color, 0 otherwise. (*Ideally a color is a special input color only if it appears in all input examples but no output example but current implementation only looks at the colors of a single example*). **Length:10, Value:{0,1}**.
* Output special colors: Index i is 1 if the color is a special output color, 0 otherwise. **Length:10, Value:{0,1}**. 
* Dimension change: 1 if the output grid dimensions are differne than the input grid dimensions, 0 otherwise. **Length:1, Value:{0,1}**. 
* Pixel error: Number between 0 and 1 representing the pixel accuracy between the input and output grids. Assumes that these are the same size. If not defaults to 0. **Length 1, Value:[0,1]**. 
* Use split blocks: 1 if either the width or the height of the input grid is twice that of the output grid, otherwise 0. **Length:1, Value:{0,1}**
* Fraction of black input grid tiles: Number between 0 and 1, representing the fraction of input grid tiles that are black. **Length:1, Value:[0,1]**.
* Fraction of black output grid tiles: Number between 0 and 1, representing the fraction of input grid tiles that are black. **Length:1, Value:[0,1]**.

##### Next Steps
There are a lot of possible ways to extend the current architecture which is merely a baseline. Experimenting with RNNs, CNNs, better feature design are all promising future next steps.

### Helmholtz Enumeration
Required for the recognition model to work well this should definitely be a priority moving forward. Being able to train the neural network on dream tasks and just solved tasks is really important for bootstrapping and library learning to work well. This requires both a way to sample programs as well as a way to sample input grids. Currently the only progress to this end is code to permute grid non-special color (so that the program found is correct for the color-permuted grid too. This code can be found in `dreamcoder/domains/arc/taskGeneration.py`.

### Compression
From the experiments run, it seems that compression often learns highly specific long programs due to the presence of very similar tasks. Thinking about and modifying compression to encourage learning shorter and more general primitives is a promising direction forward.


