# Installation

Requirements `c++17`.

After cloning the repository and entering into its folder run:

```
pip install .
```

The `setup.py` has an option to compile a version that runs on GPU, check the script to set it up.
A CUDA 12 or above compiler is needed.

# Running an example

Download the example data set from [here](https://github.com/ajjimeno/list-data).

Run the following code, updating the path to the location of the downloaded set:

```
import SimulatorCPU as simulator
n=300
s = simulator.Runner("data/sorted/training")
s.run(["testing_output_write(get9())"]*n)
```

# Citation

If you use this software, cite as follow:

```
@article{yepes2025evolutionary,
  title={Evolutionary thoughts: integration of large language models and evolutionary algorithms},
  author={Jimeno Yepes, Antonio and Barnard, Pieter},
  journal={arXiv preprint arXiv:2505.05756},
  year={2025}
}
```
