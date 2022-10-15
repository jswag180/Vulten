# Vulten
A Vulkan PluggableDevice for Tensorflow

To make the Vulkan side more manageable Vulten uses [Kompute](https://github.com/KomputeProject/kompute)

# Why?
Well, I started this project because AMD does not have great support for consumer cards (except the 6900 XT) for Tensorflow.
I wanted to be able to use my 6700 XT in Tensorflow and what better api then Vulkan! I think this project has far more use the AMD cards
because there is no easy way to use Arm or RISC-V gpus in Tensorflow.

# Compatibility
It should **hopefully** work with any Vulkan 1.2 complient device on Arm or x86 (idk about RISC-V).
I have not tested this on Windows or MacOs.

# Ops / kernels
- Conv2d
    - no dilations
    - no explicit padding
- Relu
- SoftMax
- BiasAdd
- MatMul
- AssignAddVariable
- ReadVariableOp
- Identity
- Pow
- ResourceApplyAdam
- Cast
    - uint
    - int
    - float
    - uint64
    - int64
- Mul
- Add
- Sub
- Div
- DivNoNan

# Example
```
import tensorflow as tf
import numpy as np

tf.config.run_functions_eagerly(False)

batch    = 1
height    = 512
width    = 512
channels = 3
tensor = tf.random.normal((batch*height*width*channels,), dtype=tf.float32)
tensor = tf.reshape(tensor, (batch, height, width, channels))

with tf.device('VULK:0'):
    tensorRelu = tf.nn.relu(tensor)
```

# Build dependencies
- Python 3.10 with venv
- Vulkan SDK
# Build instructions
1. Run `./scripts/create_venv.sh` to create venv with Tensorflow and other dependencies needed
2. `cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug .` or `cmake -Bbuild -DCMAKE_BUILD_TYPE=Release .`
3. `cmake --build build/`
4. `./scripts/package_wheel.sh` builds .whl and installs it in the venv

# FAQ
- Is it fast?
    - Nope not enough ops are implmented so it spends ~90% of the time transfering to and from host.
