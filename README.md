# Vulten
A Vulkan PluggableDevice for Tensorflow

# Compatibility
It should **hopefully** work with any Vulkan 1.2 complient device.

# Ops / kernels
- Relu
- ReluGrad
- SoftMax
- BiasAdd
- BiasAddGrad
- MatMul
- AssignAddVariable
- AssignSubVariable
- ReadVariableOp
- Identity
- IdentityN
- Pow
- ResourceApplyAdam
- Cast
- Mul
- Add
- AddV2
- Sub
- Div
- DivNoNan
- AssignVariable
- Sum
- Addn
- Exp
- Sqrt

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

# Env Variables
- VULTEN_DUMP_SPV=(true/false)
- VULTEN_DISABLE_FLOAT64=(true/false)
- VULTEN_DISABLE_FLOAT16=(true/false)
- VULTEN_DISABLE_INT64=(true/false)
- VULTEN_DISABLE_INT16=(true/false)
- VULTEN_DISABLE_INT8=(true/false)

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
