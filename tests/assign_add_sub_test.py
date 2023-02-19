import utills
import numpy as np
import tensorflow as tf
import pytest

tf.config.run_functions_eagerly(False)

@pytest.mark.parametrize("data_type", [(np.float32), (np.float64), (np.float16), (np.int64), (np.uint64), (np.int32), (np.uint32), (np.int16),
                                        (np.uint16), (np.int8), (np.uint8), (np.complex64), (np.complex128)])
def test_assign_add(data_type):
    MAX_BATCH = 5
    MAX_HIGHT = 5
    MAX_WIDTH = 5
    MAX_CHANNELS = 5

    for batchNum in range(1, MAX_BATCH + 1):
        for hight in range(1, MAX_HIGHT + 1):
            for width in range(1, MAX_WIDTH + 1):
                for channels in range(1, MAX_CHANNELS + 1):
                    if(not np.issubdtype(data_type, np.unsignedinteger)):
                        total_elements = (batchNum*hight*width*channels)
                        start = -int(total_elements / 2)
                        stop = total_elements / 2
                        resource = np.arange(start, stop, dtype=data_type).reshape(batchNum, hight, width, channels)
                        value = np.flip(resource.copy().flatten()).reshape(batchNum, hight, width, channels)
                    else:
                        resource = np.arange(0, (batchNum*hight*width*channels), dtype=data_type).reshape(batchNum, hight, width, channels)
                        value = np.flip(resource.copy().flatten()).reshape(batchNum, hight, width, channels)

                    with tf.device('CPU:0'):
                        res1 = tf.Variable(resource)
                        res1.assign_add(value)
                        
                    with tf.device(utills.DEVICE_NAME + ':0'):
                        res2 = tf.Variable(resource)
                        res2.assign_add(value)
                    
                    if not tf.reduce_all(res1 == res2):
                        print(f' B:{batchNum} H:{hight} W:{width} C:{channels}')
                        print('Input tensor resource:')
                        print(resource)
                        print('Input tensor value:')
                        print(value)
                        print('Expected output:')
                        print(res1)
                        print('Got:')
                        print(res2)
                        assert False

@pytest.mark.parametrize("data_type", [(np.float32), (np.float64), (np.float16), (np.int64), (np.uint64), (np.int32), (np.uint32), (np.int16),
                                        (np.uint16), (np.int8), (np.uint8), (np.complex64), (np.complex128)])
def test_assign_sub(data_type):
    MAX_BATCH = 5
    MAX_HIGHT = 5
    MAX_WIDTH = 5
    MAX_CHANNELS = 5

    for batchNum in range(1, MAX_BATCH + 1):
        for hight in range(1, MAX_HIGHT + 1):
            for width in range(1, MAX_WIDTH + 1):
                for channels in range(1, MAX_CHANNELS + 1):
                    if(not np.issubdtype(data_type, np.unsignedinteger)):
                        total_elements = (batchNum*hight*width*channels)
                        start = -int(total_elements / 2)
                        stop = total_elements / 2
                        resource = np.arange(start, stop, dtype=data_type).reshape(batchNum, hight, width, channels)
                        value = np.flip(resource.copy().flatten()).reshape(batchNum, hight, width, channels)
                    else:
                        resource = np.arange(0, (batchNum*hight*width*channels), dtype=data_type).reshape(batchNum, hight, width, channels)
                        value = np.flip(resource.copy().flatten()).reshape(batchNum, hight, width, channels)

                    with tf.device('CPU:0'):
                        res1 = tf.Variable(resource)
                        res1.assign_sub(value)
                        
                    with tf.device(utills.DEVICE_NAME + ':0'):
                        res2 = tf.Variable(resource)
                        res2.assign_sub(value)
                    
                    if not tf.reduce_all(res1 == res2):
                        print(f' B:{batchNum} H:{hight} W:{width} C:{channels}')
                        print('Input tensor resource:')
                        print(resource)
                        print('Input tensor value:')
                        print(value)
                        print('Expected output:')
                        print(res1)
                        print('Got:')
                        print(res2)
                        assert False