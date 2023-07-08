import utills
import numpy as np
import tensorflow as tf
import pytest

tf.config.run_functions_eagerly(False)

def param_types(op):
    return [(np.float32, op), (np.float64, op), (np.float16, op), (np.int64, op), (np.uint64, op), (np.int32, op), (np.uint32, op), (np.int16, op),
            (np.uint16, op), (np.int8, op), (np.uint8, op), (np.complex64, op), (np.complex128, op)]

@pytest.mark.parametrize("data_type, op", param_types(tf.raw_ops.Mul) + param_types(tf.raw_ops.AddV2) + param_types(tf.raw_ops.Sub))
def test_basic(data_type, op):
    MAX_BATCH = 5
    MAX_HIGHT = 5
    MAX_WIDTH = 5
    MAX_CHANNELS = 5

    for batchX in range(1, MAX_BATCH + 1):
        for batchY in range(1, MAX_BATCH + 1):
            for heightX in range(1, MAX_HIGHT + 1):
                for heightY in range(1, MAX_HIGHT + 1):
                    for widthX in range(1, MAX_WIDTH + 1):
                        for widthY in range(1, MAX_WIDTH + 1):
                            for channelX in range(1, MAX_CHANNELS + 1):
                                for channelY in range(1, MAX_CHANNELS + 1):
                                    if(batchX != batchY and min(batchX, batchY) != 1):
                                        pass
                                    elif(heightX != heightY and min(heightX, heightY) != 1):
                                        pass
                                    elif(widthX != widthY and min(widthX, widthY) != 1):
                                        pass
                                    elif(channelX != channelY and min(channelX, channelY) != 1):
                                        pass
                                    else:
                                        if(not np.issubdtype(data_type, np.unsignedinteger)):
                                            total_elements = (batchX*heightX*widthX*channelX)
                                            start = -int(total_elements / 2)
                                            stop = total_elements / 2
                                            x = np.arange(start, stop, dtype=data_type).reshape(batchX, heightX, widthX, channelX)
                                            y = np.arange(0, (batchY*heightY*widthY*channelY), dtype=data_type).reshape(batchY, heightY, widthY, channelY)
                                        else:
                                            x = np.arange(0, (batchX*heightX*widthX*channelX), dtype=data_type).reshape(batchX, heightX, widthX, channelX)
                                            y = y = np.arange(0, (batchY*heightY*widthY*channelY), dtype=data_type).reshape(batchY, heightY, widthY, channelY)

                                        with tf.device('CPU:0'):
                                            res1 = op(x=x, y=y)
                                            
                                        with tf.device(utills.DEVICE_NAME + ':0'):
                                            res2 = op(x=x, y=y)
                                        
                                        if not tf.reduce_all(res1 == res2):
                                            print(f' B:{batchX} H:{heightX} W:{widthX} C:{channelX}')
                                            print('Input tensor x:')
                                            print(x)
                                            print(f' B:{batchY} H:{heightY} W:{widthY} C:{channelY}')
                                            print('Input tensor y:')
                                            print(y)
                                            print('Expected output:')
                                            print(res1)
                                            print('Got:')
                                            print(res2)
                                            assert False

@pytest.mark.parametrize("data_type, op", param_types(tf.raw_ops.Mul) + param_types(tf.raw_ops.AddV2) + param_types(tf.raw_ops.Sub))
def test_basic_2_scalars(data_type, op):
    x = data_type(1)
    y = data_type(2)
    
    with tf.device('CPU:0'):
        res1 = op(x=x, y=y)
                                            
    with tf.device(utills.DEVICE_NAME + ':0'):
        res2 = op(x=x, y=y)
    
    if not tf.reduce_all(res1 == res2):
        print('Input tensor x:')
        print(x)
        print('Input tensor y:')
        print(y)
        print('Expected output:')
        print(res1)
        print('Got:')
        print(res2)
        assert False

@pytest.mark.parametrize("data_type, op", param_types(tf.raw_ops.Mul) + param_types(tf.raw_ops.AddV2) + param_types(tf.raw_ops.Sub))
def test_basic_1_scalar(data_type, op):
    x = data_type(1)
    y = np.array([1, 2, 3, 4, 5], dtype=data_type)
    
    with tf.device('CPU:0'):
        res1 = op(x=x, y=y)
                                            
    with tf.device(utills.DEVICE_NAME + ':0'):
        res2 = op(x=x, y=y)
    
    if not tf.reduce_all(res1 == res2):
        print('Input tensor x:')
        print(x)
        print('Input tensor y:')
        print(y)
        print('Expected output:')
        print(res1)
        print('Got:')
        print(res2)
        assert False

@pytest.mark.parametrize("data_type, op", param_types(tf.raw_ops.Mul) + param_types(tf.raw_ops.AddV2) + param_types(tf.raw_ops.Sub))
def test_basic_2_empty(data_type, op):
    x = np.array([], dtype=data_type)
    y = np.array([], dtype=data_type)
    
    with tf.device('CPU:0'):
        res1 = op(x=x, y=y)
                                            
    with tf.device(utills.DEVICE_NAME + ':0'):
        res2 = op(x=x, y=y)
    
    if not tf.reduce_all(res1 == res2):
        print('Input tensor x:')
        print(x)
        print('Input tensor y:')
        print(y)
        print('Expected output:')
        print(res1)
        print('Got:')
        print(res2)
        assert False

@pytest.mark.parametrize("data_type, op", param_types(tf.raw_ops.Mul) + param_types(tf.raw_ops.AddV2) + param_types(tf.raw_ops.Sub))
def test_basic_1_empty(data_type, op):
    x = np.array([], dtype=data_type)
    y = np.array([1,], dtype=data_type)
    
    with tf.device('CPU:0'):
        res1 = op(x=x, y=y)
                                            
    with tf.device(utills.DEVICE_NAME + ':0'):
        res2 = op(x=x, y=y)
    
    if not tf.reduce_all(res1 == res2):
        print('Input tensor x:')
        print(x)
        print('Input tensor y:')
        print(y)
        print('Expected output:')
        print(res1)
        print('Got:')
        print(res2)
        assert False

