import utills
import numpy as np
import tensorflow as tf
import pytest

tf.config.run_functions_eagerly(False)

@pytest.mark.parametrize("data_type", [(np.float32), (np.float64), (np.int32), (np.uint8), (np.int16), (np.int8), (np.int64),
                                        (np.uint16), (np.uint32), (np.uint64)])
def test_sum(data_type):
    MAX_RANKS = 5
    MAX_DIM_PER_RANK = 5

    for ranks in range(1, MAX_RANKS + 1):
        dims = np.ones(ranks, dtype=np.int32)
        for j in range(ranks):
            for k in range(MAX_DIM_PER_RANK):
                dims[j] += 1 * (k != 0)
                for i in range(1, ranks + 1):
                    if(not np.issubdtype(data_type, np.unsignedinteger)):
                        total_elements = np.prod(dims)
                        start = -int(total_elements / 2)
                        stop = total_elements / 2
                        tens = np.arange(start, stop, dtype=data_type).reshape(*dims)
                    else:
                        tens = np.arange(0, np.prod(dims), dtype=data_type).reshape(*dims)

                    axis = tf.convert_to_tensor(np.arange(i, dtype=np.int32), dtype=tf.int32)
                    keep_dims = False

                    with tf.device('CPU:0'):
                        res1 = tf.raw_ops.Sum(input=tens, axis=axis, keep_dims=keep_dims)
                
                    with tf.device(utills.DEVICE_NAME + ':0'):
                        res2 = tf.raw_ops.Sum(input=tens, axis=axis, keep_dims=keep_dims)
            
                    if not tf.reduce_all(res1 == res2):
                        print(f' Shape: {dims} axis: {axis}')
                        print('Input tensor:')
                        print(tens)
                        print('Expected output:')
                        print(res1)
                        print('Got:')
                        print(res2)
                        assert False
