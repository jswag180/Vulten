import utills
import numpy as np
import tensorflow as tf
import pytest

tf.config.run_functions_eagerly(False)

#float16 will overflow and causes propblems for testing
#, (np.float16)
@pytest.mark.parametrize("data_type", [(np.float32), (np.float64), (np.int32), (np.int64), (np.complex64), (np.complex128)])
def test_mat_mul(data_type):
    MAX_X = 10
    MAX_Y = 10

    for x in range(1, MAX_X + 1):
        for y in range(1, MAX_Y + 1):
            for x2 in range(1, MAX_X + 1):
                for y2 in range(1, MAX_Y + 1):
                    for tx in range(2):
                        for ty in range(2):

                            if (y if not bool(tx) else x) != (x2 if not bool(ty) else y2):
                                break

                            if(not np.issubdtype(data_type, np.unsignedinteger)):
                                total_elements = (x*y)
                                start = -int(total_elements / 2)
                                stop = total_elements / 2
                                a = np.arange(start, stop, dtype=data_type).reshape(x, y)
                                b = np.arange(0, (x2*y2), dtype=data_type).reshape(x2, y2)
                            else:
                                a = np.arange(0, (x*y), dtype=data_type).reshape(x, y)
                                b = y = np.arange(0, (x2*y2), dtype=data_type).reshape(x2, y2)
                            
                            with tf.device('CPU:0'):
                                res1 = tf.raw_ops.MatMul(a=a, b=b, transpose_a=bool(tx), transpose_b=bool(ty))

                            with tf.device('VULK:0'):
                                res2 = tf.raw_ops.MatMul(a=a, b=b, transpose_a=bool(tx), transpose_b=bool(ty))
                            
                            if not tf.reduce_all(res1 == res2):
                                print(f'(X:{x}, Y:{y}) (X:{x2}, Y:{y2}) tx{tx} ty:{ty}')
                                print('Input tensor a:')
                                print(a)
                                print('Input tensor b:')
                                print(b)
                                print('Expected output:')
                                print(res1)
                                print('Got:')
                                print(res2)
                                assert False