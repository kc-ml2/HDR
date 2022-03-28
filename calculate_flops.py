import os
import argparse
import datetime
import tensorflow as tf
from tqdm import tqdm
from model_ATT import *
from models_residual import *
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


def flop_keras(model):
    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                         options=ProfileOptionBuilder.float_operation())

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    flops = graph_info.total_float_ops // 2
    print('Flops: {:,}'.format(flops/10**9))


def get_runtime(model, input_size=(1, 3, 1060, 1900, 6), num_reps=100):
    """ This function calculates the mean runtime of a given pytorch model.
    More info: https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/

    Args:
        model: A pytorch model object
        input_size: (batch_size, num images, channels, height, width) - input dimensions for a single NTIRE test scene
        num_reps: The number of repetitions over which to calculate the mean runtime

    Returns:
        mean_runtime: The everage runtime of the model over num_reps iterations

    """
    # Define input, for this example we will use a random dummy input
    in_data = Input(batch_shape=input_size)
    # Define start and stop cuda events
    times = np.zeros((num_reps, 1))
    # Perform warm-up runs (that are normally slower)
    for _ in range(10):
        # _ = model.predict(input)
        _ = model_x.main_model(in_data)
    # Measure actual runtime
    for it in tqdm(range(num_reps)):
        starter = datetime.datetime.now()
        _ = model_x.main_model(in_data)
        ender = datetime.datetime.now()
        # Await for GPU to finish the job and sync
        curr_time = ender - starter
        # Convert from miliseconds to seconds
        times[it] = curr_time.seconds
    # Average all measured times
    mean_runtime = np.sum(times) / num_reps
    return mean_runtime


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--filter', type=int, default=8)
    parser.add_argument('--attention_filter', type=int, default=16)
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--encoder_kernel', type=int, default=3)
    parser.add_argument('--decoder_kernel', type=int, default=3)
    parser.add_argument('--triple_pass_filter', type=int, default=16)

    config = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model_x = Net(config)
    in_data = Input(batch_shape=(None, 3, 1060, 1900, 6))
    model = Model(inputs=in_data, outputs=model_x.main_model(in_data))
    print('run time: ', get_runtime(model=model_x))
    flop_keras(model)
