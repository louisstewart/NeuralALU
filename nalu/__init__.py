from nalu.NeuralAccumulator import NeuralAccumulator
from nalu.NeuralArithmeticLogicUnit import NeuralALU


def neural_accumulator(inputs, outputs, name=None, kernel_initializer=None):
    """
    Create a neural accumulator cell
    Args:
        inputs: 2D Tensor [batch_size, features]
        outputs: number of output features
        name: name of op
        kernel_initializer: initializer for weights of NAC

    Returns:
        NAC step applied to inputs
    """
    cell = NeuralAccumulator(outputs, kernel_initializer=kernel_initializer, name=name)
    outputs = cell(inputs)

    return outputs


def neural_alu(inputs, outputs, name=None, kernel_initializer=None):
    """
    Create a neural arithmetic logic cell
    Args:
        inputs: 2D Tensor [batch_size, features]
        outputs: number of output features
        name: name of op
        kernel_initializer: initializer for weights of NAC

    Returns:
        Neural ALU step applied to inputs
    """
    cell = NeuralALU(outputs, kernel_initializer=kernel_initializer, name=name)
    outputs = cell(inputs)

    return outputs
