# ErrorCorrectionLearning

The main function in this script is error_correction_learning(), which takes the following parameters:

- inputs: A list of input vectors.
- desired_outputs: A list of desired output values corresponding to each input vector.
- max_iterations: The maximum number of iterations for the learning process.
- learning_rate: The learning rate for weight adjustments.
- random_weights: A boolean value indicating whether to initialize the weights randomly or not.
- external_weights: An optional parameter for providing external weights.

The function initializes the weights either randomly or using the provided external weights. It then iterates over the inputs, calculates the actual output based on the current weights, and adjusts the weights based on the error between the desired and actual output. The function returns the learned weights after convergence or reaching the maximum number of iterations.

The script also includes two sets of input/output mappings, which are used to test the learning algorithm.

This will print out the learned weights after running the error correction learning algorithm with random starting weights. You can adjust the inputs and parameters as needed for your specific use case.

Please note that this is a simple implementation of the Error Correction Learning algorithm and may not be suitable for complex or large-scale problems.
