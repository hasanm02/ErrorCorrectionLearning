import numpy as np

def error_correction_learning(inputs, desired_outputs, max_iterations, learning_rate, random_weights=True, external_weights=None):
    num_inputs = len(inputs[0]) + 1

    if random_weights:
        # Generate random starting weights in the range [-0.5, 0.5] including bias weight
        weights = np.random.uniform(-0.5, 0.5, size=num_inputs)
    else:
        if external_weights is None:
            raise ValueError("External weights must be provided if use_random_weights is set to False.")
        weights = external_weights

    iterations = 0
    flag = True

    while iterations < max_iterations:
        flag = True
        for i in range(len(inputs)):
            input_vector = np.append(inputs[i], 1)  #bias
            weighted_sum = np.dot(input_vector, weights)
            actual_output = 1 if weighted_sum >= 0 else -1  
            error = desired_outputs[i] - actual_output

            if error != 0:
                # Adjust weights using the error-correction rule with learning rate
                weights += learning_rate * error * input_vector
                flag = False

        iterations += 1

        if flag:
            break

    if iterations == max_iterations and not flag:
        print("Maximum iterations reached without convergence.")
    
    return weights

#Input/output mappings from slide 16
inputs_16 = [[1, 1, 1, 1], [1, 1, -1, 1], [1, -1, 1, 1], [1, -1, -1, 1], [-1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1]]
desired_outputs_16 = [1, 1, 1, -1, 1, 1, -1, -1]

# Input/output mappings from slide 17
inputs_17 = [[1, 1, 1, 1], [1, 1, -1, 1], [1, -1, 1, 1], [1, -1, -1, 1], [-1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [-1, -1, -1, 1]]
desired_outputs_17 = [1, -1, -1, -1, 1, -1, 1, -1]

# including bias for external weights
external_weights = np.zeros(len(inputs_17[0]) + 1)

# Slide 16: Learning from random starting weights
weights_random = error_correction_learning(inputs_16, desired_outputs_16, max_iterations=1000, learning_rate=0.1, random_weights=True)
print("Weights learned from random starting weights (Slide 16):", weights_random)

# Slide 17: Learning from Hebbian starting weights
weights_hebbian = error_correction_learning(inputs_17, desired_outputs_17, max_iterations=1000, learning_rate=0.1, random_weights=False, external_weights=external_weights)
print("Weights learned from Hebbian starting weights (Slide 17):", weights_hebbian)
