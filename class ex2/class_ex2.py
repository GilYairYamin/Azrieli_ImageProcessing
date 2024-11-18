# Step 1: Basic Setup
"""
First, we'll set up our basic imports and create a simple class structure.
Student Task: Import numpy and create an empty Perceptron class
"""
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, threshold=0.5):
        #  Task: Initialize the following:
        # - weights as numpy array of zeros with size 3 (for w0, w1, w2)
        # - learning_rate
        # - threshold
        self.weights = np.array([0, 0, 0])
        self.learning_rate = learning_rate
        self.threshold = threshold
        pass

    # Step 2: Implement the Prediction Function
    """
    Now implement the function that calculates the sum and makes a prediction.
    Formula: sum = x0*w0 + x1*w1 + x2*w2
    If sum > threshold, predict 1; otherwise predict 0
    """

    def predict(self, inputs):
        #  Task:
        # 1. Calculate dot product of inputs and weights (np.dot)
        # 2. Compare with threshold and return binary result
        val = np.dot(self.weights, inputs)
        return 1 if val > self.threshold else 0 

    # Step 3: Implement Single Training Step
    """
    This is the core learning function. For each training example:
    1. Make prediction
    2. Calculate error
    3. Update weights
    """

    def train_step(self, inputs, desired_output):
        #  Task:
        # 1. Calculate current prediction using predict()
        # 2. Calculate error (desired_output - prediction)
        # 3. Calculate correction (learning_rate * error)
        # 4. Update each weight: w_new = w_old + correction * input

        pred = self.predict(inputs)
        error = desired_output - pred
        correlation = self.learning_rate * error
        self.weights = self.weights + correlation * inputs
        result = {
            "prediction": pred,
            "error": error,
            "weights": self.weights
        }
        return result

# Step 4: Complete Implementation
"""
Here's the complete implementation for reference
"""


# Step 5: Training Data Setup
"""
Create training data for the logic function you want to learn
Example: Learning OR function
"""


def create_training_data():
    # Student Task:
    # Create training data as list of tuples: (inputs, desired_output)
    # Remember to include bias input (x0) as 1
    # Format: [(x0, x1, x2), desired_output]
    training_data = [
        ([1, 0, 0], 1),  # Example: First row from your table
        ([1, 0, 1], 1),  # Add more rows...
        ([1, 1, 0], 1),
        ([1, 1, 1], 0),
    ]
    return training_data


# Step 6: Training Loop
"""
Implement the training loop that processes all examples
"""


def train_perceptron(training_data, epochs=4):
    # Student Task:
    # 1. Create perceptron instance
    # 2. Loop through epochs
    # 3. In each epoch, loop through training data
    # 4. Call train_step for each training example
    # 5. Optional: Store training history for visualization
    p = Perceptron()
    for _ in range(epochs):
        for inputs, desired_output in training_data:
            result = p.train_step(np.array(inputs), desired_output)

            # Print training step details
            print(f"\nInputs: {inputs}")
            print(f"Desired Output: {desired_output}")
            print(f"Prediction: {result['prediction']}")
            print(f"Error: {result['error']}")
            print(f"Updated Weights: {result['weights']}")
    return p


# Step 7: Example Usage and Visualization
def main():
    # Create training data
    training_data = [([1, 0, 0], 1), ([1, 0, 1], 1), ([1, 1, 0], 1), ([1, 1, 1], 0)]

    # Create and train perceptron
    p = Perceptron()

    # Training loop
    print("Training Process:")
    print("=" * 50)

    for epoch in range(4):
        print(f"\nEpoch {epoch + 1}")
        for inputs, desired in training_data:
            result = p.train_step(np.array(inputs), desired)

            # Print training step details
            print(f"\nInputs: {inputs}")
            print(f"Desired Output: {desired}")
            print(f"Prediction: {result['prediction']}")
            print(f"Error: {result['error']}")
            print(f"Updated Weights: {result['weights']}")


# Run the example
if __name__ == "__main__":
    main()
