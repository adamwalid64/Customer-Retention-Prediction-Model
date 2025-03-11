"""
Multiple Linear Regression with Gradient Descent
This implementation includes data preprocessing, training, evaluation and prediction capabilities.
Author: Your Name
Date: 2024
"""

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Import custom data loading utilities
from data_loader import load_csv_data, load_excel_data, split_data

class MultipleLinearRegression:
    """A Multiple Linear Regression implementation using gradient descent optimization."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """Initialize the model parameters."""
        # Model parameters
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
        # Training history
        self.cost_history = []
        
        # Scalers for features and target
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def preprocess_data(self, X, y):
        """Scale the features and target variable."""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
        return X_scaled, y_scaled
    
    def initialize_parameters(self, n_features):
        """Initialize model weights and bias to zeros."""
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
    
    def compute_cost(self, X, y):
        """Calculate the Mean Squared Error cost."""
        m = X.shape[0]
        predictions = np.dot(X, self.weights) + self.bias
        cost = (1/(2*m)) * np.sum(np.square(predictions - y))
        return cost
    
    def gradient_descent_step(self, X, y):
        """Perform one iteration of gradient descent."""
        m = X.shape[0]
        predictions = np.dot(X, self.weights) + self.bias
        
        # Calculate gradients
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def train_model(self, X_train, y_train):
        """Train the model using gradient descent."""
        # Initialize model parameters
        self.initialize_parameters(X_train.shape[1])
        
        # Gradient descent iterations
        for i in range(self.n_iterations):
            self.gradient_descent_step(X_train, y_train)
            cost = self.compute_cost(X_train, y_train)
            self.cost_history.append(cost)
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f'Iteration {i+1}/{self.n_iterations}, Cost: {cost:.6f}')
    
    def evaluate_model(self, X_test, y_test):
        """Calculate model performance metrics."""
        y_pred = np.dot(X_test, self.weights) + self.bias
        
        metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2 Score': r2_score(y_test, y_pred)
        }
        return metrics
    
    def predict(self, X_new):
        """Make predictions on new data."""
        X_scaled = self.scaler_X.transform(X_new)
        y_pred_scaled = np.dot(X_scaled, self.weights) + self.bias
        return self.scaler_y.inverse_transform(y_pred_scaled)

def create_sample_data(n_samples=100):
    """Create synthetic data for demonstration."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.rand(n_samples, 3),
        columns=['feature1', 'feature2', 'feature3']
    )
    # True relationship: y = 3*x1 + 2*x2 - x3 + noise
    y = pd.Series(
        3 * X['feature1'] + 2 * X['feature2'] - X['feature3'] + 
        np.random.normal(0, 0.1, n_samples)
    )
    return X, y

def run_model_with_data(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Run the complete modeling process with provided data.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    learning_rate : float, optional
        Learning rate for gradient descent
    n_iterations : int, optional
        Number of training iterations
    """
    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Initialize and train the model
    print("\nInitializing the model...")
    model = MultipleLinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
    
    # Preprocess the data
    print("Preprocessing the data...")
    X_train_scaled, y_train_scaled = model.preprocess_data(X_train, y_train)
    X_test_scaled, y_test_scaled = model.preprocess_data(X_test, y_test)
    
    # Train the model
    print("\nTraining the model...")
    model.train_model(X_train_scaled, y_train_scaled)
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    metrics = model.evaluate_model(X_test_scaled, y_test_scaled)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(model.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Training Cost History')
    plt.grid(True)
    plt.show()
    
    return model

def main():
    """Main function demonstrating different ways to use the model."""
    
    # Example 1: Using sample data
    print("\nExample 1: Using sample data")
    print("-" * 50)
    X, y = create_sample_data(n_samples=100)
    model1 = run_model_with_data(X, y)
    
    # Example 2: Loading data from CSV
    print("\nExample 2: Loading data from CSV")
    print("-" * 50)
    try:
        # Replace with your CSV file path and column names
        X, y = load_csv_data(
            file_path="your_data.csv",
            target_column="target",
            feature_columns=None  # Use all columns except target
        )
        if X is not None and y is not None:
            model2 = run_model_with_data(X, y)
    except Exception as e:
        print(f"Skipping CSV example: {str(e)}")
    
    # Example 3: Loading data from Excel
    print("\nExample 3: Loading data from Excel")
    print("-" * 50)
    try:
        # Replace with your Excel file path and column names
        X, y = load_excel_data(
            file_path="your_data.xlsx",
            target_column="target",
            feature_columns=None,  # Use all columns except target
            sheet_name=0  # First sheet
        )
        if X is not None and y is not None:
            model3 = run_model_with_data(X, y)
    except Exception as e:
        print(f"Skipping Excel example: {str(e)}")

if __name__ == "__main__":
    main() 