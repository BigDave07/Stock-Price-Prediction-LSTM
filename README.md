# Stock Price Prediction using LSTM

This project demonstrates how to use a Long Short-Term Memory (LSTM) neural network to predict the closing price of a stock, specifically NVIDIA (NVDA), using historical daily price data.

## Project Overview

The notebook performs the following steps:

1.  **Data Acquisition**: Downloads historical daily stock price data for a specified ticker (NVDIA in this case) using the `yfinance` library.
2.  **Data Preprocessing**:
    *   Keeps only the 'Close' price column.
    *   Handles missing values.
    *   Scales the 'Close' prices to a range between 0 and 1 using `MinMaxScaler` for stable model training.
    *   Creates time sequences (windows) of the data for training the LSTM model. Each window consists of the previous 30 days' closing prices to predict the next day's closing price.
3.  **Train/Test Split**: Splits the data into training and testing sets based on time, ensuring that the model is trained on past data and tested on future data.
4.  **Model Building**: Constructs a Sequential LSTM model using TensorFlow/Keras. The model consists of:
    *   Two stacked LSTM layers with Dropout for regularization.
    *   A Dense layer with a single output node to predict the next day's closing price.
5.  **Model Training**: Trains the LSTM model on the training data using Mean Squared Error (MSE) as the loss function and the Adam optimizer. Includes validation on the test set to monitor performance during training.
6.  **Model Evaluation**:
    *   Predicts the closing prices for the test set.
    *   Inverse-scales the predictions back to the original dollar values.
    *   Calculates the Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) to evaluate the model's performance.
    *   Visualizes the actual vs. predicted closing prices for the test set.
7.  **Future Prediction**: Uses the trained model and the last 30 days of the entire dataset to forecast the closing price for the next trading day.

## Requirements

The following Python libraries are required:

*   `yfinance`: To download historical stock data.
*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `matplotlib`: For plotting and visualization.
*   `scikit-learn`: For data splitting and scaling.
*   `tensorflow`: For building and training the LSTM model.

You can install these libraries using pip:
