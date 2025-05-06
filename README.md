# Credit card Eligibility Application

This repository contains a machine learning application for credit card eligibility checker. It uses a RandomForest classifier to predict whther a user is eligible for a credit card and also a chatbot 
that gives tips for the user on how to improve their chances of getting one and is built with Streamlit, deployed using Docker, and orchestrated with GitHub Actions for CI/CD. MLflow is used for model tracking and management.

## Key Features
- Automated detection of whether a user is eligible for a credit card.
- Regular retraining capabilities to adapt to data drift.
- Streamlit API for accessing model predictions.
- MLflow integration for model performance monitoring.
- Automated workflow using GitHub Actions for model retraining and deployment.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.8 or higher
- Docker installed
- Access to a terminal/command line interface
- GitHub account (for CI/CD using GitHub Actions)

## Installation

To install the Fraud Detection Application, follow these steps:

1. Clone the repository:
   ```
   git clone [repository-url]
   ```

2. Navigate to the cloned directory:
   ```
   cd [local-repository]
   ```

3. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install required packages:
   ```
   pip install -r requirements.txt
   ```


## Training and Evaluating the Model and 

To train the model, run:

```
python /model/model.py
```

This will load the test data and the trained model, then output evaluation metrics.



## Running the Application
1. **Train the model**:
   ```
   python model/model.py
   ```
2. **Start the Streamlit app**:
   ```
   Streamlit run api/app.py
   ```


## Deploying with Docker

To deploy the application using Docker:

1. Build the Docker image:
   ```
   docker build -t credit-eligibility .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8080 credit-eligibility
   ```

The application will be available at `http://localhost:8080`.

## CI/CD with GitHub Actions

The `.github/workflows/ml-ops-workflow.yml` file defines the GitHub Actions workflow for continuous integration and deployment. Pushing changes to the main branch will trigger the workflow. Also the workflow is triggered every month to retrain the model.

## Using MLflow for Model Tracking

To track models using MLflow, ensure MLflow server is running and accessible. The training and evaluation scripts are set up to log metrics and parameters to MLflow. 
```
mlflow ui
```

## License
This project is licensed under the terms of the MIT License


