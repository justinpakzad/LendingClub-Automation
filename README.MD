# Lending Club

## Overview
This project is designed to automate the loan decision-making process utilizing the Lending Club dataset. Through the development and deployment of machine learning models, we aim to classify loans as accepted or rejected and predict loan grades, subgrades, and interest rates.

### Installation
Install the required Python packages:
  ```
  pip install -r requirements.txt
  ```

## Project Structure
The project is structured into a series of Jupyter notebooks that focus on different aspects of the loan decision process. These notebooks are stored in the `notebooks` folder. Our project also consists of a flask application for model deployment, which lives in the `flask_app` folder.

### Part One: Data Cleaning
- **Notebook**: `Data_Cleaning.ipynb`
- **Description**: Basic cleaning and preprocessing of the datasets, addressing missing values, duplicates, and data inconsistencies.

### Part Two: Loan Acceptance Model
- **Notebook**: `Accepted_Rejected Loans.ipynb`
- **Description**: Performed exploratory data analysis and developed a model to classify loans as accepted or rejected, establishing the basis for further analysis.

### Part Three: Loan Grading Model
- **Notebook**: `Loan Grades & Subgrades.ipynb`
- **Description**:Further analyzed our data and developed models to predict the grades and subgrades for loans.

### Part Four: Interest Rate Prediction
- **Notebook**: `Interest Rates.ipynb`
- **Description**: Finished our automation task by analyzing and developing a model to predict interest rates.

### Part Five: Model Deployment
- **Folder**: `flask_app`
- **Description**: This folder contains all the code for our model deployment and flask application.


## API Usage
Our API endpoints can be accessed using various tools such as Postman or `curl`.

**URL**:
https://lending-club-automation-rcmhkbbwgq-uc.a.run.app

**Endpoints**:
- `POST /predict/loan-approval`
- `POST /predict/grade-and-rate`


**Example Request:**
```bash
curl -X POST https://lending-club-automation-rcmhkbbwgq-uc.a.run.app/predict/loan-approval \
-H "Content-Type: application/json" \
-d '{
      "loan_amnt": 5000.0,
      "purpose": "credit_card",
      "dti": 18,
      "addr_state": "CA",
      "zip_code": "940xx",
      "emp_length": "2 years",
      "fico_range_low": 700.0,
      "fico_range_high": 720.0
    }'
