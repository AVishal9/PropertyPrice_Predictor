
library(tidyverse) 
library(dplyr)
library(naniar)
library(ggplot2)
library(formattable)
library(caret)
library(MASS)
library(kableExtra)
library(xgboost)


####
# XG boost
####

#Read the sampled data
stratified_sample <- read.csv("sampled_data.csv", sep = ",")%>%
                    dplyr::select(-X)
                

# Label encode categorical variables
# Extract column names that are not numeric
categorical <- names(stratified_sample)[sapply(stratified_sample, function(x) !is.numeric(x))]

# Convert these columns to factor 
stratified_sample[categorical] <- lapply(stratified_sample[categorical], as.factor)
stratified_sample[categorical] <- lapply(stratified_sample[categorical], as.integer)


# Creating an index to split the dataset into training dataset (70%)
Index <- createDataPartition(
  y = stratified_sample$price,
  p = .70, # The percentage of data in the training set
  list = FALSE 
)

# Splitting into training and test dataset using the Index created above
TrainDataXG <- stratified_sample[Index, ]
TestDataXG <- stratified_sample[-Index, ]

# Separate the features and target variables
x_train <- as.matrix(TrainDataXG %>%
                       dplyr::select(-price)) # Exclude target variable

y_train <- TrainDataXG$price

x_test <- as.matrix(TestDataXG %>%
                      dplyr::select(-price))
y_test <- TestDataXG$price

# Convert data into DMatrix, a special format for XGBoost
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)


###
# Training the XGBoost model using hyperparameters using grid search & CV
###
set.seed(1)

# Define hyperparameters for XGBoost using grid search
param_grid <- expand.grid(
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.1, 0.2),
  subsample = c(0.6, 0.8, 1.0),
  colsample_bytree = c(0.6, 0.8, 1.0)
)


# Variables to store results
best_rmse <- Inf
best_params <- NULL

# Loop through parameter grid
for (i in 1:nrow(param_grid)) {
  params <- list(
    objective = "reg:squarederror",
    eta = param_grid[i, "eta"],
    max_depth = param_grid[i, "max_depth"],
    subsample = param_grid[i, "subsample"],
    colsample_bytree = param_grid[i, "colsample_bytree"]
  )
  
  # Performing  cross-validation on training dataset
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 150,
    nfold = 5,
    metrics = "rmse",
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  # Tracking the best parameters
  mean_rmse <- min(cv_results$evaluation_log$test_rmse_mean)
  if (mean_rmse < best_rmse) {
    best_rmse <- mean_rmse
    best_params <- params
  }
}

# Best parameters and RMSE
print(best_params)
print(best_rmse)


####
# Training the model using 5 fold cross-validation after Random search
####

set.seed(42)

# Generate random parameter combinations
param_samples <- data.frame(
  max_depth = sample(3:7, 10, replace = TRUE),
  eta = runif(10, 0.01, 0.3),
  subsample = runif(10, 0.6, 1.0),
  colsample_bytree = runif(10, 0.6, 1.0)
)

best_rmse_RS <- Inf
best_params_RS <- NULL

for (i in 1:nrow(param_samples)) {
  params_RS <- list(
    objective = "reg:squarederror",
    eta = param_samples[i, "eta"],
    max_depth = param_samples[i, "max_depth"],
    subsample = param_samples[i, "subsample"],
    colsample_bytree = param_samples[i, "colsample_bytree"]
  )
  
  cv_results_RS <- xgb.cv(
    params_RS = params_RS,
    data = dtrain,
    nrounds = 150,
    nfold = 5,
    metrics = "rmse",
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  mean_rmse <- min(cv_results_RS$evaluation_log$test_rmse_mean)
  if (mean_rmse < best_rmse_RS) {
    best_rmse_RS <- mean_rmse
    best_params_RS <- params_RS
  }
}

# Printing the best parameter from Random Search
print(best_params_RS)
print(best_rmse_RS)

###
# Training on the Final Model
###

# The Grid Search had the lower RMSE so we will evaluate the final model using those parameters
final_params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 7,
  subsample = 0.8,
  colsample_bytree = 0.6
)

# Train the model with the 150 boosting rounds

set.seed(17)
xgb_model_cv <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = 150
)

# Predict on test data
y_pred_cv <- predict(xgb_model_cv, newdata = dtest)


# Calculate evaluation metrics
mae_cv <- mean(abs(y_pred_cv - TestDataXG$price))
mse_cv <- mean((y_pred_cv - TestDataXG$price)^2)
rmse_cv <- sqrt(mse_cv)
r2_cv <- 1 - (sum((TestDataXG$price - y_pred_cv)^2) / sum((TestDataXG$price - mean(TestDataXG$price))^2))


# Print evaluation metrics
cat("MAE:", mae_cv, "\nMSE:", mse_cv, "\nRMSE:", rmse_cv, "\nR2:", r2_cv)

#MAE: 103769.6 
#MSE: 26226219981 
#RMSE: 161945.1 
#R2: 0.7028739
