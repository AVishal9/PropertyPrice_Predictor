library(tidyverse)
library(caret)
library(corrplot)
library(GGally)
library(scales)
library(plotly)
library(glmnet)

# Assume sample_data is already in the environment

# Convert acres to square feet and rename the column
sample_data <- sample_data %>%
  mutate(land_size = acre_lot * 43560) 

# Apply log transformations
sample_data <- sample_data %>%
  mutate(
    log_price = log1p(price),         # Log transformation for price
    log_house_size = log1p(house_size),  # Log transformation for house size
    log_land_size = log1p(land_size)    # Log transformation for land size
  )

# Remove rows where land_size is 0 or land_size is not greater than house_size
sample_data <- sample_data %>%
  filter(land_size > house_size & price > house_size & city != "")

# Check for missing values and handle them
sample_data <- sample_data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Feature scaling for numeric columns
scaled_data <- sample_data %>%
  mutate(
    house_size = scale(log_house_size),
    price = scale(log_price),
    land_size = scale(log_land_size)
  )

# One-hot encode categorical variables (e.g., state)
dummies <- dummyVars(~ state, data = sample_data)
state_dummies <- predict(dummies, newdata = sample_data)
sample_data <- cbind(sample_data, state_dummies)

# Feature engineering
sample_data <- sample_data %>%
  mutate(
    price_per_sqft = price / house_size,                # Price per square foot
    bed_bath_ratio = ifelse(bath > 0, bed / bath, NA),  # Avoid division by zero
    land_to_house_ratio = land_size / house_size,       # Land-to-house size ratio
    bed_bath_interaction = bed * bath,                 # Interaction between number of bedrooms and bathrooms
    total_rooms = bed + bath,                           # Total number of rooms
    rooms_size_interaction = total_rooms * house_size   # Interaction between total rooms and house size
  ) %>%
  drop_na()

# Remove outliers for price and price_per_sqft
remove_outliers <- function(data, column) {
  Q1 <- quantile(data[[column]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[column]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  data %>%
    filter(data[[column]] >= (Q1 - 1.5 * IQR) & data[[column]] <= (Q3 + 1.5 * IQR))
}

sample_data <- sample_data %>%
  remove_outliers("price") %>%
  remove_outliers("price_per_sqft")

# Split the data into training and testing sets
set.seed(123)
train_index <- sample(1:nrow(sample_data), size = 0.8 * nrow(sample_data))
train_data <- sample_data[train_index, ]
test_data <- sample_data[-train_index, ]

# Prepare features and target for Lasso regression
x_train <- train_data %>%
  select(log_house_size, log_land_size, bed, bath, 
         bed_bath_ratio, land_to_house_ratio, 
         bed_bath_interaction, total_rooms, rooms_size_interaction) %>%
  as.matrix()

y_train <- train_data$log_price

x_test <- test_data %>%
  select(log_house_size, log_land_size, bed, bath, 
         bed_bath_ratio, land_to_house_ratio, 
         bed_bath_interaction, total_rooms, rooms_size_interaction) %>%
  as.matrix()

y_test <- test_data$log_price

# Train the Lasso model with cross-validation
lasso_model_fine <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,  # Lasso regression
  standardize = TRUE,
  nfolds = 10,
  lambda = 10^seq(-4, 1, length = 100)  # Expanded range of lambda values
)

# Get the best lambda
best_lambda_fine <- lasso_model_fine$lambda.min
cat("Fine-Tuned Best Lambda:", best_lambda_fine, "\n")

# Predict on the test set
lasso_predictions_fine <- predict(lasso_model_fine, s = best_lambda_fine, newx = x_test)

# Calculate performance metrics
mse_fine <- mean((lasso_predictions_fine - y_test)^2)  # Mean Squared Error (MSE)
rmse_fine <- sqrt(mse_fine)                           # Root Mean Squared Error (RMSE)
mae_fine <- mean(abs(lasso_predictions_fine - y_test)) # Mean Absolute Error (MAE)
r_squared_fine <- 1 - sum((lasso_predictions_fine - y_test)^2) / sum((y_test - mean(y_test))^2) # R-squared

# Print metrics
cat("Fine-Tuned Lasso MSE:", mse_fine, "\n")
cat("Fine-Tuned Lasso RMSE:", rmse_fine, "\n")
cat("Fine-Tuned Lasso MAE:", mae_fine, "\n")
cat("Fine-Tuned Lasso R-squared:", r_squared_fine, "\n")

# Store metrics in a data frame
results <- data.frame(
  Metric = c("MSE", "RMSE", "MAE", "R-Squared"),
  Value = c(mse_fine, rmse_fine, mae_fine, r_squared_fine)
)

# Print the results data frame
print(results)

# Plot the Lasso model
plot(lasso_model_fine)

# Extract and print coefficients
coefficients <- coef(lasso_model_fine, s = best_lambda_fine)
print(coefficients)
