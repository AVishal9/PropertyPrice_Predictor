library(tidyverse)
library(caret)
library(corrplot)
library(GGally)
library(scales)
library(plotly)
library(glmnet)

# Load the data
house_data <- readRDS("data_processed/house_data.rds")

# Apply log transformations
house_data <- house_data %>%
  mutate(
    log_price = log1p(price),       # Log transformation for price
    log_house_size = log1p(house_size),  # Log transformation for house size
    log_land_size = log1p(land_size)  # Log transformation for land size
  )

# Remove rows where land_size is 0 or land_size is not greater than house_size
house_data <- house_data %>%
  filter(land_size > house_size & price > house_size & city!="")

sum(is.na(house_data$city))

# Check the cleaned data
summary(house_data)


# 1. Handling Missing Values
# Check for missing values in the dataset
sum(is.na(house_data))  # Number of missing values
# For simplicity, let's impute missing values using median for numerical columns
house_data <- house_data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 2. Scaling Numerical Features
# Log transformations were already applied for price, house_size, and land_size.
# Now, let's scale the numerical variables for algorithms that require it (e.g., linear regression)
scaled_data <- house_data %>%
  mutate(
    house_size = scale(log_house_size), 
    price = scale(log_price),  # scaling log-transformed price
    land_size = scale(log_land_size)
  )

# 3. Encoding Categorical Variables
# If you have categorical variables, you can encode them:
# One-hot encoding for a categorical variable (e.g., state)
house_data <- house_data %>%
  mutate(across(where(is.factor), as.character)) %>%
  mutate(across(where(is.character), as.factor))

# One-hot encode 'state' using dummyVars from caret package
# Create dummy variables for 'state'
dummies <- dummyVars(~ state, data = house_data)

# Predict the dummy variables
state_dummies <- predict(dummies, newdata = house_data)

# Add the dummy variables to the original dataset without overwriting
house_data <- cbind(house_data, state_dummies)


# Feature Engineering
house_data <- house_data %>%
  mutate(
    price_per_sqft = price / house_size,                # Price per square foot
    bed_bath_ratio = ifelse(bath > 0, bed / bath, NA),  # Avoid division by zero
    land_to_house_ratio = land_size / house_size,       # Land-to-house size ratio
    bed_bath_interaction = bed * bath,                 # Interaction between number of bedrooms and bathrooms
    total_rooms = bed + bath,                           # Total number of rooms
    rooms_size_interaction = total_rooms * house_size   # Interaction between total rooms and house size
  )

# Remove rows with missing values created during feature engineering
house_data <- house_data %>%
  drop_na()

# Remove outliers based on IQR for 'price' and 'price_per_sqft'
remove_outliers <- function(data, column) {
  Q1 <- quantile(data[[column]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[column]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  data <- data %>%
    filter(data[[column]] >= (Q1 - 1.5 * IQR) & data[[column]] <= (Q3 + 1.5 * IQR))
  return(data)
}

house_data <- house_data %>%
  remove_outliers("price") %>%
  remove_outliers("price_per_sqft")


# Split the data into training and testing sets
set.seed(123)
train_index <- sample(1:nrow(house_data), size = 0.8 * nrow(house_data))
train_data <- house_data[train_index, ]
test_data <- house_data[-train_index, ]

# Prepare features and target for Lasso regression
# Prepare features and target for Lasso regression (exclude price_per_sqft and house_land_ratio)
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
# Fine-tune the lambda grid
lasso_model_fine <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 0.5,  # Lasso regression
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
rmse_fine <- sqrt(mean((lasso_predictions_fine - y_test)^2))  # RMSE
mae_fine <- mean(abs(lasso_predictions_fine - y_test))        # MAE
r_squared_fine <- 1 - sum((lasso_predictions_fine - y_test)^2) / sum((y_test - mean(y_test))^2) # R-squared

cat("Fine-Tuned Lasso RMSE:", rmse_fine, "\n")
cat("Fine-Tuned Lasso MAE:", mae_fine, "\n")
cat("Fine-Tuned Lasso R-squared:", r_squared_fine, "\n")

plot(lasso_model)

coefficients <- coef(lasso_model_fine, s = best_lambda_fine)
print(coefficients)

