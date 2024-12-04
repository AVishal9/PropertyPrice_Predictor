library(tidyverse)
library(caret)
library(corrplot)
library(GGally)
library(scales)
library(plotly)

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


# 4. Creating New Features
# Create a new feature: Price per Square Foot 
house_data <- house_data %>%
  mutate(price_per_sqft = price / house_size)

# Inspecting the transformed data
summary(house_data)

# Check for any missing values after transformations
sum(is.na(house_data))  # Make sure no missing values are left

# Visualize any new features
ggplot(house_data, aes(x = price_per_sqft)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  labs(title = "Distribution of Price per Square Foot")


# Select numeric columns for correlation
numeric_cols <- house_data %>%
  select_if(is.numeric) %>%                     
  mutate_if(is.integer, as.numeric) %>%
  select(-c(h_id, street, zip_code)) # Exclude any non-numeric variables like h_id, street, etc.

# Include the new features in the correlation matrix
numeric_cols <- numeric_cols %>%
  select(log_price, log_house_size, log_land_size, price_per_sqft, house_land_interaction, bath, bed, price)

# Compute correlation matrix
cor_matrix <- cor(numeric_cols, use = "pairwise.complete.obs")

# Plot correlation heatmap
corrplot(cor_matrix, method = "color", tl.cex = 0.6, addCoef.col = "white", tl.col = "black")

#feature engineering

# Adding interaction features to the dataset
house_data <- house_data %>%
  mutate(
    price_per_bed = ifelse(bed > 0, price / bed, NA),   # Avoid division by zero
    price_per_bath = ifelse(bath > 0, price / bath, NA), # Avoid division by zero
    bed_bath_ratio = ifelse(bath > 0, bed / bath, NA),  # Avoid division by zero
    land_to_house_ratio = land_size / house_size        # Land-to-House Size Ratio
  )

# Preview the new features
summary(house_data %>% select(price_per_bed, price_per_bath, bed_bath_ratio, land_to_house_ratio))

# Define a function to remove outliers based on IQR
remove_outliers <- function(data, column) {
  Q1 <- quantile(data[[column]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[column]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  data <- data %>%
    filter(data[[column]] >= lower_bound & data[[column]] <= upper_bound)
  return(data)
}

# Apply the function to remove outliers from 'price' and 'price_per_sqft'
house_data_clean <- house_data %>%
  remove_outliers("price") %>%
  remove_outliers("price_per_sqft")

cat("Dataset after removing outliers:\n")
print(dim(house_data_clean))  # Check the size of the cleaned dataset


# Split the cleaned data into train and test sets
set.seed(123)
train_index <- sample(1:nrow(house_data_clean), size = 0.8 * nrow(house_data_clean))
train_data_clean <- house_data_clean[train_index, ]
test_data_clean <- house_data_clean[-train_index, ]

x_train_clean <- as.matrix(train_data_clean %>% select(house_size, land_size, bath, bed, price_per_sqft, starts_with("state.")))
y_train_clean <- train_data_clean$price

x_test_clean <- as.matrix(test_data_clean %>% select(house_size, land_size, bath, bed, price_per_sqft, starts_with("state.")))
y_test_clean <- test_data_clean$price

# Train the Lasso model with cross-validation
lasso_model_clean <- cv.glmnet(
  x = x_train_clean,
  y = y_train_clean,
  alpha = 1,  # Lasso regression
  standardize = TRUE,
  nfolds = 10
)

# Get the best lambda
best_lambda_clean <- lasso_model_clean$lambda.min
cat("Best Lambda (Clean Data):", best_lambda_clean, "\n")

# Predict on the test set
lasso_predictions_clean <- predict(lasso_model_clean, s = best_lambda_clean, newx = x_test_clean)

# Calculate RMSE and MSE
lasso_rmse_clean <- sqrt(mean((lasso_predictions_clean - y_test_clean)^2))
lasso_mse_clean <- mean((lasso_predictions_clean - y_test_clean)^2)

cat("Lasso Model RMSE (Clean Data):", lasso_rmse_clean, "\n")
cat("Lasso Model MSE (Clean Data):", lasso_mse_clean, "\n")

# Calculate MAE for the cleaned data
lasso_mae_clean <- mean(abs(lasso_predictions_clean - y_test_clean))

cat("Lasso Model MAE (Clean Data):", lasso_mae_clean, "\n")


# Calculate R-squared
lasso_r_squared_clean <- 1 - sum((lasso_predictions_clean - y_test_clean)^2) /
  sum((y_test_clean - mean(y_test_clean))^2)

cat("Lasso Model R-squared (Clean Data):", lasso_r_squared_clean, "\n")

# Define the threshold
threshold <- 200000

# Convert actual and predicted prices into classes
actual_classes <- ifelse(y_test_clean < threshold, "Affordable", "Luxury")
predicted_classes <- ifelse(lasso_predictions_clean < threshold, "Affordable", "Luxury")

# Create a confusion matrix
library(caret)
confusion_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(actual_classes))

# Extract the table from the confusion matrix
conf_matrix_table <- as.table(confusion_matrix$table)

# Plot the confusion matrix as a heatmap
library(ggplot2)
ggplot(data = as.data.frame(conf_matrix_table), aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted Class", y = "Actual Class") +
  theme_minimal()

