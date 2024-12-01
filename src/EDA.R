####
# Exploratory Data Analysis 
####

# Loading all the necessary libraries 

library(tidyverse) 
library(dplyr)
library(naniar)
library(ggplot2)
library(fastDummies)
library(corrplot)
library(randomForest)

# Defining the file path to the dataset
file_path <- "Final Project/realtorData.csv"

# Read and preview the dataset
data_og <- read.csv(file_path, sep = ",")
View(data_og)
str(data_og) #There are 5 numeric variables, 4 variables of class character and 3 of integer class.


# Dropping the column prev_sold_date; only keeping the latest entries of a particular listing 
#Adding a variable called 'sale frequency'
data_og <- data_og %>%
  group_by(house_size, bed, bath, street, city, state, zip_code, acre_lot)%>%
  mutate(sale_frequency = n()) %>%
  ungroup()

# Arranging by prev_sold_date in descending order
data_og <- data_og %>%
  arrange(desc(prev_sold_date))

# Keeping the latest record for each listing
data_og <- data_og %>%
  distinct(house_size, bed, bath, street, city, state, zip_code, acre_lot, .keep_all = TRUE)

# Dropping the prev_sold_date column
data_og <- data_og %>%
    dplyr::select(-prev_sold_date)

# Making a composite key since there isn't any unique id
data_og$unique_id <- paste(data_og$street, data_og$city, data_og$house_size, 
                           data_og$zip_code, data_og$price, data_og$bed, data_og$bath, sep = "_")

# Checking if there are any duplicates
duplicate_check <- sum(duplicated(data_og$unique_id))

# Removing the duplicates 
data_og <- data_og[!duplicated(data_og$unique_id), ]

#Creating a unique id for the each composite key
data_og$unique_id <- seq_len(nrow(data_og))

###
# Sampling the Data 
###

#Sampling a little more than 10000 as they would reduce after cleaning the data 
set.seed(123)
sampled_data <- data_og[sample(nrow(data_og), 50000), ]

# Only Keeping Relevant US states

#Checking the unique states in the sampled dataset 
unique(sampled_data$state) #We would only be modelling mainland US so territories that are not in US neet to be eliminated

#Eliminating empty state fields
sampled_data <- sampled_data[sampled_data$state != "", ]

#Make a vector that stores all the mainland US states
valid_states <- c("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
                  "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
                  "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
                  "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
                  "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
                  "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
                  "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
                  "Wisconsin", "Wyoming", "District of Columbia")

#Filtering the dataset to only include valid states 
sampled_data <- sampled_data %>%
  filter(state %in% valid_states) 

# To ensure unit consistency converting acres into square feet
# and changing the feature name to land_size

sampled_data <- sampled_data %>% 
  mutate(land_size = acre_lot *43560) %>%
  dplyr::select(-acre_lot)

####################
# Dealing with NAs
#################### 

summary(sampled_data) #Our target variable which is price contains 29 NAs

#Eliminating rows in Price variable that contain NA
sampled_data <- sampled_data[!is.na(sampled_data$price), ]

#Getting a summary of features with missing data 
(colSums(is.na(sampled_data)) / nrow(sampled_data))

#Visualizing the distribution of missing data
gg_miss_var(sampled_data)

# Since we cannot remove columns as they are critical, removing rows with NAs
sampled_data <- sampled_data[!is.na(sampled_data$bed) & !is.na(sampled_data$bath) & 
                            !is.na(sampled_data$house_size) & !is.na(sampled_data$land_size) & 
                            !is.na(sampled_data$brokered_by) & !is.na(sampled_data$zip_code)&
                            !is.na(sampled_data$street), ]
#Now we are left with 30,064 observations

####
#Visualising Data Relationships 
###

#Comparing the variable distribution before removing NAs with after removing NAs
#to ensure there isn't any skewness

#Distribution of bedroom variable
ggplot(data = data_og2, aes(x = bed)) + geom_histogram() + ggtitle("Before Dropping NAs")
ggplot(data = sampled_data, aes(x = bed)) + geom_histogram() + ggtitle("After Dropping NAs")

#Distribution of bathroom variable 
ggplot(data = data_og2, aes(x = bath)) + geom_histogram() + ggtitle("Before Dropping NAs")
ggplot(data = sampled_data, aes(x = bath)) + geom_histogram() + ggtitle("After Dropping NAs")

#Distribution of house size variable 
ggplot(data = data_og2, aes(x = house_size)) + geom_histogram() + ggtitle("Before Dropping NAs")
ggplot(data = sampled_data, aes(x = house_size)) + geom_histogram() + ggtitle("After Dropping NAs")

#Distribution of land size variable 
ggplot(data = data_og2, aes(x = acre_lot)) + geom_histogram() + ggtitle("Before Dropping NAs")
ggplot(data = sampled_data, aes(x = land_size)) + geom_histogram() + ggtitle("After Dropping NAs")


#Checking relationship between numeric variables

#Correlation Matrix of Numerical Data 
cor_mat_data <- sampled_data %>% select_if(., is.numeric) # Dataset of numerical variables to be used for the calculation of correlations
cor_mat_data <- cor_mat_data %>% mutate_if(is.integer, as.numeric) # The calculation of correlations requires all variables to be numeric
str(cor_mat_data) #Only 9 variables included 
cor_mat <- cor(cor_mat_data[,!colnames(cor_mat_data) %in% c("unique_id", "brokered_by","street","zip_code")], use="pairwise.complete.obs") #take out unique_id, and other numerically encoded categorical variables from here

# Finding higher correlations within the correlation matrix of numeric data (i.e., absolute correlation >= 0.95)
high_cor <- data.frame(row=rownames(cor_mat)[row(cor_mat)[upper.tri(cor_mat)]], 
                       col=colnames(cor_mat)[col(cor_mat)[upper.tri(cor_mat)]], 
                       corr=cor_mat[upper.tri(cor_mat)]) # Correlation matrix transformed into vector form
high_cor$corr <- as.numeric(high_cor$corr) # Correlations need to be numeric, not character
order_cor <- order(abs(high_cor$corr), decreasing = TRUE) # Order rows to have the highest absolute values at the top
high_cor <- high_cor[order_cor, ]
high_cor <- high_cor %>% filter(abs(corr) >= 0.95) # Bath-bed have 76% correlation and bed-bath with house_size 23 & 25%. 
                                                   # No variable is perfectly correlated that is has a correlation of 100%

# Making a heat map of correlations between numeric variable 
corrplot(cor_mat, method="color",tl.cex = 0.6, tl.col = "black")


###
#Feature Engineering 
###
