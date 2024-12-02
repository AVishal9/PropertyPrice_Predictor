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
                           data_og$zip_code, data_og$bed, data_og$bath, sep = "_")

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
# Outlier Detection and Elimination 
###

# Examining the distribution of non binary numeric variables

# Convert summary to a data frame
summary_df <- as.data.frame.matrix(summary(sampled_data))

# Create a table that shows the summary statistics to evaluate the distribution of each variable.
kable(summary_df, format = "html") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))

# Histogram of numerical non-dummy variables
sampled_data %>% 
  dplyr::select(bath, bed, house_size, land_size, price) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_boxplot()

# The variables house_size, price, land_size seem to have extreme values
# Our target variable price has a long right tail(the mean is higher than the median). The minimum value seems to be $1 which doesn't seem practical
# House sizes also have unrealistically large value, the max value which of 598,697 is uncommon for residential areas. This is either a very rare case of luxury house 
# or a data entry, thus can be removed. Even here the mean > than the median thus indicating a longer right tail.
# Land size also has a large range, with the minimium being 0.00 acres and the maximum being 4.356 billion sq. ft. 
# The mean exceeds the median by a lot pointing towards an extreme upper value.

unique(sampled_data$bath) #Since the data sampling wasn't balanced. 
#There seems to be an extreme entry of 198 baths
# The bed ranges from 1 to 99.
# However, both bed and bath have similar mean and median indicating their distributions are 
# relatively similar 

###
# House_size to price ratio. 
###
#To assess how much square footage can be bought for a dollar
ratio <- sampled_data%>%
  mutate(ratio_hp = house_size/price)

#Summary statistics of this new variable
summary(ratio$ratio_hp)
ratio %>% 
  dplyr::select(ratio_hp) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_boxplot()


#Grouped by state to analyse the trend 
ratio %>%
  dplyr::select(ratio_hp) %>%
  dplyr::group_by(ratio$state) %>%
  summarise(mean_ratio = mean(ratio_hp)) %>%
  arrange(desc(mean_ratio)) %>%
  ungroup()%>%
  print(n = 51)

# Michigan and Texas have extremely high ratios 2.98 and 0.474 respectively before outlier removal
# This might be because they have cheaper reale state prices (More square footage for less price) or it is just a data problem
# Properties with unusually low ratios could indicate overpriced properties or luxury homes.

michigan_data <- sampled_data %>% filter(state == "Michigan")
texas_data <- sampled_data %>% filter(state == "Texas")

# Identifying extreme values given the distributions

#Creating a subset of data with high outlier values
outlier_subset <- subset(sampled_data, select = c(price, house_size, land_size))

# Since around 99% of the observations lie within 3 standard deviations, thus any value that is 
# more than 3 sd above or below the mean will be a potential outlier.
# (mean - 3 * sd) calculates the lower bound.
# (mean + 3 * sd) calculates the upper bound.
#Finding out share of observations that are below the 0.15th percentile.
#The share of observations above 99.85th percentile

# Remove extreme outliers beyond the 98th percentile
remove_extreme <- function(outlier_subset, column) {
  lower_bound <- quantile(outlier_subset[[column]], 0.01, na.rm = TRUE)
  upper_bound <- quantile(outlier_subset[[column]], 0.99, na.rm = TRUE)
  
  outlier_subset %>% filter(outlier_subset[[column]] >= lower_bound & outlier_subset[[column]] <= upper_bound)
}

#removing extreme values from the subset
outlier_subset <- outlier_subset %>%
  remove_extreme("price") %>%
  remove_extreme("house_size") %>%
  remove_extreme("land_size")

# Using Histogram to check the distribution of these variables, there still seems to be a right tail, especially in land size
outlier_subset %>% 
  dplyr::select(house_size, land_size, price) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

# while the mean and median of house_size and price are still greater than median,the mean and median of land_size
# has significant difference 
summary(outlier_subset) 

#Since they are still positively skewed a log transformation might be beneficial while building the model

#Now removing extreme values from the sampled_data 
sampled_data<- sampled_data %>%
  remove_extreme("price") %>%
  remove_extreme("house_size") %>%
  remove_extreme("land_size")

#Using Histogram to check the distribution all the variables in sampled data after outlier elimination
sampled_data %>% 
  dplyr::select(bed, bath, house_size, land_size, price) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

# Inspecting Summary statistics
summary(sampled_data)

###
# One-Hot Encoding of Categorical Variables
###
#As the dataset has high-cardinality categorical variables, it was difficult to one hot encode them, as the computational power
#wasn't sufficient. 

#Summarising unique values of each variable in the daatset
col_no_unique <- data.frame(sampled_data[,-1] %>% summarise(across(everything(), n_distinct)) %>% pivot_longer(everything()))

#Creating dummy variables for categorical variables

# Dummy variables for status column 
status <- unique(sampled_data$status) #Two unique variables "sold" and "for_sale"
status_dummies <- dummy_cols(sampled_data, select_columns = 'status')[,(ncol(sampled_data)+1):(ncol(sampled_data)+length(unique(sampled_data$status)))]
status_dummies <- status_dummies[,-ncol(status_dummies)] #Reducing one column to avoid multicollinearity 

#Dummy Variable for state column 

state <- unique(sampled_data$state) #51 unique values
state_dummies <- dummy_cols(sampled_data, select_columns = 'state')[,(ncol(sampled_data)+1):(ncol(sampled_data)+length(unique(sampled_data$state)))]
state_dummies <- state_dummies[,-ncol(state_dummies)] #Reducing one column to avoid multicollinearity 


#Since city, street, and zip code, brokered_by have extremely high unique values, it is better to frequency encode them to reduce the
#high dimensionality

#Grouping Variables for dimensionality reduction

# Grouping zip code below frequency 100
zip_code_stats <- sampled_data %>%
  group_by(zip_code) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter zip code with occurrences <= 19
zip_stats_less_than_19 <- zip_code_stats %>%
  filter(count <= 19) %>%
  pull(zip_code)

# Replace zip codes with 'other' if they occur <= 10 times
sampled_data$zip_code <- ifelse(sampled_data$zip_code %in% zip_stats_less_than_19 , 'other', sampled_data$zip_code)

# Get the number of unique zip_codes
length(unique(sampled_data$zip_code))

#Grouping cities Below frequency 100
city_stats <- sampled_data %>%
  group_by(city) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter cities with occurrences <= 100
city_stats_less_than_100 <- city_stats %>%
  filter(count <= 100) %>%
  pull(city)

# Replace city names with 'other' if they occur <= 10 times
sampled_data$city <- ifelse(sampled_data$city %in% city_stats_less_than_100, 'other', sampled_data$city)

# Get the number of unique locations
length(unique(sampled_data$city))

# Street below frequency 1
street_stats <- sampled_data %>%
  group_by(street) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter streets with occurrences <= 1
street_stats_less_than_1<- street_stats %>%
  filter(count <= 1) %>%
  pull(street)

# Replace street with 'other' if they occur <= 10 times
sampled_data$street <- ifelse(sampled_data$street %in% street_stats_less_than_1, 'other', sampled_data$street)

# Get the number of unique stret
length(unique(sampled_data$street))

# Grouping Brokered by below frequency 100
broker_stats <- sampled_data %>%
  group_by(brokered_by) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Filter broker with occurrences <= 100
broker_stats_less_than_100<- broker_stats %>%
  filter(count <= 100) %>%
  pull(brokered_by)

# Replace broker with 'other' if they occur <= 10 times
sampled_data$brokered_by <- ifelse(sampled_data$brokered_by %in% broker_stats_less_than_100, 'other', sampled_data$brokered_by)

# Get the number of unique brokers
length(unique(sampled_data$brokered_by))

# Dummy Variable for city column 
city <- unique(sampled_data$city) #6012 unique Variables before grouping.After grouping 13
city_dummies <- dummy_cols(sampled_data, select_columns = 'city')[,(ncol(sampled_data)+1):(ncol(sampled_data)+length(unique(sampled_data$city)))]
city_dummies <- city_dummies[,-ncol(city_dummies)] #Reducing one column to avoid multicollinearity 

#Dummy Variable for brokered_by 
brokered_by <- unique(sampled_data$brokered_by) #13816 unique values. After grouping 5 
broker_dummies <- dummy_cols(sampled_data, select_columns = 'brokered_by')[,(ncol(sampled_data)+1):(ncol(sampled_data)+length(unique(sampled_data$brokered_by)))]
broker_dummies <- broker_dummies[,-ncol(broker_dummies)] 

#Dummy Variable for street
street <- unique(sampled_data$street) #28412 unique values. 8 after grouping 
street_dummies <- dummy_cols(sampled_data, select_columns = 'street')[,(ncol(sampled_data)+1):(ncol(sampled_data)+length(unique(sampled_data$street)))]
steet_dummies <- street_dummies[,-ncol(street_dummies)]  #too large for my laptop to run. We want to keep the analysis state level so we will exclude streets.

#Dummy Variable for zip_code
zip_code <- unique(sampled_data$zip_code) #10602 unique values. 9 after grouping 
zipcode_dummies <- dummy_cols(sampled_data, select_columns = 'zip_code')[,(ncol(sampled_data)+1):(ncol(sampled_data)+length(unique(sampled_data$zip_code)))]
zipcode_dummies <- zipcode_dummies[,-ncol(zipcode_dummies)]


#Removing the categorical columns and attaching the binary variable columns 
data2 <- subset(sampled_data, select = -c(status, state, city, brokered_by, street, zip_code))
data2 <- cbind(data2, state_dummies, status_dummies,city_dummies, broker_dummies,street_dummies,zipcode_dummies)

### I think it is better to not one hot encode categorical variables for tree based algorithms.


