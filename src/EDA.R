####
# Exploratory Data Analysis 
####

# Loading all the necessary libraries 
library(tidyverse)
library(kableExtra)
library(naniar)
library(fastDummies)
library(corrplot)
library(randomForest)


house_data <- readRDS("data_processed/house_data.rds")

View(house_data)
str(house_data) 


####
#Visualising Data Relationships 
###


#Checking relationship between numeric variables

#Correlation Matrix of Numerical Data 
# Dataset of numerical variables to be used for the calculation of correlations
cor_mat_data <- house_data %>% select_if(., is.numeric) 

# The calculation of correlations requires all variables to be numeric
cor_mat_data <- cor_mat_data %>% mutate_if(is.integer, as.numeric) 
str(cor_mat_data) #Only 9 variables included 

#drop unique_id, and other numerically encoded categorical variables
exclude_cols <- c("h_id", "brokered_by", "street", "zip_code")

cor_mat <- cor(
  cor_mat_data[,!colnames(cor_mat_data) %in% exclude_cols], 
  use="pairwise.complete.obs") 

# Identify high correlations (absolute correlation >= 0.95) in the correlation matrix
high_cor <- data.frame(
  row = rownames(cor_mat)[row(cor_mat)[upper.tri(cor_mat)]],  
  col = colnames(cor_mat)[col(cor_mat)[upper.tri(cor_mat)]],  
  corr = as.numeric(cor_mat[upper.tri(cor_mat)]) # Correlation matrix transformed into vector form
) 

high_cor <- high_cor %>%
  arrange(desc(abs(corr))) %>%  # Arrange rows by descending absolute correlation
  filter(abs(corr) >= 0.95)

order_cor <- order(abs(high_cor$corr), decreasing = TRUE) # Order rows to have the highest absolute values at the top
high_cor <- high_cor[order_cor, ]
high_cor <- high_cor %>% filter(abs(corr) >= 0.95) # Bath-bed have 76% correlation and bed-bath with house_size 23 & 25%. 
# No variable is perfectly correlated that is has a correlation of 100%

# Making a heat map of correlations between numeric variable 
corrplot(cor_mat, method="color",tl.cex = 0.6, addCoef.col = "white", tl.col = "black")

###
# Outlier Detection and Elimination 
###

# Examining the distribution of non binary numeric variables
# Convert summary to a data frame
summary_df <- as.data.frame.matrix(summary(house_data))

# Create a table that shows the summary statistics to evaluate the distribution of each variable.
kable(summary_df, format = "html") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))

# Histogram of numerical non-dummy variables
house_data %>% 
  select(bath, bed, house_size, land_size, price) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_boxplot()

####MODIFY COMMENTS
# The variables house_size, price, land_size seem to have extreme values
# Our target variable price has a long right tail(the mean is higher than the median). The minimum value seems to be $1 which doesn't seem practical
# House sizes also have unrealistically large value, the max value which of 598,697 is uncommon for residential areas. This is either a very rare case of luxury house 
# or a data entry, thus can be removed. Even here the mean > than the median thus indicating a longer right tail.
# Land size also has a large range, with the minimium being 0.00 acres and the maximum being 4.356 billion sq. ft. 
# The mean exceeds the median by a lot pointing towards an extreme upper value.

unique(house_data$bath) #Since the data sampling wasn't balanced. 
#There seems to be an extreme entry of 198 baths
# The bed ranges from 1 to 99.
# However, both bed and bath have similar mean and median indicating their distributions are 
# relatively similar 

###
# House_size to price ratio. 
###
#To assess how much square footage can be bought for a thousand dollars
ratio <- house_data%>%
  mutate(ratio_hp = 1000* (house_size/price))

#Summary statistics of this new variable
summary(ratio$ratio_hp)

ratio_filtered <- ratio %>%
  filter(ratio_hp <= quantile(ratio_hp, 0.99, na.rm = TRUE))

ratio_filtered %>%
  ggplot(aes(x = "", y = ratio_hp)) +  
  geom_boxplot() +
  labs(
    title = "Square Footage per 1000 Dollars",
    y = "Square Footage per Dollar",
    x = NULL
  ) +
  theme_minimal()

#Grouped by state to analyse the trend 
state_price <- ratio %>%
  select(ratio_hp) %>%
  group_by(ratio$state) %>%
  summarise(mean_ratio = mean(ratio_hp)) %>%
  arrange(desc(mean_ratio)) %>%
  ungroup()

#Ohio has unusually large mean size per dollar ratio, wrong data entry/outlier

###MODIFY COMMENTS
# Michigan and Texas have extremely high ratios 2.98 and 0.474 respectively before outlier removal
# This might be because they have cheaper reale state prices (More square footage for less price) or it is just a data problem
# Properties with unusually low ratios could indicate overpriced properties or luxury homes.

michigan_data <- ratio_filtered %>% 
  filter(state == "Michigan")

texas_data <- ratio %>%
  filter(state == "Texas")

# Identifying extreme values given the distributions

#Creating a subset of data with high outlier values
outlier_subset <- subset(house_data, select = c(price, house_size, land_size))

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
  select(house_size, land_size, price) %>% 
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
col_no_unique <- data.frame(house_data[,-1] %>% summarise(across(everything(), n_distinct)) %>% pivot_longer(everything()))

#Creating dummy variables for categorical variables

# Dummy variables for status column 
status <- unique(house_data$status) #Two unique variables "sold" and "for_sale"
status_dummies <- dummy_cols(house_data, select_columns = 'status')[,(ncol(house_data)+1):(ncol(house_data)+length(unique(house_data$status)))]
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

###
#Feature Engineering 
###
