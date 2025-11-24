### Project: Spotify User Churn Prediction Using Machine Learning
### Course: DTI5126 – Fundamental of Applied Data Science
### Date: November 18th, 2025

### Group members: 

#Ngoc Khanh Trinh - ID: 300448582 - Tasks: Designed & populated presentation slides. Modified & optimized feature engineering & supervised learning models. Contributed to unsupervised learning models. Led final conclusions. 
#Juan Sebastian Ramirez Acua - ID: 300459414 - Tasks: Led supervised learning models. Contributed to presentation slides & content.
#Ignancya Michelle George - ID: 300460808 - Tasks: Led unsupervised learning models. Contributed to presentation slides & content.
#Udi Bhasin - ID: 300475136 - Tasks: Led unsupervised learning models. Contributed to presentation slides & content.
#Ferdous Pathan - ID: 300513282 - Tasks: Led supervised learning models. Contributed to presentation slides & content.
#Julian Vu Hoang - ID: 300514180 - Tasks: Created outline for presentation slides. Modified & optimized supervised learning models. Contributed to unsupervised learning models. Finalized R code file. 

### Problem Statement:

#User churn represents a critical business challenge for subscription-based streaming platforms such as Spotify. 
#When users are no longer engaged with the platform or cancel subscription plans, 
#the company loses revenue and needs to invest more in customer acquisition, 
#which impacts their profitability. In this project, we will address a real business question: 
#“Can we anticipate churn for Spotify users based on important factors and identify actionable segments to enhance retention?

### Approach:

#We will use various machine learning techniques to identify at-risk users early and identify cancellation drivers. 
#In addition, the model will help to develop retention strategies that will minimize churn. 
#Our measure for success will include strong predictive performance on held-out users, 
#as well as clear actionable insights that product and marketing teams can capitalize on.


### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####

# I. IMPORT PACKAGES & DATASET
install.packages(c("dplyr", "stringr", "tidyr", "forcats", "janitor"))
install.packages(c("readxl","rpart.plot","randomForest","pROC","xgboost"))
install.packages("factoextra")
install.packages("caret")
install.packages("corrplot")
install.packages(c("textclean"))
install.packages("readr")
install.packages("tidyverse")
install.packages("Rtsne")

#Packages for data cleaning & EDA
library(dplyr)
library(stringr)
library(tidyr)
library(janitor)
library(textclean)
library(readxl)
library(forcats)
library(readr)
library(ggplot2)
library(lubridate)
library(scales)
library(tidyverse)
library(gridExtra)

# Packages for Supervised learning 
library(rpart)
library(rpart.plot)
library(randomForest)
library(class)
library(caret)
library(corrplot)
library(pROC)
library(xgboost)

# Packages for Unsupervised learning 
library(cluster)
library(factoextra)
library(ggplot2)
library(Rtsne)


#Load data from Github
spotify_dirty <- read.csv("https://raw.githubusercontent.com/jvuhoang/DTI5126/main/spotify_churn_dataset_dirty_binary_bool.csv", header = TRUE, stringsAsFactors = TRUE)

head(spotify_dirty)


### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####

# II. DATA PREPROCESSING

# Cleaning column names
spotify_clean <- spotify_dirty %>% janitor::clean_names()

# Inspecting the dataset
glimpse(spotify_clean)
summary(spotify_clean)
head(spotify_clean)

# Checking unique values in key categorical columns
unique(spotify_clean$gender)
unique(spotify_clean$country)
unique(spotify_clean$subscription_type)
unique(spotify_clean$device_type)

# Changing the case
spotify_clean <- spotify_clean %>% mutate(
  gender=str_to_title(gender),
  country = str_to_title(country),
  subscription_type = str_to_title(subscription_type),
  device_type = str_to_title(device_type))

unique(spotify_clean$gender)
unique(spotify_clean$country)
unique(spotify_clean$subscription_type)
unique(spotify_clean$device_type)

# Manual Mapping Dictionaries
gender_map <- c("M" = "Male", "F"="Female", "O" = "Other")
device_map <- c("Moble" = "Mobile", "Moblie" = "Mobile", "Mobliee" = "Mobile",
                "Desktp" = "Desktop", "Desk" = "Desktop", "Wb" = "Web")
subscription_map <- c("Premum" = "Premium", "Premiun" = "Premium", "F Ree" = "Free",
                      "F ree" = "Free", "Famly" = "Family", "Famil" = "Family", "Studnt" = "Student", "Studntt" = "Student")
country_map <- c("Usa"="USA", "Us" = "USA", "U.s.a." = "USA", "U.s.a" = "USA", "U.S.A." = "USA", "U.s." = "USA", "U.S." = "USA", "United States" = "USA", "Uk" = "United
Kingdom", "United \nKingdom" = "United Kingdom", "United\nKingdom" =  "United Kingdom", "U.k." = "United Kingdom", "U.K." = "United Kingdom", "Britain" = "United Kingdom", "De" =
                   "Germany", "Ger" = "Germany","Pk" = "Pakistan", "In" = "India", "Can" = "Canada", "Ca" = "Canada", "Au"="Australia", "Aus" = "Australia", "Fr" = "France")


spotify_clean <- spotify_clean %>% mutate(
  gender = recode(gender, !!!gender_map),
  device_type = recode(device_type, !!!device_map),
  subscription_type = recode(subscription_type, !!!subscription_map),
  country = recode(country, !!!country_map))

spotify_clean$country <- gsub("\\\\n|\\n", " ", spotify_clean$country) #Fix United Kingdom values which contains a new line

unique(spotify_clean$gender)
unique(spotify_clean$country)
unique(spotify_clean$subscription_type)
unique(spotify_clean$device_type)

#OUTLIER DETECTION AND MISSING VALUES
#CLEANING AGE COLUMN
# Age Cleaning
summary(spotify_clean$age)

# Viewing first few entries
head(spotify_clean$age)

# Looking at the unique values
unique(spotify_clean$age)

# Converting age to numeric data type
spotify_clean <- spotify_clean %>% mutate(
  age = str_trim(age),
  age = na_if(age, "N/A"),
  age = na_if(age, " "),
  age = na_if(age, "999"),
  age = na_if(age, "##"),
  age = na_if(age, "unknown"),
  age = as.numeric(age))

# Handling outliers in age
spotify_clean <- spotify_clean %>% mutate(
  age = ifelse(age<10 | age > 100, NA, age))

# Detecting statistical outliers
Q1 <- quantile(spotify_clean$age, 0.25, na.rm = TRUE)
Q3 <- quantile(spotify_clean$age, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
cat("Lower Bound:", lower_bound, "\nUpper Bound:", upper_bound, "\n")

# Treating outliers
# Capping to the IQR boundaries
spotify_clean <- spotify_clean %>%
  mutate(
    age = ifelse(age < lower_bound, lower_bound,
                 ifelse(age > upper_bound, upper_bound, age)))

# Removing missing values from age
median_age <- median(spotify_clean$age, na.rm=TRUE)
spotify_clean <- spotify_clean %>% mutate(age=ifelse(is.na(age),median_age,age))

# Checking if it is cleaned properly
unique(spotify_clean$age)

#CLEANING LISTENING TIME COLUMN
# Listening time Cleaning
summary(spotify_clean$listening_time)

# Viewing first few entries
head(spotify_clean$listening_time)

# Looking at the unique values
unique(spotify_clean$listening_time)

# Converting listening time to numeric data type
spotify_clean <- spotify_clean %>%
  mutate(
    listening_time = str_trim(listening_time),
    listening_time = str_to_lower(listening_time),
    listening_time = na_if(listening_time, ""),
    listening_time = na_if(listening_time, "n/a"),
    listening_time = na_if(listening_time, "unknown"),
    listening_time = as.numeric(listening_time))

# Checking structure to confirm its numeric
str(spotify_clean$listening_time)

# Handling outliers in listening time
spotify_clean <- spotify_clean %>%
  mutate(
    listening_time = ifelse(listening_time < 0 | listening_time > 1440, NA, listening_time))

# Detecting statistical outliers
Q1 <- quantile(spotify_clean$listening_time, 0.25, na.rm = TRUE)
Q3 <- quantile(spotify_clean$listening_time, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
cat("Lower Bound:", lower_bound, "\nUpper Bound:", upper_bound, "\n")

# Treating outliers
# Capping to the IQR boundaries
spotify_clean <- spotify_clean %>%
  mutate(
    listening_time_outlier = ifelse(listening_time < lower_bound | listening_time > upper_bound, TRUE, FALSE),
    listening_time = ifelse(listening_time < lower_bound, lower_bound,
                            ifelse(listening_time > upper_bound, upper_bound, listening_time)))

# Treating missing values from listening time
median_listening_time <- median(spotify_clean$listening_time, na.rm = TRUE)
spotify_clean <- spotify_clean %>%
  mutate(listening_time = ifelse(is.na(listening_time), median_listening_time, listening_time))

# Checking if it is cleaned properly
unique(spotify_clean$listening_time)

#CLEANING SONGS PLAYED PER DAY COLUMN
# Songs played per day Cleaning
summary(spotify_clean$songs_played_per_day)
# Viewing first few entries
head(spotify_clean$songs_played_per_day)

# Looking at the unique values
unique(spotify_clean$songs_played_per_day)

# Converting this column to numeric data type
spotify_clean <- spotify_clean %>%
  mutate(
    songs_played_per_day = str_trim(songs_played_per_day),     # removing spaces
    songs_played_per_day = str_to_lower(songs_played_per_day), # normalizing case
    
    # handle non-numeric words explicitly
    songs_played_per_day = case_when(
      songs_played_per_day %in% c("many", "a lot", "lots", "several") ~ "100",  # assigned a reasonable numeric estimate
      songs_played_per_day %in% c("none", "zero") ~ "0",
      TRUE ~ songs_played_per_day
    ),
    
    # replace missing and placeholder text with NA
    songs_played_per_day = na_if(songs_played_per_day, ""),
    songs_played_per_day = na_if(songs_played_per_day, "n/a"),
    songs_played_per_day = na_if(songs_played_per_day, "unknown"),
    
    # finally convert to numeric
    songs_played_per_day = as.numeric(songs_played_per_day))

# Checking structure to confirm its numeric
str(spotify_clean$listening_time)

# Handling outliers in songs played
spotify_clean <- spotify_clean %>%
  mutate(
    songs_played_per_day = ifelse(songs_played_per_day < 0 | songs_played_per_day > 300, NA, songs_played_per_day))

# Detecting statistical outliers
Q1 <- quantile(spotify_clean$songs_played_per_day, 0.25, na.rm = TRUE)
Q3 <- quantile(spotify_clean$songs_played_per_day, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
cat("Lower Bound:", lower_bound, "\nUpper Bound:", upper_bound, "\n")

# Treating outliers
# Capping to the IQR boundaries
spotify_clean <- spotify_clean %>%
  mutate(
    songs_played_outlier = ifelse(songs_played_per_day < lower_bound | songs_played_per_day > upper_bound, TRUE, FALSE),
    songs_played_per_day = ifelse(songs_played_per_day < lower_bound, lower_bound,
                                  ifelse(songs_played_per_day > upper_bound, upper_bound, songs_played_per_day)))

# Treating missing values from songs played column
median_songs <- median(spotify_clean$songs_played_per_day, na.rm = TRUE)
spotify_clean <- spotify_clean %>%
  mutate(songs_played_per_day = ifelse(is.na(songs_played_per_day), median_songs, songs_played_per_day))

# Checking if it is cleaned properly
unique(spotify_clean$songs_played_per_day)

# skip_rate Cleaning
summary(spotify_clean$skip_rate)

# Viewing first few entries
head(spotify_clean$skip_rate)

# Looking at the unique values
unique(spotify_clean$skip_rate)

# Convert the factor to character to work with the text.
spotify_clean$skip_rate_char <- as.character(spotify_clean$skip_rate)

# Find all unique non-numeric strings to target for cleaning.
# We will check the levels for anything that isn't a simple number (0-9, decimal).
cat("--- Non-Numeric Levels to Clean ---\n")
sort(unique(spotify_clean$skip_rate_char[!grepl("^[0-9.]+$", spotify_clean$skip_rate_char)]))

# Corrected Cleaning Code
spotify_clean <- spotify_clean %>%
  mutate( # CLEAN THE TEXT AND CONVERT TO NUMERIC
    skip_rate_cleaned = str_replace_all(skip_rate_char,
                                        pattern = c("##" = "",
                                                    "-" = "",
                                                    ":" = "",
                                                    "high" = "")),
    
    # Convert to numeric (this turns the values like '10' and '2' and remaining text into NA)
    skip_rate = as.numeric(skip_rate_cleaned),
    
    # OUTLIER HANDLING
    # Convert values outside the [0, 1] range into NA
    skip_rate = ifelse(skip_rate < 0 | skip_rate > 1, NA_real_, skip_rate)
  )

# IMPUTE MISSING VALUES
median_skip <- median(spotify_clean$skip_rate, na.rm = TRUE)

spotify_clean <- spotify_clean %>%
  mutate(
    skip_rate = ifelse(is.na(skip_rate), median_skip, skip_rate)
  )

# Verification
cat("--- Final skip_rate Summary After Correction ---\n")
summary(spotify_clean$skip_rate)

# Checking if it is cleaned properly
unique(spotify_clean$skip_rate)
table(spotify_clean$skip_rate)

# ads_listened_per_week Cleaning
summary(spotify_clean$ads_listened_per_week)

# Viewing first few entries
head(spotify_clean$ads_listened_per_week)

# Looking at the unique values
unique(spotify_clean$ads_listened_per_week)

# Converting this column to numeric data type
spotify_clean <- spotify_clean %>%
  mutate(
    # handle non-numeric words explicitly
    ads_listened_per_week = case_when(
      ads_listened_per_week %in% c("many", "a lot", "lots", "several") ~ "50",
      ads_listened_per_week %in% c("none", "zero") ~ "0",
      TRUE ~ ads_listened_per_week
    ),
    
    # replace missing and placeholder text with NA
    ads_listened_per_week = na_if(ads_listened_per_week, ""),
    ads_listened_per_week = na_if(ads_listened_per_week, "##"),
    ads_listened_per_week = na_if(ads_listened_per_week, "unknown"),
    
    # finally convert to numeric
    ads_listened_per_week = as.numeric(ads_listened_per_week))

# Checking structure to confirm its numeric
str(spotify_clean$ads_listened_per_week)

# Treating missing values in ads_listened_per_week
median_ads <- median(spotify_clean$ads_listened_per_week, na.rm = TRUE)
spotify_clean <- spotify_clean %>%
  mutate(ads_listened_per_week = ifelse(is.na(ads_listened_per_week), median_ads, ads_listened_per_week))

# Checking if it is cleaned properly
unique(spotify_clean$ads_listened_per_week)

#CLEANING OFFLINE LISTENING COLUMN
# offline_listening Cleaning
summary(spotify_clean$offline_listening)

# Viewing first few entries
head(spotify_clean$offline_listening)

# Looking at the unique values
unique(spotify_clean$offline_listening)

# Checking for class imbalance
table(spotify_clean$offline_listening)
prop.table(table(spotify_clean$offline_listening)) * 100

# Finding mode
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

mode_value <- get_mode(spotify_clean$offline_listening)
cat("Mode of offline_listening is:", mode_value, "\n")

# Treating the missing values
spotify_clean <- spotify_clean %>%
  mutate(
    offline_listening = ifelse(is.na(offline_listening), mode_value, offline_listening),
    offline_listening = as.logical(offline_listening))

# Checking if it is cleaned properly
unique(spotify_clean$offline_listening)

#CLEANING IS CHURNED COLUMN
# is_churned Cleaning
summary(spotify_clean$is_churned)

# Converting this column to boolean type from character
spotify_clean <- spotify_clean %>%
  mutate(offline_listening = as.logical(as.numeric(offline_listening)))

# Viewing first few entries
head(spotify_clean$is_churned)

# Looking at the unique values
unique(spotify_clean$is_churned)

#Since we have Yes and No we replace that with 1 and 0 respectively.

spotify_clean <- spotify_clean %>%
  mutate(
    # Converting to lowercase
    is_churned = tolower(as.character(is_churned)),
    
    # Mapping Yes/No to 1/0
    is_churned = case_when(
      is_churned %in% c("yes", "1") ~ 1,
      is_churned %in% c("no", "0")  ~ 0,
      TRUE ~ NA_real_   # treat anything else as missing
    ),
    
    # Ensure numeric type
    is_churned = as.numeric(is_churned))

# Checking results
unique(spotify_clean$is_churned)

# Treating the missing values
# Finding the mode
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

mode_churn <- get_mode(spotify_clean$is_churned)
cat("Mode of is_churned:", mode_churn, "\n")

# Replace missing values with mode
spotify_clean <- spotify_clean %>%
  mutate(
    is_churned = ifelse(is.na(is_churned), mode_churn, is_churned))

# Checking the changes made
unique(spotify_clean$is_churned)

#CLEANING GENDER COLUMN

# gender column Cleaning
summary(spotify_clean$gender)

# Viewing first few entries
head(spotify_clean$gender)

# Looking at the unique values
unique(spotify_clean$gender)

spotify_clean <- spotify_clean %>%
  mutate(
    # Trim spaces in gender values
    gender = trimws(gender))

# Looking at the unique values
unique(spotify_clean$gender)

spotify_clean <- spotify_clean %>%
  mutate(
    # Replacing invalid entries (???, Unknown, blank) with NA
    gender = ifelse(
      gender %in% c("???", "Unknown",''), NA, gender))

# Checking the changes made
unique(spotify_clean$gender)

# Treating the missing values
# Finding the mode
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

mode_churn <- get_mode(spotify_clean$gender)
cat("Mode of gender:", mode_churn, "\n")

# Replace missing values with mode
spotify_clean <- spotify_clean %>%
  mutate(
    gender = ifelse(is.na(gender), mode_churn, gender))

# Checking the changes made
unique(spotify_clean$gender)

#CLEANING COUNTRY COLUMN
# country column Cleaning
summary(spotify_clean$country)

# Viewing first few entries
head(spotify_clean$country)

# Looking at the unique values
unique(spotify_clean$country)

spotify_clean <- spotify_clean %>%
  mutate(
    # Replacing invalid entries (???, blank) with NA
    country = ifelse(
      country %in% c("???",""), NA, country))

# Checking the changes made
unique(spotify_clean$country)

# Treating the missing values
# Finding the mode
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

mode_churn <- get_mode(spotify_clean$country)
cat("Mode of country:", mode_churn, "\n")

# Replace missing values with mode
spotify_clean <- spotify_clean %>%
  mutate(
    country = ifelse(is.na(country), mode_churn, country))

# Checking the changes made
unique(spotify_clean$country)

# subscription_type column Cleaning
summary(spotify_clean$subscription_type)

# Viewing first few entries
head(spotify_clean$subscription_type)

# Looking at the unique values
unique(spotify_clean$subscription_type)

spotify_clean <- spotify_clean %>%
  mutate(
    # Trim spaces in subscription_type values
    subscription_type = trimws(subscription_type))

spotify_clean <- spotify_clean %>%
  mutate(
    # Replacing invalid entries (???, blank) with NA
    subscription_type = ifelse(
      subscription_type%in% c("???",""), NA, subscription_type))

# Treating the missing values
# Finding the mode
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

mode_churn <- get_mode(spotify_clean$subscription_type)
cat("Mode of subscription_type:", mode_churn, "\n")

# Replace missing values with mode
spotify_clean <- spotify_clean %>%
  mutate(
    subscription_type = ifelse(is.na(subscription_type), mode_churn, subscription_type))

# Checking the changes made
unique(spotify_clean$subscription_type)

#CLEANING DEVICE TYPE COLUMN

# device_type column Cleaning
summary(spotify_clean$device_type)

# Viewing first few entries
head(spotify_clean$device_type)

# Looking at the unique values
unique(spotify_clean$device_type)

spotify_clean <- spotify_clean %>%
  mutate(
    # Trim spaces in subscription_type values
    device_type = trimws(device_type))

spotify_clean <- spotify_clean %>%
  mutate(
    # Replacing invalid entries (Unknown, blank) with NA
    device_type = ifelse(
      device_type %in% c("Unknown",""), NA, device_type))

# Checking the changes made
unique(spotify_clean$device_type)
table(spotify_clean$device_type)

# Treating the missing values
# Finding the mode
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

mode_churn <- get_mode(spotify_clean$device_type)
cat("Mode of device_type:", mode_churn, "\n")

# Replace missing values with mode
spotify_clean <- spotify_clean %>%
  mutate(
    device_type = ifelse(is.na(device_type), mode_churn, device_type))

# Checking the changes made
unique(spotify_clean$device_type)

#DUPLICATES DETECTION AND REMOVAL

# Checking for exact duplicates
# Counting total duplicate rows
sum(duplicated(spotify_clean))

# Viewing duplicate rows (if any)
spotify_clean[duplicated(spotify_clean), ]

#So no complete duplicate rows in our dataset.

# Checking for near duplicates using user_id
dup_users <- spotify_clean %>%
  group_by(user_id) %>%
  filter(n() > 1) %>%
  arrange(user_id)

# Viewing them
dup_users

# Final check
sum(duplicated(spotify_clean))

#Hence, no duplicates of any sort are there in our dataset.

#FEATURE ENGINEERING & TRANSFORMATION (creates new, potentially more predictive 
#features and transforms existing ones to simplify analysis.)

#Age Group
# Age Binning: Convert continuous 'age' into categorical groups

spotify_new <- spotify_clean %>%
  mutate(
    
    age_group = case_when(
      age < 18 ~ "Teenager",
      age >= 18 & age < 25 ~ "Young Adult",
      age >= 25 & age < 35 ~ "Middle Adult",
      age >= 35 ~ "Senior Adult",
      TRUE ~ "Unknown" # Handle any potential NA/missing ages
    )
  )

#Listening time (Hour) and Ad sensitive
# Listening Time Conversion: Convert minutes to hours
# and Ad Sensitivity: Combine ads listened with subscription type

spotify_new <- spotify_new %>%
  mutate(
    listening_time_hours = listening_time / 60,
    ad_sensitive = case_when(
      subscription_type == "Free" & ads_listened_per_week > 0 ~ "High Ad Exposure",
      subscription_type == "Premium" ~ "No Ad Exposure",
      TRUE ~ "Low/Unknown Exposure"
    )
  )

#Check new features

cat("\n\n Structure of New DataFrame (spotify_new) \n")
print(str(spotify_new))

cat("\n\n Distribution of New Age Group Feature \n")
print(table(spotify_new$age_group))

cat("\n\n Distribution of New Ad Sensitive Feature \n")
print(table(spotify_new$ad_sensitive))

cat("\n\n Summary of New Listening Time (in Hours) \n")
print(summary(spotify_new$listening_time_hours))






### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####

#III. EXPLORATORY DATA ANALYSIS

#. Univariate Analysis (Individual Variables)

#A. NUMERIC VARIABLES
#Overview of all numeric variables
# Calculate Summary Statistics for all Numeric Features
# This gives Mean, Median, Min, Max, Q1, Q3, and NA count for each.
spotify_clean %>%
  select(age, listening_time, songs_played_per_day, skip_rate, ads_listened_per_week) %>%
  summary()

#Age
# Age Distribution
ggplot(spotify_clean, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "#1DB954", color = "black") +
  labs(
    title = "Distribution of User Age",
    x = "Age (Years)",
    y = "Count"
  ) +
  theme_minimal()

#Songs Played Per Day
# Songs Played Per Day
ggplot(spotify_clean, aes(x = songs_played_per_day)) +
  geom_histogram(binwidth = 10, fill = "#1DB954", color = "black") +
  labs(
    title = "Distribution of Songs Played Per Day",
    x = "Songs Played Per Day",
    y = "Count"
  ) +
  theme_minimal()

#Listening time
# Listening Time Outliers
ggplot(spotify_clean, aes(y = listening_time)) +
  geom_boxplot(fill = "#1DB954") +
  labs(
    title = "Box Plot of User Listening Time",
    y = "Listening Time (e.g., Minutes/Week)"
  ) +
  theme_minimal()

# Skip Rate
# Density Plot for Skip Rate
ggplot(spotify_clean, aes(x = skip_rate)) +
  geom_density(fill = "#1DB954", color = "black", alpha = 0.6) +
  labs(
    title = "Distribution of Skip Rate",
    x = "Skip Rate (Proportion)",
    y = "Density"
  ) +
  scale_x_continuous(labels = scales::percent) + # Format x-axis as percentage
  theme_minimal()

#Ads Listened Per Week
# Ads Listened Per Week
ggplot(spotify_clean, aes(y = ads_listened_per_week)) +
  geom_boxplot(fill = "#1DB954", color = "black", width = 0.5) +
  labs(
    title = "Box Plot of Ads Listened Per Week",
    y = "Ads Listened Per Week",
    x = ""
  ) +
  theme_minimal()

# The distribution of the free-tier behavior, excluding the zeros.
spotify_free_users <- spotify_clean[spotify_clean$ads_listened_per_week > 0, ]

if (nrow(spotify_free_users) > 0) {
  ggplot(spotify_free_users, aes(y = ads_listened_per_week)) +
    geom_boxplot(fill = "#1DB954", color = "black", width = 0.5) +
    labs(
      title = "Box Plot of Ads Listened Per Week (Excluding Zero-Ad Users)",
      y = "Ads Listened Per Week",
      x = ""
    ) +
    theme_minimal() +
    theme(axis.text.x = element_blank())
}

#B. ANALYSIS OF CATEGORICAL & BINARY VARIABLES

# Summary Statistics for all Categorical & Binary features
spotify_clean %>%
  select(subscription_type, device_type, gender, country, is_churned, offline_listening) %>%
  summary()

#Subscription Type
# Frequency and Proportion Table for Subscription Type
subscription <- spotify_clean %>%
  count(subscription_type, sort = TRUE) %>%
  mutate (percentage = n/ sum(n)*100)
subscription

#User count by Subscription Type
ggplot(spotify_clean, aes(x = subscription_type, fill = subscription_type)) +
  geom_bar() +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "User Count by Subscription Type", x = "Type", y = "Count") +
  theme_minimal()

# Songs Played Per Day by Subscription Type

ggplot(spotify_clean, aes(x = subscription_type, y = songs_played_per_day, fill = subscription_type)) +
  geom_boxplot() +
  labs(title = "Songs Played Per Day by Subscription Type",
       x = "Subscription Type", y = "Songs Played Per Day") +
  theme(legend.position = "none")

#Device Type
#Frequency and Proportion Table for Device Type
# Device type distribution
device <- spotify_clean %>%
  count(device_type, sort = TRUE) %>%
  mutate (percentage = n/ sum(n)*100)
device

#User count by Device Type
ggplot(spotify_clean, aes(y = device_type, fill = device_type)) +
  geom_bar() +
  geom_text(stat = 'count', aes(label = after_stat(count)), hjust = 1.5) +
  labs(title = "User Count by Device Type", x = "Count", y = "Type") +
  theme_minimal()

#Gender and Offline Listening
# Frequency Table for Gender and Offline Listening
cat("\n\n Gender and Offline Listening Counts \n")
print(table(spotify_clean$gender))
print(table(spotify_clean$offline_listening))

#Gender
# Visualization: Bar Chart for Gender
ggplot(spotify_clean, aes(x = gender)) +
  geom_bar(fill = "#1DB954", color = "black") +
  labs(
    title = "Distribution of User Gender",
    x = "Gender",
    y = "Count"
  ) +
  theme_minimal()

#Offline Listening
# Visualization: Bar Chart for Offline Listening Status
ggplot(spotify_clean, aes(x = offline_listening)) +
  geom_bar(fill = "#1DB954", color = "black") +
  labs(
    title = "Offline Listening Status",
    x = "Listens Offline (TRUE/FALSE)",
    y = "Count"
  ) +
  theme_minimal()

#Country
# User Activity by Country

country_engagement <- spotify_clean %>%
  group_by(country) %>%
  summarise(
    avg_listening_time = mean(listening_time, na.rm = TRUE),user_count = n()
  ) %>%
  arrange(desc(avg_listening_time))

country_engagement

ggplot(country_engagement, aes(x = reorder(country, avg_listening_time), y = avg_listening_time, fill = country)) +
  geom_col() +
  coord_flip() +  # makes it easier to read country names
  labs(
    title = "Average Listening Time by Country",
    x = "Country",
    y = "Average Listening Time (minutes or hours)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

#Churn rate

# Distribution of churn_rate
n_churned <- spotify_clean %>% count(is_churned)
n_churned
distribution_churned <- n_churned %>% mutate(percentage = n / sum (n)*100)
distribution_churned

# Churn Status summary

ggplot(spotify_clean, aes(x = factor(is_churned, levels = c(0, 1), labels = c("Not Churned", "Churned")))) +
  geom_bar(fill = "#1DB954", color = "black") +
  labs(
    title = "User Churn Status (Churn Rate)",
    x = "Churn Status",
    y = "Count"
  ) +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +
  theme_minimal()

#2. Bivariate and Multivariate Analysis
#(focuses on finding relationships and trends between the features with target variable, is_churned)

#Correlation between numerical values and target variable

cor(spotify_clean[, c("listening_time", "songs_played_per_day", "skip_rate", "ads_listened_per_week","is_churned")] %>%
      mutate(skip_rate = as.numeric(skip_rate),
             is_churned = as.numeric(as.character(is_churned))))

#A. Numeric Features vs. Churn

# Average Metrics by Churn Status -  crucial for identifying which behaviors (metrics) are predictive of churn.
spotify_clean %>%
  group_by(is_churned) %>%
  summarise(
    Avg_Listening_Time = mean(listening_time, na.rm = TRUE),
    Avg_Songs_Per_Day = mean(songs_played_per_day, na.rm = TRUE),
    Avg_Skip_Rate = mean(skip_rate, na.rm = TRUE),
    Avg_Ads_Per_Week = mean(ads_listened_per_week, na.rm = TRUE),
    Median_Age = median(age, na.rm = TRUE)
  )

#Age vs Churn rate
# Age vs. Churn Status (Box Plot)
ggplot(spotify_clean, aes(
  x = factor(is_churned),
  y = age,
  fill = factor(is_churned)
)) +
  geom_boxplot(color = "black", alpha = 0.8) +
  scale_fill_manual(values = c("0" = "#1DB954", "1" = "#FF4B4B")) +
  labs(
    title = "Age by Churn Status",
    x = "Churn Status",
    y = "Age"
  ) +
  theme_minimal()

#Listening Time vs. Churn
# Listening Time vs. Churn (Box Plot)
ggplot(spotify_clean, aes(x = as.factor(is_churned), y = listening_time, fill = as.factor(is_churned))) +
  geom_boxplot() +
  scale_fill_manual(values = c("0" = "#1DB954", "1" = "#FF4B4B")) + # Spotify Green and Red
  labs(
    title = "Listening Time Distribution by Churn Status",
    x = "Churn Status",
    y = "Listening Time",
    fill = "Churn Status"
  ) +
  theme_minimal()

#Songs Played Per Day vs. Churn Rate
# Songs Played Per Day vs. Churn Status (Box Plot)
ggplot(spotify_clean, aes(
  x = factor(is_churned),
  y = songs_played_per_day,
  fill = factor(is_churned)
)) +
  geom_boxplot(color = "black", alpha = 0.8) +
  scale_fill_manual(values = c("0" = "#1DB954", "1" = "#FF4B4B")) +
  labs(
    title = "Songs Played Per Day by Churn Status",
    x = "Churn Status",
    y = "Songs Played Per Day"
  ) +
  theme_minimal()

#Skip Rate vs. Churn
# Skip Rate vs. Churn Status (Box Plot)
ggplot(spotify_clean, aes(
  x = factor(is_churned),
  y = skip_rate,
  fill = factor(is_churned)
)) +
  geom_boxplot(color = "black", alpha = 0.8) +
  scale_fill_manual(values = c("0" = "#1DB954", "1" = "#FF4B4B")) +
  labs(
    title = "Skip Rate by Churn Status",
    x = "Churn Status",
    y = "Skip Rate (Proportion)"
  ) +
  scale_y_continuous(labels = scales::percent) + # Format y-axis as percentage
  theme_minimal()

#Listened Per Week vs. Churn
# Listened Per Week vs. Churn Status (Box Plot)
ggplot(spotify_clean, aes(
  x = factor(is_churned),
  y = ads_listened_per_week,
  fill = factor(is_churned)
)) +
  geom_boxplot(color = "black", alpha = 0.8) +
  scale_fill_manual(values = c("0" = "#1DB954", "1" = "#FF4B4B")) +
  labs(
    title = "Ads Listened Per Week by Churn Status",
    x = "Churn Status",
    y = "Ads Listened Per Week"
  ) +
  theme_minimal()

#B. Categorical Features vs. Churn

#Gender vs Churn
#User Churn rate by Gender

ggplot(spotify_clean, aes(x = gender, fill = factor(is_churned))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Proportion of Churned vs Active Users by Gender",
       x = "Gender", y = "Percentage", fill = "Churned (1 = Yes)") +
  scale_fill_manual(values = c("0" = "#2b8cbe", "1" = "#f03b20"))


# Churn Rate by Subscription Type
cat("\n\n--- Churn Rate by Subscription Type ---\n")

spotify_clean %>%
  group_by(subscription_type) %>%
  summarise(
    Total_Users = n(),
    Churned_Users = sum(is_churned == 1),
    Churn_Rate_Pct = round(Churned_Users / Total_Users * 100, 2)
  ) %>%
  arrange(desc(Churn_Rate_Pct)) %>%
  print()

#Churn Proportion by Subscription Type
ggplot(spotify_clean, aes(x = subscription_type, fill = factor(is_churned))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Churn Proportion by Subscription Type",
       x = "Subscription Type",
       y = "Proportion of Users",
       fill = "Churned (1 = Yes)") +
  scale_fill_manual(values = c("0" = "#2b8cbe", "1" = "#f03b20"))


# Churn Rate by Device Type
spotify_clean %>%
  group_by(device_type) %>%
  summarise(
    Total_Users = n(),
    Churned_Users = sum(is_churned == 1),
    Churn_Rate_Pct = round(Churned_Users / Total_Users * 100, 2)
  ) %>%
  arrange(desc(Churn_Rate_Pct)) %>%
  print()

# Churn Proportion by Device Type

ggplot(spotify_clean, aes(x = device_type, fill = as.factor(is_churned))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  labs(
    title = "Proportion of Churn by Device Type",
    x = "Device Type",
    y = "Proportion of Users",
    fill = "Churned (1 = Yes)") +
  scale_fill_manual(values = c("0" = "#2b8cbe", "1" = "#f03b20"))

#Country vs Churn
# Churn Rate country
country_group <- spotify_clean %>%
  group_by(country) %>%
  summarise(
    Total_Users = n(),
    Churned_Users = sum(is_churned == 1),
    Churn_Rate_Pct = round(Churned_Users / Total_Users * 100, 2)
  ) %>%
  arrange(desc(Churn_Rate_Pct)) %>%
  print()

# Churn Proportion by Country
ggplot(country_group, aes(x = reorder(country, Churn_Rate_Pct),
                          y = Churn_Rate_Pct,
                          fill = Churn_Rate_Pct)) +
  geom_col() + # Use geom_col for pre-calculated heights
  scale_fill_gradient(low = "#1DB954", high = "#FF4B4B") +
  coord_flip() +
  labs(
    title = "Churn Rate by Country (Top 10)",
    x = "Country",
    y = "Churn Rate (%)",
    fill = "Churn Rate (%)"
  ) +
  theme_minimal() + # Add the percentage labels directly to the bars
  geom_text(aes(label = paste0(Churn_Rate_Pct, "%")),
            hjust = -0.1, size = 3) +
  ylim(0, max(country_group$Churn_Rate_Pct) * 1.2) # Set limits to make space for labels

#Age Group vs Churn rate
# Churn rate (percentage of churners) for age group.
spotify_new %>%
  group_by(age_group) %>%
  summarise(
    Total_Users = n(),
    Churned_Users = sum(is_churned == 1),
    Churn_Rate_Pct = round(Churned_Users / Total_Users * 100, 2)
  ) %>%
  arrange(desc(Churn_Rate_Pct)) %>%
  print()

# Age group vs. Churn Status (Box Plot)
ggplot(spotify_new, aes(x = age_group, fill = factor(is_churned))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Proportion of Churned vs Active Users by Age Group",
       x = "Age Group", y = "Percentage", fill = "Churned (1 = Yes)") +
  scale_fill_manual(values = c("0" = "#2b8cbe", "1" = "#f03b20"))


#Listening time (Hour) and Ad sensitive vs. Churn Status
# Listening Time (hours) vs. Churn status (Box plot)
ggplot(spotify_new, aes(x = as.factor(is_churned), y = listening_time_hours, fill = as.factor(is_churned))) +
  geom_boxplot() +
  scale_fill_manual(values = c("0" = "#1DB954", "1" = "#FF4B4B")) +
  labs(
    title = "Listening Time (hours) Distribution by Churn Status",
    x = "Churn Status (0: Stayed, 1: Churned)",
    y = "Listening Time (hours)",
    fill = "Churn Status"
  ) +
  theme_minimal()

# Churn rate (percentage of churners) for ads sensitive.
spotify_new %>%
  group_by(ad_sensitive) %>%
  summarise(
    Total_Users = n(),
    Churned_Users = sum(is_churned == 1),
    Churn_Rate_Pct = round(Churned_Users / Total_Users * 100, 2)
  ) %>%
  arrange(desc(Churn_Rate_Pct)) %>%
  print()

# Ads Sensitive vs. Churn Status
ggplot(spotify_new, aes(x = ad_sensitive, fill = factor(is_churned))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Proportion of Churned vs Active Users by Ads Sensitive",
       x = "Ads Sensitive", y = "Percentage", fill = "Churned (1 = Yes)") +
  scale_fill_manual(values = c("0" = "#2b8cbe", "1" = "#f03b20"))











### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####

#IV. SUPERVISED LEARNING

#A. Split Train/ Test set, One-hot Encoding & Check correlation among variables

glimpse(spotify_new)
dim(spotify_new)
n_churned <- spotify_clean %>% count(is_churned)
n_churned

# Split test/ train data
set.seed(123)
train_index <- createDataPartition(spotify_new$is_churned, p = 0.8, list = FALSE)
data_train <- spotify_new[train_index, ]
data_test  <- spotify_new[-train_index, ]
data_test <- data_test[data_test$device_type != "Smart Tv", ] #Drop an outlier record to prevent issues when predict test_set
count(data_train)
count(data_test)

# Define the columns to one-hot encode
cols_to_encode <- c("gender", "country", "subscription_type","device_type",  "age_group","ad_sensitive" )

# Convert those columns to factors
data_train[cols_to_encode] <- lapply(data_train[cols_to_encode], factor)
data_test[cols_to_encode]  <- lapply(data_test[cols_to_encode], factor)

# Create dummyVars model - excluding columns that are already numeric or logical
dummies <- dummyVars(~ gender + country + subscription_type + device_type + offline_listening + age_group + ad_sensitive, data = data_train)

# Encode the selected columns
encoded_train <- as.data.frame(predict(dummies, newdata = data_train))
encoded_test  <- as.data.frame(predict(dummies, newdata = data_test))


# Apply encode to both sets
data_train_encoded <- bind_cols(
  data_train %>% select(-all_of(cols_to_encode)),
  encoded_train
)
data_test_encoded  <- bind_cols(
  data_test %>% select(-all_of(cols_to_encode)),
  encoded_test
)


# Add target variable
data_train_encoded$is_churned <- data_train$is_churned
data_test_encoded$is_churned  <- data_test$is_churned

glimpse(data_train_encoded )

include_cols <- c("listening_time", "songs_played_per_day", "skip_rate", "ads_listened_per_week","listening_time_hours")

# Check correlation matrix for numerical data only
cor_matrix <- cor(data_train_encoded[, include_cols])
corrplot(cor_matrix, method = "color", tl.cex = 0.6)

#We don't see a strong correlation among numerical variables so it's good to proceed



###=======================####
#B. Class Balance

#Check class distribution before balancing
table(data_train_encoded$is_churned)

data_train_encoded$is_churned <- as.factor(data_train_encoded$is_churned)

count_table <- table(data_train_encoded$is_churned)
majority_class <- names(count_table)[which.max(count_table)]
n_majority <- max(count_table)

majority_df <- data_train_encoded %>% filter(is_churned == majority_class)
minority_df <- data_train_encoded %>% filter(is_churned != majority_class)

# Upsample minority to match majority
set.seed(42)
minority_upsampled <- minority_df %>% sample_n(size = n_majority, replace = TRUE)

balanced_df <- bind_rows(majority_df, minority_upsampled)

# Shuffling rows
balanced_df <- balanced_df %>% slice_sample(n = nrow(balanced_df))

table(balanced_df$is_churned)

# Verifying the results
head(balanced_df)
summary(balanced_df)


###=======================####
#C. Standardization
# Columns to include for standardization
include_cols <- c("listening_time", "songs_played_per_day", "skip_rate", "ads_listened_per_week","listening_time_hours")

# Calculate the mean and standard deviation ONLY on the BALANCED TRAINING DATA
train_means <- balanced_df %>%
  summarise(across(all_of(include_cols), mean))

train_sds <- balanced_df %>%
  summarise(across(all_of(include_cols), sd))

# Standardize the Training Data (using its own parameters)
spotify_standardized_train <- balanced_df %>%
  mutate(across(
    all_of(include_cols),
    ~ (. - as.numeric(train_means[[cur_column()]])) / as.numeric(train_sds[[cur_column()]]),
    .names = "{.col}_z"
  ))

# Standardize the Test Data (using the TRAINING Data's parameters)
spotify_standardized_test <- data_test_encoded %>%
  mutate(across(
    all_of(include_cols),
    ~ (. - as.numeric(train_means[[cur_column()]])) / as.numeric(train_sds[[cur_column()]]),
    .names = "{.col}_z"
  ))

# Columns to include for standardization
include_cols <- c("listening_time", "songs_played_per_day", "skip_rate", "ads_listened_per_week","listening_time_hours")

#Data train Standardize numeric columns and add "_z" at the end of columns' names
spotify_standardized <- balanced_df %>%
  mutate(across(
    all_of(include_cols),
    ~ scale(.) %>% as.numeric(),
    .names = "{.col}_z"
  ))

#Data test - Standardize numeric columns and add "_z" at the end of columns' names
spotify_test_standardized <- data_test_encoded %>%
  mutate(across(
    all_of(include_cols),
    ~ scale(.) %>% as.numeric(),
    .names = "{.col}_z"
  ))

# Verify the results

summary(spotify_standardized )

sapply(select(spotify_standardized, ends_with("_z")), mean)
sapply(select(spotify_standardized, ends_with("_z")), sd)






###### MODELS ##################


##D. DECISION TREE

# Define the list of original, unstandardized columns and irrelevant IDs/references
# to be removed from the standardized dataset.
cols_to_remove <- c(
  "user_id", # Irrelevant ID
  "age", # Original unstandardized column
  "ofline_listening",  #Original categorical/intermediate column
  "skip_rate_char", # Original categorical/intermediate column
  "skip_rate_cleaned", # Original categorical/intermediate column
  "skip_rate", # Original unstandardized column (now using skip_rate_z)
  "listening_time", # Original unstandardized column (now using listening_time_z)
  "songs_played_per_day", # Original unstandardized column (now using songs_played_per_day_z)
  "ads_listened_per_week", # Original unstandardized column (now using ads_listened_per_week_z)
  "listening_time_hours", # Original unstandardized column (now using listening_time_hours_z)
  "listening_time_hours_z", #duplicate columns
  # Reference categories after one-hot encoding (to be excluded)
  "gender.Male",
  "country.Canada",
  "subscription_type.Free",
  "device_type.Mobile",
  "offline_listeningFALSE",
  "age_group.Middle Adult",
  "ad_sensitive.No Ad Exposure"
)


### --- PREPARE TRAINING DATA --- ###

# Use the standardized training data (spotify_standardized_train)
data_train_final <- spotify_standardized_train %>% 
  select(-any_of(cols_to_remove))

# Ensure the target variable is a factor
data_train_final$is_churned <- as.factor(data_train_final$is_churned)

# Check the final training structure (should now primarily contain *_z columns)
glimpse(data_train_final)


### --- PREPARE TEST DATA --- ###

# Use the standardized test data (spotify_standardized_test)
data_test_final <- spotify_standardized_test %>% 
  select(-any_of(cols_to_remove))

# Ensure the target variable is a factor
data_test_final$is_churned <- as.factor(data_test_final$is_churned)

# Check the final test structure (should now primarily contain *_z columns)
glimpse(data_test_final)


# Train Decision Tree model
tree_model <- rpart(is_churned ~ .,
                    data = data_train_final,
                    method = "class",     # Equal prior weights align with the balanced dataset (0.5/0.5)
                    parms = list(prior = c(0.5, 0.5)), 
                    control = rpart.control(
                      cp = 0.001,          # Complexity parameter (controls pruning)
                      minbucket = 50,      # minimum records per terminal node
                      maxdepth = 4         # Maximum tree depth
                    ))

# Print tree structure
cat("Tree Structure:\n")
print(tree_model)

# Visualize the decision tree
rpart.plot(tree_model,
           extra = 101,                  # Display node info
           fallen.leaves = TRUE,          # Align leaves at bottom
           main = "Decision Tree for Spotify Churn Prediction")


# Pruning: Find the best complexity parameter (CP)
bestcp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"] 
cat("Best CP:", bestcp, "\n") 

tree_model_pruned <- prune(tree_model, cp = bestcp) 

rpart.plot(tree_model_pruned, extra = 106, main = "Best-Pruned Classification Tree for Churn Rate")

# Ensure the levels of the predicted and actual values are consistent
data_test_final$is_churned <- factor(data_test_final$is_churned, levels = c("0", "1"))

# Predict on test data
tree_pruned_pred_class <- predict(tree_model_pruned, 
                                  newdata = data_test_final, 
                                  type = "class")
tree_pruned_pred_class <- factor(tree_pruned_pred_class, levels = c("0", "1"))


# Create confusion matrix 
tree_pruned_cm <- confusionMatrix(tree_pruned_pred_class, 
                                  data_test_final$is_churned, 
                                  positive = "1")
print(tree_pruned_cm)

# Extract key metrics
tree_accuracy <- tree_pruned_cm$overall['Accuracy']
tree_precision <- tree_pruned_cm$byClass['Pos Pred Value']
tree_recall <- tree_pruned_cm$byClass['Sensitivity']
tree_f1 <- tree_pruned_cm$byClass['F1']

cat("\n### DECISION TREE - KEY METRICS ###\n")
cat("Accuracy: ", round(tree_accuracy * 100, 2), "%\n", sep="")
cat("Precision:", round(tree_precision * 100, 2), "%\n", sep="")
cat("Recall:   ", round(tree_recall * 100, 2), "%\n", sep="")
cat("F1-Score: ", round(tree_f1 * 100, 2), "%\n", sep="")





###=======================####


# E. RANDOM FOREST

# Columns to remove, including original unstandardized features and redundant reference categories

cols_to_remove_rf <- c(
  "user_id",
  # Original numeric columns (now replaced by *_z versions)
  "offline_listening","listening_time", "age", "skip_rate", "songs_played_per_day", 
  "ads_listened_per_week", "listening_time_hours", "listening_time_hours_z", 
  # Intermediate/Categorical columns
  "skip_rate_char", "skip_rate_cleaned",
  # Reference categories (for dummy variable trap)
  "gender.Male", "country.Canada", "subscription_type.Free", 
  "device_type.Mobile", "offline_listeningFALSE", 
  "age_group.Middle Adult", "ad_sensitive.No Ad Exposure"
)

## Prepare data
# --- TRAIN SET ---
# Use spotify_standardized_train (the balanced, scaled data)
data_train_rf <- spotify_standardized_train %>% 
  select(-any_of(cols_to_remove_rf))

# Ensure target is factor 
data_train_rf$is_churned <- as.factor(data_train_rf$is_churned)
glimpse(data_train_rf)

# --- TEST SET ---
# Use spotify_standardized_test (the scaled test data) 
data_test_rf <- spotify_standardized_test %>% 
  select(-any_of(cols_to_remove_rf))

# Ensure target is factor and levels match the training set
data_test_rf$is_churned <- factor(data_test_rf$is_churned, levels = levels(data_train_rf$is_churned))
glimpse(data_test_rf)


# Clean column names in train and test dataset
colnames(data_train_rf) <- make.names(colnames(data_train_rf))
colnames(data_test_rf) <- make.names(colnames(data_test_rf))

# Train the random forest model
rf_model <- randomForest(is_churned ~ .,
                         data = data_train_rf,
                         ntree = 500,
                         mtry = 5,
                         importance = TRUE,
                         na.action = na.omit)

# Display model summary
cat("Random Forest Model Summary:\n")
print(rf_model)


# Make predictions on test data
rf_pred_class <- predict(rf_model, newdata = data_test_rf, type = "class")

# Create confusion matrix
rf_cm <- confusionMatrix(rf_pred_class, data_test_rf$is_churned, positive = "1")

print(rf_cm)

# Extract key metrics
rf_accuracy <- rf_cm$overall['Accuracy']
rf_precision <- rf_cm$byClass['Pos Pred Value']
rf_recall <- rf_cm$byClass['Sensitivity']
rf_f1 <- rf_cm$byClass['F1']


cat("\n### RANDOM FOREST - KEY METRICS (Target: Churned=1) ###\n")
cat("Accuracy: ", round(rf_accuracy * 100, 2), "%\n", sep="")
cat("Precision:", round(rf_precision * 100, 2), "%\n", sep="")
cat("Recall:   ", round(rf_recall * 100, 2), "%\n", sep="")
cat("F1-Score: ", round(rf_f1 * 100, 2), "%\n", sep="")

# Variable Importance Plot
cat("\n### VARIABLE IMPORTANCE (Random Forest) ###\n")
varImpPlot(rf_model,
           main = "Variable Importance in Random Forest",
           n.var = 10)

# Get detailed importance scores (using Mean Decrease Gini as the primary metric)
importance_scores <- importance(rf_model)
cat("\nTop 10 Most Important Variables (by Mean Decrease Gini):\n")
# Mean Decrease Gini for classification
print(head(importance_scores[order(-importance_scores[,4]), ], 10))






###=======================####

# F. K-NEAREST NEIGHBORS (KNN)

# Prepare data

# Use data from Decision tree (has been cleaned, balanced and standardized)
data_train_knn <- data_train_final
data_test_knn <- data_test_final

# Columns to remove (unnecessary and contains NAN values)
cols_to_remove_knn <- c("listening_time_outlier", "songs_played_outlier") 

# Select, remove unnecessary columns, and ensure NO NA values remain 
data_train_knn <- data_train_knn %>% 
  select(-any_of(cols_to_remove_knn)) %>%
  na.omit() 

data_test_knn <- data_test_knn %>% 
  select(-any_of(cols_to_remove_knn)) %>%
  na.omit() 

# Separate predictors (x) and target (y)
train_x <- data_train_knn %>% select(-is_churned)
train_y <- data_train_knn$is_churned

test_x <- data_test_knn %>% select(-is_churned)
test_y <- data_test_knn$is_churned

# Rename factor levels for compatibility
train_y <- factor(train_y, levels = c("0", "1"), labels = c("NoChurn", "Churn"))
test_y <- factor(test_y, levels = c("0", "1"), labels = c("NoChurn", "Churn"))

# Ensure all predictors (X) are numeric
train_x <- as.data.frame(lapply(train_x, as.numeric))
test_x <- as.data.frame(lapply(test_x, as.numeric))


# Final N/A check for class::knn - highly strict knn() function
# Check and remove NAs from the training features and realign the target vector
na_rows_train <- !complete.cases(train_x) 
if (any(na_rows_train)) {
  cat("Removing", sum(na_rows_train), "NA rows from train_x/train_y.\n")
  train_x <- train_x[!na_rows_train, ]
  train_y <- train_y[!na_rows_train]
}

# Check and remove NAs from the test features and realign the target vector
na_rows_test <- !complete.cases(test_x)
if (any(na_rows_test)) {
  cat("Removing", sum(na_rows_test), "NA rows from test_x/test_y.\n")
  test_x <- test_x[!na_rows_test, ]
  test_y <- test_y[!na_rows_test]
}


# Due to prior tuning errors, we use a fixed, common k value (e.g., k=5)
best_k <- 5 
cat("\nSkipping automated tuning and using fixed k =", best_k, "\n")



# Train final KNN model with fixed k=5 
knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = best_k)

# Confusion Matrix and Metrics 
knn_cm <- confusionMatrix(knn_pred, test_y, positive = "Churn")

cat("\n### KNN - CONFUSION MATRIX (k=", best_k, ") ###\n", sep="")
print(knn_cm)

# Extract key metrics
knn_accuracy <- knn_cm$overall['Accuracy']
knn_precision <- knn_cm$byClass['Pos Pred Value']
knn_recall <- knn_cm$byClass['Sensitivity']
knn_f1 <- knn_cm$byClass['F1']

cat("\n### KNN - KEY METRICS (Target: Churned) ###\n")
cat("Accuracy: ", round(knn_accuracy * 100, 2), "%\n", sep="")
cat("Precision:", round(knn_precision * 100, 2), "%\n", sep="")
cat("Recall:", round(knn_recall * 100, 2), "%\n", sep="")
cat("F1-Score: ", round(knn_f1 * 100, 2), "%\n", sep="")






###=======================####

# G: LOGISTIC REGRESSION
# Logistic regression requires standardized numeric data (which we have in the final dataframes).

# Prepare data
# Start from the clean, standardized, and balanced dataframes
train_glm <- data_train_final
test_glm <- data_test_final

# Exclude unnecessary/outlier columns IF they still exist in final dataframes.
cols_to_remove_glm <- c("listening_time_outlier", "songs_played_outlier")

train_glm <- train_glm %>% select(-any_of(cols_to_remove_glm)) %>% na.omit()
test_glm <- test_glm %>% select(-any_of(cols_to_remove_glm)) %>% na.omit()


# Ensure target variable is a FACTOR with named levels (as used in KNN)
# glm() works correctly with a two-level factor, treating the second level ("Churn") as the positive class.
train_glm$is_churned <- factor(train_glm$is_churned, levels = c("0", "1"), labels = c("NoChurn", "Churn"))
test_glm$is_churned  <- factor(test_glm$is_churned, levels = c("0", "1"), labels = c("NoChurn", "Churn"))

# Fit logistic regression model
glm_model <- glm(is_churned ~ ., data = train_glm, family = binomial(link = "logit"))

cat("Logistic Regression Model Summary:\n")
summary(glm_model)


# Make predictions on test data
glm_probs <- predict(glm_model, newdata = test_glm, type = "response")

# Convert probabilities to class prediction using a 0.5 threshold
# Since "Churn" is the second factor level, it corresponds to the predicted probability.
glm_pred_class <- ifelse(glm_probs > 0.5, "Churn", "NoChurn")
glm_pred_class <- factor(glm_pred_class, levels = c("NoChurn", "Churn"))


# Evaluate model
glm_cm <- confusionMatrix(glm_pred_class, test_glm$is_churned, positive = "Churn")

cat("LOGISTIC REGRESSION - CONFUSION MATRIX\n")
print(glm_cm)

# Extract key metrics
glm_accuracy <- glm_cm$overall['Accuracy']
glm_precision <- glm_cm$byClass['Pos Pred Value']
glm_recall <- glm_cm$byClass['Sensitivity']
glm_f1 <- glm_cm$byClass['F1']

cat("LOGISTIC REGRESSION - KEY METRICS (Target: Churn)\n")
cat("Accuracy: ", round(glm_accuracy * 100, 2), "%\n", sep="")
cat("Precision:", round(glm_precision * 100, 2), "%\n", sep="")
cat("Recall:", round(glm_recall * 100, 2), "%\n", sep="")
cat("F1-Score: ", round(glm_f1 * 100, 2), "%\n", sep="")

# ROC Curve and AUC
# AUC requires the second argument (the predicted probabilities) to be numeric.
roc_obj <- roc(response = test_glm$is_churned, predictor = glm_probs)
auc_value <- auc(roc_obj)
cat("\nAUC Score:", round(auc_value, 4), "\n")

plot(roc_obj, col="#1DB954", main="ROC Curve - Logistic Regression")






###=======================####

# H: XGBOOST

# Prepare the data
# Use the cleaned/pre-processed dataframes.
train_xgb <- data_train_final %>% select(-is_churned)
test_xgb <- data_test_final %>% select(-is_churned)

# Ensure all columns in the feature matrices are numeric.
train_xgb[] <- lapply(train_xgb, as.numeric)
test_xgb[] <- lapply(test_xgb, as.numeric)

# Target variable must be numeric 0 or 1 for training
train_y_xgb <- as.numeric(as.character(data_train_final$is_churned))
test_y_xgb <- as.numeric(as.character(data_test_final$is_churned))

# Convert to DMatrix format (optimized for XGBoost)
dtrain <- xgb.DMatrix(data = as.matrix(train_xgb), label = train_y_xgb)
dtest <- xgb.DMatrix(data = as.matrix(test_xgb), label = test_y_xgb)

# Define parameters
params <- list(
  objective = "binary:logistic", 
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  nthread = 2
)

# Train model
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20,
  verbose = 0 # Suppress excessive output during training
)

cat("XGBoost Model trained with", xgb_model$best_iteration, "rounds.\n")


# Predict probabilities
xgb_pred_probs <- predict(xgb_model, dtest)

# Convert probabilities to factor class prediction (0/1)
xgb_pred_numeric <- ifelse(xgb_pred_probs > 0.5, 1, 0)


# Reference factor levels for confusion matrix
factor_levels <- c("NoChurn", "Churn")

xgb_pred_class <- factor(xgb_pred_numeric, levels = c(0, 1), labels = factor_levels)
test_y_factor <- factor(test_y_xgb, levels = c(0, 1), labels = factor_levels)


# Confusion Matrix
xgb_cm <- confusionMatrix(
  xgb_pred_class,
  test_y_factor,
  positive = "Churn" # Use the consistent positive class label
)

cat("XGBOOST - CONFUSION MATRIX\n")
print(xgb_cm)

# Metrics
xgb_accuracy <- xgb_cm$overall['Accuracy']
xgb_precision <- xgb_cm$byClass['Pos Pred Value']
xgb_recall <- xgb_cm$byClass['Sensitivity']
xgb_f1 <- xgb_cm$byClass['F1']

cat("XGBOOST - KEY METRICS (Target: Churn)\n")
cat("Accuracy: ", round(xgb_accuracy * 100, 2), "%\n", sep="")
cat("Precision:", round(xgb_precision * 100, 2), "%\n", sep="")
cat("Recall:", round(xgb_recall * 100, 2), "%\n", sep="")
cat("F1-Score: ", round(xgb_f1 * 100, 2), "%\n", sep="")

# ROC Curve and AUC
roc_xgb <- roc(test_y_xgb, xgb_pred_probs) # Use numeric target for pROC
auc_xgb <- auc(roc_xgb)
cat("\nAUC Score (XGBoost):", round(auc_xgb, 4), "\n")

# Variable Importance
importance_matrix <- xgb.importance(model = xgb_model)
cat("\nTop 10 Most Important Features (XGBoost):\n")
print(head(importance_matrix, 10))






###=======================####
# I. MODEL COMPARISON

# Decision Tree 
tree_accuracy_final <- tree_accuracy
tree_precision_final <- tree_precision
tree_recall_final <- tree_recall
tree_f1_final <- tree_f1

# Random Forest
rf_accuracy_final <- rf_accuracy
rf_precision_final <- rf_precision
rf_recall_final <- rf_recall
rf_f1_final <- rf_f1


#Create Comparison Data Frame

model_comparison_ext <- data.frame(
  Model = c("Decision Tree",
            "Random Forest",
            paste0("KNN (k=", best_k, ")"),
            "Logistic Regression",
            "XGBoost"),
  Accuracy = c(tree_accuracy_final, rf_accuracy_final, knn_accuracy, glm_accuracy, xgb_accuracy),
  Precision = c(tree_precision_final, rf_precision_final, knn_precision, glm_precision, xgb_precision),
  Recall = c(tree_recall_final, rf_recall_final, knn_recall, glm_recall, xgb_recall),
  F1_Score = c(tree_f1_final, rf_f1_final, knn_f1, glm_f1, xgb_f1)
)

# Convert to percentages and round
model_comparison_ext <- model_comparison_ext %>%
  mutate(across(where(is.numeric), ~ round(. * 100, 2)))

cat("\n=== FINAL MODEL COMPARISON ===\n")
print(model_comparison_ext)

# Bar Plot Comparison
model_long <- model_comparison_ext %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1_Score),
               names_to = "Metric", values_to = "Value")


# Find the best F1-Score model 
best_f1_model <- model_comparison_ext$Model[which.max(model_comparison_ext$F1_Score)]

ggplot(model_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Comparison: Churn Prediction Performance",
       subtitle = paste("Best F1-Score Model:", best_f1_model),
       y = "Score (%)", x = "Metric") +
  theme_minimal() +
  scale_fill_brewer(palette =  "Paired") + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1))



#MODEL COMPARISION IN DETAILS

# Create Comparison Data Frame (Values in percentage form)
model_comparison_ext <- data.frame(
  Model = c("Decision Tree",
            "Random Forest",
            paste0("KNN (k=", best_k, ")"),
            "Logistic Regression",
            "XGBoost"),
  Accuracy = c(tree_accuracy_final, rf_accuracy_final, knn_accuracy, glm_accuracy, xgb_accuracy),
  Precision = c(tree_precision_final, rf_precision_final, knn_precision, glm_precision, xgb_precision),
  Recall = c(tree_recall_final, rf_recall_final, knn_recall, glm_recall, xgb_recall),
  F1_Score = c(tree_f1_final, rf_f1_final, knn_f1, glm_f1, xgb_f1)
) %>% 
  mutate(across(where(is.numeric), ~ . * 100)) # Convert to percentage

# Pivot Data for Plotting

model_long <- model_comparison_ext %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1_Score),
               names_to = "Metric", values_to = "Value")

# Customize Metric Names for clarity on the plot
model_long$Metric <- factor(model_long$Metric,
                            levels = c("Accuracy", "Precision", "Recall", "F1_Score"),
                            labels = c("Accuracy", "Precision", "Recall", "F1-Score"))

# Define the soft pastel color palette
pastel_colors <- c("Accuracy" = "#B3CDE3", 
                   "Precision" = "#FDBF6F", 
                   "Recall" = "#B2DF8A", 
                   "F1-Score" = "#FB9A99")

# Base Plot Function
create_model_plot <- function(df, selected_model) {
  df_filtered <- df %>% filter(Model == selected_model)
  
  # Find the highest F1 
  f1_value <- df_filtered$Value[df_filtered$Metric == "F1-Score"]
  
  ggplot(df_filtered, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity") +
    labs(
      title = paste("Performance Metrics for:", selected_model),
      subtitle = paste("F1-Score:", round(f1_value, 2), "%"),
      x = "Metric",
      y = "Score (%)"
    ) +
    # Add labels above the bars
    geom_text(aes(label = paste0(round(Value, 2), "%"), y = Value + 1.5), 
              size = 3, color = "black") +
    scale_fill_manual(values = pastel_colors) +
    scale_y_continuous(limits = c(0, max(df_filtered$Value) * 1.15)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 15, hjust = 1, size = 10), 
          legend.position = "none") 
}

# Generate and Display All Five Plots using gridExtra

model_list <- unique(model_comparison_ext$Model)
plot_list <- list()

# Generate plots
for (i in seq_along(model_list)) {
  model_name <- model_list[i]
  p <- create_model_plot(model_long, model_name)
  plot_list[[i]] <- p # Store plot in the list
}

# Display all plots in a grid 
cat("\n=== Displaying All Five Model-Centric Plots ===\n")
grid.arrange(grobs = plot_list, ncol = 3, nrow = 2)







### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####
### ==================================================================================#####
### V. UNSUPERVISED LEARNING


#A. PREPARING THE DATA

clustering_features <- spotify_new %>%
  select(age, listening_time, songs_played_per_day, skip_rate, ads_listened_per_week) %>%
  na.omit() %>%
  scale() %>%
  as.data.frame()

spotify_for_profiling <- spotify_new %>%
  select(age, listening_time, songs_played_per_day, skip_rate,
         ads_listened_per_week, is_churned) %>%
  na.omit()

cat("Total observations for clustering:", nrow(clustering_features), "\n")



###===========================###

#B. DETERMINING THE OPTIMAL NUMBER OF CLUSTERS

set.seed(123)

# Sample data if too large (for faster computation)
if(nrow(clustering_features) > 5000) {
  sample_data <- clustering_features[sample(1:nrow(clustering_features), 5000), ]
} else {
  sample_data <- clustering_features
}

# Elbow Method
fviz_nbclust(sample_data, kmeans, method = "wss", k.max = 10) +
  labs(title = "Elbow Method")

# Silhouette Method
fviz_nbclust(sample_data, kmeans, method = "silhouette", k.max = 5) +
  labs(title = "Silhouette Method")

# Gap Statistic
fviz_nbclust(sample_data, kmeans, method = "gap_stat",
             nboot = 50, k.max = 5) +
  labs(title = "Gap Statistic Method")


###===========================###

#C. APPLICATION OF K-MEANS CLUSTERING

set.seed(123)
kmeans_fit <- kmeans(clustering_features, centers = 3, nstart = 25)

# Cluster sizes
cat("\nCluster Sizes:\n")
print(table(kmeans_fit$cluster))

# Add clusters to data
spotify_for_profiling$cluster <- as.factor(kmeans_fit$cluster)
clustering_features$cluster <- as.factor(kmeans_fit$cluster)



###===========================###

#D. DIMENSIONALITY REDUCTION

# PCA
pca_fit <- prcomp(clustering_features[, -ncol(clustering_features)], scale. = TRUE)

# Variance explained
cat("\nPCA Variance Explained:\n")
print(summary(pca_fit))

# PCA visualization
pca_df <- data.frame(
  PC1 = pca_fit$x[,1],
  PC2 = pca_fit$x[,2],
  Cluster = clustering_features$cluster
)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "PCA - K-Means Clusters", x = "PC1", y = "PC2") +
  theme_minimal() +
  scale_color_manual(values = c("#1DB954", "#FF4B4B", "#1E90FF"))


# t-SNE Visualization
set.seed(123)

# Remove duplicates for t-SNE
clustering_unique <- clustering_features[!duplicated(clustering_features[, -ncol(clustering_features)]), ]
cat("\nOriginal rows:", nrow(clustering_features), "| Unique rows:", nrow(clustering_unique), "\n")

# Run t-SNE
tsne_fit <- Rtsne(clustering_unique[, -ncol(clustering_unique)],
                  dims = 2, perplexity = 30,verbose = FALSE,
                  check_duplicates = FALSE)

# Create visualization data
tsne_df <- data.frame(
  tSNE1 = tsne_fit$Y[,1],
  tSNE2 = tsne_fit$Y[,2],
  Cluster = clustering_unique$cluster
)

ggplot(tsne_df, aes(x = tSNE1, y = tSNE2, color = Cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "t-SNE - K-Means Clusters") +
  theme_minimal() +
  scale_color_manual(values = c("#1DB954", "#FF4B4B", "#1E90FF"))


###===========================###
# E. CLUSTER PROFILING

# Checking the clusters profile
cluster_profile <- spotify_for_profiling %>%
  group_by(cluster) %>%
  summarise(
    across(c(age, listening_time, songs_played_per_day, skip_rate,
             ads_listened_per_week),
           mean, .names = "avg_{.col}"),
    Churn_Rate = mean(is_churned)
  )

cat("\n===== CLUSTER PROFILING =====\n")
print(cluster_profile)


###===========================###
#F. CHURN VERIFICATION

cat("\n===== CHURN DISTRIBUTION BY CLUSTER =====\n")

churn_by_cluster <- spotify_for_profiling %>%
  group_by(cluster, is_churned) %>%
  summarise(Count = n(), .groups = "drop") %>%
  pivot_wider(names_from = is_churned, values_from = Count, values_fill = 0) %>%
  mutate(
    Total = `0` + `1`,
    Pct_Not_Churned = round(`0` / Total * 100, 1),
    Pct_Churned = round(`1` / Total * 100, 1)
  )

print(churn_by_cluster)

ggplot(spotify_for_profiling, aes(x = cluster, fill = factor(is_churned))) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = c("0" = "#1DB954", "1" = "#FF4B4B"),
                    labels = c("Not Churned", "Churned")) +
  labs(title = "Churn Distribution Within Each Cluster",
       x = "Cluster", y = "Percentage", fill = "Churn Status") +
  theme_minimal()


###===========================###
#G. VALIDATION

sil <- silhouette(kmeans_fit$cluster, dist(clustering_features[, -ncol(clustering_features)]))

cat("\n=== VALIDATION ===\n")
cat(sprintf("Average Silhouette Width: %.3f\n", mean(sil[,3])))
cat(sprintf("Within-Cluster SS: %.2f\n", kmeans_fit$tot.withinss))
cat(sprintf("Between-Cluster SS: %.2f\n", kmeans_fit$betweenss))
cat(sprintf("BSS/TSS Ratio: %.2f%%\n", 100 * kmeans_fit$betweenss / kmeans_fit$totss))

fviz_silhouette(sil) +
  labs(title = "Silhouette Plot")

#H. CHURN SUMMARY

cat("\n========================================================\n")
cat("SUMMARY: USER PERSONAS & CHURN INSIGHTS\n")
cat("========================================================\n\n")

#User Personas & Churn Insights Summary
high_risk_cluster <- cluster_profile$cluster[which.max(cluster_profile$Churn_Rate)]
high_risk_rate <- max(cluster_profile$Churn_Rate)

low_risk_cluster <- cluster_profile$cluster[which.min(cluster_profile$Churn_Rate)]
low_risk_rate <- min(cluster_profile$Churn_Rate)

cat(sprintf("HIGH-RISK PERSONA: Cluster %s (%.1f%% churn rate)\n",
            high_risk_cluster, high_risk_rate * 100))
cat(sprintf("LOW-RISK PERSONA: Cluster %s (%.1f%% churn rate)\n\n",
            low_risk_cluster, low_risk_rate * 100))


cat("========================================================\n")
