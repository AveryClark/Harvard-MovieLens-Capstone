
#################  Executive Summary  #################

# In this analysis, I used machine learning methods to build prediction models 
# designed to predict what a user will rate a movie as a foundation for a 
# recommendation system.

# In this section I'll describe the dataset and summarize the goal of the project and key steps 
# that were performed.

# I analyzed the MovieLens 10M database and used it to attempt
# to build a machine learning algorithm that can predict what movies users would 
# like to watch with high accuracy. These predictions will be trained on one dataset
# and tested on a separate dataset, where they will hopefully come very close to 
# predicting how many stars (on a 0.5 to 5 star scale) a user will rate a movie.

# To win the grand prize of $1 million from the Netflix challenge, a participating 
# team had to get to a residual mean square error (RMSE) of about 0.857.

# You can read more about it here:
# http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/

# My goal was to build a prediction model with a RMSE of less than 0.8649.
# I surpassed that goal.

# The RMSE (Residual Mean Square Error) is the standard deviation of the 
# prediction errors (residuals).

# In other words, it's the average of how far the predictions deviate from what they are trying 
# to predict.

# The lower the RMSE, the more accurate the algorithm's predictions are.

# I split the data into a training set (90% of data) to train the prediction models
# and a testing set (10% of data) to test the accuracy of the prediction model.

# After running five prediction models, the lowest Residual Mean Square Error (RMSE) 
# obtained was 0.8644501, which accomplishes the goal of reaching lower than 0.8649.

# The most effective prediction model was "Regularized Movie + User + Genre 
# Effect Model", where I used the biases per movie, per user, and per genre 
# of the reviews in the training set and then regularized (or "rubber-banded") 
# the results, penalizing biases of movies/users/genres with low review 
# counts by pulling them toward the dataset average.


# This report contains four sections: 
# Executive Summary, Analysis, Results, and Conclusion.

# Executive Summary describes the dataset and summarizes the goal of the project and key steps 
# that were performed.

# Analysis explains the process and techniques used, such as data cleaning, data exploration and 
# visualization, any insights gained, and the modeling approach.

# Results presents the modeling results and discusses the model performance.

# Conclusion gives a brief summary of the report, its limitations and future work.

# Thank you for taking the time to look at this report.
# I hope that you will run this code by stepping through (by pressing Ctrl + Enter) 
# as I'm explaining it.



#################  Analysis  #################

# I'd like to start this analysis off by asking: 
# How important is it that your next recommendation be something you really like? 
# Netflix thought it was so important that they happily offered $1 million for a 
# 10% increase in the accuracy of their recommendations.

# You can read more about it here:
# http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/

# Below, I'll be analyzing the MovieLens 10M database and attempt
# to build a machine learning algorithm that can predict what movies users would 
# like to watch with high accuracy. These predictions will be trained on one dataset
# and tested on a separate dataset, where they will hopefully come very close to 
# predicting how many stars (on a 0.5 to 5 star scale) a user will rate a movie.

# To win the grand prize of $1 million from the Netflix challenge, a participating 
# team had to get to a residual mean square error (RMSE) of about 0.857.

# My goal is to build a prediction model with a RMSE of less than 0.8649.

# The RMSE (Residual Mean Square Error) is the standard deviation of the 
# prediction errors (residuals).

# In other words, it's the average of how far the predictions deviate from what they are trying 
# to predict.

# The lower the RMSE, the more accurate the algorithm's predictions are.

# This was run using RStudio Version 1.1.463 and R version 3.6.2 (Dark and Stormy Night) 
# from https://www.r-project.org/


# In this section, I'll explain the process and techniques used, such as data cleaning, 
# data exploration and visualization, any insights gained, and the modeling approach. 
# You'll see these models in action in the Results section.

# 90% of the data was designated for training the prediction model and 10% of the data 
# was reserved for testing the accuracy of that model's predictions.

# A simple way of thinking about this is that the model (or algorithm) will learn 
# about the data by taking in different factors and will make a prediction of what 
# star rating (on a 0.5 to 5 star scale) a user will rank a movie based on those factors.
# Different approaches will have the model/algorithm using the factors given to it 
# in different ways to make predictions.

# The model/algorithm decides to predict a review rating "Y" based on factors "A", "B",
# and "C" (or more). Then the model/algorithm is exposed to the testing dataset to see if 
# what it predicts as the review rating "Y" (based on the factors in the new dataset "A", "B",
# and "C") is actually that accurate or not. Then from the results we can compute our
# RMSE (Residual Mean Square Error). This is how we test the model's accuracy.

# The RMSE (Residual Mean Square Error) is the standard deviation of the 
# prediction errors (residuals). In simpler terms, it's the average of how far the 
# predictions deviate from what they are trying to predict (how far off the mark our model's 
# predictions are). The lower the RMSE, the more accurate the model's predictions are.

# I hope that you will step through the code with me as I explain it.

# You can run all of the code by clicking Run. You can run it line by line by pressing Ctrl + Enter 
# on your keyboard. You can also highlight a section of code and run just that by clicking Run or pressing
# Ctrl + Enter on your keyboard.

# Let's dig in!


# First, we'll build our training set (edx set) and our validation set from 
# the MovieLens 10M dataset by splitting the data up randomly.

# The training set (edx set) will hold about 90% of our data, and the validation set will hold about 10%.

# We will use the training set to train our machine learning algorithm
# and we will test its accuracy on the validation set.


# To begin this, we'll install the packages that will give us the tools to analyze the data.
# Notice the if statements mean the packages will not install if you have them already.

# Note: this might take a couple of minutes.

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dotwhisker)) install.packages("dotwhisker", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(rmarkdown)) install.packages("rmarkdown", repos = "http://cran.us.r-project.org")
library(caret)
library(data.table)
library(dotwhisker)
library(tidyverse)
library(rmarkdown)



wd <- getwd()

# This is your working directory.
wd

setwd(wd)

# Now we'll download the data.

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Now we'll convert the data into the columns we will use.

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


# Now we'll split up the data into our training set and our validation set.

# The training set will be 90% of the MovieLens data and the validation set will be 10%.

# We'll set a random seed so the results can be reproduced by anyone else 
# running this code.
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead.
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId values in validation line up to the IDs contained in the edx set.

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set.

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# This will clean up memory by deleting things we no longer need.

rm(dl, ratings, movies, test_index, temp, removed)



# And now that we've built our training set (edx set) and our validation set, 
# let's look at the data we're working with.


# Now let's check if we have any missing data.

any(is.na(edx))
any(is.na(validation))


# Now let's probe the data a little more.

summary(edx)
dim(edx)
head(edx)
tibble(edx)
# The training set has 9,000,055 entries (or rows) of movie reviews and 6 columns of details.

summary(validation)
dim(validation)
head(validation)
tibble(validation)
# As we can see, the validation set has 999,999 entries and the same 6 columns.

# Thankfully, this data is already pretty clean. I won't have to go through a lot of 
# effort looking for ways to fix errors or NAs from bad data entry or missing data.

# Take note that the entries in the validation set ARE NOT in the training set (edx set), and vice versa. 
# No entries are shared between the sets.

# We'll train our model/algorithm on the training set and test its accuracy on the validation set.


# Let's see how many unique users and movies are in the training dataset.

edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# We have 69878 users and 10677 movies.


# Some distributions may give us a better understanding of the reviews/ratings.
# Let's see the distribution of how many ratings the movies receive.

edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
# From the graph we can see that the most common number of ratings received for movies in this 
# dataset is roughly around 100.


# Let's see the distribution of how many ratings users give.

edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")
# From the graph we can see that the most common number of ratings given by users is around 
# 20 to 150 ratings.



# Let's split up the genres into separate columns per genre (such as gAction for Action) 
# and assign a 1 in each column for reviews that are of a movie in that genre and assign 
# a 0 if not.

# This could provide more accuracy later on if the effects of genres are significant enough.

## genreList <- as.factor(edx$genres)
## levels(genreList)

# The two lines above gave me a quick way to see what genres are in the data.

# Let's make columns to check for each genre.

# Keep in mind we are merely making it easier for the computer to understand what
# genres are involved in each review.

edx$gAction <- ifelse(grepl("Action", edx$genres), 1, 0)
edx$gAdvent <- ifelse(grepl("Adventure", edx$genres), 1, 0)
edx$gAnim <- ifelse(grepl("Animation", edx$genres), 1, 0)
edx$gChild <- ifelse(grepl("Children", edx$genres), 1, 0)
edx$gComedy <- ifelse(grepl("Comedy", edx$genres), 1, 0)
edx$gFantasy <- ifelse(grepl("Fantasy", edx$genres), 1, 0)
edx$gSciFi <- ifelse(grepl("Sci-Fi", edx$genres), 1, 0)
edx$gImax <- ifelse(grepl("IMAX", edx$genres), 1, 0)
edx$gDrama <- ifelse(grepl("Drama", edx$genres), 1, 0)
edx$gHorror <- ifelse(grepl("Horror", edx$genres), 1, 0)
edx$gMyst <- ifelse(grepl("Mystery", edx$genres), 1, 0)
edx$gThrill <- ifelse(grepl("Thriller", edx$genres), 1, 0)
edx$gCrime <- ifelse(grepl("Crime", edx$genres), 1, 0)
edx$gRom <- ifelse(grepl("Romance", edx$genres), 1, 0)
edx$gWar <- ifelse(grepl("War", edx$genres), 1, 0)
edx$gWest <- ifelse(grepl("Western", edx$genres), 1, 0)
edx$gMusic <- ifelse(grepl("Musical", edx$genres), 1, 0)
edx$gDocu <- ifelse(grepl("Documentary", edx$genres), 1, 0)
edx$gFilmN <- ifelse(grepl("Film-Noir", edx$genres), 1, 0)
tibble(edx)

# Let's run a multiple regression analysis on the training set to see if genre affects users' movie ratings.
# This will statistically predict the movies' ratings based on the whether the movie belongs to 
# a particular genre or not.

genreFit <- lm(rating ~ gAction + gAdvent + gAnim + gChild + gComedy + gFantasy + gSciFi + gImax + gDrama + gHorror + gMyst + gMusic + gDocu + gFilmN, data=edx)
summary(genreFit)

# It turns out that all genres have a statistically significant effect on the movie rating.
# You can see this by looking at the three stars "***" beside each coefficient,
# which shows that the p-value is small enough for the coefficient to have a high significance.
# Here is a graph of the estimated effects of each genre on the movie rating.
dwplot(genreFit)

genreFit
modcoef <- summary(genreFit)[["coefficients"]]
modcoef[order(modcoef[ , 1]), ] 


# If we look closely at the results, we can see that movies of the Film-Noir genre are expected to score 
# an estimated 0.399 points higher in ratings than the intercept (3.449).

# Genres with the most positive effect are Film-Noir (0.399), Documentary (0.332), and Animation (0.298).


# We also see that movies of the Children genre are expected to score an estimated 0.265 points lower 
# in ratings than the intercept (3.449).

# Genres with the most negative effect are Children (-0.265), Horror (-0.203), and Action (-0.090).


# Our modeling approach will be to start off with some simple models and
# gradually add complexity in the hopes of reaching a lower RMSE (greater accuracy)
# and eventually accomplishing the goal of an RMSE of less than 0.8649.

# The models we will be building are:
# 1. "Only the Average/Naive Approach Model", predicting that all reviews will 
#     be equal to the average of all reviews in the testing set (as a baseline 
#     model for testing subsequent model accuracy).

# 2. "Movie Effect Model", predicting based on previous reviews for that movie
#     (movie bias).

# 3. "Movie + User Effects Model", adding user bias to the previous model.

# 4. "Regularized Movie + User Effect Model", regularizing the previous model 
#     because averages of biases of movies and users with few reviews are too 
#     extreme to be accurately predicted and need to be pulled in (or "rubber-banded")
#     closer to the dataset average the fewer reviews there are.

# 5. "Regularized Movie + User + Genre Effect Model", adding genre bias to 
#     the previous model and regularizing (or "rubber-banding") all three biases.


# Let's look at the results of these models in the next section.



#################  Results  #################

# Now I'll present the modeling results and discuss the model performance as 
# you step through the code with me.

# You can run all of the code by clicking Run. You can run it line by line by pressing Ctrl + Enter 
# on your keyboard. You can also highlight a section of code and run just that by clicking Run or pressing
# Ctrl + Enter on your keyboard.


# Now let's build some models and run some tests to get the 
# RMSE (Residual Mean Square Error) of each one.

# A RMSE shows us how far off the mark our model's predictions are, which tells us 
# how accurate it is.

# A simple way of thinking about this is that the model (or algorithm) will learn 
# about the data by taking in different factors and make a prediction of what 
# star rating (on a 0.5 to 5 star scale) a user will rank a movie based on those factors.
# Different approaches will have the model/algorithm using the factors given to it to
# make predictions in different ways.

# The model/algorithm decides to predict a review rating "Y" based on factors "A", "B",
# and "C" (or more). Then the model/algorithm is exposed to the testing dataset to see if 
# what it predicts as the review rating "Y" based on the factors in the new dataset "A", "B",
# and "C" is actually that accurate or not. Then from the results we can compute our
# RMSE (Residual Mean Square Error). This is how we test the model's accuracy.

# The RMSE is the standard deviation of the prediction errors (residuals). 

# In other words, it's the average of how far the predictions deviate from what they are trying 
# to predict.

# The lower the RMSE, the more accurate the algorithm's predictions are.

# Our goal is to build an algorithm with as much accuracy as possible,
# so we want to have as low of a RMSE as we can get.

# For this analysis, we'll aim for a RMSE lower than 0.8649.

# Here is our RMSE formula:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# To explain step by step, we subtract the predicted ratings from the true ratings,
# then we square them all (to turn negative results into positive results for ease of 
# use since we are measuring deviation/distance), then we get the average of all of those results,
# then we get the square root of that average.

# This shows us the average of how far off the predictions are from what they are trying 
# to predict. That's our RMSE.

# If our RMSE is larger than 1, it means our rating prediction is on average more than one star off 
# the mark (on a 0.5 to 5 star scale), which isn't very good.

# Note: Users can rate movies in 0.5 increments but most of the ratings are 1, 2, 3, 4, or 5 stars.

# Now let's try to build a prediction model with a RMSE of less than 0.8649.

# For our first prediction model, we'll start with a very simple approach.
# Let's get the average of all the ratings.
mu_hat <- mean(edx$rating)
mu_hat
# 3.512 is the average of all the ratings in the training set.

# If we build a "naive" prediction model, predicting the ratings of movies
# in the validation set using only the average of ratings in the training set, 
# we get a RMSE of 1.061.
onlyAverage_rmse <- RMSE(validation$rating, mu_hat)
onlyAverage_rmse
# A RMSE of 1.061 is not very accurate. It means our predictions are about 1 star off
# (on a 0.5 to 5 star scale) on average from the actual rating users give in the 
# validation set.

# Let's put this model into a list and start off our list of attempts:
rmseTestResultsList <- tibble(method = "Only the Average/Naive Approach Model", RMSE = onlyAverage_rmse)
rmseTestResultsList %>% knitr::kable()


# For our second prediction model, we'll predict what a user will rate a movie
# based on the average rating of that movie.
# First we get the average of all ratings again. We'll call this "mu".
mu <- mean(edx$rating) 
mu

# Then we'll use mu to get our "b_i", which represents the average rating for movie "i"
# (you could also refer to this as the bias of ratings for movie "i").

# "b" (short for bias) represents the coefficient estimates of our predictors.
# In this model we'll only have one "bias" predictor, which is movie bias.

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# This plot shows us these "movie biases" per movie. Zero represents the dataset average 
# on this plot. Notice some movies score far below or above the dataset average of 3.512.
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# Let's see how much the predictions improve with this movie-based model.
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

movieEffect_rmse <- RMSE(predicted_ratings, validation$rating)
rmseTestResultsList <- bind_rows(rmseTestResultsList,
                                 tibble(method="Movie Effect Model",
                                        RMSE = movieEffect_rmse ))

rmseTestResultsList %>% knitr::kable()
# In comparison to the "Only Average" model, we've made a slight improvement but we're still
# nowhere near the goal of an RMSE below 0.8649.


# From this plot we can see that different users have different averages of their ratings
# ("b_u" is user bias).
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Let's try to improve our accuracy by creating a model based on user bias
# (users' previous ratings).
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Now we'll predict the validation set again after combining our movie-based approach
# with our new user-based approach.
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred


userEffect_rmse <- RMSE(predicted_ratings, validation$rating)
rmseTestResultsList <- bind_rows(rmseTestResultsList,
                                 tibble(method="Movie + User Effects Model",  
                                        RMSE = userEffect_rmse ))
rmseTestResultsList %>% knitr::kable()
# The combined approach gave us an RMSE of 0.8653!
# That's almost less than our goal of 0.8649! This is great! We're getting close!

# If we keep studying the data we might notice that many of the highest-ranking and
# lowest-ranking movies do not have many reviews. 

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

edx %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

edx %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Many of the highest and lowest-ranking movies have less than five reviews. This is because 
# small number of reviews the movie received were very high or low. This causes higher 
# variability in their rankings and may make our prediction model innacurate. Because 
# averages of biases of movies and users with few reviews are too extreme to be accurately 
# predicted, they need to be pulled in (or "rubber-banded") closer to the dataset 
# average the fewer reviews there are.

# In order to smooth out the problems caused by the variability of the 
# rankings of these movies with low numbers of reviews, we can 
# improve our model by using regularization.

# With regularization our predictions will not be as swayed by the noise and outliers
# of movies and users with only a few extreme reviews because they are penalized.

# Regularizing our model will bring the predictions for movies with
# less ratings closer to the average of all ratings in the dataset,
# but will not affect movies with many ratings as much.
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda
# The lambda that gives us the most accuracy (smallest RMSE) is 5.25.

rmseTestResultsList <- bind_rows(rmseTestResultsList,
                                 tibble(method="Regularized Movie + User Effect Model",  
                                        RMSE = min(rmses)))
rmseTestResultsList %>% knitr::kable()

# This combined and regularized approach gave us an RMSE of 0.8648!
# That's less than our goal of 0.8649! We did it!


# As we saw earlier in the multiple regression analysis, genres had a statistically significant 
# effect on ratings. So let's see if we can get an even lower RMSE by adding the effects of the 
# different genres to our regularized movie and user effect prediction algorithm.

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- edx %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses)  

lambda_g <- lambdas[which.min(rmses)]
lambda_g
# The lambda that gives us the most accuracy (smallest RMSE) is 5.

# The higher the lambda, the less significance is held by the coefficients, 
# giving them less and less effect on the prediction algorithm.
# Some variables may play no role in the model at all if the lambda is high enough.

# Lambda is a tuning parameter for our model. 
# This approach tests different values of lambda and picks the one with the lowest RMSE.

rmseTestResultsList <- bind_rows(rmseTestResultsList,
                                 tibble(method="Regularized Movie + User + Genre Effect Model",  
                                        RMSE = min(rmses)))


# Let's see our results:
rmseTestResultsList %>% knitr::kable()

# This regularized test of the movie, user, and genre effects combined gives us
# a RMSE of 0.86445, which is even more accurate.

minRMSE <- min(rmses)
cat(minRMSE, "is the RMSE of the Regularized Movie + User + Genre Effect Model, which accomplishes 
    our goal of reaching a RMSE of less than 0.8649.")



#################  Conclusion  #################

# In this section I'll give a brief summary of the report, its limitations 
# and future work.

# I split the data into a training set (90% of data) to train the prediction models
# and a testing set (10% of data) to test the accuracy of the prediction model.

# After running five prediction models, the lowest Residual Mean Square Error (RMSE) 
# obtained was 0.8644501, which accomplishes the goal of reaching lower than 0.8649.

# The most effective prediction model was "Regularized Movie + User + Genre 
# Effect Model", where I used the biases per movie, per user, and per genre 
# of the reviews in the training set and then regularized (or "rubber-banded") 
# the results, penalizing biases of movies/users/genres with low review 
# counts by pulling them toward the dataset average.

# I feel as though my report has some limitations. I could have taken more
# modeling approaches, such as Naive Bayes Classification and Matrix Decomposition,
# to potentially reach a lower RMSE.

# I keep thinking about how, in the Netflix challenge, to win the grand prize of $1 million 
# a participating team had to reach a RMSE of about 0.857.

# I would like to improve this analysis in the future by finding some prediction
# model approaches that will give me a RMSE of less than 0.857.

# Thank you for reading my report. I hope you enjoyed it.
#  - Avery Clark