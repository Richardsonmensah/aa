Please i am working on a project where i have to use neural network model on chainladder to predict claim reserves. i am new to neural work and i need help. bellow is the code i am using. anybody with knowledge in this can help. it can come at a cost. 




# Installing necessary libraries
library(keras)

# Test Input
file_raw_data <- file.choose()
file_triangular_data <- file.choose()
path_result <- "lob1_nnn.csv"

# Reading raw data - saving it under dat.x
dat.x <- read.csv(file_raw_data)

# Reading triangular data - saving it under dat
dat <- read.csv(file_triangular_data)

# Observed responses
dat.Y <- as.matrix(dat$Pay01/sqrt(dat$Pay00))

# Volumes used as offset
dat.W <- as.matrix(sqrt(dat$Pay00))

# Number of hidden neurons
q <- 20

# Definition of neural network

# 1) We build network with q hidden neurons and excluding the offset
features <- layer_input(shape = c(ncol(dat.x)))
features

net <- features %>%
  layer_dense(units = q, activation = 'tanh')%>%
  layer_dense(units = 1, activation = k_exp )  

# 2) We build the offset part
volumes <- layer_input(shape = c(1))
offset <- volumes %>%
  layer_dense(units = 1, 
              activation = 'linear', 
              use_bias = FALSE, 
              trainable = FALSE,
              weights = list(array (1, dim = c(1 ,1))))

# 3) We merge the two parts to regression function
merged <- list(net , offset) %>%
  layer_multiply () 
model <- keras_model(inputs = list(features , volumes), 
                     outputs = merged)

# 4) We compile the model using square loss function and optimizer 'rmsprop'
model %>% compile(loss = 'mse', optimizer = 'rmsprop')

# Fitting the neural network
fit <- model %>% fit(list(dat.x, dat.W), dat.Y, epochs = 100, batch_size = 10000,
                     validation_split = 0.1)

# Predicting claims dat$C1 and in-sample loss
dat$pred <- as.vector(model %>% predict(list(dat.x, dat.W))) * sqrt(dat$Pay00)
sumSample <- sum((dat$Pay01 - dat$pred )^2/dat$Pay00)

# Write solution to csv-file
write.csv(dat$pred, path_result)

# Return results
test_result <- list("prediction" = dat$pred, "sum_sample" = sumSample))
test_result$prediction
test_result$sum_sample
