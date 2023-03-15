# install.packages("torch")
# install.packages("R.matlab")
# install.packages("pracma")
# install.packages("akima")
rm(list=ls())
# Load Matlab packages
library(R.matlab)
library(pracma)
library(torch)
library(akima)
# Set seed for R
set.seed(1234)
torch_manual_seed(1234)

nu <- 0.01 / pi
N_u <- 2000
layers <- c(2, 20, 20, 20, 20, 20, 20, 20, 20, 1)
# Load data
data_burger <- readMat("burgers_shock.mat")

t <- as.vector(data_burger$t)
x <- as.vector(data_burger$x)
Exact <- t(data_burger$usol)

grid <- meshgrid(x, t)
X <- grid$X
T <- grid$Y
X_star <- torch_stack(c(torch_flatten(torch_tensor(X)), torch_flatten(torch_tensor(T))))$t()

u_star <- torch_flatten(torch_tensor(Exact))$unsqueeze(1)$t()
# Doman bounds
lb <- apply(X_star, 2, min)
ub <- apply(X_star, 2, max)
######################################################################
############################## Model #################################
######################################################################

feed_forward_network <- nn_module(
  classname="feed_forward_network",
  initialize=function(layers) {
    self$layers <- layers
    self$layers_modules <- list()
    for (i in 1:(length(layers) - 2)) {
      self$layers_modules <- append(self$layers_modules,
                                    nn_linear(layers[i], layers[i+1]))
      self$layers_modules <- append(self$layers_modules, nn_tanh())
    }
    self$layers_modules <- append(self$layers_modules,
                                  nn_linear(layers[length(layers) - 1],
                                            layers[length(layers)]))
    self$layers_modules <- nn_module_list(self$layers_modules)
  },
  forward=function(x, t) {
    x <- torch_stack(c(x, t), 2)
    for (i in 1:length(self$layers_modules)) {
      x <- self$layers_modules[[i]]$forward(x)
    }
    x
  }
)

loss_fn <- function(u_labels, model_output) {
  u_pred <- model_output$u
  f_pred <- model_output$f
  loss <- torch_mean(torch_square(u_labels - u_pred)) + torch_mean(torch_square(f_pred))
  return(loss)
}

physics_model <- nn_module(
  classname = "physics_model",
  initialize = function(layers, lb, ub) {
    self$nn_net <- feed_forward_network(layers)
    # Initialize parameters
    self$lambda_1 <- nn_parameter(torch_zeros(size = c(1, 1)),
                                  requires_grad = TRUE)
    self$lambda_2 <- nn_parameter(torch_ones(size = c(1, 1)) * -6,
                                  requires_grad = TRUE)
    self$lb <- lb
    self$ub <- ub
    
  },
  forward = function(x) {
    x$requires_grad <- TRUE
    x_tf <- x[, 1]
    t_tf <- x[, 2]
    u_star <- self$nn_net$forward(x_tf, t_tf)
    # net_f
    u <- self$nn_net$forward(x_tf, t_tf)
    u_grad_x <- autograd_grad(u, x_tf, grad_outputs = torch_ones_like(u),
                              create_graph = TRUE, retain_graph = TRUE)
    u_grad_t <- autograd_grad(u, t_tf, grad_outputs = torch_ones_like(u),
                              create_graph = TRUE, retain_graph = TRUE)
    u_x <- u_grad_x[[1]]
    u_t <- u_grad_t[[1]]
    u_x_grad <- autograd_grad(u_x, x_tf, grad_outputs = torch_ones_like(u_x),
                              create_graph = TRUE, retain_graph = TRUE)
    u_xx <- u_x_grad[[1]]
    f_star <- u_t + (self$lambda_1 * u_star[, 1] * u_x) - (torch_exp(self$lambda_2) * u_xx)
    list(u = u_star, f = f_star)
  }
)
######################################################################
######################## Noiseles Data ###############################
######################################################################
idx <- sample(dim(X_star)[1], N_u, replace = FALSE)

X_u_train <- X_star[idx, ]
u_train <- u_star[idx]

### network parameters ---------------------------------------------------------
model_phy <- physics_model(layers, lb, ub)
optimizer <- optim_adam(model_phy$parameters)
model_phy$train()
for (t in 1:1000) {
  ### -------- Forward pass --------
  y_pred <- model_phy$forward(torch_tensor(X_u_train))
  ### -------- compute loss --------
  loss <- loss_fn(u_train, y_pred)
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(),
        "Lambda", model_phy$lambda_1$item(),
        "Lambda 2", exp(model_phy$lambda_2$item()), "\n")
  ### -------- Backpropagation --------
  optimizer$zero_grad()
  # gradients are still computed on the loss tensor (no change here)
  loss$backward()
  ### -------- Update weights --------
  # use the optimizer to update model parameters
  optimizer$step()
}

lbfgs_optimizer <- optim_lbfgs(model_phy$parameters,
                               max_iter = 50000,
                               max_eval = 50000,
                               history_size = 200)
lbfgs_optimizer$step(function() {
  lbfgs_optimizer$zero_grad()
  ### -------- Forward pass --------
  y_pred <- model_phy$forward(torch_tensor(X_u_train))
  ### -------- compute loss --------
  loss <- loss_fn(u_train, y_pred)
  cat("Loss: ", loss$item(),
      "Lambda", model_phy$lambda_1$item(),
      "Lambda 2", exp(model_phy$lambda_2$item()), "\n")
  ### -------- Backpropagation --------
  loss$backward()
  loss
})


prediction <- model_phy$forward(torch_tensor(X_star))

u_pred <- prediction$u
f_pred <- prediction$f

error_u <- sqrt(sum(sum(torch_pow(u_star - u_pred, 2)))) / sqrt(sum(sum(torch_pow(u_star, 2))))

lambda_1_value <- model_phy$lambda_1$item()
lambda_2_value <- model_phy$lambda_2$item()

lambda_2_value <- exp(lambda_2_value)
error_lambda_1 <- abs(lambda_1_value - 1.0) * 100
error_lambda_2 <- abs(lambda_2_value - nu) / nu * 100

print(c("Error u:", error_u$item()))
print(c("Error l1:", error_lambda_1))
print(c("Error l2:", error_lambda_2))
######################################################################
############################# Plotting ###############################
######################################################################
library(ggplot2)
griddata <- function(coords, values, x_length, y_length) {
  x <- coords[, 2]
  y <- coords[, 1]
  z <- values
  grid <- interp(x, y, z,
                 xo = seq(min(x), max(x), length = x_length),
                 yo = seq(min(y), max(y), length = y_length),
                 linear = FALSE)
  grid
}
U_pred <- griddata(as.matrix(X_star), as.array(u_pred), dim(X)[1], dim(T)[2])
heatmap_data <- interp2xyz(U_pred, data.frame = TRUE)
scatter_data <- data.frame(as.matrix(X_u_train))
# Plot the results + scatter

ggplot(data = heatmap_data, aes(x = x, y = y)) + 
  geom_tile(data = heatmap_data, 
            aes(fill = z)) +
  scale_fill_distiller(palette = "Set1") +
  geom_point(data = scatter_data, 
             aes(x = X2, 
                 y = X1), color = "black")


plot(x, Exact[25, ], col="blue", type="l") +
  lines(x, U_pred$z[25, ], col="red", type="l", lty=2)
plot(x, Exact[50, ], col="blue", type="l") +
  lines(x, U_pred$z[50, ], col="red", type="l", lty=2)
plot(x, Exact[75, ], col="blue", type="l") +
  lines(x, U_pred$z[75, ], col="red", type="l", lty=2)



######################################################################
######################## Noise Data ##################################
######################################################################
noise <- 0.01

idx <- sample(dim(X_star)[1], N_u, replace = FALSE)

X_u_train <- X_star[idx, ]
u_train <- u_star[idx]
u_train <- u_train + noise * torch_std(u_train) * torch_randn(dim(u_train))

### network parameters ---------------------------------------------------------
model_phy <- physics_model(layers, lb, ub)

optimizer <- optim_adam(model_phy$parameters)
model_phy$train()
for (t in 1:1000) {
  ### -------- Forward pass --------
  y_pred <- model_phy$forward(torch_tensor(X_u_train))
  ### -------- compute loss --------
  loss <- loss_fn(u_train, y_pred)
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(),
        "Lambda", model_phy$lambda_1$item(),
        "Lambda 2", exp(model_phy$lambda_2$item()), "\n")
  ### -------- Backpropagation --------
  optimizer$zero_grad()
  # gradients are still computed on the loss tensor (no change here)
  loss$backward()
  ### -------- Update weights --------
  # use the optimizer to update model parameters
  optimizer$step()
}

lbfgs_optimizer <- optim_lbfgs(model_phy$parameters,
                               max_iter = 50000,
                               max_eval = 50000,
                               history_size = 200)
lbfgs_optimizer$step(function() {
  lbfgs_optimizer$zero_grad()
  ### -------- Forward pass --------
  y_pred <- model_phy$forward(torch_tensor(X_u_train))
  ### -------- compute loss --------
  loss <- loss_fn(u_train, y_pred)
  cat("Loss: ", loss$item(),
      "Lambda", model_phy$lambda_1$item(),
      "Lambda 2", exp(model_phy$lambda_2$item()), "\n")
  ### -------- Backpropagation --------
  loss$backward()
  loss
})


prediction <- model_phy$forward(torch_tensor(X_star))

u_pred <- prediction$u
f_pred <- prediction$f

error_u <- sqrt(sum(sum(torch_pow(u_star - u_pred, 2)))) / sqrt(sum(sum(torch_pow(u_star, 2))))

lambda_1_value <- model_phy$lambda_1$item()
lambda_2_value <- model_phy$lambda_2$item()

lambda_2_value <- exp(lambda_2_value)
error_lambda_1 <- abs(lambda_1_value - 1.0) * 100
error_lambda_2 <- abs(lambda_2_value - nu) / nu * 100

print(c("Error u:", error_u$item()))
print(c("Error l1:", error_lambda_1))
print(c("Error l2:", error_lambda_2))

griddata <- function(coords, values, x_length, y_length) {
  x <- coords[, 2]
  y <- coords[, 1]
  z <- values
  grid <- interp(x, y, z,
                 xo = seq(min(x), max(x), length = x_length),
                 yo = seq(min(y), max(y), length = y_length),
                 linear = FALSE)
  grid
}
U_pred <- griddata(as.matrix(X_star), as.array(u_pred), dim(X)[1], dim(T)[2])
heatmap_data <- interp2xyz(U_pred, data.frame = TRUE)
scatter_data <- data.frame(as.matrix(X_u_train))
# Plot the results + scatter

ggplot(data = heatmap_data, aes(x = x, y = y)) + 
  geom_tile(data = heatmap_data, 
            aes(fill = z)) +
  scale_fill_distiller(palette = "Set1") +
  geom_point(data = scatter_data, 
             aes(x = X2, 
                 y = X1), color = "black")


plot(x, Exact[25, ], col="blue", type="l") +
  lines(x, U_pred$z[25, ], col="red", type="l", lty=2)
plot(x, Exact[50, ], col="blue", type="l") +
  lines(x, U_pred$z[50, ], col="red", type="l", lty=2)
plot(x, Exact[75, ], col="blue", type="l") +
  lines(x, U_pred$z[75, ], col="red", type="l", lty=2)

