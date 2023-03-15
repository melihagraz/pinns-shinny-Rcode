
rm(list = ls())
# first:   install_tensorflow(version = "1.15.0")
# second:  need to create a virtual env namely r-eticulate
# than you can run the code below
library(reticulate)
use_condaenv('r-reticulate')
library(tensorflow)
tf$compat$v1$logging$set_verbosity(tf$compat$v1$logging$ERROR)

tf$set_random_seed(1234)
layers <- c(1, rep(50, 5), 1)
N_residual <- 100
N_test <- 80

x_f <- seq(0, 1, length.out=N_residual)

x <- c(0, 1)
y <- c(1, exp(1))
x_train <- matrix(x, nrow = 2, ncol = 1)
y_train <- matrix(y, nrow = 2, ncol = 1)
x_f_train <- matrix(x_f, nrow = length(x_f), ncol=1)

xavier_init <- function(size){
  
  in_dim <- size[1]
  out_dim <- size[2]
  xavier_stddev <- sqrt(2 / (in_dim + out_dim))
  return (tf$Variable(tf$truncated_normal(shape = shape(in_dim, out_dim),
         stddev=xavier_stddev, dtype=tf$float64), dtype=tf$float64))

}

initialize_NN <- function(layers){
  weights <-c()
  biases <- c()
  num_layers <- length(layers)-1
  for (i in seq(num_layers)){
    W <- xavier_init(c(layers[i], layers[i+1]))
    weights <- c(weights, W)
    b <- tf$Variable(tf$zeros(shape = shape(1, layers[i+1]),
                           dtype=tf$float64), dtype=tf$float64)
    biases <- c(biases, b)
}
  return(list(Weights = weights, Biases = biases))
}


neural_net <- function(x_, weights, biases){
  num_layers <- length(weights) + 1
  H <- x_
  for (l in seq(num_layers-2)){
    W <- weights[[l]]
    b <- biases[[l]]
    H <- tf$nn$tanh(tf$add(tf$matmul(H, W), b))
  }
  W <- tail(weights, n=1)[1]
  b <- tail(biases, n=1)[1]
  Y <- tf$add(tf$matmul(H, W), b)
  return(Y)
}

net_y <- function(x_){
  out <- neural_net(x_, weights, biases)
  return(out)
}


net_f <- function(x_){
  y_ <- net_y(x_)
  y_x <- tf$gradients(y_, x_)[[1]]
  # Residuals
  f <- y_x - y_
  return(f)
}

obj <- initialize_NN(layers)
weights <- obj$Weights
biases <- obj$Biases

sess <- tf$Session(config=tf$compat$v1$ConfigProto(allow_soft_placement=T,
                   log_device_placement=T))
saver <- tf$compat$v1$train$Saver()

x_tf <- tf$placeholder(tf$float64, shape=shape(NULL, dim(x_train)[2]))
y_tf <- tf$placeholder(tf$float64, shape=shape(NULL, dim(y_train)[2]))

x_f_tf <- tf$placeholder(tf$float64, shape=shape(NULL, dim(x_f_train)[2]))
y_pred <- net_y(x_tf)
f_pred <- net_f(x_f_tf)
loss_bd <- tf$reduce_mean(tf$square(y_pred - y_train))
loss_res <- tf$reduce_mean(tf$square(f_pred))
loss <- loss_bd + loss_res
optimizer_Adam <- tf$compat$v1$train$AdamOptimizer(1e-3)
train_op_Adam <- optimizer_Adam$minimize(loss)
init <- tf$global_variables_initializer()
sess$run(init)
iter <- 10000
total_loss <- c()
bd_loss <- c()
res_loss <- c()
for (i in seq(iter+1)){
  
  sess$run(train_op_Adam, feed_dict = dict(x_tf= x_train,
           y_tf= y_train, x_f_tf= x_f_train))
  bd <- as.numeric(sess$run(loss_bd, feed_dict = dict(x_tf= x_train,
              y_tf= y_train, x_f_tf= x_f_train)))
  res <- as.numeric(sess$run(loss_res, dict(x_tf= x_train,
               y_tf= y_train, x_f_tf= x_f_train)))
  bd_loss <- c(bd_loss, bd)
  res_loss <- c(res_loss, res)
  total <- sum(c(bd, res))
  total_loss <- c(total_loss, total)
  cat("\nIteration", i, "\t BD Loss | ", bd, "\t Res Loss | ",
      res, "\t Total Loss | ", total)
}

# save all losses in dataframe
loss_history <- list(Total_Loss = total_loss,
                      BD_loss = bd_loss,
                      Res_loss = res_loss)

loss_history$Epochs <- seq(0, 10000, by= 1)

# create sequence of slected epochs
x <- seq(0, 10000, by= 100)
x_df <- do.call(cbind.data.frame, loss_history)
rownames(x_df) <- seq(0, 10000, by= 1)

# filter by selected epochs
df <- x_df[x_df$Epochs  %in% x,]
head(df)

graphics.off()
par(mfrow=c(1,3))
plot(df$Epochs, df$Total_Loss, pch=2,
     type="l", lty=2, col="red",
     ylim = c(min(df$Total_Loss), .02),
     xlab = "Epochs", ylab="Total Loss")
plot(df$Epochs, df$BD_loss, pch=2, type="l",
     lty=2, col="red",
     ylim = c(min(df$BD_loss), .009),
     xlab = "Epochs", ylab="BD Loss")
plot(df$Epochs, df$Res_loss,
     pch=2, type="l", lty=2, col="red",
     ylim = c(min(df$Res_loss), .02),
     xlab = "Epochs", ylab="Res Loss")


predict_test <- function(x_star){
  y_star = sess$run(y_pred, dict(x_tf= x_star))
  return (y_star)
}

exact_solution <- function(data){
  return (exp(data))
}

test = seq(0, 1, length.out=N_test)
x_test <- matrix(test, nrow = length(test), ncol=1)

# apply functions
y <- predict_test(x_test)
y_e <- exact_solution(x_test)
yy <- matrix(y, ncol = dim(y)[1], nrow = dim(y)[2])
mse =sqrt(rowMeans((y_e - yy)^2)) / sqrt(rowMeans((y_e)^2))
line_df <- data.frame(x_test = x_test, Exact = y_e,
                      PINNs = yy)

# plots
library(reshape2)
library(ggplot2)
library(ggpubr)
library(facetscales)
mdf <- melt(df, id.vars="Epochs", value.name="value",
            variable.name="Losses")

scales_y <- list(
  `Total Loss` = scale_y_continuous(limits = c(min(df$Total_Loss), .02)),
  `BD Loss` = scale_y_continuous(limits = c(min(df$BD_loss), .009)),
  `Res Loss` = scale_y_continuous(limits = c(min(df$Res_loss), .02))
)

levels(mdf$Losses) <- c("Total Loss", "BD Loss", "Res Loss")

# ggplot(mdf, aes(x=Epochs, y=value,
#        group = Losses, colour = Losses)) +
#   geom_line()+
#   theme_pubr()+
#   labs(x = "Epochs", y = "Value", title = "Losses")+
#   facet_grid(.~ Losses, scales = "free_y")+
#   facet_grid_sc(rows = vars(Losses), scales = list(y = scales_y))+
#   theme(panel.spacing = unit(0.1, "lines"),
#         panel.border = element_rect(color = "black",
#          fill = NA, size = 1), 
#         axis.text.y = element_text(size=8),
#         strip.background = element_rect(color = "black", size = 1))+
#   scale_colour_manual(values=c("grey39","goldenrod",
#                                "dodgerblue"))
# save plot
# ggsave("plot1.pdf", width = 20,  height = 18,
#        units = "cm")
dev.off()

# line plot
line_df <- data.frame(x_test = x_test, Exact = y_e,
                      PINNs = yy)
df1 <- melt(line_df, id.vars="x_test", value.name="value",
            variable.name="Test")

ggplot(df1, aes(x=x_test, y=value, color=Test)) + 
  geom_line(aes(linetype=Test), size=1) +
  geom_point(aes(shape=Test), size=1) +
  scale_linetype_manual(values = c(1,2)) +
  scale_shape_manual(values=c(0,1))+
  scale_colour_manual(values=c("darkred",
          "skyblue"))+
  labs(x = "X Test", y = "Solution")+
  theme_pubclean()

# save plot
ggsave("plot2.pdf", width = 15,  height = 10,  units = "cm")
dev.off()


df2 <- data.frame(X_test = x_test, Error = abs(y_e - yy))
ggpubr::ggline(df2, "X_test", "Error",
               numeric.x.axis=T, color="darkgreen",
               shape=10, point.size=0.5,
               xlab = "X test", ylab = "Error",
    title = paste("MSE = ",format(mse, scientific = TRUE)))
ggsave("plot3.pdf", width = 15,  height = 10,  units = "cm")
dev.off()
