# Basic Operations in R

my_vec = c(1,2,3,4,5,6)

my_seq=seq(from=1.5, to=4.2, by=0.1)

my_rep= rep("A", 10)

y=1:100


# r is indexed from 1 and not from 0

y[1:3] # will get first 3 values thus the indexing in inclusive

my_vec[my_vec>2] # sub setting the results based on some condition

sum(my_vec>2) # no of places the condition is satisfied

which(my_vec>2) # get the index of the resultant output

as.numeric(my_vec>2) # convert the values to numeric type

max(my_vec)

which.max(my_vec) # get the index of max element

length(my_vec) # get the length of the vector

############################   Conditional Opeartions in R

x=2
y=2

if (x==1) {
  print("x is equal to 1")
} else {
  print("x is not 1")
}

ifelse(x==2, 1, 0)

################################  Data strcutures in R

x=1:9

my_matrix= matrix(x, nrow = 3, ncol = 3) # create a matrix 

my_matrix[1,]

cbind(col1=x, col2=rev(x), col3= rep(1,9)) # create a matrix by doing column binding

# For matrix multiplication you should use %*%

dim(my_matrix) # get the shape
nrow(my_matrix) # no of rows
ncol(my_matrix) # no of col
rowSums(my_matrix) # summing each row
colMeans(my_matrix) # mean of each col

list(42, "Mohit", TRUE) # list enable to have multiple data types


# create a data frame with 3 columns
my_data_frame <- data.frame(
  x= c(1,2,3,4,5,6,7,8,9,0),
  y= c(rep("Hello", 9), "Goodbye"),
  z= rep(c(TRUE, FALSE), 5)
)


# generate a named list
list(
  x= c(1,2,3,4,5,6,7,8,9,0),
  y= c(rep("Hello", 9), "Goodbye"),
  z= rep(c(TRUE, FALSE), 5)
)

str(my_data_frame) # lets you know about the structure of your data frame

install.packages("mosaicData")
library("mosaicData")

View(Galton) # have a look at the entire dataset
head(Galton, n=10) # get first 10 rows
names(Galton) # get the columns names in the dataframe
Galton$family # get a column from a data frame
Galton["father"] # will give the datframe with father column
Galton[["father"]] # will give me the vectors with values
subset(Galton, subset = height>70) # subseting your data based upon a condition


library(tibble)
Galton = as_tibble(Galton) # a better form of a vanialla data frame
Galton$family # will give you a vector
# tibble is faster and allows us to keep char information as char whereas dataframe converted any char 
# column to level


# dont use attach() method since it will make the data frame variable directly accesible in enviroment and cause a 
#lot of conflict when dealing with multiple dataset or other variables


###############  Use summary statistics in R
install.packages("ggplot2")
library(ggplot2)
mpg = ggplot2::mpg
?mpg # get the description of your dataset, where it came from, its source etc...

# center
mean(mpg$cty) # get the average
median(mpg$cty) # get the middle value


# spread
var(mpg$cty) # get the variance
sd(mpg$cty) # get the standard deviation
range(mpg$cty) # get the max-min value
IQR(mpg$cty) # get interquartile range q3-q1


# summary
summary(mpg$cty) # get the summary statictic like quartiles, median etc

table(mpg$drv) # summarise the values in columns, like a value count in python

######## Visualisation in R

# when you have continous variable
hist(mpg$cty, 
     xlab="Miles per Gallon (City)",
     main= "Histogram of MPG (City)",
     breaks=12,
     col="darkorange",
     border = "dodgerblue")

# when you have categorical variable
barplot(table(mpg$drv),
        xlab="Drivetrain", 
        main="Drivetrains",
        border="darkorange",
        col = "dodgerblue")

# when you have continous vs categorical variab;e
boxplot(hwy ~ drv, data=mpg,
        xlab="Drivetrain",
        ylab="Miles per Gallon",
        main="Mpg Highway vs Drivetrains",
        border="darkorange",
        col = "dodgerblue", 
        pch=20,
        cex=2)


# when you have 2 continous variable
plot(hwy ~ displ, data=mpg,
     xlab="Engine Displacement",
     ylab="Miles per Gallon",
     main="Mpg Highway vs Engine displacement",
     border="darkorange",
     col = "dodgerblue", 
     pch=20,
     cex=2)

# other plotting library options
library(lattice)
xyplot(hwy~displ, data=mpg)

library(ggplot2)
ggplot(mpg, aes(x=displ, y= hwy)) + geom_point()



install.packages("rmarkdown")


######################

dnorm(x, mean, sd) # helps calculate the pdf , d stands for density
pnorm(x, mean, sd) # helps calculate the cdf, p stands for prob
qnorm(prob, mean, sd) # gives u the value of x, q stands for quantile
rnorm(n, mean, sd) # will generate random observation from a distribution, r stands for random

# these above functions can be easily vectorised for ex:
dnorm(c(1,2,3), mean= c(5,7,9), sd= c(2,1,4))


################## Linear modelling 

dist_model= lm(dist ~ speed, data=cars) # dist is the target and speed in the explanatory variable

plot(dist~speed, data=cars,
     xlab="Speed",
     ylab="Stopping distance",
     main="Stopping distance vs Speed",
     pch=20,
     cex=2,
     col="grey")
abline(dist_model, lwd=3, col="darkorange") # this will add the fitted line to the plot

names(dist_model) # will get you the content names of your fitted model
dist_model$coefficients # will give you the fitted cofficeints
dist_model$residuals # will give you the residual
dist_model$fitted.values # will give you the predicted values of training set


coef(dist_model) # similar to dist_model$coefficients
# similary we have fitted() and resid() functions

summary(dist_model) # gives u a lot of useful info about your fitted model
names(summary(dist_model)) # gives access to some additional summary values


# use these methods to get the prediction from your model
predict(dist_model, newdata = data.frame(speed=8))
predict(dist_model, newdata = data.frame(speed=c(8,21, 50)))
  
  

# when working with t distribution, always use pt() - for getting the prob at a critical value
# and qt() to get a critical value at a prob 


# confidence intervals

confint(stop_dist_model, level = 0.99) # calculates 99% confidence intervals for beta_0, beta_1

# calculate critical value for two-sided 99% CI
crit = qt(0.995, df = length(resid(stop_dist_model)) - 2) # always do it for 2 tail , thereby adding 0.99+ ((1-0.99)/2)

predict(stop_dist_model, newdata = new_speeds, 
        interval = c("confidence"), level = 0.99) # ci for mean response

predict(stop_dist_model, newdata = new_speeds, 
        interval = c("prediction"), level = 0.99) # prediction interval for new observation

# in the regression test we always do a 2 tail  test thats the reason when you have a test scatistic
# you should multiply the value of pt() by 2
# 2*(pt(abs(t_value), df=5, lower.tail=FALSE))


####  MLR
mpg_model= lm(mpg~wt+year, data=autompg)

# Ordinary least square
beta_hat= solve(t(X)%*%X) %*% t(X) %*% y


  
# for dummy variables first make sure those as coded as factor variables and then simply pass it to lm, the
#first one gets assigned 0 value and then +1 for every succesive class based on alphabetical order, thus is 
# this case for k classes in categorical variable you need k-1 dummy variable where the intercept will act as a
# reference level and you will have cofficeint for every other class.


#Inorder to have each of the regression line to take different slope we can add interactions to the model
mpg_disp_int = lm(mpg ~ disp + domestic + disp:domestic, data = autompg) # intercation between disp and domestic
# or
mpg_disp_int2 = lm(mpg ~ disp * domestic, data = autompg) # same as above
disp*domestic*mpg # get 3 way interaction model


# 4 ways of writing the same model
lm(mpg~disp*cyl,data=autompg)
lm(mpg~0+cyl+disp:cyl)
lm(mpg~0+cyl*disp)
lm(mpg~0+cyl+disp+disp:cyl)

# to create 3 way interactions you can also write it as 
lm(mpg~(disp+hp+domestic)^3,data=autompg)
#and to create all possible 2 way interaction you can simply right
lm(mpg~(disp+hp+domestic)^2,data=autompg)



  
#### Model Diaginostic
library(lmtest)
bptest(fit_1) # check hetrodecasity

qqnorm(resid(fit_1), main = "Normal Q-Q Plot, fit_1", col = "darkgrey")
qqline(resid(fit_1), col = "dodgerblue", lwd = 2)

airquality = na.omit(airquality)
mod=lm(log(Ozone)~Temp, data=airquality) # check for normality of errors
shapiro.test(resid(mod)) # check normality of errors


##### Outliers
hatvalues(model_1) # get the leverages for each observation
hatvalues(model_1) > 2 * mean(hatvalues(model_1)) # flag out high leverage

resid(model_1) # get residuals
rstandard(model_1) # get standardised residual

cooks.distance(model_1) > 4 / length(cooks.distance(model_1)) # flag observation with large cooks distance



cd_mpg_hp_add = cooks.distance(mpg_hp_add)
large_cd_mpg = cd_mpg_hp_add > 4 / length(cd_mpg_hp_add)
cd_mpg_hp_add[large_cd_mpg]
mpg_hp_add_fix = lm(mpg ~ hp + am,
                    data = mtcars,
                    subset = cd_mpg_hp_add <= 4 / length(cd_mpg_hp_add))
plot(mpg_hp_add) # gives you residual vs fitted , qq plot, standardised residual , residual vs leverage

##### Transformations

initech_fit_log = lm(log(salary) ~ years, data = initech) # log transoformation
mark_mod_poly2 = lm(sales ~ advert + I(advert ^ 2), data = marketing) # polynomial 
# or you can get it done using poly function
mod=lm(y~poly(x,4), data=xyz) # this will add intercept, 1st order, 2nd order, 3 rd order, 4th order term


library(faraway)
hip_model = lm(hipcenter ~ ., data = seatpos)
car::vif(hip_model)
#or
car::vif # any one can be used

# partial corelation cofficeint
mod=lm(y~., data=quiz_data)
x1_model=lm(x1~.-y, data=quiz_data)
cor(resid(mod), resid(x1_model))


extractAIC(mod) # to get the AIC from a model

# to do a exhaustive search for the best model
library(leaps)
regsubsets(hip~., data=abc)


hipcenter_mod_back_aic = step(hipcenter_mod, direction = "backward", trace=0)# backward selection using AIC
hipcenter_mod_back_bic = step(hipcenter_mod, direction = "backward", trace=0, k=log(n))# backward selection using BIC


# function to calculate loocv
calc_loocv_rmse = function(model) {
  sqrt(mean((resid(model) / (1 - hatvalues(model))) ^ 2))
}
calc_loocv_rmse(hipcenter_mod_back_aic)


# perform forward selection
hipcenter_mod_start = lm(hipcenter ~ 1, data = seatpos)
hipcenter_mod_forw_aic = step(
  hipcenter_mod_start, 
  scope = hipcenter ~ Age + Weight + HtShoes + Ht + Seated + Arm + Thigh + Leg, 
  direction = "forward")


# Logistic regression

fit_glm = glm(y ~ x, data = example_data, family = binomial)

fit_glm = glm(y ~ x, data = example_data, family = binomial(link = "logit")) # you can use link as response to get prob

anova(chd_mod_ldl, chd_mod_additive, test = "LRT") # compare 2 nested models
lrtest(mod1) # likelihood ratio test


predict(chd_mod_selected, new_obs, type = "link") # get the value of linear combination
predict(chd_mod_selected, new_obs, type = "response") # get the probability


eta_hat = predict(chd_mod_selected, new_obs, se.fit = TRUE, type = "link") # get the SE for the mean response   
z_crit = round(qnorm(0.975), 2) # get critical value
eta_hat$fit + c(-1, 1) * z_crit * eta_hat$se.fit # get CI for log odds
boot::inv.logit(eta_hat$fit + c(-1, 1) * z_crit * eta_hat$se.fit) # get CI for prob that y=1 

confint(logistic_model, level = 0.99) # get the CI for params

boot::cv.glm(spam_trn, fit_caps, K = 5)$delta[1] # cv for logistic model



