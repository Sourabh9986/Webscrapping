library(MASS)
library(car)
library(caret)
library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)

webscrap<-read.csv("Book1.csv")

View(webscrap)

#Change target variable to factors and binary form 
webscrap$isscrappy<- factor(webscrap$isscrappy)
webscrap$isscrappy<- ifelse(webscrap$isscrappy=="Y",1,0)

#Check Scrapping rate
Scrapping_rate <- sum(webscrap$isscrappy)/nrow(webscrap)

#Change categorical variables to dummy 
webscrap$Webserver <- factor(webscrap$Webserver, levels = c("A",levels(webscrap$Webserver)))
dummy_1 <- data.frame(model.matrix( ~Webserver, data = webscrap))
dummy_1 <- dummy_1[,-1]
Scrap_1 <- cbind(webscrap[,-2], dummy_1)
View(Scrap_1)

webscrap$CMS <- factor(webscrap$CMS, levels = c("A",levels(webscrap$CMS)))
dummy_1 <- data.frame(model.matrix( ~CMS, data = webscrap))
dummy_1 <- dummy_1[,-1]
Scrap_2 <- cbind(Scrap_1[,-2], dummy_1)
View(Scrap_2)

Fscrap<- Scrap_2[,-1]
View(Fscrap)


########################################################################
# splitting the data between train and test
set.seed(100)

# randomly generate row indices for train dataset
trainindices= sample(1:nrow(Fscrap), 0.7*nrow(Fscrap))
# generate the train data set
train = Fscrap[trainindices,]

#Similarly store the rest of the observations into an object "test".
test = Fscrap[-trainindices,]

########################################################################
# Logistic Regression: 

#Initial model
model_1 = glm(isscrappy ~ ., data = train, family = "binomial")
summary(model_1) 

# Stepwise selection
library("MASS")
model_2<- stepAIC(model_1, direction="both")

summary(model_2)


# Removing multicollinearity through VIF check
library(car)
vif(model_2)


#exclude WebserverGWS


model_3<- glm(formula = isscrappy ~ CMSContao + CMSDotclear + 
                CMSJahia.Community.Distribution + CMSJamroom + CMSNucleus.CMS + 
                CMSOpenWGA, family = "binomial", data = train)
summary(model_3)
vif(model_3)

#exclude CMSContao


model_4<- glm(formula = isscrappy ~ CMSDotclear + 
                CMSJahia.Community.Distribution + CMSJamroom + CMSNucleus.CMS + 
                CMSOpenWGA, family = "binomial", data = train)
summary(model_4)
vif(model_4)


#exclude CMSNucleus.CMS

model_5<- glm(formula = isscrappy ~ CMSDotclear + 
                CMSJahia.Community.Distribution + CMSJamroom + 
                CMSOpenWGA, family = "binomial", data = train)
summary(model_5)
vif(model_5)


final_model<- model_5


### Model Evaluation

### Test Data ####

#predicted probabilities of scrapped websites for test data

test_pred = predict(final_model, type = "response", 
                    newdata = test[,-1])


# Let's see the summary 

summary(test_pred)

test$prob <- test_pred
View(test)

# Let's use the probability cutoff of 40%.

test_pred_at <- factor(ifelse(test_pred >= 0.40, "Yes", "No"))
test_actual_at <- factor(ifelse(test$isscrappy==1,"Yes","No"))


table(test_actual_at,test_pred_at)

#######################################################################
test_pred_at <- factor(ifelse(test_pred >= 0.43, "Yes", "No"))

test_conf <- confusionMatrix(test_pred_at, test_actual_at, positive = "Yes")
test_conf


# Let's Choose the cutoff value. 
# 

# Let's find out the optimal probalility cutoff 

perform_fn <- function(cutoff) 
{
  predicted_at <- factor(ifelse(test_pred >= cutoff, "Yes", "No"))
  conf <- confusionMatrix(predicted_at, test_actual_at, positive = "Yes")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  out <- t(as.matrix(c(sens, spec, acc))) 
  colnames(out) <- c("sensitivity", "specificity", "accuracy")
  return(out)
}


# Summary of test probability

summary(test_pred)

s = seq(.01,.80,length=100)

OUT = matrix(0,100,3)


for(i in 1:100)
{
  OUT[i,] = perform_fn(s[i])
} 


plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1), type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))


cutoff <- s[which(abs(OUT[,1]-OUT[,2])<0.01)]

# Let's choose a cutoff value of 0.44 for final model

test_cutoff_at <- factor(ifelse(test_pred >=0.44, "Yes", "No"))

conf_final <- confusionMatrix(test_cutoff_at, test_actual_at, positive = "Yes")

acc <- conf_final$overall[1]

sens <- conf_final$byClass[1]

spec <- conf_final$byClass[2]

acc

sens

spec

