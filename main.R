# ======================================
#    (Problem klasyfikacji binarnej)
# ======================================
#install.packages("neuralnet")
#install.packages("scales")
#install.packages("caret")
#install.packages("corrplot")
#install.packages("GGally")
#install.packages("fastDummies")

setwd("C:/Users/cuker/Desktop/R/MachineLearning datasets")

# Wczytanie danych do kwalifikacji binarnej
waterQuality <- read.csv("waterQuality1.csv", header = TRUE, sep = ",")
head(waterQuality)
str(waterQuality)

waterQuality$ammonia <- as.numeric(waterQuality$ammonia)
waterQuality$is_safe <- as.numeric(waterQuality$is_safe)


str(waterQuality)

sum(is.na(waterQuality))

waterQuality = na.omit(waterQuality)

library(caret)

# Sprawdzenie wariancji każdej zmiennej
nzv <- nearZeroVar(waterQuality, saveMetrics = TRUE)

# Wyświetlenie zmiennych z niską wariancją
print(nzv)

# Usunięcie tych zmiennych
waterQuality <- waterQuality[, !nzv$nzv]

# Załaduj pakiety
library(corrplot)
library(GGally)
# Obliczenie macierzy korelacji
corr_matrix <- cor(waterQuality, use = "complete.obs")

# Wizualizacja macierzy korelacji
corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.8)

# Znalezienie zmiennych silnie skorelowanych (np. korelacja > 0.9)
high_corr <- findCorrelation(corr_matrix, cutoff = 0.9)

# Wyświetlenie indeksów zmiennych do usunięcia
print(high_corr)

#Nie występują silnie skolerowane zmienne
#Usunięcie silnie skorelowanych zmiennych
#df_reduced <- waterQuality[, -high_corr]

### Algorytm KNN ########################################################################################

library(caret)
library(dplyr)

set.seed(234)

sample_waterQuality <- waterQuality[sample(1:nrow(waterQuality), 400, replace = FALSE), ]



folds <- createFolds(sample_waterQuality$is_safe, k = 5)

kTune <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


miaryKNN_B_wlasna_cv <- data.frame(k = integer(),
                                   AUC = numeric(),
                                   Czulosc = numeric(),
                                   Specyficznosc = numeric(),
                                   Jakosc = numeric())

miaryKNN_B_caret_cv <- data.frame(k = integer(),
                                  AUC = numeric(),
                                  Czulosc = numeric(),
                                  Specyficznosc = numeric(),
                                  Jakosc = numeric())


for (i in kTune) {
  

  auc_vec_wlasna  <- c()
  sens_vec_wlasna <- c()
  spec_vec_wlasna <- c()
  jakosc_vec_wlasna <- c()
  
  auc_vec_caret  <- c()
  sens_vec_caret <- c()
  spec_vec_caret <- c()
  jakosc_vec_caret <- c()
  

  for(f in 1:5){
    

    test_index <- folds[[f]]
    
    train_data <- sample_waterQuality[-test_index, ]

    test_data  <- sample_waterQuality[ test_index, ]
    
    train_x <- model.matrix(is_safe ~ ., train_data)[, -1]
    test_x  <- model.matrix(is_safe ~ ., test_data)[, -1]
    
    train_y <- as.factor(train_data$is_safe)
    test_y  <- as.factor(test_data$is_safe)
    

    modelK <- KNNtrain(train_x, train_y, k = i, Xmin = 0, Xmax = 1)
    
    modelPredK <- KNNpred(modelK, test_x)
    
    y_hat <- as.numeric(modelPredK[, 2])
    

    ocena_wlasna <- ModelOcena(test_y, y_hat)

    auc_vec_wlasna  <- c(auc_vec_wlasna,  ocena_wlasna$Miary["AUC"])
    sens_vec_wlasna <- c(sens_vec_wlasna, ocena_wlasna$Miary["Czulosc"])
    spec_vec_wlasna <- c(spec_vec_wlasna, ocena_wlasna$Miary["Specyficznosc"])
    jakosc_vec_wlasna <- c(jakosc_vec_wlasna, ocena_wlasna$Miary["Jakosc"])
    

    train_x_norm <- normalize_data(train_x, XminNew = 0, XmaxNew = 1)
    test_x_norm  <- normalize_data(test_x,  XminNew = 0, XmaxNew = 1)
    modelPredCaret <- predict(
      knn3(x = train_x_norm, y = train_y, k = i),
      test_x_norm
    )
    
    y_hat_caret <- as.numeric(modelPredCaret[, 2])
    
    ocena_caret <- ModelOcena(test_y, y_hat_caret)
    auc_vec_caret  <- c(auc_vec_caret,  ocena_caret$Miary["AUC"])
    sens_vec_caret <- c(sens_vec_caret, ocena_caret$Miary["Czulosc"])
    spec_vec_caret <- c(spec_vec_caret, ocena_caret$Miary["Specyficznosc"])
    jakosc_vec_caret <- c(jakosc_vec_caret, ocena_caret$Miary["Jakosc"])
    
  }
  

  miaryKNN_B_wlasna_cv <- rbind(
    miaryKNN_B_wlasna_cv,
    data.frame(k = i,
               AUC = mean(auc_vec_wlasna),
               Czulosc = mean(sens_vec_wlasna),
               Specyficznosc = mean(spec_vec_wlasna),
               Jakosc = mean(jakosc_vec_wlasna))
  )
  
  miaryKNN_B_caret_cv <- rbind(
    miaryKNN_B_caret_cv,
    data.frame(k = i,
               AUC = mean(auc_vec_caret),
               Czulosc = mean(sens_vec_caret),
               Specyficznosc = mean(spec_vec_caret),
               Jakosc = mean(jakosc_vec_caret))
  )
  
} 
miaryKNN_B_wlasna_cv
miaryKNN_B_caret_cv

### Algorytm Drzewa Decyzyjnego ########################################################################################

library(caret)
library(rpart)


set.seed(234)


waterQuality$is_safe <- as.factor(waterQuality$is_safe)

depthTune <- c(2, 3, 4, 6)
minobsTune <- c(2, 3, 6)


miaryTree_B_wlasna <- data.frame(depth = integer(),
                                 minobs = integer(),
                                 AUC = numeric(),
                                 Czulosc = numeric(),
                                 Specyficznosc = numeric(),
                                 Jakosc = numeric()
)
miaryTree_B_rpart <- data.frame(depth = integer(),
                                minobs = integer(),
                                AUC = numeric(),
                                Czulosc = numeric(),
                                Specyficznosc = numeric(),
                                Jakosc = numeric()
)


set.seed(234)
folds <- createFolds(waterQuality$is_safe, k = 5)


for (d in depthTune) {
  for (m in minobsTune) {
    

    auc_wlasne_vec  <- c()
    sens_wlasne_vec <- c()
    spec_wlasne_vec <- c()
    jakosc_wlasne_vec <- c()
    

    auc_rpart_vec  <- c()
    sens_rpart_vec <- c()
    spec_rpart_vec <- c()
    jakosc_rpart_vec <- c()
    

    for(f in 1:5){

      test_index <- folds[[f]]
      

      train_data <- waterQuality[-test_index, ]
      test_data  <- waterQuality[ test_index, ]
      

      Yname <- "is_safe"
      Xnames <- c("aluminium","ammonia","arsenic","barium","cadmium",
                  "chloramine","chromium","copper","flouride","viruses",
                  "lead","nitrates","nitrites","mercury","perchlorate",
                  "radium","selenium","silver","uranium",
                  "is_safe") 

      tree_binary <- Tree(
        Y = Yname,
        X = Xnames,
        data = train_data,
        type = "Gini",       
        depth = d,        
        minobs = m,         
        overfit = "none",   
        cf = 0.01           
      )
      

      pred_binary <- PredictTree(tree_binary, test_data)
      y_hat_wlasne <- as.numeric(as.character(pred_binary$Klasa))
      y_test <- as.numeric(as.character(test_data$is_safe))
      

      ocena_wlasna <- ModelOcena(test_data$is_safe, y_hat_wlasne)
      auc_wlasne_vec  <- c(auc_wlasne_vec,  ocena_wlasna$Miary["AUC"])
      sens_wlasne_vec <- c(sens_wlasne_vec, ocena_wlasna$Miary["Czulosc"])
      spec_wlasne_vec <- c(spec_wlasne_vec, ocena_wlasna$Miary["Specyficznosc"])
      jakosc_wlasne_vec <- c(jakosc_wlasne_vec, ocena_wlasna$Miary["Jakosc"])
      
      f <- as.formula(paste(Yname, "~", paste(Xnames, collapse = " + ")))
      
      tree_rpart_binary <- rpart(
        f,
        data = train_data,
        method = "class",
        control = rpart.control(maxdepth = d, minsplit = m, cp = 0.01)
      )
      
      pred_rpart_binary <- predict(tree_rpart_binary, test_data, type = "class")
      y_hat_rpart <- as.numeric(as.character(pred_rpart_binary))
      
      ocena_rpart <- ModelOcena(test_data$is_safe, y_hat_rpart)
      auc_rpart_vec  <- c(auc_rpart_vec,  ocena_rpart$Miary["AUC"])
      sens_rpart_vec <- c(sens_rpart_vec, ocena_rpart$Miary["Czulosc"])
      spec_rpart_vec <- c(spec_rpart_vec, ocena_rpart$Miary["Specyficznosc"])
      jakosc_rpart_vec <- c(jakosc_rpart_vec, ocena_rpart$Miary["Jakosc"])
      
    } 
    
    miaryTree_B_wlasna <- rbind(
      miaryTree_B_wlasna,
      data.frame(depth = d,
                 minobs = m,
                 AUC = mean(auc_wlasne_vec),
                 Czulosc = mean(sens_wlasne_vec),
                 Specyficznosc = mean(spec_wlasne_vec),
                 Jakosc = mean(jakosc_wlasne_vec))
    )
    
    miaryTree_B_rpart <- rbind(
      miaryTree_B_rpart,
      data.frame(depth = d,
                 minobs = m,
                 AUC = mean(auc_rpart_vec),
                 Czulosc = mean(sens_rpart_vec),
                 Specyficznosc = mean(spec_rpart_vec),
                 Jakosc = mean(jakosc_rpart_vec))
    )
    

  }
}

miaryTree_B_wlasna_cv <- miaryTree_B_wlasna
miaryTree_B_rpart_cv <- miaryTree_B_rpart

miaryTree_B_wlasna_cv
miaryTree_B_rpart_cv
### Algorytm Sieci Neuronowej ########################################################################################

library(caret)
library(nnet)  

waterQuality$is_safe <- as.factor(waterQuality$is_safe)

set.seed(234)


Yname <- "is_safe"
Xnames <- c("aluminium", "ammonia", "arsenic", "barium", "cadmium",
            "chloramine", "chromium", "copper", "flouride", "viruses",
            "lead", "nitrates", "nitrites", "mercury", "perchlorate",
            "radium", "selenium", "silver", "uranium") 

hTune       <- list(c(5), c(10), c(10,5), c(20,10))
iterTune    <- c(2000, 5000)
lrTune      <- c(0.001, 0.01, 0.05)


hTune_nnet  <- c(5, 10, 20)  
iterTune_nnet <- c(1000, 2000)



miaryNN_B_wlasna_cv <- data.frame(
  hTune       = I(list()),
  iterTune    = integer(),
  lrTune      = numeric(),
  AUC         = numeric(),
  Czulosc     = numeric(),
  Specyficznosc = numeric(),
  Jakosc      = numeric()
)

miaryNN_B_nnet_cv <- data.frame(
  hTune       = integer(),
  iterTune    = integer(),
  AUC         = numeric(),
  Czulosc     = numeric(),
  Specyficznosc = numeric(),
  Jakosc      = numeric()
)


set.seed(123)
folds <- createFolds(waterQuality$is_safe, k = 5)


for (iter in iterTune) {
  for (h in hTune) {
    for (lr in lrTune) {
      

      auc_vec  <- c()
      sens_vec <- c()
      spec_vec <- c()
      jakosc_vec <- c()
      

      for (f in 1:5) {
        

        test_idx  <- folds[[f]]

        train_data <- waterQuality[-test_idx, ]
        test_data  <- waterQuality[ test_idx, ]
        
        
        train_data[Xnames] <- normalize_data(train_data[Xnames], XminNew = 0, XmaxNew = 1)
        test_data[Xnames]  <- normalize_data(test_data[Xnames],  XminNew = 0, XmaxNew = 1)
        

        nn_model <- trainNN(
          Yname  = Yname,
          Xnames = Xnames,
          data   = train_data,
          h      = h,         
          type   = "binary",
          lr     = lr,        
          iter   = iter,      
          seed   = 234
        )
        
        pred_prob <- predNN(nn_model, test_data[, Xnames], type = "binary")
        


        ocena <- ModelOcena(test_data[[Yname]], pred_prob)
        
        auc_vec  <- c(auc_vec,  ocena$Miary["AUC"])
        sens_vec <- c(sens_vec, ocena$Miary["Czulosc"])
        spec_vec <- c(spec_vec, ocena$Miary["Specyficznosc"])
        jakosc_vec <- c(jakosc_vec, ocena$Miary["Jakosc"])
        
      } 
      
      miaryNN_B_wlasna_cv <- rbind(
        miaryNN_B_wlasna_cv,
        data.frame(hTune = I(list(h)),
                   iterTune = iter,
                   lrTune = lr,
                   AUC = mean(auc_vec),
                   Czulosc = mean(sens_vec),
                   Specyficznosc = mean(spec_vec),
                   Jakosc = mean(jakosc_vec))
      )
      
    }
  }
}



for (h in hTune_nnet) {
  for (iter in iterTune_nnet) {
    

    auc_vec_nnet  <- c()
    sens_vec_nnet <- c()
    spec_vec_nnet <- c()
    jakosc_vec_nnet <- c()
    
    for (f in 1:5) {
      
      test_idx  <- folds[[f]]
      train_data <- waterQuality[-test_idx, ]
      test_data  <- waterQuality[ test_idx, ]
      

      train_data[Xnames] <- normalize_data(train_data[Xnames], XminNew = 0, XmaxNew = 1)
      test_data[Xnames]  <- normalize_data(test_data[Xnames],  XminNew = 0, XmaxNew = 1)
      

      set.seed(234)
      nn_nnet <- nnet(
        is_safe ~ .,
        data  = train_data,
        size  = h,         
        linout = FALSE,    
        maxit = iter,
        trace = FALSE     
      )
      

      pred_prob_nnet <- predict(nn_nnet, newdata = test_data, type = "raw")
      

      ocena_nnet <- ModelOcena(test_data[[Yname]], pred_prob_nnet)
      
      auc_vec_nnet  <- c(auc_vec_nnet,  ocena_nnet$Miary["AUC"])
      sens_vec_nnet <- c(sens_vec_nnet, ocena_nnet$Miary["Czulosc"])
      spec_vec_nnet <- c(spec_vec_nnet, ocena_nnet$Miary["Specyficznosc"])
      jakosc_vec_nnet <- c(jakosc_vec_nnet, ocena_nnet$Miary["Jakosc"])
      
    } 
    
    miaryNN_B_nnet_cv <- rbind(
      miaryNN_B_nnet_cv,
      data.frame(
        hTune = h,
        iterTune = iter,
        AUC = mean(auc_vec_nnet),
        Czulosc = mean(sens_vec_nnet),
        Specyficznosc = mean(spec_vec_nnet),
        Jakosc = mean(jakosc_vec_nnet)
      )
    )
    
  }
}


miaryNN_B_wlasna_cv
miaryNN_B_nnet_cv


# ======================================
#    (Problem klasyfikacji wieloklasowej)
# ======================================

bodyPerformance <- read.csv("bodyPerformance.csv", header = TRUE, sep = ",")
head(bodyPerformance)
sum(is.na(bodyPerformance))
str(bodyPerformance)
# Nie ma braków w danych

library(fastDummies)

bodyPerformance <- dummy_cols(bodyPerformance, select_columns = c("gender"), 
                              remove_selected_columns = TRUE, 
                              remove_first_dummy = TRUE)

library(caret)

# Sprawdzenie wariancji każdej zmiennej
nzv <- nearZeroVar(bodyPerformance, saveMetrics = TRUE)

# Wyświetlenie zmiennych z niską wariancją
print(nzv)

# Usunięcie tych zmiennych
bodyPerformance <- bodyPerformance[, !nzv$nzv]

# Załaduj pakiety
library(corrplot)
library(GGally)
# Obliczenie macierzy korelacji

bodyPerformance_x = model.matrix(class~., bodyPerformance)[,-1] # przycinam pierwszą kolumnę

corr_matrix <- cor(bodyPerformance_x, use = "complete.obs")

# Wizualizacja macierzy korelacji
corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.8)

# Znalezienie zmiennych silnie skorelowanych (np. korelacja > 0.9)
high_corr <- findCorrelation(corr_matrix, cutoff = 0.9)

# Wyświetlenie indeksów zmiennych do usunięcia
print(high_corr)

#Nie występują silnie skolerowane zmienne
#Usunięcie silnie skorelowanych zmiennych
#df_reduced <- bodyPerformance_x[, -high_corr]

### Algorytm KNN ########################################################################################


library(caret)
library(dplyr)


set.seed(234)
sample_bodyPerformance <- bodyPerformance[sample(1:nrow(bodyPerformance), 400, replace = FALSE), ]

folds <- createFolds(sample_bodyPerformance$class, k = 5)


kTune <- c(1,2,3,4,5,6,7,8,9,10)


miaryKNN_M_wlasna_cv <- data.frame(k = integer(),
                                   AUC = numeric(),
                                   Czulosc = numeric(),
                                   Specyficznosc = numeric(),
                                   Jakosc = numeric())
miaryKNN_M_caret_cv  <- data.frame(k = integer(),
                                   AUC = numeric(),
                                   Czulosc = numeric(),
                                   Specyficznosc = numeric(),
                                   Jakosc = numeric())

for(k in kTune) {
  

  auc_wlasne_vec  <- c()
  sens_wlasne_vec <- c()
  spec_wlasne_vec <- c()
  jakosc_wlasne_vec <- c()
  
  auc_caret_vec  <- c()
  sens_caret_vec <- c()
  spec_caret_vec <- c()
  jakosc_caret_vec <- c()
  

  for(f in 1:5) {
    

    test_idx <- folds[[f]]
    

    train_data <- sample_bodyPerformance[-test_idx, ]
    test_data  <- sample_bodyPerformance[ test_idx, ]
    

    train_data$class <- as.factor(train_data$class)
    test_data$class  <- as.factor(test_data$class)
    

    train_x <- model.matrix(class ~ ., data=train_data)[,-1]
    train_x <- as.data.frame(train_x)
    train_y <- train_data$class
    

    test_x <- model.matrix(class ~ ., data=test_data)[,-1]
    test_x <- as.data.frame(test_x)
    test_y <- test_data$class
    

    train_x <- normalize_data(train_x, XminNew = 0, XmaxNew = 1)
    test_x  <- normalize_data(test_x,  XminNew = 0, XmaxNew = 1)
    

    modelK <- KNNtrain(train_x, train_y, k)

    predK  <- KNNpred(modelK, test_x)

    y_hat_wlasne <- predK$Klasa
    

    ocena_wlasne <- ModelOcena(test_y, y_hat_wlasne)
    auc_wlasne_vec  <- c(auc_wlasne_vec,  ocena_wlasne$Miary["AUC"])
    sens_wlasne_vec <- c(sens_wlasne_vec, ocena_wlasne$Miary["Czulosc"])
    spec_wlasne_vec <- c(spec_wlasne_vec, ocena_wlasne$Miary["Specyficznosc"])
    jakosc_wlasne_vec <- c(jakosc_wlasne_vec, ocena_wlasne$Miary["Jakosc"])
    

    model_caret <- knn3(x=train_x, y=train_y, k=k)

    pred_caret_matrix <- predict(model_caret, test_x)

    y_hat_caret <- as.factor(
      colnames(pred_caret_matrix)[max.col(pred_caret_matrix, ties.method = "first")]
    )
    

    ocena_caret <- ModelOcena(test_y, y_hat_caret)
    auc_caret_vec  <- c(auc_caret_vec,  ocena_caret$Miary["AUC"])
    sens_caret_vec <- c(sens_caret_vec, ocena_caret$Miary["Czulosc"])
    spec_caret_vec <- c(spec_caret_vec, ocena_caret$Miary["Specyficznosc"])
    jakosc_caret_vec <- c(jakosc_caret_vec, ocena_caret$Miary["Jakosc"])
    
  } 
  

  miaryKNN_M_wlasna_cv <- rbind(
    miaryKNN_M_wlasna_cv,
    data.frame(
      k = k,
      AUC = mean(auc_wlasne_vec),
      Czulosc = mean(sens_wlasne_vec),
      Specyficznosc = mean(spec_wlasne_vec),
      Jakosc = mean(jakosc_wlasne_vec)
    )
  )
  
  miaryKNN_M_caret_cv <- rbind(
    miaryKNN_M_caret_cv,
    data.frame(
      k = k,
      AUC = mean(auc_caret_vec),
      Czulosc = mean(sens_caret_vec),
      Specyficznosc = mean(spec_caret_vec),
      Jakosc = mean(jakosc_caret_vec)
    )
  )
}


miaryKNN_M_wlasna_cv
miaryKNN_M_caret_cv



### Algorytm Drzewa Decyzyjnego ########################################################################################

library(caret)
library(rpart)



set.seed(234)


Yname <- "class"
Xnames <- c("age","height_cm","weight_kg","body.fat_.","diastolic","systolic","gripForce",
            "sit.and.bend.forward_cm","sit.ups.counts","broad.jump_cm","gender_M")

depthTune <- c(2, 3, 4, 6)
minobsTune <- c(2, 3, 6)

miaryTree_M_wlasna_cv <- data.frame(depth = integer(),
                                    minobs = integer(),
                                    AUC = numeric(),
                                    Czulosc = numeric(),
                                    Specyficznosc = numeric(),
                                    Jakosc = numeric())

miaryTree_M_rpart_cv <- data.frame(depth = integer(),
                                   minobs = integer(),
                                   AUC = numeric(),
                                   Czulosc = numeric(),
                                   Specyficznosc = numeric(),
                                   Jakosc = numeric())


set.seed(234)
folds <- createFolds(bodyPerformance$class, k = 5)


for(d in depthTune) {
  for(m in minobsTune) {
    

    auc_wlasne_vec  <- c()
    sens_wlasne_vec <- c()
    spec_wlasne_vec <- c()
    jakosc_wlasne_vec <- c()
    

    auc_rpart_vec  <- c()
    sens_rpart_vec <- c()
    spec_rpart_vec <- c()
    jakosc_rpart_vec <- c()
    

    for(f in 1:5) {
      

      test_idx <- folds[[f]]
      

      train_data <- bodyPerformance[-test_idx, ]
      test_data  <- bodyPerformance[test_idx, ]
      

      train_data$class <- as.factor(train_data$class)
      test_data$class  <- as.factor(test_data$class)
      

      tree_multi <- Tree(
        Y = Yname,
        X = Xnames,
        data = train_data,
        type = "Entropy",  
        depth = d,
        minobs = m,
        overfit = "none",
        cf = 0.01
      )
      

      pred_multi <- PredictTree(tree_multi, test_data)
      y_hat_wlasne <- pred_multi$Klasa
      

      ocena_wlasne <- ModelOcena(test_data[[Yname]], y_hat_wlasne)
      auc_wlasne_vec  <- c(auc_wlasne_vec,  ocena_wlasne$Miary["AUC"])
      sens_wlasne_vec <- c(sens_wlasne_vec, ocena_wlasne$Miary["Czulosc"])
      spec_wlasne_vec <- c(spec_wlasne_vec, ocena_wlasne$Miary["Specyficznosc"])
      jakosc_wlasne_vec <- c(jakosc_wlasne_vec, ocena_wlasne$Miary["Jakosc"])
      

      fmla <- as.formula(paste(Yname, "~", paste(Xnames, collapse = " + ")))
      
      tree_rpart_multi <- rpart(
        formula = fmla,
        data    = train_data,
        method  = "class",
        parms   = list(split = "information"),  
        control = rpart.control(maxdepth = d, minsplit = m, cp = 0.01)
      )
      

      y_hat_rpart <- predict(tree_rpart_multi, test_data, type = "class")
      

      ocena_rpart <- ModelOcena(test_data[[Yname]], y_hat_rpart)
      auc_rpart_vec  <- c(auc_rpart_vec,  ocena_rpart$Miary["AUC"])
      sens_rpart_vec <- c(sens_rpart_vec, ocena_rpart$Miary["Czulosc"])
      spec_rpart_vec <- c(spec_rpart_vec, ocena_rpart$Miary["Specyficznosc"])
      jakosc_rpart_vec <- c(jakosc_rpart_vec, ocena_rpart$Miary["Jakosc"])
      
    } 
    

    miaryTree_M_wlasna_cv <- rbind(
      miaryTree_M_wlasna_cv,
      data.frame(
        depth = d,
        minobs = m,
        AUC = mean(auc_wlasne_vec),
        Czulosc = mean(sens_wlasne_vec),
        Specyficznosc = mean(spec_wlasne_vec),
        Jakosc = mean(jakosc_wlasne_vec)
      )
    )
    
    miaryTree_M_rpart_cv <- rbind(
      miaryTree_M_rpart_cv,
      data.frame(
        depth = d,
        minobs = m,
        AUC = mean(auc_rpart_vec),
        Czulosc = mean(sens_rpart_vec),
        Specyficznosc = mean(spec_rpart_vec),
        Jakosc = mean(jakosc_rpart_vec)
      )
    )
    
  } 
}

miaryTree_M_wlasna_cv
miaryTree_M_rpart_cv


### Algorytm Sieci Neuronowej ########################################################################################

library(caret)
library(nnet)  

hTune       <- list(c(5), c(10), c(10,5), c(20,10))  
hTune_nnet  <- c(5, 10, 20)                         
iterTune    <- c(2000, 5000)
lrTune      <- c(0.001, 0.01, 0.05)


Yname  <- "class"
Xnames <- c("age","height_cm","weight_kg","body.fat_.","diastolic","systolic",
            "gripForce","sit.and.bend.forward_cm","sit.ups.counts","broad.jump_cm",
            "gender_M")


bodyPerformance$class <- as.factor(bodyPerformance$class)


set.seed(234)


miaryNN_M_wlasna_cv <- data.frame(hTune = I(list()),
                                  iterTune = integer(),
                                  lrTune = numeric(),
                                  AUC = numeric(),
                                  Czulosc = numeric(),
                                  Specyficznosc = numeric(),
                                  Jakosc = numeric())

miaryNN_M_nnet_cv <- data.frame(hTune = numeric(),  
                                iterTune = numeric(),
                                AUC = numeric(),
                                Czulosc = numeric(),
                                Specyficznosc = numeric(),
                                Jakosc = numeric())


set.seed(234)
folds <- createFolds(bodyPerformance$class, k=5)


for (iter in iterTune) {
  for (h in hTune) {
    for (lr in lrTune) {
      
      
      auc_vec  <- c()
      sens_vec <- c()
      spec_vec <- c()
      jakosc_vec <- c()
      

      for (f in 1:5) {
        test_idx  <- folds[[f]]
        

        train_data <- bodyPerformance[-test_idx, ]
        test_data  <- bodyPerformance[ test_idx, ]
        
        train_data$class <- as.factor(train_data$class)
        test_data$class  <- as.factor(test_data$class)
        
        train_data[Xnames] <- normalize_data(train_data[Xnames], XminNew = 0, XmaxNew = 1)
        test_data[Xnames]  <- normalize_data(test_data[Xnames],  XminNew = 0, XmaxNew = 1)
        

        nn_model <- trainNN(
          Yname  = Yname,
          Xnames = Xnames,
          data   = train_data,
          h      = h,
          type   = "multiclass",  
          lr     = lr,
          iter   = iter,
          seed   = 234
        )
        

        pred_probs <- predNN(nn_model, test_data[, Xnames, drop=FALSE], type="multiclass")
        

        class_levels <- levels(test_data[[Yname]])
        colnames(pred_probs) <- class_levels
        

        pred_class <- apply(pred_probs, 1, function(x) class_levels[which.max(x)])
        pred_class <- as.factor(pred_class)
        

        ocena <- ModelOcena(test_data[[Yname]], pred_class)
        
        auc_vec  <- c(auc_vec,  ocena$Miary["AUC"])
        sens_vec <- c(sens_vec, ocena$Miary["Czulosc"])
        spec_vec <- c(spec_vec, ocena$Miary["Specyficznosc"])
        jakosc_vec <- c(jakosc_vec, ocena$Miary["Jakosc"])
      }
      

      miaryNN_M_wlasna_cv <- rbind(
        miaryNN_M_wlasna_cv,
        data.frame(
          hTune    = I(list(h)),
          iterTune = iter,
          lrTune   = lr,
          AUC         = mean(auc_vec),
          Czulosc     = mean(sens_vec),
          Specyficznosc = mean(spec_vec),
          Jakosc      = mean(jakosc_vec)
        )
      )
      
    } 
  }   
}     




for (iter in iterTune) {
  for (h in hTune_nnet) {
    
    auc_vec_nnet  <- c()
    sens_vec_nnet <- c()
    spec_vec_nnet <- c()
    jakosc_vec_nnet <- c()
    

    for (f in 1:5) {
      test_idx  <- folds[[f]]
      

      train_data <- bodyPerformance[-test_idx, ]
      test_data  <- bodyPerformance[ test_idx, ]
      
      train_data$class <- as.factor(train_data$class)
      test_data$class  <- as.factor(test_data$class)
      

      train_data[Xnames] <- normalize_data(train_data[Xnames], XminNew = 0, XmaxNew = 1)
      test_data[Xnames]  <- normalize_data(test_data[Xnames],  XminNew = 0, XmaxNew = 1)
      

      set.seed(234)
      nn_nnet <- nnet(
        formula = class ~ .,
        data    = train_data,
        size    = h,        
        linout  = FALSE,   
        maxit   = iter,
        trace   = FALSE
      )
      

      pred_probs_nnet <- predict(nn_nnet, newdata = test_data[, Xnames, drop=FALSE], type="raw")
      

      class_levels <- levels(test_data[[Yname]])
      colnames(pred_probs_nnet) <- class_levels
      

      pred_class_nnet <- apply(pred_probs_nnet, 1, function(x) class_levels[which.max(x)])
      pred_class_nnet <- as.factor(pred_class_nnet)
      

      ocena_nnet <- ModelOcena(test_data[[Yname]], pred_class_nnet)
      
      auc_vec_nnet  <- c(auc_vec_nnet,  ocena_nnet$Miary["AUC"])
      sens_vec_nnet <- c(sens_vec_nnet, ocena_nnet$Miary["Czulosc"])
      spec_vec_nnet <- c(spec_vec_nnet, ocena_nnet$Miary["Specyficznosc"])
      jakosc_vec_nnet <- c(jakosc_vec_nnet, ocena_nnet$Miary["Jakosc"])
    }
    

    miaryNN_M_nnet_cv <- rbind(
      miaryNN_M_nnet_cv,
      data.frame(
        hTune = h,
        iterTune = iter,
        AUC = mean(auc_vec_nnet),
        Czulosc = mean(sens_vec_nnet),
        Specyficznosc = mean(spec_vec_nnet),
        Jakosc = mean(jakosc_vec_nnet)
      )
    )
    
  } 
}   


miaryNN_M_wlasna_cv
miaryNN_M_nnet_cv



# ======================================
#    (Problem regresji)
# ======================================

#Wczytanie danych do regresji
food_Delivery <- read.csv("Food_Delivery_Times.csv", header = TRUE, sep = ",")
head(food_Delivery)

# usuwanie brakujących wierszy,

food_Delivery = na.omit(food_Delivery)
food_Delivery <- subset(food_Delivery, Weather != "" & !is.na(Weather))
food_Delivery <- subset(food_Delivery, Traffic_Level != "" & !is.na(Traffic_Level))
food_Delivery <- subset(food_Delivery, Time_of_Day != "" & !is.na(Time_of_Day))

food_Delivery <- food_Delivery[, -1] #usuwanie orderID

#117 objects removed

sum(is.na(food_Delivery))
str(food_Delivery)

#library(fastDummies)

#food_Delivery <- dummy_cols(food_Delivery, select_columns = c("Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"), 
#                            remove_selected_columns = TRUE, 
#                            remove_first_dummy = FALSE)
# Załaduj pakiet
library(caret)

# Sprawdzenie wariancji każdej zmiennej
nzv <- nearZeroVar(food_Delivery, saveMetrics = TRUE)

# Wyświetlenie zmiennych z niską wariancją
print(nzv)

# Usunięcie tych zmiennych
food_Delivery <- food_Delivery[, !nzv$nzv]

# Załaduj pakiety
library(corrplot)
library(GGally)

# Wybór tylko kolumn numerycznych
numeric_cols <- sapply(food_Delivery, is.numeric)
food_Delivery_numeric <- food_Delivery[, numeric_cols, drop = FALSE]

# Obliczenie macierzy korelacji
corr_matrix <- cor(food_Delivery_numeric, use = "complete.obs")

# Wizualizacja macierzy korelacji
corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.8)

# Znalezienie zmiennych silnie skorelowanych (np. korelacja > 0.9)
high_corr <- findCorrelation(corr_matrix, cutoff = 0.9)

# Wyświetlenie indeksów zmiennych do usunięcia
print(high_corr)


### Algorytm KNN ########################################################################################


set.seed(234) 

sample_food_Delivery <- food_Delivery[sample(1:nrow(food_Delivery), 400, replace = FALSE), ]

library(caret)
library(dplyr)



kTune <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


miaryKNN_R_wlasna_cv <- data.frame(k = integer(),
                                   MAE = numeric(),
                                   MSE = numeric(),
                                   MAPE = numeric())

miaryKNN_R_caret_cv  <- data.frame(k = integer(),
                                   MAE = numeric(),
                                   MSE = numeric(),
                                   MAPE = numeric())


set.seed(234)
folds <- createFolds(sample_food_Delivery$Delivery_Time_min, k=5)


for (k in kTune) {
  

  mae_vec_wlasne  <- c()
  mse_vec_wlasne  <- c()
  mape_vec_wlasne <- c()
  

  mae_vec_caret  <- c()
  mse_vec_caret  <- c()
  mape_vec_caret <- c()
  

  for (f in 1:5) {
    

    test_idx <- folds[[f]]
    

    train_data <- sample_food_Delivery[-test_idx, ]
    test_data  <- sample_food_Delivery[ test_idx, ]
    

    train_y <- train_data$Delivery_Time_min
    test_y  <- test_data$Delivery_Time_min
    

    train_x <- train_data %>% select(-Delivery_Time_min)
    test_x  <- test_data  %>% select(-Delivery_Time_min)
    

    train_x_mat <- model.matrix(~ . -1, data=train_x)
    test_x_mat  <- model.matrix(~ . -1, data=test_x)
    

    modelK <- KNNtrain(train_x_mat, train_y, k)

    predK <- KNNpred(modelK, test_x_mat)
    
    ocena_wlasne <- ModelOcena(test_y, predK)
    
    mae_vec_wlasne  <- c(mae_vec_wlasne,  ocena_wlasne["MAE"])
    mse_vec_wlasne  <- c(mse_vec_wlasne,  ocena_wlasne["MSE"])
    mape_vec_wlasne <- c(mape_vec_wlasne, ocena_wlasne["MAPE"])
    

    train_x_nor <- normalize_data(train_x_mat, XminNew = 0, XmaxNew = 1)
    test_x_nor  <- normalize_data(test_x_mat,  XminNew = 0, XmaxNew = 1)
    
    model_knnreg <- knnreg(x = train_x_nor, y = train_y, k = k)
    pred_caret   <- predict(model_knnreg, newdata = test_x_nor)
    
    ocena_caret <- ModelOcena(test_y, pred_caret)
    
    mae_vec_caret  <- c(mae_vec_caret,  ocena_caret["MAE"])
    mse_vec_caret  <- c(mse_vec_caret,  ocena_caret["MSE"])
    mape_vec_caret <- c(mape_vec_caret, ocena_caret["MAPE"])
    
  } 
  

  miaryKNN_R_wlasna_cv <- rbind(
    miaryKNN_R_wlasna_cv,
    data.frame(
      k = k,
      MAE = mean(mae_vec_wlasne),
      MSE = mean(mse_vec_wlasne),
      MAPE = mean(mape_vec_wlasne)
    )
  )
  
  miaryKNN_R_caret_cv <- rbind(
    miaryKNN_R_caret_cv,
    data.frame(
      k = k,
      MAE = mean(mae_vec_caret),
      MSE = mean(mse_vec_caret),
      MAPE = mean(mape_vec_caret)
    )
  )
  
} 


miaryKNN_R_wlasna_cv
miaryKNN_R_caret_cv

### Algorytm Drzewa Decyzyjnego ########################################################################################

library(fastDummies)

food_Delivery <- dummy_cols(food_Delivery, select_columns = c("Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"), 
                            remove_selected_columns = TRUE, 
                            remove_first_dummy = FALSE)

food_Delivery <- model.matrix(~ . -1, data = food_Delivery)
food_Delivery = as.data.frame(food_Delivery)

library(caret)
library(rpart)


depthTune <- c(2, 3, 4, 6)
minobsTune <- c(2, 3, 6)


miaryTree_R_wlasna_cv <- data.frame(depth = integer(),
                                    minobs = integer(),
                                    MAE = numeric(),
                                    MSE = numeric(),
                                    MAPE = numeric())

miaryTree_R_rpart_cv <- data.frame(depth = integer(),
                                   minobs = integer(),
                                   MAE = numeric(),
                                   MSE = numeric(),
                                   MAPE = numeric())


set.seed(234)
folds <- createFolds(food_Delivery$Delivery_Time_min, k = 5)


for(d in depthTune) {
  for(m in minobsTune) {
    

    mae_vec_wlasne  <- c()
    mse_vec_wlasne  <- c()
    mape_vec_wlasne <- c()
    

    mae_vec_rpart  <- c()
    mse_vec_rpart  <- c()
    mape_vec_rpart <- c()
    

    for(f in 1:5) {
      

      test_idx <- folds[[f]]
      

      train_data <- food_Delivery[-test_idx, ]
      test_data  <- food_Delivery[ test_idx, ]
      

      train_data$Delivery_Time_min <- as.numeric(train_data$Delivery_Time_min)
      test_data$Delivery_Time_min  <- as.numeric(test_data$Delivery_Time_min)
      

      tree_reg <- Tree(
        Y = "Delivery_Time_min",
        X = c("Distance_km","Preparation_Time_min","Courier_Experience_yrs",
              "Weather_Clear","Weather_Foggy","Weather_Rainy","Weather_Snowy","Weather_Windy",
              "Traffic_Level_High","Traffic_Level_Low","Traffic_Level_Medium",
              "Time_of_Day_Afternoon","Time_of_Day_Evening","Time_of_Day_Morning","Time_of_Day_Night",
              "Vehicle_Type_Bike","Vehicle_Type_Car","Vehicle_Type_Scooter"),
        data = train_data,
        type = "SS",       
        depth = d,
        minobs = m,
        overfit = "none",
        cf = 0.01
      )
      

      pred_reg <- PredictTree(tree_reg, test_data)
     
      ocena_wlasne <- ModelOcena(test_data$Delivery_Time_min, pred_reg)
      
      mae_vec_wlasne  <- c(mae_vec_wlasne,  ocena_wlasne["MAE"])
      mse_vec_wlasne  <- c(mse_vec_wlasne,  ocena_wlasne["MSE"])
      mape_vec_wlasne <- c(mape_vec_wlasne, ocena_wlasne["MAPE"])
      

      fmla <- as.formula(
        paste("Delivery_Time_min ~",
              paste(c("Distance_km","Preparation_Time_min","Courier_Experience_yrs",
                      "Weather_Clear","Weather_Foggy","Weather_Rainy","Weather_Snowy","Weather_Windy",
                      "Traffic_Level_High","Traffic_Level_Low","Traffic_Level_Medium",
                      "Time_of_Day_Afternoon","Time_of_Day_Evening","Time_of_Day_Morning","Time_of_Day_Night",
                      "Vehicle_Type_Bike","Vehicle_Type_Car","Vehicle_Type_Scooter"),
                    collapse = " + "))
      )
      
      tree_rpart_reg <- rpart(
        formula = fmla,
        data = train_data,
        method = "anova",  
        control = rpart.control(maxdepth = d, minsplit = m, cp = 0.01)
      )
      

      pred_rpart_reg <- predict(tree_rpart_reg, test_data)
      

      ocena_rpart <- ModelOcena(test_data$Delivery_Time_min, pred_rpart_reg)
      
      mae_vec_rpart  <- c(mae_vec_rpart,  ocena_rpart["MAE"])
      mse_vec_rpart  <- c(mse_vec_rpart,  ocena_rpart["MSE"])
      mape_vec_rpart <- c(mape_vec_rpart, ocena_rpart["MAPE"])
      
    } 

    miaryTree_R_wlasna_cv <- rbind(
      miaryTree_R_wlasna_cv,
      data.frame(
        depth = d,
        minobs = m,
        MAE = mean(mae_vec_wlasne),
        MSE = mean(mse_vec_wlasne),
        MAPE = mean(mape_vec_wlasne)
      )
    )
    
    miaryTree_R_rpart_cv <- rbind(
      miaryTree_R_rpart_cv,
      data.frame(
        depth = d,
        minobs = m,
        MAE = mean(mae_vec_rpart),
        MSE = mean(mse_vec_rpart),
        MAPE = mean(mape_vec_rpart)
      )
    )
    
  } 
} 



miaryTree_R_wlasna_cv
miaryTree_R_rpart_cv


### Algorytm Sieci Neuronowej ########################################################################################

library(caret)
library(nnet)


hTune       <- list(c(5), c(10), c(10,5), c(20,10))  
hTune_nnet  <- c(5, 10, 20)                         
iterTune    <- c(2000, 5000)
lrTune      <- c(0.001, 0.01, 0.05)


Yname <- "Delivery_Time_min"
Xnames <- c("Distance_km","Preparation_Time_min","Courier_Experience_yrs",
            "Weather_Clear","Weather_Foggy","Weather_Rainy","Weather_Snowy","Weather_Windy",
            "Traffic_Level_High","Traffic_Level_Low","Traffic_Level_Medium",
            "Time_of_Day_Afternoon","Time_of_Day_Evening","Time_of_Day_Morning","Time_of_Day_Night",
            "Vehicle_Type_Bike","Vehicle_Type_Car","Vehicle_Type_Scooter")


food_Delivery$Delivery_Time_min <- as.numeric(food_Delivery$Delivery_Time_min)


miaryNN_R_wlasna_cv <- data.frame(hTune   = I(list()),
                                  iterTune = integer(),
                                  lrTune   = numeric(),
                                  MAE = numeric(),
                                  MSE = numeric(),
                                  MAPE = numeric())

miaryNN_R_nnet_cv   <- data.frame(hTune   = numeric(),
                                  iterTune = numeric(),
                                  MAE = numeric(),
                                  MSE = numeric(),
                                  MAPE = numeric())


set.seed(234)
folds <- createFolds(food_Delivery$Delivery_Time_min, k = 5)


for (iter in iterTune) {
  for (h in hTune) {
    for (lr in lrTune) {
      

      mae_vec  <- c()
      mse_vec  <- c()
      mape_vec <- c()
      

      for (f in 1:5) {
        

        test_idx <- folds[[f]]
        

        train_data <- food_Delivery[-test_idx, ]
        test_data  <- food_Delivery[ test_idx, ]
        

        
        train_data[Xnames] <- normalize_data(train_data[Xnames], XminNew=0, XmaxNew=1)
        test_data[Xnames]  <- normalize_data(test_data[Xnames],  XminNew=0, XmaxNew=1)
        

        nn_reg <- trainNN(
          Yname  = Yname,
          Xnames = Xnames,
          data   = train_data,
          h      = h,              
          type   = "regression",
          lr     = lr,
          iter   = iter,
          seed   = 234
        )
        

        pred_reg <- predNN(nn_reg, test_data[, Xnames, drop = FALSE], type="regression")
        

        ocena <- ModelOcena(test_data[[Yname]], pred_reg)
        
        mae_vec  <- c(mae_vec,  ocena["MAE"])
        mse_vec  <- c(mse_vec,  ocena["MSE"])
        mape_vec <- c(mape_vec, ocena["MAPE"])
      }
      

      miaryNN_R_wlasna_cv <- rbind(
        miaryNN_R_wlasna_cv,
        data.frame(
          hTune   = I(list(h)),
          iterTune = iter,
          lrTune   = lr,
          MAE  = mean(mae_vec),
          MSE  = mean(mse_vec),
          MAPE = mean(mape_vec)
        )
      )
      
    } 
  }   
}    



for (iter in iterTune) {
  for (h in hTune_nnet) {
    
    mae_vec_nnet  <- c()
    mse_vec_nnet  <- c()
    mape_vec_nnet <- c()
    
    for (f in 1:5) {
      
      test_idx   <- folds[[f]]
      train_data <- food_Delivery[-test_idx, ]
      test_data  <- food_Delivery[ test_idx, ]

      train_data[Xnames] <- normalize_data(train_data[Xnames], XminNew=0, XmaxNew=1)
      test_data[Xnames]  <- normalize_data(test_data[Xnames],  XminNew=0, XmaxNew=1)
      

      set.seed(234)
      nn_nnet <- nnet(
        formula = Delivery_Time_min ~ .,
        data    = train_data,
        size    = h,         
        linout  = TRUE,      
        maxit   = iter,
        trace   = FALSE
      )
      

      pred_reg_nnet <- predict(nn_nnet, newdata = test_data[, Xnames, drop=FALSE])

      ocena_nnet <- ModelOcena(test_data[[Yname]], pred_reg_nnet)
      
      mae_vec_nnet  <- c(mae_vec_nnet,  ocena_nnet["MAE"])
      mse_vec_nnet  <- c(mse_vec_nnet,  ocena_nnet["MSE"])
      mape_vec_nnet <- c(mape_vec_nnet, ocena_nnet["MAPE"])
    }
    

    miaryNN_R_nnet_cv <- rbind(
      miaryNN_R_nnet_cv,
      data.frame(
        hTune   = h,
        iterTune = iter,
        MAE  = mean(mae_vec_nnet),
        MSE  = mean(mse_vec_nnet),
        MAPE = mean(mape_vec_nnet)
      )
    )
    
  } 
}   


miaryNN_R_wlasna_cv
miaryNN_R_nnet_cv

####################################### Porównywanie modeli ####################################################


print("Klasyfikacja binarna") #-----------------------------------------------------------------

miaryKNN_B_wlasna_cv
plot_metrics(
  df = miaryKNN_B_wlasna_cv,
  param_cols = "k",  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc")
)

miaryKNN_B_caret_cv
plot_metrics(
  df = miaryKNN_B_caret_cv,
  param_cols = "k",  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc")
)


miaryTree_B_wlasna_cv
plot_metrics(
  df = miaryTree_B_wlasna_cv,
  param_cols = c("depth", "minobs"),  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc"),
  sep = "_"  
)

miaryTree_B_rpart_cv
plot_metrics(
  df = miaryTree_B_wlasna_cv,
  param_cols = c("depth", "minobs"),  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc"),
  sep = "_"  
)

miaryNN_B_wlasna_cv
plot_metrics(
  df = miaryNN_B_wlasna_cv,
  param_cols = c("hTune", "iterTune",'lrTune'),  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc"),
  sep = "_"  
)

miaryNN_B_nnet_cv
str(miaryNN_B_nnet_cv)
plot_metrics(
  df = miaryNN_B_nnet_cv,
  param_cols = c("hTune", "iterTune"), 
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc"),
  sep = "_"  
)

print("Klasyfikacja wieloklasowa") #-----------------------------------------------------------------

miaryKNN_M_wlasna_cv
plot_metrics(
  df = miaryKNN_M_wlasna_cv,
  param_cols = "k",  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc")
)

miaryKNN_M_caret_cv
plot_metrics(
  df = miaryKNN_M_caret_cv,
  param_cols = "k",  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc")
)

miaryTree_M_wlasna_cv
plot_metrics(
  df = miaryTree_M_wlasna_cv,
  param_cols = c("depth", "minobs"),  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc"),
  sep = "_"  
)

miaryTree_M_rpart_cv
plot_metrics(
  df = miaryTree_M_rpart_cv,
  param_cols = c("depth", "minobs"),  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc"),
  sep = "_"  
)


miaryNN_M_wlasna_cv
plot_metrics(
  df = miaryNN_M_wlasna_cv,
  param_cols = c("hTune", "iterTune",'lrTune'),  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc"),
  sep = "_"  
)

miaryNN_M_nnet_cv
plot_metrics(
  df = miaryNN_M_nnet_cv,
  param_cols = c("hTune", "iterTune"),  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc"),
  sep = "_"  
)

print("Regresja") #-----------------------------------------------------------------

miaryKNN_R_wlasna_cv
Prez_miary <- miaryKNN_R_wlasna_cv
Prez_miary$MAPE <- Prez_miary$MAPE*100

plot_metrics(
  df = Prez_miary,
  param_cols = "k",  
  metric_cols = c("MAE", "MSE", "MAPE")
)

miaryKNN_R_caret_cv
Prez_miary <- miaryKNN_R_caret_cv
Prez_miary$MAPE <- Prez_miary$MAPE*100

plot_metrics(
  df = Prez_miary,
  param_cols = "k",  
  metric_cols = c("MAE", "MSE", "MAPE")
)

miaryTree_R_wlasna_cv
Prez_miary <- miaryTree_R_wlasna_cv
Prez_miary$MAPE <- Prez_miary$MAPE*100

plot_metrics(
  df = Prez_miary,
  param_cols = c("depth", "minobs"),  
  metric_cols = c("MAE", "MSE", "MAPE"),
  sep = "_"  
)

miaryTree_R_rpart_cv
Prez_miary <- miaryTree_R_rpart_cv
Prez_miary$MAPE <- Prez_miary$MAPE*100

plot_metrics(
  df = Prez_miary,
  param_cols = c("depth", "minobs"),  
  metric_cols = c("MAE", "MSE", "MAPE"),
  sep = "_"  
)

miaryNN_R_wlasna_cv
Prez_miary <- miaryNN_R_wlasna_cv
Prez_miary$MAPE <- Prez_miary$MAPE*100

plot_metrics(
  df = Prez_miary,
  param_cols = c("hTune", "iterTune",'lrTune'),  
  metric_cols = c("MAE", "MSE", "MAPE"),
  sep = "_"  
)


miaryNN_R_nnet_cv
Prez_miary <- miaryNN_R_nnet_cv
Prez_miary$MAPE <- Prez_miary$MAPE*100
plot_metrics(
  df = Prez_miary,
  param_cols = c("hTune", "iterTune"),  
  metric_cols = c("MAE", "MSE", "MAPE"),
  sep = "_"  
)

#-------------------------------ANALIZA METOD DLA DANEGO problemu -------------------------

print("Klasyfikacja binarna") #-----------------------------------------------------------------

miaryNN_B_wlasna_cv$hTune <- as.character(miaryNN_B_wlasna_cv$hTune)

library(dplyr)


wlasna_extract <- miaryKNN_B_wlasna_cv %>%
  mutate(Metoda = "KNN_Własna") %>%
  filter(k %in% c(7, 9))


caret_extract <- miaryKNN_B_caret_cv %>%
  mutate(Metoda = "KNN_Caret") %>%
  filter(k %in% c(7, 10))



wyniki_final <- bind_rows(wlasna_extract, caret_extract)


wyniki_final <- wyniki_final %>%
  mutate(parametry = paste0("k=", k)) %>%  
  select(-k)     


wyniki_knn <- wyniki_final


tree_wlasna <- miaryTree_B_wlasna_cv %>%
  filter(depth == 6, minobs == 2) %>%
  transmute(
    AUC          = AUC,
    Czulosc      = Czulosc,
    Specyficznosc= Specyficznosc,
    Jakosc       = Jakosc,
    Metoda       = "Tree_Własna",
    parametry    = paste0("depth=", depth, ", minobs=", minobs)
  )


tree_rpart <- miaryTree_B_rpart_cv %>%
  filter(depth == 6, minobs == 2) %>%
  transmute(
    AUC          = AUC,
    Czulosc      = Czulosc,
    Specyficznosc= Specyficznosc,
    Jakosc       = Jakosc,
    Metoda       = "Tree_rpart",
    parametry    = paste0("depth=", depth, ", minobs=", minobs)
  )


wyniki_all <- bind_rows(wyniki_knn, tree_wlasna, tree_rpart)


wyniki_all

wlasna_extract <- miaryNN_B_wlasna_cv %>%
  filter(
    (hTune == "c(20, 10)" & iterTune == 5000 & lrTune == 0.001) |
      (hTune == "c(10, 5)"  & iterTune == 5000 & lrTune == 0.001)
  ) %>%
  mutate(
    Metoda    = "NN_Własna",

    parametry = paste0("hTune=", hTune,
                       ", iterTune=", iterTune,
                       ", lrTune=", lrTune)
  ) %>%
  select(AUC, Czulosc, Specyficznosc, Jakosc, Metoda, parametry)



nnet_extract <- miaryNN_B_nnet_cv %>%
  filter(hTune == 5, iterTune == 2000) %>%
  mutate(
    Metoda    = "NN_nnet",
    parametry = paste0("hTune=", hTune, ", iterTune=", iterTune)
  ) %>%
  select(AUC, Czulosc, Specyficznosc, Jakosc, Metoda, parametry)



wyniki_final <- bind_rows(wyniki_all, wlasna_extract, nnet_extract)


wyniki_final_bin <- wyniki_final
wyniki_final_bin

##########################################################################################################
print("Klasyfikacja wieloklasowa") #-----------------------------------------------------------------

miaryNN_M_wlasna_cv$hTune <- as.character(miaryNN_M_wlasna_cv$hTune)

library(dplyr)


wlasna_extract <- miaryKNN_M_wlasna_cv %>%
  mutate(Metoda = "KNN_Własna") %>%
  filter(k %in% c(7))


caret_extract <- miaryKNN_M_caret_cv %>%
  mutate(Metoda = "KNN_Caret") %>%
  filter(k %in% c(7))



wyniki_final <- bind_rows(wlasna_extract, caret_extract)


wyniki_final <- wyniki_final %>%
  mutate(parametry = paste0("k=", k)) %>%  
  select(-k)     


wyniki_knn <- wyniki_final


tree_wlasna <- miaryTree_M_wlasna_cv %>%
  filter(depth == 6, minobs == 2) %>%
  transmute(
    AUC          = AUC,
    Czulosc      = Czulosc,
    Specyficznosc= Specyficznosc,
    Jakosc       = Jakosc,
    Metoda       = "Tree_Własna",
    parametry    = paste0("depth=", depth, ", minobs=", minobs)
  )


tree_rpart <- miaryTree_M_rpart_cv %>%
  filter(depth == 6, minobs == 2) %>%
  transmute(
    AUC          = AUC,
    Czulosc      = Czulosc,
    Specyficznosc= Specyficznosc,
    Jakosc       = Jakosc,
    Metoda       = "Tree_rpart",
    parametry    = paste0("depth=", depth, ", minobs=", minobs)
  )


wyniki_all <- bind_rows(wyniki_knn, tree_wlasna, tree_rpart)


wyniki_all

wlasna_extract <- miaryNN_M_wlasna_cv %>%
  filter(
      (hTune == "c(20, 10)"  & iterTune == 5000 & lrTune == 0.05)
  ) %>%
  mutate(
    Metoda    = "NN_Własna",

    parametry = paste0("hTune=", hTune,
                       ", iterTune=", iterTune,
                       ", lrTune=", lrTune)
  ) %>%
  select(AUC, Czulosc, Specyficznosc, Jakosc, Metoda, parametry)



nnet_extract <- miaryNN_M_nnet_cv %>%
  filter(hTune == 5, iterTune == 2000) %>%
  mutate(
    Metoda    = "NN_nnet",
    parametry = paste0("hTune=", hTune, ", iterTune=", iterTune)
  ) %>%
  select(AUC, Czulosc, Specyficznosc, Jakosc, Metoda, parametry)



wyniki_final <- bind_rows(wyniki_all, wlasna_extract, nnet_extract)


wyniki_final_multi <- wyniki_final
wyniki_final_multi

plot_metrics(
  df = wyniki_final_multi,
  param_cols = c("Metoda", "parametry"),  
  metric_cols = c("AUC", "Czulosc", "Specyficznosc", "Jakosc"),
  sep = " "  
)
##########################################################################################################
print("Regresja") #-----------------------------------------------------------------

miaryNN_R_wlasna_cv$hTune <- as.character(miaryNN_R_wlasna_cv$hTune)

library(dplyr)


wlasna_extract <- miaryKNN_R_wlasna_cv %>%
  mutate(Metoda = "KNN_Własna") %>%
  filter(k %in% c(7,8,9))


caret_extract <- miaryKNN_R_caret_cv %>%
  mutate(Metoda = "KNN_Caret") %>%
  filter(k %in% c(8,10))



wyniki_final <- bind_rows(wlasna_extract, caret_extract)


wyniki_final <- wyniki_final %>%
  mutate(parametry = paste0("k=", k)) %>%  
  select(-k)     


wyniki_knn <- wyniki_final


tree_wlasna <- miaryTree_R_wlasna_cv %>%
  filter(depth == 6, minobs == 6) %>%
  transmute(
    MAE          = MAE,
    MSE      = MSE,
    MAPE= MAPE,
    Metoda       = "Tree_Własna",
    parametry    = paste0("depth=", depth, ", minobs=", minobs)
  )


tree_rpart <- miaryTree_R_rpart_cv %>%
  filter(depth == 6, minobs == 2) %>%
  transmute(
    MAE          = MAE,
    MSE      = MSE,
    MAPE = MAPE,
    Metoda       = "Tree_rpart",
    parametry    = paste0("depth=", depth, ", minobs=", minobs)
  )

wyniki_all <- bind_rows(wyniki_knn, tree_wlasna, tree_rpart)

wyniki_all

wlasna_extract <- miaryNN_R_wlasna_cv %>%
  filter(
    (hTune == "c(10)"  & iterTune == 5000 & lrTune == 0.001)
  ) %>%
  mutate(
    Metoda    = "NN_Własna",
    parametry = paste0("hTune=", hTune,
                       ", iterTune=", iterTune,
                       ", lrTune=", lrTune)
  ) %>%
  select(MAE, MSE, MAPE, Metoda, parametry)



nnet_extract <- miaryNN_R_nnet_cv %>%
  filter(hTune == 5, iterTune == 2000) %>%
  mutate(
    Metoda    = "NN_nnet",
    parametry = paste0("hTune=", hTune, ", iterTune=", iterTune)
  ) %>%
  select(MAE, MSE, MAPE, Metoda, parametry)


wyniki_final <- bind_rows(wyniki_all, wlasna_extract, nnet_extract)

wyniki_final_reg <- wyniki_final
wyniki_final_reg
wyniki_final_reg$MAPE <- wyniki_final_reg$MAPE*100

plot_metrics(
  df = wyniki_final_reg,
  param_cols = c("Metoda", "parametry"),  
  metric_cols = c("MAE", "MSE", "MAPE"),
  sep = " "  
)