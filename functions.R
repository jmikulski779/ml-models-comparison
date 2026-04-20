
library(scales)

normalize_data <- function(X, XminNew = 0, XmaxNew = 1) {
  X <- as.data.frame(X)
  
  numeric_cols <- sapply(X, is.numeric)
  
  minOrg <- sapply(X[, numeric_cols, drop = FALSE], min)
  maxOrg <- sapply(X[, numeric_cols, drop = FALSE], max)
  
  # Skalowanie kolumn numerycznych
  for (col_name in names(X)[numeric_cols]) {
    col_min <- min(X[[col_name]])
    col_max <- max(X[[col_name]])
    
    if (col_min == col_max) {
      X[[col_name]] <- XminNew
    } else {
      X[[col_name]] <- rescale(X[[col_name]], to = c(XminNew, XmaxNew))
    }
  }
  

  attr(X, "minOrg")   <- minOrg
  attr(X, "maxOrg")   <- maxOrg
  attr(X, "minmaxNew") <- c(XminNew, XmaxNew)
  
  return(X)
}
########################################################################## modelOcena ##############################################################

#################################################################################################
# Funkcje pomocnicze - regresja
#################################################################################################
mae_func <- function(y_tar, y_hat) {
  mean(abs(y_tar - y_hat))
}

mse_func <- function(y_tar, y_hat) {
  mean((y_tar - y_hat)^2)
}

mape_func <- function(y_tar, y_hat) {
  mean(abs((y_tar - y_hat) / y_tar))
}

#################################################################################################
# Funkcje pomocnicze - klasyfikacja binarna
#################################################################################################
auc_func <- function(y_tar, y_hat) {
  
  y_num <- as.numeric(y_tar) - 1
  
  idx_sort <- order(y_hat, decreasing = TRUE)
  y_hat_sorted <- y_hat[idx_sort]
  y_num_sorted <- y_num[idx_sort]
  
  P <- sum(y_num == 1)
  N <- sum(y_num == 0)
  
  tpr_values <- c(0)
  fpr_values <- c(0)
  
  tp <- 0
  fp <- 0
  
  for (i in seq_along(y_hat_sorted)) {
    if (y_num_sorted[i] == 1) {
      tp <- tp + 1
    } else {
      fp <- fp + 1
    }
    tpr_values <- c(tpr_values, tp / P)
    fpr_values <- c(fpr_values, fp / N)
  }
  
  auc_value <- 0
  for (i in 2:length(tpr_values)) {
    x_diff <- fpr_values[i] - fpr_values[i - 1]
    y_mean <- (tpr_values[i] + tpr_values[i - 1]) / 2
    auc_value <- auc_value + x_diff * y_mean
  }
  
  return(auc_value)
}

youden_threshold <- function(y_tar, y_hat) {
  y_num <- as.numeric(y_tar) - 1
  uniq_vals <- sort(unique(y_hat))
  
  best_thr <- uniq_vals[1]
  best_youden <- -Inf
  
  for (thr in uniq_vals) {
    pred_class <- as.numeric(y_hat >= thr)
    
    TP <- sum(pred_class == 1 & y_num == 1)
    FP <- sum(pred_class == 1 & y_num == 0)
    P  <- sum(y_num == 1)
    N  <- sum(y_num == 0)
    
    TPR <- TP / P
    FPR <- FP / N
    youden_val <- TPR - FPR
    
    if (youden_val > best_youden) {
      best_youden <- youden_val
      best_thr <- thr
    }
  }
  
  return(list(threshold = best_thr, youden = best_youden))
}

#################################################################################################
# Funkcje pomocnicze - metryki oparte na macierzy pomyłek
#################################################################################################
conf_mat_func <- function(y_tar, y_pred_class) {
  poziomy <- levels(y_tar)
  y_pred_class <- factor(y_pred_class, levels = poziomy)
  table(y_tar, y_pred_class)
}

sens_func <- function(conf_mat, class_idx = 2) {
  # czułość = TP / (TP + FN)
  TP <- conf_mat[class_idx, class_idx]
  FN <- sum(conf_mat[class_idx, ]) - TP
  sensitivity <- if ((TP + FN) > 0) TP / (TP + FN) else 0
  return(sensitivity)
}

spec_func <- function(conf_mat, class_idx = 2) {
  # specyficzność = TN / (TN + FP)
  TN <- sum(conf_mat[-class_idx, -class_idx])
  FP <- sum(conf_mat[-class_idx, class_idx])
  specificity <- if ((TN + FP) > 0) TN / (TN + FP) else 0
  return(specificity)
}

acc_func <- function(conf_mat) {
  suma_diag <- sum(diag(conf_mat))
  suma_all <- sum(conf_mat)
  return(suma_diag / suma_all)
}

#################################################################################################
# Funkcje pomocnicze - wieloklasowa "one-vs-rest"
#################################################################################################

multi_class_auc_matrix <- function(y_tar, y_hat_mat) {

  classes <- levels(y_tar)
  auc_vals <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    klasa_i <- classes[i]

    y_num <- as.numeric(y_tar == klasa_i)
    y_factor <- factor(y_num, levels = c(0,1))
    

    y_hat_i <- y_hat_mat[, i]
    
    auc_vals[i] <- auc_func(y_factor, y_hat_i)
  }
  
  mean(auc_vals)  
}

multi_class_auc_factor <- function(y_tar, y_hat_factor) {
  classes <- levels(y_tar)
  auc_vals <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    klasa_i <- classes[i]
    
    y_num <- as.numeric(y_tar == klasa_i)  
    y_factor_bin <- factor(y_num, levels = c(0,1))
    
    y_hat_num <- as.numeric(y_hat_factor == klasa_i)
    
    auc_vals[i] <- auc_func(y_factor_bin, y_hat_num)
  }
  
  mean(auc_vals)
}


multi_sens_spec <- function(y_tar, y_pred, typ = c("sens","spec")) {
  typ <- match.arg(typ)
  classes <- levels(y_tar)
  vals <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    klasa_i <- classes[i]
    

    y_bin_tar  <- factor(ifelse(y_tar == klasa_i, 1, 0), levels = c(0,1))
    y_bin_pred <- factor(ifelse(y_pred == klasa_i, 1, 0), levels = c(0,1))
    
    cm_i <- table(y_bin_tar, y_bin_pred)
    
    TP <- cm_i[2,2]
    FN <- cm_i[2,1]
    TN <- cm_i[1,1]
    FP <- cm_i[1,2]
    
    if (typ == "sens") {
      vals[i] <- if ((TP + FN) > 0) TP/(TP + FN) else 0
    } else {
      vals[i] <- if ((TN + FP) > 0) TN/(TN + FP) else 0
    }
  }
  mean(vals)
}

#################################################################################################
# Główna funkcja: ModelOcena
#################################################################################################
ModelOcena <- function(y_tar, y_hat) {
  
  # 1. REGRESJA
  if (is.numeric(y_tar)) {
    mae_val  <- mae_func(y_tar, y_hat)
    mse_val  <- mse_func(y_tar, y_hat)
    mape_val <- mape_func(y_tar, y_hat)
    
    wynik <- c(MAE = mae_val, MSE = mse_val, MAPE = mape_val)
    return(wynik)
  }
  
  # 2. KLASYFIKACJA
  if (!is.factor(y_tar)) {
    stop("Nieobsługiwany typ zmiennej docelowej: ani numeric, ani factor.")
  }
  
  k <- length(levels(y_tar))
  
  # Klasyfikacja binarna
  if (k == 2) {
    auc_val <- auc_func(y_tar, y_hat)
    
    thr_info <- youden_threshold(y_tar, y_hat)
    thr <- thr_info$threshold
    j_val <- thr_info$youden
    
    poziomy <- levels(y_tar)
    pred_class_num <- ifelse(y_hat >= thr, 1, 0)
    pred_class_fac <- factor(pred_class_num, levels = c(0,1), labels = poziomy)
    
    mat_cf <- conf_mat_func(y_tar, pred_class_fac)
    
    sens_val <- sens_func(mat_cf, class_idx = 2)
    spec_val <- spec_func(mat_cf, class_idx = 2)
    acc_val  <- acc_func(mat_cf)
    
    miary_vec <- c(AUC = auc_val,
                   Czulosc = sens_val,
                   Specyficznosc = spec_val,
                   Jakosc = acc_val)
    
    wynik_list <- list(
      ConfMat = mat_cf,
      YoudenJ = j_val,
      bestThreshold = thr,
      Miary   = miary_vec
    )
    return(wynik_list)
  }
  

  if (is.factor(y_hat)) {
    y_pred_class <- y_hat
    mat_cf <- conf_mat_func(y_tar, y_pred_class)
    accuracy <- acc_func(mat_cf)
    
    sens_macro <- multi_sens_spec(y_tar, y_pred_class, typ = "sens")
    spec_macro <- multi_sens_spec(y_tar, y_pred_class, typ = "spec")
    

    auc_macro <- multi_class_auc_factor(y_tar, y_pred_class)
    
    wynik_list <- list(
      ConfMat = mat_cf,
      Miary = c(
        AUC = auc_macro,
        Czulosc = sens_macro,
        Specyficznosc = spec_macro,
        Jakosc = accuracy
        
      )
    )
    return(wynik_list)
    
  } else if (is.matrix(y_hat)) {

    pred_idx <- apply(y_hat, 1, which.max)  
    classes <- levels(y_tar)
    y_pred_class <- factor(classes[pred_idx], levels = classes)
    
    mat_cf <- conf_mat_func(y_tar, y_pred_class)
    accuracy <- acc_func(mat_cf)
    
    auc_macro <- multi_class_auc_matrix(y_tar, y_hat)
    
    sens_macro <- multi_sens_spec(y_tar, y_pred_class, typ = "sens")
    spec_macro <- multi_sens_spec(y_tar, y_pred_class, typ = "spec")
    
    wynik_list <- list(
      ConfMat = mat_cf,
      Miary = c(
        AUC = auc_macro,
        Czulosc = sens_macro,
        Specyficznosc = spec_macro,
        Jakosc = accuracy
        
      )
    )
    return(wynik_list)
    
  } else {
    stop("Dla klasyfikacji wieloklasowej oczekuję factor (etykiety) lub matrix (prawdopodobieństwa).")
  }
}

################################################### KNN ###########################################################################



KNNtrain <- function(X, y_tar, k, XminNew = 0, XmaxNew = 1) {
  

  if (!(is.data.frame(X) || is.matrix(X))) {
    stop("Błąd: 'X' musi być macierzą lub data.frame.")
  }
  
  if (anyNA(X)) {
    stop("Błąd: 'X' zawiera braki danych (NA).")
  }
  if (anyNA(y_tar)) {
    stop("Błąd: 'y_tar' zawiera braki danych (NA).")
  }
  
  if (k <= 0) {
    stop("Błąd: 'k' musi być większe od 0.")
  }
  
  # ----- 2) Normalizacja zmiennych ilościowych -----
  
  X <- as.data.frame(X)
  

  numeric_cols <- sapply(X, is.numeric)
  

  minOrg <- sapply(X[, numeric_cols, drop = FALSE], min)
  maxOrg <- sapply(X[, numeric_cols, drop = FALSE], max)
  
  X_s <- X
  
  for (col_name in names(X)[numeric_cols]) {
    col_min <- min(X[[col_name]])
    col_max <- max(X[[col_name]])
    
    if (col_max == col_min) {
      X_s[[col_name]] <- XminNew
    } else {
      X_s[[col_name]] <- XminNew + (X[[col_name]] - col_min) * 
        (XmaxNew - XminNew) / (col_max - col_min)
    }
  }
  

  attr(X_s, "minOrg")   <- minOrg
  attr(X_s, "maxOrg")   <- maxOrg
  attr(X_s, "minmaxNew") <- c(XminNew, XmaxNew)
  
  # ----- 3) Przygotowanie listy wyjściowej -----
  
  model_list <- list(
    X = X_s,
    y = y_tar,
    k = k
  )
  
  return(model_list)
}


KNNpred <- function(KNNmodel, X) {

  if (anyNA(X)) {
    stop("Błąd: 'X' zawiera braki danych (NA).")
  }
  
  model_X <- KNNmodel$X
  model_y <- KNNmodel$y
  k       <- KNNmodel$k
  
  if (!setequal(colnames(X), colnames(model_X))) {
    stop("Błąd: kolumny w 'X' nie pasują do kolumn w 'KNNmodel$X'.")
  }

  X <- X[, colnames(model_X), drop = FALSE]
  
  # ----------------------------
  # 2) Normalizacja  'X' 
  # ----------------------------
  
  X_s <- as.data.frame(X)
  

  minOrg    <- attr(model_X, "minOrg")
  maxOrg    <- attr(model_X, "maxOrg")
  minmaxNew <- attr(model_X, "minmaxNew") 
  
  numeric_mask <- sapply(model_X, is.numeric)
  
  for (col_name in names(model_X)[numeric_mask]) {
    old_min <- minOrg[col_name]
    old_max <- maxOrg[col_name]
    new_min <- minmaxNew[1]
    new_max <- minmaxNew[2]
    

    if (old_max == old_min) {
      X_s[[col_name]] <- new_min
    } else {
      X_s[[col_name]] <- new_min + (X_s[[col_name]] - old_min) *
        (new_max - new_min) / (old_max - old_min)
    }
  }
  
  # ----------------------------
  # 3) Funkcja pomocnicza do liczenia odległości
  # ----------------------------
  calc_distance <- function(rowA, rowB, numeric_mask) {
    
    # Sprawdzamy 3 przypadki:
    #  a) wszystkie kolumny numeryczne -> Euklides
    #  b) wszystkie kolumny nienumeryczne -> Hamming
    #  c) mieszane -> Gower
    
    if (all(numeric_mask)) {
      # --- Euklides ---
      diff_sq <- 0
      for (j in seq_along(numeric_mask)) {
        diff_val <- as.numeric(rowA[[j]]) - as.numeric(rowB[[j]])
        diff_sq  <- diff_sq + diff_val^2
      }
      return( sqrt(diff_sq) )
      
    } else if (all(!numeric_mask)) {
      # --- Hamming ---
      diff_sum <- 0
      for (j in seq_along(numeric_mask)) {
        valA <- rowA[[j]]
        valB <- rowB[[j]]
        if (valA != valB) diff_sum <- diff_sum + 1
      }
      return(diff_sum)
      
    } else {
      # --- Gower (mieszane kolumny) ---
      dist_sum <- 0
      dist_count <- 0
      
      for (j in seq_along(numeric_mask)) {
        if (numeric_mask[j]) {

          valA <- as.numeric(rowA[[j]])
          valB <- as.numeric(rowB[[j]])
          d_j  <- abs(valA - valB)
          dist_sum <- dist_sum + d_j
          dist_count <- dist_count + 1
        } else {

          valA <- rowA[[j]]
          valB <- rowB[[j]]
          d_j  <- ifelse(valA == valB, 0, 1)
          dist_sum <- dist_sum + d_j
          dist_count <- dist_count + 1
        }
      }
      return(dist_sum / dist_count)
    }
  }
  
  # ----------------------------
  # 4) KNN predykcja 
  # ----------------------------
  
  n_model <- nrow(model_X)
  n_new   <- nrow(X_s)
  
  is_regression <- is.numeric(model_y)
  
  if (is_regression) {

    pred_values <- numeric(n_new)
    
    for (i in seq_len(n_new)) {
      x_i <- X_s[i, , drop = FALSE]
      
      dist_vec <- numeric(n_model)
      for (m in seq_len(n_model)) {
        row_m <- model_X[m, , drop = FALSE]
        dist_vec[m] <- calc_distance(row_m, x_i, numeric_mask)
      }
      
      nn_idx <- order(dist_vec)[1:k]
      neigh_y <- model_y[nn_idx]
      pred_values[i] <- mean(neigh_y)
    }
    
    return(pred_values)
    
  } else {
    # --------------------
    # *Klasyfikacja*
    # --------------------
    
    class_levels <- levels(model_y)  
    n_class <- length(class_levels)
    
    out_df <- data.frame(matrix(0, nrow = n_new, ncol = n_class))
    colnames(out_df) <- class_levels
    

    out_df[["Klasa"]] <- factor(NA, levels = class_levels)
    
    for (i in seq_len(n_new)) {
      x_i <- X_s[i, , drop = FALSE]
      

      dist_vec <- numeric(n_model)
      for (m in seq_len(n_model)) {
        dist_vec[m] <- calc_distance(model_X[m, , drop = FALSE], x_i, numeric_mask)
      }
      
      nn_idx <- order(dist_vec)[1:k]
      neigh_y <- model_y[nn_idx]  
      

      tab_class <- table(neigh_y)
      freq_class <- as.numeric(tab_class)
      

      p_vec <- numeric(n_class)
      names(p_vec) <- class_levels
      

      p_vec[names(tab_class)] <- freq_class

      p_vec <- p_vec / k
      

      out_df[i, class_levels] <- p_vec
      
      best_class <- names(which.max(p_vec))
      out_df$Klasa[i] <- best_class
    }
    
    return(out_df)
  }
}

####################################### NN ############################################################
# Zadanie 1:
# a) Opracuj uogólnienie funkcji "trainNN", "wprzod", "wstecz", tak aby rozwiązywały one problemy 
#    klasyfikacji binarnej, wieloklasowej oraz regresji. 

#    "trainNN" przyjmuje nastęujące parametry: "Yname", "Xnames", "data", "h", "lr", "iter", "seed".
#    Znaczenie parametrów: Yname - nazwa zmiennej celu z parametru data.
#                          Xnames - nazwy potencjalnych zmiennych objaśniających z parametru data.
#                          data - analizowany zbiór danych.
#                          h - wektor wskazujący liczbę warst ukrytych oraz liczbę neuronów ukrytych,
#                              np. c(3,2) definiuje dwie warstwy ukryte, odpowiednio z trzema oraz dwoma neuronami.
#                          lr - szybkość uczenia.
#                          iter - maksymalna liczba iteracji.
#                          seed - punkt początkowy dla PRNG.




sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

d_sigmoid <- function(x) {

  x * (1 - x)
}

ReLu <- function(x) {
  ifelse(x <= 0, 0, x)
}

d_ReLu <- function(x) {
  ifelse(x <= 0, 0, 1)
}

SoftMax <- function(x_vec) {

  ex <- exp(x_vec)
  ex / sum(ex)
}


MSE <- function(y_tar, y_hat) {

  mean( (y_tar - y_hat)^2 )
}

#forward pass

wprzod <- function(X, W, type = c("binary","multiclass","regression")) {
  type <- match.arg(type)
  

  L <- length(W)  
  A <- vector("list", L)  
  Z <- vector("list", L)  
  

  A0 <- X  
  
  for (l in seq_len(L)) {

    inputWithBias <- cbind(1, A0)  
    Zl <- inputWithBias %*% W[[l]] 
    
    if (l < L) {

      Al <- sigmoid(Zl)
    } else {
      if (type == "binary") {

        Al <- sigmoid(Zl)  
      } else if (type == "multiclass") {

        Al <- t( apply(Zl, 1, SoftMax) )
      } else {

        Al <- Zl  
      }
    }
    
    Z[[l]] <- Zl
    A[[l]] <- Al
    A0 <- Al  
  }
  

  return(list(A = A, Z = Z, y_hat = A[[L]]))
}


wstecz <- function(X, y_tar, W, fw, type = c("binary","multiclass","regression"), lr = 0.01) {
  type <- match.arg(type)
  
  A <- fw$A
  Z <- fw$Z
  y_hat <- fw$y_hat
  
  n <- nrow(X)
  L <- length(W)
  

  D <- vector("list", L)
  

  if (type == "binary") {

    D[[L]] <- (y_hat - y_tar) * d_sigmoid(y_hat)
  } else if (type == "multiclass") {

    D[[L]] <- (y_hat - y_tar)
  } else {
    
    D[[L]] <- (y_hat - y_tar)
  }
  

  if (L > 1) {
    for (l in seq(L-1, 1, by = -1)) {

      W_no_bias <- W[[l+1]][-1,,drop=FALSE]  

      dL <- (D[[l+1]] %*% t(W_no_bias)) * d_sigmoid(A[[l]])
      D[[l]] <- dL
    }
  }
  
  A0 <- X
  
  for (l in seq_len(L)) {
    if (l == 1) {
      inputWithBias <- cbind(1, A0)  
    } else {
      inputWithBias <- cbind(1, A[[l-1]])  
    }
    
    dW <- t(inputWithBias) %*% D[[l]] / n
    

    W[[l]] <- W[[l]] - lr * dW
  }
  
  return(W)
}


initWeights <- function(p, hidden, out_size, seed = 123) {
  set.seed(seed)
  
  layer_dims <- c(p, hidden, out_size)
  
  L <- length(layer_dims) - 1
  W <- vector("list", L)
  
  for (l in seq_len(L)) {
    in_dim <- layer_dims[l]
    out_dim <- layer_dims[l+1]
    W[[l]] <- matrix(runif((in_dim + 1)*out_dim, min=-1, max=1),
                     nrow = in_dim + 1,
                     ncol = out_dim)
  }
  return(W)
}

trainNN <- function(Yname, Xnames, data, h = c(5,5),
                    type = c("binary","multiclass","regression"),
                    lr = 0.01, iter = 1000, seed = 123)
{
  type <- match.arg(type)
  set.seed(seed)
  

  Xmat <- as.matrix(data[, Xnames, drop=FALSE])
  
  if (type == "binary") {

    y_tar_vec <- as.numeric(data[[Yname]])

    y_tar <- matrix(y_tar_vec, ncol=1)
    out_size <- 1
    
  } else if (type == "multiclass") {

    classes <- factor(data[[Yname]])
    k <- length(levels(classes))

    y_tar <- model.matrix(~ classes - 1)
    out_size <- k
    
  } else {

    y_tar <- as.matrix(data[[Yname]])
    out_size <- 1
  }
  
  p <- ncol(Xmat)
  

  W <- initWeights(p = p, hidden = h, out_size = out_size, seed = seed)

  loss_vec <- numeric(iter)
  
  for (i in seq_len(iter)) {

    fw <- wprzod(Xmat, W, type=type)
    

    y_hat <- fw$y_hat

    if (type == "binary" || type == "regression") {

      loss_vec[i] <- MSE(y_tar, y_hat)
    } else {
      
      loss_vec[i] <- MSE(y_tar, y_hat)
    }

    W <- wstecz(Xmat, y_tar, W, fw, type=type, lr=lr)
    
    if (i %% 100 == 0) {
      cat("Iter:", i, "Loss:", round(loss_vec[i], 6), "\n")
    }
  }
  
  return(list(
    W = W,
    loss = loss_vec,
    final_yhat = y_hat
  ))
}



predNN <- function(NN, Xnew, type=c("binary","multiclass","regression")) {
  type <- match.arg(type)
  W <- NN$W
  
  Xmat <- as.matrix(Xnew)
  fw <- wprzod(Xmat, W, type=type)
  y_hat <- fw$y_hat
  
  return(y_hat)
}

######################################################################### TREE #################################################


Prob <- function(y) {
  tb <- table(y)
  p <- tb / sum(tb)
  return(as.numeric(p))
}


Gini <- function(y) {
  p <- Prob(y)
  return(1 - sum(p^2))
}

Entropy <- function(y) {
  p <- Prob(y)
  p_no0 <- p[p > 0]
  return(-sum(p_no0 * log2(p_no0)))
}


SS <- function(y) {

  return(sum((y - mean(y))^2))
}


Count <- function(data, Y) {
  nrow(data)
}

# Zadanie 1:
# a) Stwórz funkcję "StopIfNot" przyjmującą nastęujące parametry: "Y", "X", "data", "type", "depth", "minobs", "overfit", "cf".
# b) Funkcja powinna sprawdzać czy nauka modelu jest możliwa do wykonania, tj:
#    - czy "data" jest ramką danych,
#    - czy wszystkie wymienione zmienne ("Y", "X") istnieją w "data",
#    - czy zmienna "Y" oraz zmienne "X" w tabeli "data" nie ma braków danych,
#    - czy "depth" oraz "minobs" są większe od 0,
#    - czy "type" przyjmuje watrtość "Gini", "Entropy", "SS",
#    - czy "overfit" przyjmuje watrtość "none" lub "prune",
#    - czy "cf" jest w przedziale (0,0.5],
#    - czy możliwe kombinacje parametrów mają sens, np. "type = SS" kiedy "Y" jest faktorem.
# c) W przypadku niespełniania któregoś z warunków, funkcja powinna wyświetlić w konsoli, czego dotyczy problem.
# d) Funkcja zwraca "TRUE", jeżeli nauka jest możliwa, w przeciwnym wypadku "FALSE". 

StopIfNot <- function(Y, X, data, type, depth, minobs, overfit, cf) {

  if (!is.data.frame(data)) {
    message("Błąd: 'data' nie jest ramką danych.")
    return(FALSE)
  }
  

  all_vars <- c(Y, X)
  if (!all(all_vars %in% colnames(data))) {
    message("Błąd: niektóre z zmiennych (Y,X) nie istnieją w 'data'.")
    return(FALSE)
  }
  

  if (anyNA(data[, all_vars])) {
    message("Błąd: zmienne Y lub X zawierają wartości NA.")
    return(FALSE)
  }
  

  if (depth <= 0) {
    message("Błąd: 'depth' musi być większe od 0.")
    return(FALSE)
  }
  if (minobs <= 0) {
    message("Błąd: 'minobs' musi być większe od 0.")
    return(FALSE)
  }
  

  if (!type %in% c("Gini","Entropy","SS")) {
    message("Błąd: 'type' musi być jedną z: 'Gini', 'Entropy', 'SS'.")
    return(FALSE)
  }
  

  if (!overfit %in% c("none","prune")) {
    message("Błąd: 'overfit' musi być 'none' lub 'prune'.")
    return(FALSE)
  }
  

  if (!(cf > 0 && cf <= 0.5)) {
    message("Błąd: 'cf' musi być w (0,0.5].")
    return(FALSE)
  }
  

  if (type == "SS") {

    if (is.factor(data[[Y]])) {
      message("Błąd: 'type=SS' dla Y będącej factorem. Niespójność.")
      return(FALSE)
    }
  } else {

    if (!is.factor(data[[Y]])) {
      message("Błąd: 'type=Gini' lub 'Entropy' dla Y nienależącej do factor. Niespójność.")
      return(FALSE)
    }
  }
  
  return(TRUE)  
}

# Zadanie 2:
# a) Stwórz funkcję "AssignInitialMeasures" przyjmującą nastęujące parametry: "tree", "Y", "data", "type", "depth".
# b) Funkcja powinna na podstawie parametrów wejściowych przypisywać do obiektu "tree" (czyli korzenia) wartości początkowe:
#    - "depth" = 0.
#    - w zależności od "type" wartość miary Gini, Entropy, SS dla calej populacji (bo to korzeń).

AssignInitialMeasures <- function(tree, Y, data, type, depth) {

  tree$depth <- 0

  if (type == "Gini") {
    tree$measure <- Gini(data[[Y]])
  } else if (type == "Entropy") {
    tree$measure <- Entropy(data[[Y]])
  } else {
    tree$measure <- SS(data[[Y]])
  }

  tree$n <- nrow(data)
  

  
  return(tree)
}

# Zadanie 3:
# a) Stwórz funkcję "AssignInfo" przyjmującą nastęujące parametry: "tree", "Y", "X", "data", "type", "depth", "minobs", "overfit", "cf".
# b) Funkcja powinna na podstawie parametrów wejściowych przypisywać do obiektu "tree" (jako attrybuty obiektu) wartości owych parametrów.

AssignInfo <- function(tree, Y, X, data, type, depth, minobs, overfit, cf) {
  attr(tree, "Y")       <- Y
  attr(tree, "X")       <- X
  attr(tree, "data")    <- data
  attr(tree, "type")    <- type
  attr(tree, "depth")   <- depth
  attr(tree, "minobs")  <- minobs
  attr(tree, "overfit") <- overfit
  attr(tree, "cf")      <- cf
  return(tree)
}

#
# Zadanie 4:
# a) Stwórz funkcję "FindBestSplit" przyjmującą nastęujące parametry: "Y", "X", "data", "parentVal", "type", "minobs".
# b) Funkcja powinna zwracać tabelę z wynikami najlepszego możliwego podziału, zawierjącą:
#    - "infGain" - zysk informacyjny dla podziału, 
#    - "lVal" - miarę niejednorodności dla lewego węzła, 
#    - "rVal" - miarę niejednorodności dla prawego węzła,
#    - "point" - punkt (lub zbiór punktów dla zmiennych kategorycznych) podzału,
#    - "Ln" - liczbę obserwacji w lewym węźle, 
#    - "Rn" - liczbę obserwacji w prawym węźle. 

FindBestSplit <- function(Y, X, data, parentVal, type, minobs) {

  
  results <- data.frame(
    infGain = numeric(0),
    lVal = numeric(0),
    rVal = numeric(0),
    point = numeric(0),
    Ln = integer(0),
    Rn = integer(0),
    var = character(0),  
    stringsAsFactors = FALSE
  )
  

  n_all <- nrow(data)
  
  measureFun <- switch(type,
                       "Gini"    = Gini,
                       "Entropy" = Entropy,
                       "SS"      = SS
  )
  
  for (varName in X) {
    x_vec <- data[[varName]]
    

    if (is.numeric(x_vec)) {

      s_unique <- sort(unique(x_vec))

      candidates <- (s_unique[-1] + s_unique[-length(s_unique)]) / 2
      

      if (length(candidates) < 1) {
        next
      }
      
      for (sp in candidates) {
        idx_left <- x_vec <= sp
        Ln <- sum(idx_left)
        Rn <- n_all - Ln
        

        if (Ln < minobs || Rn < minobs) {
          
          next
        }
        
        left_data <- data[idx_left, ]
        right_data <- data[!idx_left, ]
        
        lVal <- measureFun(left_data[[Y]])
        rVal <- measureFun(right_data[[Y]])
        

        infGain <- parentVal - (Ln / n_all * lVal + Rn / n_all * rVal)
        
        tmp <- data.frame(
          infGain = infGain,
          lVal = lVal,
          rVal = rVal,
          point = sp,
          Ln = Ln,
          Rn = Rn,
          var = varName,
          stringsAsFactors = FALSE
        )
        results <- rbind(results, tmp)
      }
      
    } else {

    }
  }
  
  if (nrow(results) == 0) {

    return(data.frame(
      infGain = 0, lVal = NA, rVal = NA,
      point = NA, Ln = 0, Rn = 0,
      var = NA
    ))
  }
  

  best_idx <- which.max(results$infGain)
  best_row <- results[best_idx, , drop = FALSE]
  return(best_row)
}
#
# Zadanie 5:
# a) Stwórz funkcję "Tree" przyjmującą nastęujące parametry: "Y", "X", "data", "type", "depth", "minobs", "overfit", "cf".
# b) Jest to rozwinięcie funkcji ze slajdu nr 19. Funckja powinna po kolei wywoływać pozostałe funkcje:
#    - "StopIfNot", jeżeli zwracana wartość to "FALSE" to kończymy działanie całej funkcji (zwracamy obiekt niewidzialny),
#    - tworzenie obiektu "tree",
#    - "AssignInitialMeasures",
#    - "BuildTree",
#    - "PruneTree", na tę chwilę ta funkcja jest pusta PruneTree<-function(){},
#    - "AssignInfo".
# c) Funkcja powwina zwracać obiekt "tree".


NodeCost <- function(node, cf) {
  if (isTRUE(node$isLeaf)) {
    return(node$n * node$measure + cf)  
  } else {
    leftCost <- NodeCost(node$left, cf)
    rightCost <- NodeCost(node$right, cf)
    return(leftCost + rightCost)
  }
}

PruneTree <- function(node, cf) {

  if (!isTRUE(node$isLeaf)) {

    node$left <- PruneTree(node$left, cf)
    node$right <- PruneTree(node$right, cf)
    

    costChildren <- NodeCost(node$left, cf) + NodeCost(node$right, cf)
    costAsLeaf <- node$n * node$measure + cf
    
    if (costAsLeaf <= costChildren) {

      node$isLeaf <- TRUE
      node$left <- NULL
      node$right <- NULL
    }
  }
  return(node)
}


BuildTree <- function(node, Y, X, data, type, depth, minobs, currentDepth = 0) {

  node$n <- nrow(data)
  

  if (currentDepth >= depth) {
    node <- MakeLeaf(node, data, Y, type)
    return(node)
  }
  

  if (node$measure <= 1e-15) {
    node <- MakeLeaf(node, data, Y, type)
    return(node)
  }
  

  bestSplit <- FindBestSplit(Y, X, data, parentVal = node$measure, type, minobs)
  

  if (bestSplit$infGain <= 0) {
    node <- MakeLeaf(node, data, Y, type)
    return(node)
  }
  

  node$var <- bestSplit$var
  node$point <- bestSplit$point
  node$infGain <- bestSplit$infGain
  node$Leaf <- NULL   
  

  x_vec <- data[[ node$var ]]
  idx_left <- x_vec <= node$point
  data_left <- data[idx_left, ]
  data_right <- data[!idx_left, ]
  

  left_child <- list()
  left_child$depth <- currentDepth + 1
  

  if (type == "Gini") {
    left_child$measure <- Gini(data_left[[Y]])
  } else if (type == "Entropy") {
    left_child$measure <- Entropy(data_left[[Y]])
  } else {
    left_child$measure <- SS(data_left[[Y]])
  }
  

  node$left <- BuildTree(
    node        = left_child,
    Y           = Y,
    X           = X,
    data        = data_left,
    type        = type,
    depth       = depth,
    minobs      = minobs,
    currentDepth = currentDepth + 1
  )
  

  right_child <- list()
  right_child$depth <- currentDepth + 1
  
  if (type == "Gini") {
    right_child$measure <- Gini(data_right[[Y]])
  } else if (type == "Entropy") {
    right_child$measure <- Entropy(data_right[[Y]])
  } else {
    right_child$measure <- SS(data_right[[Y]])
  }
  

  node$right <- BuildTree(
    node        = right_child,
    Y           = Y,
    X           = X,
    data        = data_right,
    type        = type,
    depth       = depth,
    minobs      = minobs,
    currentDepth = currentDepth + 1
  )
  
  return(node)
}

MakeLeaf <- function(node, data, Y, type) {
  node$Leaf <- "*"

  node$isLeaf <- TRUE
  
  if (type == "SS") {

    node$Prediction <- mean(data[[Y]])
  } else {

    freq <- table(data[[Y]])
    p <- freq / sum(freq)
    node$Prob <- as.numeric(p)
    names(node$Prob) <- names(freq)
    node$Class <- names(which.max(freq))
  }
  

  node$left <- NULL
  node$right <- NULL
  node$infGain <- 0 
  
  return(node)
}

Tree <- function(Y, X, data, type = "Gini", depth = 3, minobs = 2, overfit = "none", cf = 0.01) {
  ok <- StopIfNot(Y, X, data, type, depth, minobs, overfit, cf)
  if (!ok) {
    message("Parametry niepoprawne. Zwracam niewidoczny NULL.")
    return(invisible(NULL))
  }
  
  tree <- list()
  

  tree <- AssignInitialMeasures(tree, Y, data, type, depth)
  

  tree <- BuildTree(tree, Y, X, data, type, depth, minobs, currentDepth = 0)
  

  if (overfit == "prune") {
    tree <- PruneTree(tree, cf)
  }

  tree <- AssignInfo(tree, Y, X, data, type, depth, minobs, overfit, cf)

  return(tree)
}

# Zadanie 6:
# a) Dokonaj integracji opracowanej funkcji "FindBestSplit" z funkcjami "Tree" oraz "BuildTree".

PredictTree <- function(tree, data) {

  type <- attr(tree, "type")
  

  used_vars <- attr(tree, "X")
  

  if (!all(used_vars %in% colnames(data))) {
    stop("Nie wszystkie zmienne użyte w drzewie istnieją w nowym zbiorze danych!")
  }
  

  

  predictOne <- function(obs, node) {

    if (isTRUE(node$isLeaf)) {
      
      if (type == "SS") {

        return(node$Prediction) 
      } else {
        
        return(list(prob = node$Prob, class = node$Class))
      }
    }
    

    splitVar <- node$var
    splitPoint <- node$point
    

    xval <- obs[[splitVar]]
    if (xval <= splitPoint) {
      return(predictOne(obs, node$left))
    } else {
      return(predictOne(obs, node$right))
    }
  }
  

  n <- nrow(data)
  if (n == 0) {
    if (type == "SS") {
      return(numeric(0))
    } else {
      return(data.frame())
    }
  }
  

  if (type == "SS") {

    preds <- numeric(n)
    for (i in seq_len(n)) {
      obs_i <- data[i, , drop=FALSE]
      val <- predictOne(obs_i, tree)
      preds[i] <- val
    }
    return(preds)
    
  } else {

    
    getClassesFromAnyLeaf <- function(node) {
      if (isTRUE(node$isLeaf)) {

        return(names(node$Prob))
      }

      return(getClassesFromAnyLeaf(node$left))
    }
    
    classNames <- getClassesFromAnyLeaf(tree)
    k <- length(classNames)
    

    probMat <- matrix(0, nrow = n, ncol = k)
    predictedClass <- character(n)
    
    for (i in seq_len(n)) {
      obs_i <- data[i, , drop=FALSE]
      rez <- predictOne(obs_i, tree)  


      probMat[i, ] <- rez$prob
      predictedClass[i] <- rez$class
    }
    

    dfPred <- as.data.frame(probMat)
    colnames(dfPred) <- classNames
    
    dfPred$Klasa <- factor(predictedClass, levels = classNames)  
    return(dfPred)
  }
}
#______________________________________WIZUALIZACJA___________________________________________________________
# install.packages("ggplot2")   
# install.packages("tidyr")     
# install.packages("dplyr")     

library(ggplot2)
library(tidyr)
library(dplyr)

plot_metrics <- function(df, param_cols, metric_cols, sep = "_") {
  

  df_params <- df %>%
    mutate(
      param_label = if (length(param_cols) == 1) {

        as.character(.data[[param_cols]])
      } else {

        do.call(paste, c(across(all_of(param_cols)), sep = sep))
      }
    )
  

  df_long <- df_params %>%
    pivot_longer(
      cols = all_of(metric_cols),
      names_to = "Metryka",
      values_to = "Wartosc"
    )

  ggplot(df_long, aes(x = param_label, y = Wartosc, fill = Metryka)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
    labs(
      x = paste(param_cols, collapse = " / "),
      y = "Wartość metryki",
      fill = "Metryka",
      title = "Porównanie metryk dla różnych wariantów parametrów"
    ) +
    theme_minimal() +

    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}