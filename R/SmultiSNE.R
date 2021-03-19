#' S-Multi-SNE: Semi-supervised multi-SNE
#'
#' This function performs S-multi-SNE, a visualization and classification algorithm for multi-view data.
#' One of the data-views in the input list contains the labeling information with dummy variables representing the different clusters. Unlabeled samples are represented with NA in all variables of the labeling data-view.
#' 
#' 
#' @param X A list with each element representing a data-view. All data-views must have the same number of rows.
#' @param initial_config Initialization matrix, specifying the initial embedding for each data-view. Should be a list of the same length as X. Default is NULL.
#' @param k Number of dimensions in the lhydatent embeddings. Default is 2.
#' @param whitening If True, whitening process to reduce the dimensions of the input data-views will be applied prior to multi-SNE. Default is FALSE.
#' @param initial_dims The number of dimensions of the whitened data-views, if whitening True. Default is 30.
#' @param max_iter Maximum number of iterations to run. Default is 1000.
#' @param perplexity Value of perplexity parameter. Default is 30.
#' @param min_cost Minimum cost to halt iteration Default is 0.
#' @param epoch_callback 	A callback function used after each epoch (an epoch here means a set number of iterations). Default is NULL.
#' @param epoch The number of iterations in between update messages. Default is 100.
#' @param weights Initialization of the weights. A vector of the same length as X. Default is NULL.
#' @param weightUpdating Boolean. If True, weights will be updated at each iteration. Default is FALSE.
#' @param lambdaParameter Parameter to indicate the contribution of weight update. Default is 1.
#' 
#' @return Y : The latent embeddings.
#' @return Weights : A matrix containing the weights used for each iteration. Rows represent the iteration and columns the data-view.
#' @return Errors : A matrix containing the errors used for each iteration. Rows represent the iteration and columns the data-view.
#' 
#'  
#' @keywords tsne, multisne, visualisation, dimensionality reduction, nonlinear
#' @export
#' @examples
#' # Get sample data
#' X <- vector("list")
#' X$first_dataView <- rbind(matrix(rnorm(10000),nrow=500,ncol=20), matrix(rnorm(5000, mean=1,sd=2),nrow=250,ncol=20))
#' X$second_dataView <- rbind(matrix(rpois(20000, lambda = 1),nrow=500,40), matrix(rpois(10000, lambda=3),250,40))
#' X$labeling <- rbind(cbind(rep(1,500), rep(0,500)), cbind(rep(0,250), rep(1,250)))
#' nan_index <- sample(seq(1,750,1),250)
#' X$labeling[nan_index,] <- c(NA,NA)
#' # Run S-multi-SNE
#' Y <- SmultiSNE(X, max_iter = 200)
#' true_labels <- c(rep(2,500), rep(3,250))
#' missing_labels <- true_labels
#' missing_labels[nan_index] <- 1
#' par(mfrow=c(1,2))
#' plot(Y$Y, col = true_labels, pch=19)
#' plot(Y$Y, col = missing_labels, pch=19)
#' 
#' 

## Semi-supervised ##
# S-multi-SNE #
SmultiSNE <- function (X, initial_config = NULL, k = 2, initial_dims = 30,
                               perplexity = 30, max_iter = 1000, min_cost = 0, epoch_callback = NULL,
                               whitening = FALSE, epoch = 100, weights=NULL, weightUpdating=TRUE, lambdaParameter=1){
  M <- length(X)
  for (i in 1:M){
    if ("dist" %in% class(X[[i]])) {
      n = attr(X[[i]], "Size")
    }
    else {
      if (any(is.na(X[[i]]))){
        r <- apply(X[[i]], 1, function(x) any(is.na(x)))
        Xnan <- X[[i]][r,]
        Xclean <- X[[i]][!r,]
        Xclean = as.matrix(Xclean)
        Xclean = Xclean - min(Xclean)
        Xclean = Xclean/max(Xclean)
        initial_dims = min(initial_dims, ncol(Xclean))
        if (whitening){
          Xclean <- whiten(as.matrix(Xclean), n.comp = initial_dims)
        }
        X[[i]][r,]  <- Xnan
        X[[i]][!r,] <- Xclean
      } else {
        X[[i]] = as.matrix(X[[i]])
        X[[i]] = X[[i]] - min(X[[i]])
        X[[i]] = X[[i]]/max(X[[i]])
        initial_dims = min(initial_dims, ncol(X[[i]]))
        if (whitening){
          X[[i]] <- whiten(as.matrix(X[[i]]), n.comp = initial_dims)
        }
      }
    }
  }
  # Prepare weights
  if (is.null(weights)){
    w <- rep(1,M)
  } else {
    w <- weights
  }
  w <- w/sum(w, na.rm=T)
  Ctemp = rep(0,M)
  z = rep(0,M) #  Used in the automatic update of weights
  Weights = matrix(0, nrow = max_iter, ncol = M)
  Errors = matrix(0, nrow = max_iter, ncol = M)
  # Initialisation
  n = nrow(X[[1]])
  momentum = 0.5
  final_momentum = 0.8
  mom_switch_iter = 250
  epsilon = 500
  min_gain = 0.01
  initial_P_gain = 4
  eps = 2^(-52)
  if (!is.null(initial_config) && is.matrix(initial_config)) {
    if (nrow(initial_config) != n | ncol(initial_config) !=
        k) {
      stop("initial_config argument does not match necessary configuration for X")
    }
    ydata = initial_config
    initial_P_gain = 1
  } else {
    ydata = matrix(rnorm(k * n), n)
  }
  P <- vector("list")
  for (i in 1:M){
    if (any(is.na(X[[i]]))){
      r <- apply(X[[i]], 1, function(x) any(is.na(x)))
      Xnan <- X[[i]][r,]
      Xclean <- X[[i]][!r,]
      Xclean = as.matrix(Xclean)
      Pclean = x2p(Xclean, perplexity, 1e-05)$P
      P[[i]] <- matrix(NA, nrow = nrow(X[[i]]), ncol = nrow(X[[i]]))
      #P[[i]][r,] <- Xnan
      P[[i]][!r,!r] <- Pclean
    }else{
      P[[i]] = x2p(X[[i]], perplexity, 1e-05)$P
    }
    P[[i]] = 0.5 * (P[[i]] + t(P[[i]]))
    P[[i]][P[[i]] < eps] <- eps
    P[[i]] = P[[i]]/sum(P[[i]], na.rm=T)
    P[[i]] = P[[i]] * initial_P_gain
  }
  #P = lapply(X, function(x) x2p(x,perplexity,1e-05)$P)
  grads = matrix(0, nrow(ydata), ncol(ydata))
  incs = matrix(0, nrow(ydata), ncol(ydata))
  gains = matrix(1, nrow(ydata), ncol(ydata))
  for (iter in 1:max_iter) {
    if (iter%%epoch == 0) {
      total_cost = 0
      for (i in 1:M){
        temp_cost = 0
        temp_cost = sum(apply(P[[i]] * log((P[[i]] + eps)/(Q + eps)), 1,
                              sum), na.rm=T)
        message("Epoch: Iteration #", iter, ", data-view #",i, ", error is: ",
                temp_cost)
        total_cost = total_cost + temp_cost
      }
      message("Epoch: Iteration #", iter,  ", total error is: ",
              total_cost)
      if (total_cost < min_cost)
        break
      if (!is.null(epoch_callback))
        epoch_callback(ydata)
    }
    sum_ydata = apply(ydata^2, 1, sum)
    num = 1/(1 + sum_ydata + sweep(-2 * ydata %*% t(ydata),
                                   2, -t(sum_ydata)))
    diag(num) = 0
    Q = num/sum(num, na.rm=T)
    if (any(is.nan(num))){
      message("NaN in grad. descent")
    }
    Q[Q < eps] = eps
    for (i in 1:M){
      stiffnesses = 4 * (P[[i]] - Q) * num
      if (i==1){
        for (j in 1:n) {
          if (!any(is.na(X[[i]][j,]))){
            grads[j, ] = w[i]*apply(sweep(-ydata, 2, -ydata[j, ]) *
                                      stiffnesses[, j], 2, function(x)sum(x, na.rm = T))
          }
        }
      } else {
        for (j in 1:n) {
          if (!any(is.na(X[[i]][j,]))){
            grads[j, ] = grads[j,] + w[i]*apply(sweep(-ydata, 2, -ydata[j, ]) *
                                                  stiffnesses[, j], 2, function(x)sum(x, na.rm = T))
          }
        }
      }
    }
    gains = ((gains + 0.2) * abs(sign(grads) != sign(incs)) +
               gains * 0.8 * abs(sign(grads) == sign(incs)))
    gains[gains < min_gain] = min_gain
    incs = momentum * incs - epsilon * (gains * grads)
    ydata = ydata + incs
    ydata = sweep(ydata, 2, apply(ydata, 2, mean))
    if (iter == mom_switch_iter)
      momentum = final_momentum
    for (i in 1:M){
      C = sum(P[[i]]*log(P[[i]]/Q), na.rm=T)
      if (is.na(C)){
        Ctemp[i] <- 0
      } else {
        Ctemp[i] <- C
      }
      if (iter == 100 && is.null(initial_config)){
        P[[i]] = P[[i]]/4
      }
    }
    if (weightUpdating == TRUE){
      wc = Ctemp
      if (sum(wc, na.rm=T)==0){
        wc <- rep(1,M)
      }
      wc <- wc/sum(wc, na.rm=T)
      w <- 1-wc
      w <- w/sum(w, na.rm=T)
    } else{
      w <- rep(1,M)
      w <- w/sum(, na.rm=T)
    }
    Weights[iter,] <- w
    Errors[iter,] <- Ctemp
  }
  return(list("Y" = ydata, "Weights" = Weights, "Errors" = Errors))
}
