#' The t-SNE method for dimensionality reduction
#'
#' This function is copied from the tsne package found in CRAN. 
#' Provides a simple function inteface for specifying t-SNE dimensionality reduction on R matrices or "dist" objects.
#' 
#' @param X The R matrix or "dist" object
#' @param initial_config An argument providing a matrix specifying the initial embedding for X. See Default is NULL.
#' @param k  The dimension of the resulting embedding. Default is 2.
#' @param whitening A boolean value indicating whether the matrix data should be whitened. Default is FALSE.
#' @param initial_dims The number of dimensions to use in reduction method, if whitening True. Default is 30.
#' @param max_iter Maximum number of iterations to perform. Default is 1000.
#' @param perplexity Perplexity parameter. (optimal number of neighbors). Default is 30.
#' @param min_cost The minimum cost value (error) to halt iteration. Default is 0.
#' @param epoch_callback 	A callback function used after each epoch (an epoch here means a set number of iterations). Default is NULL.
#' @param epoch The number of iterations in between update messages. Default is 100.
#' 
#' @return ydata : An R object containing a ydata embedding matrix, as well as a the matrix of probabilities P
#' 
#'  
#' @keywords tsne, multisne, visualisation, dimensionality reduction, nonlinear
#' @export
#' @examples
#' ## Not run: 
#' colors = rainbow(length(unique(iris$Species)))
#' names(colors) = unique(iris$Species)
#' ecb = function(x,y){ plot(x,t='n'); text(x,labels=iris$Species, col=colors[iris$Species]) }
#' tsne_iris = tsne(iris[,1:4], epoch_callback = ecb, perplexity=50)
#'
#' # compare to PCA
#' dev.new()
#' pca_iris = princomp(iris[,1:4])$scores[,1:2]
#' plot(pca_iris, t='n')
#' text(pca_iris, labels=iris$Species,col=colors[iris$Species])

## End(Not run)
#' 


# t-SNE #
tSNE <- function(X, initial_config = NULL, k = 2, initial_dims = 30,
                 perplexity = 30, max_iter = 1000, min_cost = 0, epoch_callback = NULL,
                 whitening = FALSE, epoch = 100){
  if (any("dist" == class(X))) {
    n = attr(X, "Size")
  } else {
    X = as.matrix(X)
    X = X - min(X)
    X = X/max(X)
    initial_dims = min(initial_dims, ncol(X))
    if (whitening){
      X <- whiten(as.matrix(X), n.comp = initial_dims)
    }
    n = nrow(X)
  }
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
  P = x2p(X, perplexity, 1e-05)$P
  P = 0.5 * (P + t(P))
  P[P < eps] <- eps
  P = P/sum(P, na.rm=T)
  P = P * initial_P_gain
  grads = matrix(0, nrow(ydata), ncol(ydata))
  incs = matrix(0, nrow(ydata), ncol(ydata))
  gains = matrix(1, nrow(ydata), ncol(ydata))
  for (iter in 1:max_iter) {
    if (iter%%epoch == 0) {
      cost = sum(apply(P * log((P + eps)/(Q + eps)), 1,
                       sum), na.rm=T)
      message("Epoch: Iteration #", iter, " error is: ",
              cost)
      if (cost < min_cost){
        break
      }
      if (!is.null(epoch_callback)){
        epoch_callback(ydata)
      }
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
    stiffnesses = 4 * (P - Q) * num
    for (i in 1:n) {
      grads[i, ] = apply(sweep(-ydata, 2, -ydata[i, ]) *
                           stiffnesses[, i], 2, sum)
    }
    gains = ((gains + 0.2) * abs(sign(grads) != sign(incs)) +
               gains * 0.8 * abs(sign(grads) == sign(incs)))
    gains[gains < min_gain] = min_gain
    incs = momentum * incs - epsilon * (gains * grads)
    ydata = ydata + incs
    ydata = sweep(ydata, 2, apply(ydata, 2, mean))
    if (iter == mom_switch_iter){
      momentum = final_momentum
    }
    if (iter == 100 && is.null(initial_config)){
      P = P/4
    }
  }
  return(ydata)
}