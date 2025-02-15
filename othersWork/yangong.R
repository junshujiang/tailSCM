## R CODE SUPPORTING THE FOLLOWING PAPER:
## Partial Tail-Correlation Coefficient, Applied to Extremal-Network Learning 
## Authors: Yan Gong^{1;2}, Peng Zhong^3, Thomas Opitz^4 and Raphael Huser^1 
##  1. King Abdullah University of Science and Technology (KAUST), Thuwal, Saudi Arabia; 
##  2. Harvard School of Public Health, Boston, US
##  3. University of New South Wales, Sydney, Australia
##  4. INRAE, Biostatistics and Spatial Processes, Avignon, France

###################################
## transformed-linear operations ##
###################################
library(spectralGraphTopology)
library(igraph)
library(evd)
library(quadprog)

t_fun <- function(y){
  res <- vector()
  for(k in 1:length(y)){
    if(y[k]>15){
      res[k] = y[k] 
    } else{
      res[k] =  log(1 + exp(y[k]))
    }
  }
  return(res)
}

t_inv <- function(x){
  res <- vector()
  for(k in 1:length(x)){
    if(x[k]>15){
      res[k] = x[k] 
    } else{
      res[k] =  log(exp(x[k])-1) 
    }
  }
  return(res)
}

plus <- function(x1, x2){ t_fun(t_inv(x1) + t_inv(x2)) }
minus <- function(x1, x2){ t_fun(t_inv(x1) - t_inv(x2)) }
multiply <- function(c, x){ t_fun(c*t_inv(x)) }
mat.multiply <- function(x1.vec, x2.vec){ t_fun(x1.vec%*%t_inv(x2.vec))}

###########
## Utils ##
###########
mynorm <- function(x){sqrt(sum(x^2))}
my_trans <- function(x){
  x_trans = x-mean(x)
  for(i in 1:length(x_trans)){
    if(x_trans[i] < 0){x_trans[i] <- 0}
  }
  return(x_trans)
}

