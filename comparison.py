import pandas as pd
import numpy as np
from rpy2 import  robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


    # 激活 pandas 和 R 的自动转换

def ComparisonBodik(data:pd.DataFrame,lag=2)->np.array: ##shape=(n_samples,n_features)
    pandas2ri.activate()
    robjects.r('''

        source("othersWork/bodik.R")
        library(igraph)   # For visualizing the final graph estimates
        library(EnvStats) # Or any other package that can generate Pareto noise
        a_func <- function(w,lag_future=2) {
            G <- Extreme_causality_full_graph_estimate(w, lag_future = lag_future) 
            if(class(G)== "character"  && grepl("Empty", G)){
                return (NULL)
            }else{
                
                edge<-G$G
                return(edge)
            }
        }

    ''')


    function=robjects.r['a_func']
    with localconverter(robjects.default_converter + pandas2ri.converter):
        edge  = function(data,lag)
        if isinstance(edge, robjects.rinterface.NULLType):
            return np.zeros((data.shape[1],data.shape[1],2),dtype='<U3')
        else:
            edge=np.array(edge)-1
            if edge.ndim==1:
                edge=edge[np.newaxis,:]
            
    adj_matrix=np.zeros((data.shape[1],data.shape[1],2),dtype='<U3')
    try:
        for i,j in edge:
            adj_matrix[i,j,1]="-->"   
    except Exception as e:
        print(adj_matrix)
        print(str(e))
        print(edge)
        raise e
    return adj_matrix
        
def ComparisonGong(data:pd.DataFrame,alpha=0.04,beta=1.26)->np.array: ##shape=(n_samples,n_features)
    robjects.r(""" 

    source("othersWork/yangong.R")

    gong<-function(input,alpha,beta){

    d = dim(input)[2]
    n = dim(input)[1]

    ## marginal transformation by sites, to Frechet(2) ##
    input.trans = input
    for(i in 1:d){
        u = rank(input[,i], ties.method = "random")/(n+1)
        input.trans[,i] = sqrt(-1/log(u))
    }

    ######################
    ## compute the TPDM ##
    ######################
    x = input.trans
    p = dim(x)[2]
    mynorm <- function(x){sqrt(sum(x^2))}
    r = apply(x, 1, mynorm)
    w = x/r
    n.samp = n
    r0 = quantile(r, 0.99) # 0.99 is the threshold to be specified
    n_exc = sum(r>r0)
    m = p
    TPDM <- matrix(nrow = p, ncol = p)
    idx = which(r>r0)
    for(i in 1:p){
        for(k in i:p){
        sum_one = vector()
        for(t in 1:length(idx)){
            sum_one[t] = w[idx[t],i]*w[idx[t],k]
        }
        TPDM[i,k] <- TPDM[k,i] <- m/length(idx)*sum(sum_one) 
        }
    }



    graph_onecomponent = learn_k_component_graph(S = TPDM, k = 1, w0 = "qp",
                                                    alpha = alpha, beta = beta,
                                                    verbose = F, fix_beta = TRUE, abstol = 1e-3)
    graph.input = graph_onecomponent
    graph.input$adjacency[which(graph.input$adjacency!=0, arr.ind = T)] <- 1

    graph = graph_from_adjacency_matrix(graph.input$adjacency, mode = "undirected")
    ##file_name <- paste0("tmp/graph_plot_", alpha, "_", beta, ".pdf"
    return_result<-graph.input$adjacency
    return (return_result)
    }

    """)

    function=robjects.r["gong"]
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_data= function(data,alpha,beta)
    adj_matrix=np.array(r_data)
    results=adj_matrix[:,:,np.newaxis]
    x=np.zeros_like(results,dtype='<U3')
    x[results==1]="o-o"
    return x
