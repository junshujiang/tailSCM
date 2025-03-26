import causaldag as cd
import pandas as pd
import numpy as np 
from threadpoolctl import threadpool_limits
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from helper_util import *
from scipy import stats
import mkl
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects import r
numpy2ri.activate()
mkl.set_num_threads(1)
import os 
os.environ["OMP_NUM_THREADS"] = "1"
np.seterr(all='ignore')
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"




def simulation(n,d,alpha=2):
    latents=np.random.uniform(0,1,size=(n,np.eye(d) .shape[0]))
    ferchet=np.power(1-latents,-1/alpha)
    return ferchet




'''
Author: Angus
Description:
    This function generates a directed acyclic graph (DAG) with a specified number of nodes and a given edge probability (<0.1).
    The function can also generate a lagged causal DAG if specified (with contemporaneous edges unexists).
    The generated DAG is represented as a an adjacency matrix, and a edge matrix

Parameters:
    num_nodes (int): The number of nodes in the DAG.
    edge_probability (float): The probability of an edge existing between any two nodes.
    lagged_causal (bool): If True, generates a lagged causal DAG. Default is False.

Returns:
    adjacency_matrix (numpy.ndarray): The adjacency matrix of the DAG. (shape: num_nodes * num_nodes). for lagged causal, the adjacency matrix is B[1:num_nodes,num_nodes:2*num_nodes] (the part of one-step lagged causal).
    edges matrix(numpy.ndarray): 
'''

def generate_dag(num_nodes, edge_probability=0.3,lagged_causal=False):


    # generate adjacency matrix B
    ## for contemporaneous causal, B is upper triangular to avoid cycle
    adjacency_matrix=np.zeros((num_nodes,num_nodes))
    for i in range(num_nodes):
        if lagged_causal: 
            start_index=0
        else:
            start_index=i+1
        for j in range(start_index, num_nodes):
            if np.random.rand() < edge_probability:
                adjacency_matrix[i, j] = np.random.rand()


    adjacency_matrix=adjacency_matrix.T
    


    edge_shape=np.zeros(shape=(num_nodes,num_nodes,1), dtype='<U3')
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] != 0:
                edge_shape[j,i,0]="-->"
                if not lagged_causal:
                    edge_shape[i,j,0]="<--"
    if lagged_causal:
        edge_shape_2=np.zeros(shape=(num_nodes, num_nodes, 2), dtype='<U3')
        edge_shape_2[:,:,[1]]=edge_shape
        edge_shape=edge_shape_2


    return adjacency_matrix,edge_shape


'''
Author: ANgus
Description:
    This function generates a time series of Directed Acyclic Graphs (DAGs) with both contemporaneous and lagged causal relationships.
    The function first generates a contemporaneous adjacency matrix and its corresponding ground truth graph.
    Then, it generates lagged adjacency matrices and their corresponding ground truth graphs for each lag up to tau.
    Finally, it stacks the contemporaneous and lagged adjacency matrices and ground truth graphs into 3D arrays.
Parameters:
    num_nodes (int): Number of nodes in the graph.
    sparsity_lag (float): Sparsity level for the lagged causal relationships.
    sparsity_contemp (float): Sparsity level for the contemporaneous causal relationships.
    tau (int): Number of lags to consider.
Returns:
    tuple: A tuple containing the stacked adjacency matrix and the stacked ground truth graph.
'''


def generate_dag_timeseries(num_nodes,sparsity_lag,sparsity_contemp,tau):

    adjacency_matrix_contemp,ground_true_graph_contemp = generate_dag(num_nodes,edge_probability=sparsity_contemp,lagged_causal=False)


    adjacency_matrix_laggeds,ground_true_graph_laggeds=[],[]
    for i in range(tau):
        adjacency_matrix_lagged,ground_true_graph_lagged = generate_dag(num_nodes,edge_probability=sparsity_lag,lagged_causal=True)
        adjacency_matrix_laggeds.append(adjacency_matrix_lagged)
        ground_true_graph_laggeds.append(ground_true_graph_lagged)

    true_graph=np.zeros(shape=(num_nodes,num_nodes,tau+1), dtype='<U3')
    true_graph[:,:,0]=ground_true_graph_contemp[:,:,0]
    for i in range(tau):
        true_graph[:,:,i+1]=ground_true_graph_laggeds[i][:,:,1]


    adjacency_matrix=np.stack([adjacency_matrix_contemp]+adjacency_matrix_laggeds,axis=2)


    return adjacency_matrix,true_graph


def generate_dag_two_tails(num_nodes, edge_probability,lagged_causal=False):
    """Generates a DAG with two tails.

    Args:
        num_nodes (int): Number of nodes in one tail.
        edge_probability (float): Probability of edge creation between nodes.

    Returns:
        tuple: A tuple containing the adjacency matrix and edge shape matrix.
    """
    
    adjacency_matrix = np.zeros((num_nodes*2, num_nodes*2))
    # 添加有向边并确保无环
    for i in range(2*num_nodes):
        if lagged_causal: 
            start_index=0
        else:
            start_index=i+1
        for j in range(start_index, 2*num_nodes):
            if j == i + num_nodes and not lagged_causal:  # no edge between u and l if it is contemeperous
                continue
            if np.random.rand() < edge_probability:
                adjacency_matrix[i, j] = np.random.rand()

    edge_shape = np.zeros(shape=(num_nodes*2, num_nodes*2, 1), dtype='<U3')

    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] != 0:
                edge_shape[j, i, 0] = "-->"
                if not lagged_causal:
                    edge_shape[i,j,0]="<--"

    return adjacency_matrix, edge_shape









'''
Author: Angus
Description:
    This function compares two graphs and calculates the degree of mismatch between them.
    It returns the proportion of mismatched edges and the total number of mismatches.
    It ignore the direction of the edges.

Input:
    graph1: np.array, the first graph represented as an adjacency matrix.
    graph2: np.array, the second graph represented as an adjacency matrix.
    lagged: bool, indicates whether the graphs are lagged (default is False).

Output:
    mismatch_ratio: float, the ratio of mismatched edges to total edges.
    mismatch: int, the total number of mismatched edges.
'''


def compare_graphs(graph1,graph2,ignore_direction=False,ignore_comtemperous=False):

    assert graph1.shape==graph2.shape
    graph1_tmp=graph1.copy()
    graph2_tmp=graph2.copy()


    num_of_edges1=(graph1_tmp!="").sum()
    num_of_edges2=(graph2_tmp!="").sum()
    total_edges=num_of_edges1+num_of_edges2

    if ignore_direction:
        graph1_tmp[graph1_tmp!=""]="--"
        graph2_tmp[graph2_tmp!=""]="--"

    if ignore_comtemperous:
        graph1_tmp[:,:,0]=""
        graph2_tmp[:,:,0]=""

    mismatch=(graph1_tmp!=graph2_tmp).sum()

    
    
    return mismatch/(1e-5+total_edges),mismatch




def compare_graphs_without_undetectable(graph1,ground_truth):
    graph1_tmp=graph1.copy()
    graph2_tmp=ground_truth.copy()
    arcs = set()
    undetected_edges=np.zeros_like(ground_truth,dtype=int)
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[1]):
            if ground_truth[i,j,0]=="-->":
                arcs.add((i,j))
    dag = cd.DAG(arcs=arcs)
    cpdag = dag.cpdag()
    for edge in iter(cpdag.edges):

        start,end=list(edge)
        undetected_edges[start,end,0]=1
        undetected_edges[end,start,0]=1
    graph1_tmp[undetected_edges==1]=""
    graph2_tmp[undetected_edges==1]=""
    return compare_graphs(graph1_tmp,graph2_tmp,ignore_direction=False)



    


''' 


'''
def compare_timeseries_graphs(results,edges_matrix,exclude_contemp=True,exclude_self=True):
    result_here=results.copy()
    group_truth_here=edges_matrix.copy()
    if exclude_contemp:
        group_truth_here[:,:,0 ]=""
        result_here[:,:,0 ]=""
    if exclude_self:
        mask = np.eye(group_truth_here.shape[0], dtype=bool)

        mask_3d = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
        group_truth_here[mask_3d]=""
        result_here[mask_3d]=""
    return compare_graphs(result_here,group_truth_here,True)


# 使用 linspace 生成 0 到 1 的等间距值


def custom_linspace(start, stop, num, alpha=1.0):
    """
    生成一个非线性变化的数列，alpha 控制快慢变化的程度。
    起点和终点保持不变，映射回原来的区间。
    """

    linear_space = np.linspace(0, 1, num)
    

    transformed_space = linear_space**alpha

    result = start + (stop - start) * (transformed_space - transformed_space.min()) / (transformed_space.max() - transformed_space.min())
    return result





# Check if a matrix is upper triangular
def is_upper_triangular(matrix):
    return np.allclose(matrix, np.triu(matrix))


# This function simulates a time series based on the given parameters and adjacency matrix.
# It checks if the adjacency matrix is upper triangular, calculates the number of lags,
# generates simulated data, and processes it to produce a DataFrame of the time series.

def simulation_timeseries(T, burn_in, adjacency_matrix):
    # Check if the first slice of the input adjacency matrix is upper triangular
    assert is_upper_triangular(adjacency_matrix[:, :, 0].T)
    # Calculate the number of lags, tau
    tau = adjacency_matrix.shape[2] - 1
    num_nodes=adjacency_matrix.shape[0]
    # Generate simulated data with a time series length of T + tau + burn_in
    N_data = simulation(T + tau + burn_in, num_nodes).T

    # Calculate the initial condition matrix IC_0
    IC_0 = np.linalg.inv(np.eye(adjacency_matrix.shape[0]) - adjacency_matrix[:, :, 0])
    # Initialize the IC_1 list to store matrices for each lag
    IC_1_lists = []
    for i in range(tau):
        # Compute the matrix for each lag and add it to the list
        IC_1_lists.append(IC_0 @ adjacency_matrix[:, :, 1 + i])

    # Initialize the data list X_data_list
    X_data_list = []

    # Add the data for the first tau time points to X_data_list
    for i in range(tau):
        X_data_list.append(N_data[:, i])

    # Starting from the tau-th time point, compute the result for each time point and add it to X_data_list
    for i in range(tau, T + tau + burn_in):
        # Compute the result for the current time point
        results = otimes(IC_0, N_data[:, i], False)
        for j in range(tau):
            # Incorporate the effect of lags
            results = oplus(results, otimes(IC_1_lists[j], X_data_list[-j - 1], False), False)

        # Add the result to the data list
        X_data_list.append(results)

    # Convert the final time series data to DataFrame format and return
    data_df = pd.DataFrame(np.array(X_data_list[-T:]))
    return data_df


def generate_dag_timeseries_for_both_tail(num_nodes, sparsity_lag, sparsity_contemp, tau):
    """Generates DAG structure for time series data with both contemporaneous and lagged edges.
    
    Args:
        num_nodes (int): Number of nodes in the graph
        sparsity_lag (float): Edge probability for lagged connections (0-1)
        sparsity_contemp (float): Edge probability for contemporaneous connections (0-1) 
        tau (int): Number of lags to consider
        
    Returns:
        tuple: Contains:
            - np.ndarray: Combined adjacency matrix of shape (num_nodes, num_nodes, tau+1)
            - np.ndarray: Ground truth graph structure of shape (num_nodes*2, num_nodes*2, tau+1)
    
    Generates contemporaneous and multiple lagged DAG structures, then combines them into
    a single time-series compatible format.
    """
    
    # Generate contemporaneous connections (time t -> time t)
    adjacency_matrix_contemp, ground_true_graph_contemp = generate_dag_two_tails(
        num_nodes, edge_probability=sparsity_contemp, lagged_causal=False
    )

    # Initialize storage for lagged connections (time t-τ -> time t)
    adjacency_matrix_laggeds, ground_true_graph_laggeds = [], []
    
    # Generate τ lagged DAG structures
    for i in range(tau):
        adjacency_matrix_lagged, ground_true_graph_lagged = generate_dag_two_tails(
            num_nodes, edge_probability=sparsity_lag, lagged_causal=True
        )
        adjacency_matrix_laggeds.append(adjacency_matrix_lagged)
        ground_true_graph_laggeds.append(ground_true_graph_lagged)

    # Combine ground truth graphs across time lags
    true_graph = np.zeros(shape=(num_nodes*2, num_nodes*2, tau+1), dtype='<U3')
    true_graph[:, :, 0] = ground_true_graph_contemp[:, :, 0]  # Contemporaneous
    for i in range(tau):  # Lagged connections
        true_graph[:, :, i+1] = ground_true_graph_laggeds[i][:, :, 0]

    # Stack adjacency matrices along third dimension (contemporaneous + lags)
    adjacency_matrix = np.stack([adjacency_matrix_contemp] + adjacency_matrix_laggeds, axis=2)

    return adjacency_matrix, true_graph




def simulation_both_tail_cross_section(N,adjacency_matrix,A=None,switch_probability=None):
    """
    Generates simulated cross-sectional data for both upper and lower tail scenarios.
    
    Args:
        N (int): Number of observational samples
        adjacency_matrix (np.ndarray): Adjacency matrix of the causal graph (shape 2n x 2n) upper triangular matrix and no link between u and l
        switch_probability (np.ndarray): Array of switching probabilities between upper/lower tails for each node
        
    Returns:
        pd.DataFrame: Simulated data matrix with shape (N, num_nodes) containing tail-adjusted observations
    
    Description:
        Constructs upper/lower tail variables through a switching mechanism using:
        1. Base noise simulation
        2. Diagonal switching matrix construction
        3. Structural equation model computation
        4. Softplus transformation for tail separation
    """



    num_nodes = adjacency_matrix.shape[0] // 2

    A=np.diag(np.ones(num_nodes))
    A=np.concatenate([A,A],axis=0)
    if switch_probability is None:
        switch_probability=np.ones(num_nodes)*0.5

    assert is_upper_triangular(adjacency_matrix[:, :])
    assert np.all(adjacency_matrix[np.arange(num_nodes), np.arange(num_nodes) + num_nodes] == 0)


    N_data=simulation(N,num_nodes).T
    K_data=(np.random.uniform(0,1,size=(N,num_nodes))<switch_probability).astype(int)
    K_data_=np.concatenate([K_data,1-K_data],axis=1)

    K_data=np.array([np.diag(k) for k in K_data_])

    # Core computation
    I = np.diag(np.ones(num_nodes * 2))
    KA = K_data @ A
    KB = K_data @ adjacency_matrix
    IKB_inv=np.linalg.inv(I-KB)
    X_data_bar=otimes(IKB_inv @ KA,N_data,False)
    X_data_bar_=transform_softplus(X_data_bar,True)
    X_data=X_data_bar_[:num_nodes,:]-X_data_bar_[num_nodes:,:]

    data_df=pd.DataFrame(X_data.T)
    return data_df


def simulation_both_tail_cross_section_ts(T, adjacency_matrix, switch_probability=0.5, burn_in=0):
    """
    Generates time series cross-sectional data with simultaneous upper/lower tail consideration
    
    Args:
        T (int): Time series length
        adjacency_matrix (np.ndarray): Causal graph adjacency matrix (3D array shape 2n×2n×tau+1)
            Requirements: 
            - First slice must be upper triangular
            - No connections between u and l nodes
        switch_probability (float): Positive  probability (default 0.5)
        burn_in (int): Burn-in period length (default 0)
        
    Returns:
        pd.DataFrame: Simulated data matrix with shape (T, num_nodes)
        
    Methodology:
        Constructs tail-specific time series through switching mechanism:
        1. Generate base noise time series
        2. Build diagonal switching matrices
        3. Solve structural equation model
        4. Apply softplus transformation for tail separation
        5. Handle time lag effects
        6. Trim burn-in period and return final data
    """
    # Check if the first slice of the input adjacency matrix is upper triangular
    tau = adjacency_matrix.shape[2] - 1
    num_nodes = adjacency_matrix.shape[0] // 2
    assert is_upper_triangular(adjacency_matrix[:, :, 0])
    assert np.all(adjacency_matrix[np.arange(num_nodes), np.arange(num_nodes) + num_nodes,0] == 0)
    # Calculate the number of lags, tau

    # Generate simulated data with a time series length of T + tau + burn_in
    A = np.diag(np.ones(num_nodes))
    A = np.concatenate([A, A], axis=0)

    T_plus_tau_plus_burn_in = T + tau + burn_in

    N_data = simulation(T_plus_tau_plus_burn_in, num_nodes).T
    K_data = (np.random.uniform(0, 1, size=(T_plus_tau_plus_burn_in, num_nodes)) < switch_probability).astype(int)
    K_data_ = np.concatenate([K_data, 1 - K_data], axis=1)
    K_data = np.array([np.diag(k) for k in K_data_])

    IC_0_list = []

    results = np.eye(adjacency_matrix.shape[0]) - np.einsum("ijk,kl->ijl", K_data, adjacency_matrix[:, :, 0])
    for i in range(results.shape[0]):
        IC_0_list.append(np.linalg.inv(results[i]))

    IC_0 = np.array(IC_0_list)

    # Initialize the data list X_data_list
    X_data_list = []

    # Add the data for the first tau time points to X_data_list
    for i in range(tau):
        X_data_list.append(transform_softplus(K_data[i] @ A @ N_data[:, i]))

    # Starting from the tau-th time point, compute the result for each time point and add it to X_data_list
    for i in range(tau, T + tau + burn_in):
        # Compute the result for the current time point
        results = otimes(IC_0[i] @ K_data[i] @ A, N_data[:, i], False)
        for j in range(tau):
            # Incorporate the effect of lags
            results = oplus(results, otimes(IC_0[i] @ K_data[i] @ adjacency_matrix[:, :, 1 + j], X_data_list[-j - 1], False), False)
        # Add the result to the data list
        X_data_list.append(results)

    # Convert the final time series data to DataFrame format and return
    data=transform_softplus(np.array(X_data_list[-T:]), True)
    data_df = pd.DataFrame(data[:,:num_nodes]-data[:,num_nodes:])

    return data_df


# Function to expand the data frame by separating upper and lower tails

def expand_data_df(data_df):
    upper_tails = np.where(data_df.values > 0, data_df.values, 0)
    lower_tails = np.where(data_df.values < 0, -data_df.values, 0)
    X_bar = np.concatenate([upper_tails, lower_tails], axis=1)
    data_df_bar=pd.DataFrame(X_bar,columns=[f"{x}.u" for x in data_df.columns]+[f"{x}.l" for x in data_df.columns])
    return data_df_bar


def compute_spectral_radius(adjacency_matrix):
    """
    Calculate the spectral radius of the state transition matrix for given VAR(d) coefficient matrices A_1, A_2, ..., A_d.
    
    Parameters:
    A_list (list of numpy.ndarray): Contains d VAR coefficient matrices of dimension p×p.
    
    Returns:
    spectral_radius (float): The spectral radius of the state transition matrix.
    """

    IC_0 = np.linalg.inv(np.eye(adjacency_matrix.shape[0]) - adjacency_matrix[:, :, 0])
    # Initialize the IC_1 list to store matrices for each lag
    IC_1_lists = []
    tau=adjacency_matrix.shape[2]-1
    for i in range(tau):
        # Compute the matrix for each lag and add it to the list
        IC_1_lists.append(IC_0 @ adjacency_matrix[:, :, 1 + i])

    A_list = IC_1_lists

    p = A_list[0].shape[0]  # Dimension of variables
    d = len(A_list)  # Order of lags

    # Construct the state transition matrix
    top_row = np.hstack(A_list)  # Concatenate A_1, A_2, ..., A_d
    identity_blocks = np.eye(p * (d - 1))  # Generate I_p blocks
    zero_blocks = np.zeros((p * (d - 1), p))  # Generate zero matrix
    bottom_rows = np.hstack([identity_blocks, zero_blocks])  # Combine I_p and zero matrix
    
    # Form the final state transition matrix A_big
    A_big = np.vstack([top_row, bottom_rows])
    # Calculate eigenvalues and take the maximum modulus
    eigvals = np.linalg.eigvals(A_big)
    spectral_radius = max(abs(eigvals))

    return spectral_radius


def simulate_Stieltjes_sparse(num_nodes, sparsity):
    """Generate a sparse Stieltjes matrix for M-matrix simulation
    Args:
        num_nodes (int): Dimension of the square matrix
        sparsity (float): Probability of connection between nodes (0-1)
    Returns:
        np.ndarray: Generated Stieltjes matrix
    """
    # Initialize adjacency matrix
    adj = np.zeros((num_nodes, num_nodes))
    
    # Fill upper triangle with random negative values based on sparsity
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.rand() < sparsity:
                adj[i, j] = -np.random.rand()
                adj[j, i] = adj[i, j]  # Make symmetric
                
    # Calculate spectral radius and set diagonal for M-matrix property
    rho = 0.1+abs(np.linalg.eigvals(-adj)).max()
    adj[np.arange(num_nodes), np.arange(num_nodes)] = rho * 1.2  # Ensure diagonal dominance
    
    return adj


def generate_markov_network(num_nodes, edge_probability=0.3):



    precision_Matrix=simulate_Stieltjes_sparse(num_nodes,edge_probability)

    edge_shape=np.zeros(shape=(num_nodes,num_nodes,1), dtype='<U3')

    edge_shape[:,:,0][precision_Matrix!=0]="o-o"
    edge_shape[:,:,0][np.arange(num_nodes),np.arange(num_nodes)]=""

    return precision_Matrix,edge_shape



def is_positive_definite(matrix):
    """Check if a matrix is positive definite
    Args:
        matrix (np.ndarray): Input matrix to check
    Returns:
        bool: True if positive definite
    """
    # First check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False
    
    # Calculate eigenvalues using efficient Hermitian method
    eigenvalues = np.linalg.eigvalsh(matrix)
    
    # Check all eigenvalues are positive
    return np.all(eigenvalues > 0)


def is_inverse_positive_definite(matrix):
    """Check if matrix inverse is positive definite"""
    return is_positive_definite(np.linalg.inv(matrix))


def is_inverse_positive(matrix):
    """Check if matrix inverse has all non-negative elements"""
    return (np.linalg.inv(matrix) >= 0).all()


''' 
# Simulation test for matrix properties
# Generates 100 random matrices and verifies:
# 1. Matrix is positive definite
# 2. Its inverse is positive definite 
# 3. Its inverse has all non-negative elements
for i in range(100):
    example = simulate_Stieltjes_sparse(100, 0.2)
    assert is_positive_definite(example)
    assert is_inverse_positive_definite(example)
    assert is_inverse_positive(example)
'''
