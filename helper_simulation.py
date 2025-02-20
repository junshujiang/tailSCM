import pandas as pd
import numpy as np 
from threadpoolctl import threadpool_limits
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from tigramite.independence_tests import CondIndTest
from matplotlib.colors import LinearSegmentedColormap
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


'''
Author: Angus
Description:
    This function is used to regression (with tranformation) Y based on X. \cite{Transformed-linear prediction for extremes}
'''
def regression(X,Y,quantile=2,enforce_positive=False,unit_frechet=True):
    tpdmXX=estimate_tpdm1(X,quantile=quantile,unit_frechet=unit_frechet)
    tpdmXY=estimate_tpdm1(X,Y,quantile=quantile,unit_frechet=unit_frechet)
    b=np.matmul(np.linalg.inv(tpdmXX),tpdmXY)
    if enforce_positive:
        b[b<0]=0
    return b,tpdmXX,tpdmXY

'''
Author: Angus
Description:
    This function is used to apply a linear transformation to the input matrix X using the transformation matrix b.
    The transformation can be applied in both directions (forward and backward) based on the transform_back parameter.
'''


def linear_transformation(X,b,transform_back=True):
    X_r=transform_softplus(X,True)
    
    result=np.matmul(X_r,b)
    if transform_back:
        return transform_softplus(result)
    else:
        return result
    


# def dag_to_graph_for(adjacency_matrix,edges_dict,ignore_direction=False,granger=False):
#     variable_num=adjacency_matrix.shape[0]

#     if adjacency_matrix.ndim==2:
#         edge_shape=np.zeros(shape=(variable_num,variable_num,1), dtype='<U3')
#     else:
#         edge_shape=np.zeros_like(adjacency_matrix, dtype='<U3')
#     for start_i,end_i in edges_dict.keys(): ##[TODO] 确认edges_dict的key 是不是从0开始的(并且当然要为adjacency_matrix的index)
#         edge_shape[start_i,end_i,0]="-->" if not ignore_direction else "o-o"
#         if not granger:
#             edge_shape[end_i,start_i,0]="<--" if not ignore_direction else "o-o"
#     if granger:
#         edge_shape_2=np.zeros(shape=(variable_num, variable_num, 2), dtype='<U3')
#         edge_shape_2[:,:,[1]]=edge_shape
#         return edge_shape_2
#     else:   
#         return edge_shape

def draw_dag(adjacency_matrix,edges_dict,pathh,nodes):
    variable_num=adjacency_matrix.shape[0]
    edge_value=np.ones(shape=(variable_num,variable_num,2))
    edge_shape=np.zeros(shape=(variable_num,variable_num,2), dtype='<U3')
    link_attribute=np.zeros(shape=(variable_num,variable_num,2),dtype='<U3')
    for start,end in edges_dict.keys():
        start_i=nodes.index(start)
        end_i=nodes.index(end)
        edge_shape[start_i,end_i,0]="-->"
        edge_shape[end_i,start_i,0]="<--"
        link_attribute[start_i,end_i,0]=f"{edges_dict[(start,end)]:.2f}"
        link_attribute[end_i,start_i,0]=link_attribute[start_i,end_i,0]
    #edge_shape[:,:,[0,2]]=""
    #product_name=[x.split(".")[1] for x in var_names]
    G=tp.plot_graph(
        var_names=nodes,
        val_matrix=edge_value,#[userful,][:,userful],
        #label_for_for_edges=link_attribute,
        #link_width = np.abs(edge_value),
        graph=edge_shape,#[userful,][:,userful],
        arrow_linewidth=3,
        arrowhead_size=5,
        #node_pos=position_dict,
        #vmin_edges=0,
        #vmax_edges=1,
        #node_size=50,
        label_fontsize=20,
        #curved_radius=1,
        #link_label_fontsize=10000,
        figsize=(5,5),
        cmap_edges =cm1,
        cmap_nodes =cm2,
        #node_ticks =50,
        link_colorbar_label='TailCorr',
        save_name=pathh,
        show_colorbar=False 
        )
    #plt.tight_layout(pad=0)
    # plt.savefig(os.path.join(log_path,"river_gong.png"),bbox_inches='tight',dpi=200,  pad_inches=0)

    plt.show()
  


'''
Author: Angus
Description:
    This function is used to estimate the tpdm of the data between two group of variables. 

Input: 
    array1: np.array. Should be regular varying at positive infinity
    array2: np.array. Should be regular varying at positive infinity (if None, then array1 is used)
    quantile: int, if equal to 100, then use all data. This is useful if the tail data has been collected outside the function
'''
def estimate_tpdm1(array1:np.array,array2:np.array=None,quantile=10,unit_frechet=True,include_var=False)->np.array:
    if array2 is None:
        array2=array1
        p=array1.shape[1]
        radius=((array1**2).sum(axis=1))**0.5
    else:
        p=array1.shape[1]+array2.shape[1]
        radius=((array1**2).sum(axis=1)+(array2**2).sum(axis=1))**0.5

    if quantile==100:
        useful_index=np.arange(len(array1))
        needed_sample=len(array1)

    else:

        needed_sample=int(len(radius)*(quantile/100))
        useful_index = np.argpartition(radius, -needed_sample)[-needed_sample:]
        useful_index = useful_index[np.argsort(radius[useful_index])]
    filter_r=radius[useful_index]
    
    filter_1_f=array1[useful_index]/(filter_r[:,np.newaxis]+1e-9)
    filter_2_f=array2[useful_index]/(filter_r[:,np.newaxis]+1e-9)

    if unit_frechet:
        m=p
    else:
        m=filter_r[0]**2/len(array1) *needed_sample
    tpdm_=(np.matmul(filter_1_f.T,filter_2_f))/needed_sample  
    tpdm=m*tpdm_
    
    if include_var:
        ex2y2=np.matmul((filter_1_f**2).T,((filter_2_f)**2))/(needed_sample-1)
        exy2=(np.matmul(filter_1_f.T,(filter_2_f))/(needed_sample-1))**2
        tau2=m**2*(ex2y2-exy2)
        return tpdm,tau2
    else:
        return tpdm




def simulation(n,d,alpha=2):
    latents=np.random.uniform(0,1,size=(n,np.eye(d) .shape[0]))
    ferchet=np.power(1-latents,-1/alpha)
    return ferchet





class TailParCorr(CondIndTest):
    r"""Partial correlation test.

    Partial correlation is estimated through linear ordinary least squares (OLS)
    regression and a test for non-zero linear Pearson correlation on the
    residuals.

    Notes
    -----
    To test :math:`X \perp Y | Z`, first :math:`Z` is regressed out from
    :math:`X` and :math:`Y` assuming the  model

    .. math::  X & =  Z \beta_X + \epsilon_{X} \\
        Y & =  Z \beta_Y + \epsilon_{Y}

    using OLS regression. Then the dependency of the residuals is tested with
    the Pearson correlation test.

    .. math::  \rho\left(r_X, r_Y\right)

    For the ``significance='analytic'`` Student's-*t* distribution with
    :math:`T-D_Z-2` degrees of freedom is implemented.

    Parameters
    ----------
    **kwargs :
        Arguments passed on to Parent class CondIndTest.
    """
    # documentation
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self,tail_quantile=5,both_tail=False,variable_num=None,**kwargs):
        self._measure = 'Tailpar_corr'
        self.two_sided = True
        self.residual_based = True
        self.tail_quantile=tail_quantile
        self.both_tail=both_tail
        if self.both_tail:
            self.variable_num=variable_num

        CondIndTest.__init__(self, **kwargs)



    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max'):
        """Perform conditional independence test.

        Calls the dependence measure and signficicance test functions. The child
        classes must specify a function get_dependence_measure and either or
        both functions get_analytic_significance and  get_shuffle_significance.
        If recycle_residuals is True, also _get_single_residuals must be
        available.

        Parameters
        ----------
        X, Y, Z : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index and tau the time lag.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
            How many samples to cutoff at the beginning. The default is
            '2xtau_max', which guarantees that MCI tests are all conducted on
            the same samples. For modeling, 'max_lag_or_tau_max' can be used,
            which uses the maximum of tau_max and the conditions, which is
            useful to compare multiple models on the same sample.  Last,
            'max_lag' uses as much samples as possible.

        Returns
        -------
        val, pval : Tuple of floats
            The test statistic value and the p-value.
        """
        with threadpool_limits(limits=3):
            # Get the array to test on
            array, xyz, XYZ = self._get_array(X, Y, Z, tau_max, cut_off)
            X, Y, Z = XYZ
            if self.both_tail and abs(X[0][0]-Y[0][0])==self.variable_num and X[0][1]==Y[0][1]:
                return 0,1
            # Record the dimensions
            dim, T = array.shape
            # Ensure it is a valid array
            if np.any(np.isnan(array)):
                raise ValueError("nans in the array!")

            combined_hash = self._get_array_hash(array, xyz, XYZ)

            if combined_hash in self.cached_ci_results.keys():
                cached = True
                val, pval = self.cached_ci_results[combined_hash]
            else:
                cached = False
                # Get the dependence measure, reycling residuals if need be
                val = self._get_dependence_measure_recycle(X, Y, Z, xyz, array)
                # Get the p-value
                pval = self.get_significance(val, array, xyz, T, dim)
                self.cached_ci_results[combined_hash] = (val, pval)

            if self.verbosity > 1:
                self._print_cond_ind_results(val=val, pval=pval, cached=cached,
                                            conf=None)
            # Return the value and the pvalue
            

            
            return val, pval


    def get_dependence_measure(self, array, xyz):
        """Return partial correlation.

        Estimated as the Pearson correlation of the residuals of a linear
        OLS regression.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            Partial correlation coefficient.
        """
        if array.shape[0]==2:
            residuals=array.T
        else:
            Y=array[:2,:].T
            X=array[2:,:].T
            b,_,_=regression(X,Y,self.tail_quantile)
            residuals=Y-linear_transformation(X,b) 
        coeff,tau2=estimate_tpdm1(residuals,quantile=self.tail_quantile,unit_frechet=False,include_var=True)
        coeff=coeff[0,1]

            
        
        deg_f = int(array.shape[1]*self.tail_quantile/100) - (array.shape[0])

        if deg_f < 1:
            pval = np.nan

        else:
            t_statistics=coeff/(tau2[0,1]**0.5)*(deg_f**0.5)
            pval = stats.t.sf(abs(t_statistics), deg_f) * 2
        
        # if self.enhance_permutation and pval<self.pc_alpha:
        #     for i in range(self.permutation_number):
                
        self.pval=pval
        
        return coeff
        
        



    def get_analytic_significance(self, value, T, dim):
        """Returns analytic p-value from Student's t-test for the Pearson
        correlation coefficient.

        Assumes two-sided correlation. If the degrees of freedom are less than
        1, numpy.nan is returned.

        Parameters
        ----------
        value : float
            Test statistic value.

        T : int
            Sample length

        dim : int
            Dimensionality, ie, number of features.

        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """
        # Get the number of degrees of freedom

        return self.pval



    def get_model_selection_criterion(self, j, parents, tau_max=0, corrected_aic=False):
        """Returns Akaike's Information criterion modulo constants.

        Fits a linear model of the parents to variable j and returns the
        score. Leave-one-out cross-validation is asymptotically equivalent to
        AIC for ordinary linear regression models. Here used to determine
        optimal hyperparameters in PCMCI, in particular the pc_alpha value.

        Parameters
        ----------
        j : int
            Index of target variable in data array.

        parents : list
            List of form [(0, -1), (3, -2), ...] containing parents.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        Returns:
        score : float
            Model score.
        """

        Y = [(j, 0)]
        X = [(j, 0)]   # dummy variable here
        Z = parents
        array, xyz = self.dataframe.construct_array(X=X, Y=Y, Z=Z,
                                                    tau_max=tau_max,
                                                    mask_type=self.mask_type,
                                                    return_cleaned_xyz=False,
                                                    do_checks=True,
                                                    verbosity=self.verbosity)

        dim, T = array.shape

        y = self._get_single_residuals(array, target_var=1, return_means=False)
        # Get RSS
        rss = (y**2).sum()
        # Number of parameters
        p = dim - 1
        # Get AIC
        if corrected_aic:
            score = T * np.log(rss) + 2. * p + (2.*p**2 + 2.*p)/(T - p - 1)
        else:
            score = T * np.log(rss) + 2. * p
        return score

## the minimum value is 1/shape, max value is 1 ,since F^ is \sum 1<=t
def tranform_emp_cdf(array:np.ndarray,add_1:bool=False):

    sorted=array.argsort(axis=0).argsort(axis=0)
    if add_1:
        return (sorted+1)/(array.shape[0])
    else:
        return (sorted)/(array.shape[0])



def tranform_frechet(array:np.ndarray):
    tmp_array=array.copy()
    ferchet=np.power(1-tranform_emp_cdf(tmp_array),-0.5)-0.9352

    return ferchet


# refers Cooley and Thibaud (2019), a function to transform the data to the (0,\infty) 
def transform_softplus(array:np.ndarray,reverse:bool=False):
    if reverse:
        array_=np.log(np.exp(array)-1)
    else:
        array_=np.log(np.exp(array)+1)
    array_[array>709]=array[array>709] ## avoid overflow
    return array_

def otimes(coeffientMatrix,Data,back=True):
    # coeffientMatrix: p*q row vector is the coefficient of the corresponding variable
    # Data q*n
    coeffientMatrix_=coeffientMatrix
    coeffientMatrix_[coeffientMatrix_<=0]=0

    data_at_origin_scale=transform_softplus(Data, True)
    if coeffientMatrix_.ndim == 3 :
        data= np.einsum('ijk,ki->ji', coeffientMatrix_, data_at_origin_scale)
    else:
        data=coeffientMatrix_@data_at_origin_scale
    if back:
        return data
    else:
        return transform_softplus(data)

def oplus(Data1,Data2,back=True):
    result=transform_softplus(Data1,True)+transform_softplus(Data2,True)
    if back:
        return result
    else:
        return transform_softplus(result)


def A_to_A_star_plus_minus(A):
    A_plus=np.zeros_like(A)
    A_minus=np.zeros_like(A)
    A_plus[A>0]=A[A>0]
    A_minus[A<0]=-A[A<0]
    
    error_flag_pos=(A_plus<=0).any(axis=1)
    error_flag_neg=(A_minus<=0).any(axis=1)
    assert error_flag_pos.any()
    assert error_flag_neg.any()
    
    
    A_plus=A_plus.T
    c_i_plus=(A_plus**(2)).sum(axis=0)

    A_nor=c_i_plus**(-1/2)
    A_star_plus=A_plus*A_nor

    A_minus=A_minus.T
    c_i_minus=(A_minus**(2)).sum(axis=0)

    A_nor=c_i_minus**(-1/2)
    A_star_minus=A_minus*A_nor
    assert not ((A_star_plus>0)  & (A_star_minus>0)).any()
    
    A_star=A_star_plus-A_star_minus
    angular_measure_position=A_star/((A_star**2).sum(axis=1,keepdims=True)**(1/2))
    angular_measure_mass=(A_star**2).sum(axis=1)
    
    
    return A_star,angular_measure_position,angular_measure_mass






# 创建一个从浅灰色到黑色的颜色映射
colors = [(0.8, 0.8, 0.8), (0, 0, 0)]  # 浅灰色 -> 黑色
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'grey_to_black_cmap'
cm1 = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bins)

# 从浅橙色到深红色的颜色映射
colors = [(1, 0.8, 0.6), (1, 0.8, 0.6)]
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'approx_OrRd'
cm2 = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


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
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    if lagged_causal: 
        for i in range(num_nodes):
            for j in range(num_nodes):
                if np.random.rand() < edge_probability:
                    adjacency_matrix[i, j] = np.random.rand()

    else: ## for contemporaneous causal, B is upper triangular to avoid cycle
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < edge_probability:
                    adjacency_matrix[i, j] = np.random.rand()

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

    true_graph=np.concatenate([ground_true_graph_contemp,ground_true_graph_laggeds[0]],axis=2)
    return adjacency_matrix,true_graph


def generate_dag_two_tails(num_nodes,edge_probability):
    

    adjacency_matrix = np.zeros((num_nodes*2,num_nodes*2))
    # 添加有向边并确保无环
    for i in range(2*num_nodes):
        for j in range(i + 1, 2*num_nodes):
            if  np.random.rand() < edge_probability:
                adjacency_matrix[i, j] = np.random.rand()


    edge_shape=np.zeros(shape=(num_nodes*2,num_nodes*2,1), dtype='<U3')

    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] != 0:
                edge_shape[i,j,0]="<--"
                edge_shape[j,i,0]="-->"

    return adjacency_matrix,edge_shape


'''
Author: Angus
Description:
    This function is used to draw a graph based on the provided edge shape matrix. 
    It utilizes the tigramite plotting library's tp.plot_graph function to visualize the graph and can save the plot to a specified path.

Input: 
    edge_shape: np.array, the edge shape matrix representing the direction of edges.
    save_path: str, optional, the path where the graph image will be saved. If None, the graph will be displayed.
    **kwargs: additional keyword arguments for customization of the graph appearance.

Output:
    None. The function either saves the graph to a file or displays it.
'''



def draw_graph(edge_shape,save_path=None,**kwargs):
    edge_value=np.ones_like(edge_shape,dtype=float)
    tp.plot_graph(
        val_matrix=edge_value,
        graph=edge_shape,
        cmap_edges =cm1,
        cmap_nodes =cm2,
        link_colorbar_label='TailCorr',
        save_name=save_path,##
        show_colorbar=False,
        **kwargs
        )
    if save_path is not None:
        plt.savefig(save_path,dpi=200,  pad_inches=0)
    else:
        plt.show()

'''
Author: Angus
Description:
    This function is used to draw a time series graph based on the provided edge shape matrix. 
    It utilizes the tigramite plotting library's tp.plot_time_series_graph function to visualize the graph and can save the plot to a specified path.

Input: 
    edge_shape: np.array, the edge shape matrix representing the direction of edges.
    save_path: str, optional, the path where the graph image will be saved. If None, the graph will be displayed.
    **kwargs: additional keyword arguments for customization of the graph appearance.

Output:
    None. The function either saves the graph to a file or displays it.
'''
def draw_graph_timeseries(edge_shape,save_path=None,**kwargs):

    edge_value=np.ones_like(edge_shape,dtype=float)
    tp.plot_time_series_graph(
        val_matrix=edge_value,
        graph=edge_shape,
        save_name=save_path,##
        **kwargs
        )
    if save_path is not None:
        plt.savefig(save_path,dpi=200,  pad_inches=0)
    else:
        plt.show()


# def draw_graph(edge_shape,save_path=None,var_names=None,position_dict=None):
#     edge_value=np.ones_like(edge_shape,dtype=float)
#     tp.plot_graph(
#         val_matrix=edge_value,#[userful,][:,userful],
#         #link_width = np.abs(edge_value),
#         graph=edge_shape,#[userful,][:,userful],
#         var_names=var_names,
#         #label_fontsize=10,
#         arrow_linewidth=3,
#         arrowhead_size=5,
#         node_pos=position_dict,
#         #vmin_edges=0,
#         #vmax_edges=1,
#         #node_size=50,
#         label_fontsize=20,
#         #curved_radius=1,
#         #link_label_fontsize=10000,
#         figsize=(5,5),
#         cmap_edges =cm1,
#         cmap_nodes =cm2,
#         #node_ticks =50,
#         link_colorbar_label='TailCorr',
#         save_name=save_path,##
#         show_colorbar=False 
#         )
#     #plt.savefig(os.path.join(log_path,f"estimation_river_pc_alpha[rule{rule}].png"),bbox_inches='tight',dpi=200,  pad_inches=0)
#     plt.show()
    
    

'''
Author: Angus
Description:
    This function applies the method described in the paper to the given data.
    It transforms the data using the Frechet transformation, then applies the PCMCI algorithm
    with TailParCorr as the conditional independence test to identify causal relationships.

Input:
    data_df: pd.DataFrame, the input data with variables as columns.
    quantile: float, the quantile level for the TailParCorr test (default is 1).
    tau_max: int, the maximum time lag to consider (default is 0).
    pc_alpha: float, the significance level for the PCMCI algorithm (default is 0.01).
    tau_min: int, the minimum time lag to consider (default is 0).
    both_tail: bool, indicates whether to consider both tails in the TailParCorr test (default is False).
    nodes_number: int, the number of nodes/variables in the data (default is 0).
    contemp_collider_rule: str, the rule for handling contemporaneous colliders in PCMCI (default is "conservative").

Output:
    graph: np.array, the resulting graph from the PCMCI algorithm.
    results_tail: dict, the detailed results from the PCMCI algorithm.
'''


def method_this_paper(data_df,quantile=1,tau_max=0,pc_alpha=0.01,tau_min=0,both_tail=False,nodes_number=0,contemp_collider_rule="conservative"):
    data_df_=data_df.apply(tranform_frechet,axis=0,raw=True)
    dataframeRvier=pp.DataFrame(data_df_.values,var_names=data_df_.columns)
    tailparcorr = TailParCorr(quantile,both_tail=both_tail,variable_num=nodes_number)
    pcmci_parcorr = PCMCI(
        dataframe=dataframeRvier, 
        cond_ind_test=tailparcorr,
        verbosity=0)
    results_tail = pcmci_parcorr.run_pcmciplus(tau_max=tau_max, tau_min=tau_min,pc_alpha=pc_alpha,contemp_collider_rule=contemp_collider_rule)#
    return results_tail["graph"],results_tail




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


def compare_graphs(graph1,graph2,lagged=False):
    
    assert graph1.shape==graph2.shape
    edges1=np.zeros_like(graph1,dtype=int)
    edges2=np.zeros_like(graph2,dtype=int)
    edges1[graph1!=""]=1 
    edges2[graph2!=""]=1
    if lagged:
        mismatch=(edges1!=edges2).sum()
        totalEdges=(edges1.sum()+edges2.sum())
    else:
        mismatch=(edges1!=edges2).sum()/2
        totalEdges=(edges1.sum()+edges2.sum())/2
    
    return mismatch/totalEdges,mismatch



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




def extremeogram(data,quantile=0.99,maxlag=20,start=0):
    """
    Compute the extremeogram.

    Parameters:
    data (ndarray): Input data matrix （n*2).
    quantile (float): Quantile threshold, default is 0.99.
    maxlag (int): Maximum lag value, default is 20.
    start (int): Starting point, default is 0.

    Returns:
    R object: Result of the extremeogram.
    """
    x_r = r.matrix(data, nrow=data.shape[0], ncol=data.shape[1])
    robjects.globalenv['x'] = x_r 
    return robjects.r(f"""
        library(extremogram)
        extremogram2(x,{quantile},{quantile}, {maxlag}, type=1, start = {start})
    """)
