import os
import numpy as np
import logging
import sys
from tigramite.independence_tests import CondIndTest
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from threadpoolctl import threadpool_limits
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt

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
    






''' 
This function sorts the edge shape and variable names for a graph with two tails.
It rearranges the nodes such that the first half of the nodes are followed by their corresponding second half nodes.

useage:
    draw_graph(arrow_linewidth=3,arrowhead_size=5,label_fontsize=20,figsize=(5,5),**sort_name_and_edge(edge_shape,var_names))
'''
def sort_name_and_edge(edge_shape,var_names):
    num_nodes=edge_shape.shape[0]//2

    orders=[]
    for i in range(num_nodes):
        orders.append(i)
        orders.append(i+num_nodes)
    orders=np.array(orders)
    return {"edge_shape":edge_shape[orders[:, None], orders[None, :], :],"var_names":var_names[orders]}




''' 
This function sorts the edge shape and variable names for a graph with two tails.
It rearranges the nodes such that the first half of the nodes are followed by their corresponding second half nodes.

useage:
    draw_graph(arrow_linewidth=3,arrowhead_size=5,label_fontsize=20,figsize=(5,5),**sort_name_and_edge(edge_shape,var_names))
'''
def sort_name_and_edge_price_volume(edge_shape,var_names):
    num_nodes=edge_shape.shape[0]//3

    orders=[]
    for i in range(num_nodes):
        orders.append(i)
        orders.append(i+2*num_nodes)
        orders.append(i+num_nodes)

    orders=np.array(orders)
    return {"edge_shape":edge_shape[orders[:, None], orders[None, :], :],"var_names":var_names[orders]}


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
        plt.tight_layout(pad=0)
        plt.savefig(save_path,dpi=200,  pad_inches=0)
    else:
        plt.show()



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
        plt.tight_layout(pad=0)
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
    contemp_collider_rule: str, the rule for handling contemporaneous colliders in PCMCI (default is "").

Output:
    graph: np.array, the resulting graph from the PCMCI algorithm.
    results_tail: dict, the detailed results from the PCMCI algorithm.
'''


def method_this_paper(data_df,quantile=1,tau_max=0,pc_alpha=0.01,tau_min=0,both_tail_variable=0,verbosity=0):

    data_df_=data_df.apply(tranform_frechet,axis=0,raw=True)
    dataframeRvier=pp.DataFrame(data_df_.values,var_names=data_df_.columns)
    tailparcorr = TailParCorr(quantile,variable_num=both_tail_variable)
    pcmci_parcorr = PCMCI(
        dataframe=dataframeRvier, 
        cond_ind_test=tailparcorr,
        verbosity=verbosity)
    results_tail = pcmci_parcorr.run_pcmciplus(tau_max=tau_max, tau_min=tau_min,pc_alpha=pc_alpha)#
    return results_tail["graph"],results_tail

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

    def __init__(self,tail_quantile=5,variable_num=0,**kwargs):
        self._measure = 'Tailpar_corr'
        self.two_sided = True
        self.residual_based = True
        self.tail_quantile=tail_quantile
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



            if self.variable_num!=0:
                # avoid link between two tails
                if X[0][0]<2*self.variable_num and Y[0][0]<2*self.variable_num and abs(X[0][0]-Y[0][0])==self.variable_num and X[0][1]==Y[0][1]: 
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
            residuals=transform_softplus(array.T,True)
            coeff,tau2=estimate_tpdm1(residuals,quantile=self.tail_quantile,unit_frechet=True,include_var=True)
            coeff=coeff[0,1]
        else:
            Y=array[:2,:].T
            X=array[2:,:].T
            b,_,_=regression(X,Y,self.tail_quantile)
            residuals=transform_softplus(Y,True)-linear_transformation(X,b,False) 
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




dict_logger=dict()
def get_logger(log_path,log_name="xx",debug=False):
    global dict_logger
    
    if log_path+log_name not in dict_logger:
        # 第一步，创建一个logger
        logger = logging.getLogger(log_path+log_name)
        dict_logger[log_path+log_name]=logger
        logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件


        if log_path=="":
            new_log_path = os.path.abspath(".") + '/Logs/'
        else:
            new_log_path=log_path+"/"
        if not os.path.exists(new_log_path):
            os.mkdir(new_log_path)

        logfile = new_log_path+ log_name+'.log'
        print(logfile)
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s- %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面


        #if debug:
        if 0:
            ##再加上一个输出到终端
            ch=logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        logger.addHandler(fh)
        
    # 日志
    return dict_logger[log_path+log_name]


''' 
Author: Angus
Description:
    辅助函数，确保自增
    给定文件夹
        里面的内容是 1， 2，3， 或者1.jpg 2.jpg。得到最大的+1
'''
def get_max(dirpath):
    array=np.array([int(x.split(".")[0]) for x in os.listdir(dirpath) if not x.startswith(".")])
    if len(array)==0:
        return 1
    else:
        return array.max()+1





'''
The ground truth graph of the Danbube river system. 
    GROUNDTRUTH
    Engelke, S. and Hitz, A. S. (2020). Graphical models for extremes. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(4):871–932.
    Lee, J. and Cooley, D. (2022). Partial tail correlation for extremes. arXiv preprint arXiv:2210.02048.
    Gong, Y., Zhong, P., Opitz, T., and Huser, R. (2024). Partial tail-correlation coefficient applied to extremal-network learning. Technometrics, pages 1–16.

'''


start_x=271
start_y=114
if 1:
    position_dict=dict()
    position_x=dict()
    position_x[1]=[1433,421]
    position_x[2]=[1291,375]
    position_x[3]=[1189,341]
    position_x[4]=[1038,298]
    position_x[5]=[960,320]
    position_x[6]=[864,338]
    position_x[7]=[726,358]
    position_x[8]=[541,383]
    position_x[9]=[441,410]
    position_x[10]=[329,452]
    position_x[11]=[460,662]
    position_x[12]=[411,764]
    position_x[13]=[1351,448]
    position_x[14]=[1217,404]
    position_x[15]=[978,480]
    position_x[16]=[882,585]
    position_x[17]=[839,676]
    position_x[18]=[888,750]
    position_x[19]=[908,831]
    position_x[20]=[602,483]
    position_x[21]=[648,600]
    position_x[22]=[611,720]
    position_x[23]=[946,238]
    position_x[24]=[987,172]
    position_x[25]=[1061,171]
    position_x[26]=[1176,179]
    position_x[27]=[1283,176]
    position_x[28]=[1208,774]
    position_x[29]=[1176,827]
    position_x[30]=[1202,583]
    position_x[31]=[1246,668]
    position_dict={"x":[],"y":[]}
    for i in range(1,32):
        position_dict["x"].append(position_x[i][0])
        position_dict["y"].append(position_x[i][1])
    position_dict["x"]=np.array(position_dict["x"])-start_x
    position_dict["y"]=918-np.array(position_dict["y"])


    link_dict={}
    links=[]
    links.append((27,26))
    links.append((26,25))
    links.append((25,4))
    links.append((4,3))
    links.append((3,2))
    links.append((2,1))
    links.append((24,23))
    links.append((23,4))
    links.append((12,11))
    links.append((11,10))
    links.append((10,9))
    links.append((9,8))
    links.append((8,7))
    links.append((7,6))
    links.append((6,5))
    links.append((5,4))
    links.append((22,21))
    links.append((21,20))
    links.append((20,7))
    links.append((19,18))
    links.append((18,17))
    links.append((17,16))
    links.append((16,15))
    links.append((15,14))
    links.append((14,2))
    links.append((29,28))
    links.append((28,31))
    links.append((31,30))
    links.append((30,13))
    links.append((13,1))
    link_dict["GROUNDTRUTH"]=links




    links_gong=[]
    links_gong.append((12,11))
    links_gong.append((27,26))
    links_gong.append((26,25))
    links_gong.append((24,23))
    links_gong.append((10,9))
    links_gong.append((9,8))
    links_gong.append((8,7))
    links_gong.append((7,6))
    links_gong.append((6,5))
    links_gong.append((5,4))
    links_gong.append((4,3))
    links_gong.append((3,2))
    links_gong.append((2,14))
    links_gong.append((14,15))
    links_gong.append((15,16))
    links_gong.append((16,17))
    links_gong.append((17,18))
    links_gong.append((18,19))
    links_gong.append((29,28))
    links_gong.append((28,31))
    links_gong.append((31,30))
    links_gong.append((30,13))
    links_gong.append((13,1))
    links_gong.append((22,21))
    links_gong.append((21,20))

    link_dict["GONG"]=links_gong

    links_cooley=[]

    links_cooley.append((26,25))
    links_cooley.append((24,23))
    links_cooley.append((12,11))
    links_cooley.append((11,10))
    links_cooley.append((10,8))
    links_cooley.append((8,9))
    links_cooley.append((8,4))
    links_cooley.append((9,6))
    links_cooley.append((6,5))
    links_cooley.append((5,3))
    links_cooley.append((3,2))
    links_cooley.append((2,1))
    links_cooley.append((1,13))
    links_cooley.append((13,4))
    links_cooley.append((22,7))
    links_cooley.append((21,20))
    links_cooley.append((17,19))
    links_cooley.append((16,18))
    links_cooley.append((16,15))
    links_cooley.append((15,14))
    links_cooley.append((29,28))

    link_dict["COOLEY"]=links_cooley

    links_Enge=[]
    links_Enge.append((1,2))
    links_Enge.append((2,3))
    links_Enge.append((3,25))
    links_Enge.append((25,27))
    links_Enge.append((25,26))
    links_Enge.append((26,24))
    links_Enge.append((24,23))
    links_Enge.append((3,4))
    links_Enge.append((4,5))
    links_Enge.append((5,6))
    links_Enge.append((6,7))
    links_Enge.append((7,9))
    links_Enge.append((9,8))
    links_Enge.append((9,10))
    links_Enge.append((10,11))
    links_Enge.append((11,12))
    links_Enge.append((6,20))
    links_Enge.append((20,21))
    links_Enge.append((21,22))
    links_Enge.append((2,14))
    links_Enge.append((14,15))
    links_Enge.append((15,16))
    links_Enge.append((16,19))
    links_Enge.append((19,18))
    links_Enge.append((18,17))
    links_Enge.append((1,13))
    links_Enge.append((13,30))
    links_Enge.append((30,31))
    links_Enge.append((31,28))
    links_Enge.append((28,29))

    link_dict["ENGELKE"]=links_Enge



## the minimum value is 1/shape, max value is 1 ,since F^ is \sum 1<=t
def tranform_emp_cdf(array:np.ndarray,add_1:bool=False):

    sorted=array.argsort(axis=0).argsort(axis=0)
    if add_1:
        return (sorted+1)/(array.shape[0])
    else:
        return (sorted)/(array.shape[0])


def transform_regular_varing_df(df_):
    df=df_.copy()

    return_df=df.apply(transform_rv_whole,axis=0,raw=True)


    return return_df

def tranform_frechet_df(df_):

    df=df_.copy()
    return_df=df.apply(tranform_frechet,axis=0,raw=True)
    return return_df

def transform_rv_whole(X:np.ndarray):   
    """
    Transforms the input array into a Symmetric pareto distribution.
    
    Parameters:
    X (np.ndarray): The input array to be transformed.
    
    Returns:
    np.ndarray: The transformed array where values greater than the median are adjusted by subtracting the square root of 2, 
                and values less than the median are adjusted by adding the square root of 2.
    """
    new=X.copy()
    m=np.median(X)
    new[X>m]=tranform_frechet(X)[X>m]-np.sqrt(2)
    new[X<m]=-tranform_frechet(-X)[X<m]+np.sqrt(2)
    return new



def tranform_frechet(array:np.ndarray):
    """
    Converts the input array into a Frechet distribution.
    
    Parameters:
    array (np.ndarray): The input array to be transformed.
    
    Returns:
    np.ndarray: The transformed array using the empirical cumulative distribution function, 
                adjusted to follow a Frechet distribution by applying a power transformation and a constant shift.
    """
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







def get_return_from_df(df,interval,keep_segment=False,subsampling=True):


    return_dfs=[]
    for seg_id,tmp_df in df.groupby("info.segment_index"):

        tmp_df=tmp_df[[x for x in tmp_df.columns if not x.startswith("info.")]]
        if subsampling:
            sampled_df=tmp_df.iloc[::interval,].copy()
            sampled_df=sampled_df.pct_change()
        else:
            sampled_df=tmp_df.copy()
            sampled_df=sampled_df.pct_change(interval)

        if  keep_segment:
            sampled_df["info.segment_index"]=seg_id

        return_dfs.append(sampled_df)

    return_df=pd.concat(return_dfs,axis=0)
    return return_df




## categories of the futures, based on the report Citic Futures
categories={'oil crops': ['a', 'm', 'OI', 'p', 'b', 'RM',  'y'], #'RS',
 'precious metals': ['ag', 'au'],
 'nonferrous metals': ['al', 'bc', 'cu', 'ni', 'pb', 'sn', 'zn','ao'],
 'economic crops': ['AP', 'CF', 'CJ', 'CY', 'PK', 'SR'],
 'rubber&woods': [ 'br', 'fb', 'nr', 'ru', 'sp'], #"bb"
 'oil&gas': ['bu', 'fu', 'lu', 'pg', 'sc'],
 'grains': ['c', 'cs'], #'JR', 'LR', "PM", "RI", "WH", 'rr'
 'olefins': ['eb', 'l', 'pp', 'v'],
 'alcohols': ['eg', 'MA'],
 'inorganics': ['FG', 'SA', 'UR', 'SH'],
 'ferrous metals': ['hc', 'i', 'rb', 'SF', 'SM', 'ss'],#'wr'
 'equity index': ['IC', 'IF', 'IH', 'IM'],
 'coals': ['j', 'jm'], #"ZC"
 'animals': ['jd', 'lh'],
 'novel materials': ['lc', 'si'],
 'aromatics': ['PF', 'TA', 'PX'],
 'interest rates': ['T', 'TF', 'TL', 'TS'],
 'indices': ['ec']}

 ## lg and ps are not included neither (not available since the start data)




 