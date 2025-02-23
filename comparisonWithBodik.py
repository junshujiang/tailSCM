








"""
This file implements a comparison with Bodik's method on time series simulated data. 
It tests different settings and outputs the results as an error bar plot.
Author: Angus
"""






import seaborn as sns
import os
import pickle
from helper_simulation import *
from helper_util import *

import numpy as np
import pandas as pd
from comparison import ComparisonBodik
import numpy as np
from helper_simulation import *
from datetime import datetime
current_date = datetime.now().strftime("%Y%m%d")


if __name__=="__main__":

    ## PARAMS for test
    comparison_number=50
    comparison_nodes=np.array([5,9,15,35,50])
    sparcitys=np.array([0.4,0.2,0.1,0.04,0.03])
    close_contemp=False


    numeberOfLength=5000
    burn_in=1000
    tau=1



    ## PARAMS for this paper
    pc_alpha=0.005
    quantile=1
    ## PARAMS for Bodik


    max_id=get_max("exp_result")
    exp_str=""
    log_path=f"exp_result/{str(max_id)}.{current_date}.ComparisonBodik.log"

    logger=get_logger(log_path)


    logger.info(f"comparison_number: {comparison_number}")
    logger.info(f"comparison_nodes: {np.array2string(comparison_nodes, separator=', ')}")
    logger.info(f"sparcitys: {np.array2string(sparcitys, separator=', ')}")
    logger.info(f"close_contemp: {str(close_contemp)}")
    logger.info(f"numeberOfLength: {numeberOfLength}")
    logger.info(f"burn_in: {burn_in}")
    logger.info(f"tau: {tau}")
    logger.info(f"pc_alpha: {pc_alpha}")
    logger.info(f"quantile: {quantile}")


    results={}
    for config_i, nodes_number in enumerate(comparison_nodes):
        sparcity=sparcitys[config_i]
        logger.info(f"Start test for {nodes_number} nodes, Sparcity {sparcity}")
        results[config_i]={"THIS":[],"BODIK":[]}
        result_this=[]
        result_bodik=[]
        
        test_number=0
        while (test_number<comparison_number):
            logger.info(f"Test {test_number}")
            contemp_sparsity= 0 if close_contemp else sparcity
            adjacency_matrix,true_graph=generate_dag_timeseries(nodes_number,sparcity,contemp_sparsity,tau)
            spectral_radius=compute_spectral_radius(adjacency_matrix)
            logger.info(f"The spectral radius is {spectral_radius}")
            if spectral_radius>1:
                adjacency_matrix=adjacency_matrix/(spectral_radius*1.1)
                spectral_radius=compute_spectral_radius(adjacency_matrix)
                logger.info(f"adjusted spectral radius is {spectral_radius}")

            data_df=simulation_timeseries(numeberOfLength,burn_in,adjacency_matrix)


            resultsThisPaper,_=method_this_paper(data_df,tau_max=tau,tau_min=1 if close_contemp else 0,quantile=quantile,pc_alpha=pc_alpha)
            resultBodik=ComparisonBodik(data_df,tau)

            # 创建一个 [N, N] 的对角线掩码
            error_rate_this_paper,error_edges_this_paper=compare_timeseries_graphs(resultsThisPaper,true_graph,exclude_contemp=True,exclude_self=True)

            error_rate_bodik,error_edges_bodik=compare_timeseries_graphs(resultBodik,true_graph,exclude_contemp=True,exclude_self=True)
            
            result_this.append(error_rate_this_paper)
            result_bodik.append(error_rate_bodik)
            test_number=test_number+1
        results[config_i]["THIS"]=result_this
        results[config_i]["BODIK"]=result_bodik
        

    results_df=[]
    results_ori=[]
    for config_i in results:
        df_tmp_ori=pd.DataFrame({"THIS":[x for x in results[config_i]["THIS"]],
                "BODIK":[x for x in results[config_i]["BODIK"]]})
        results_ori.append(df_tmp_ori)
        mean=df_tmp_ori.mean()
        std=df_tmp_ori.std()
        
        df_tmp=pd.concat([mean,std],axis=0)
        df_tmp.index=["THIS_mean","BODIK_mean","THIS_std","BODIK_std"]
        df_tmp=pd.DataFrame(df_tmp).T    
        results_df.append(df_tmp)
    with open(os.path.join(log_path,f"result_ori.pkl"),"wb") as f:
        pickle.dump(results_ori,f)
    df_result=pd.concat(results_df,axis=0)
    df_result.index=comparison_nodes
    df_result.to_csv(os.path.join(log_path,"BODIKComparison.csv"))
    logger.info(f"Results saved to {os.path.join(log_path,'BODIKComparison.csv')}")




    ## Draw
    results = pd.concat(results_ori)

    df = pd.DataFrame()
    df["values"] = np.concatenate([results.values[:, 0], results.values[:, 1]])
    df["model"] = (['This work'] * results.shape[0] + ['method 2'] * results.shape[0])
    settings = [f"({node}, {sparcity})" for node, sparcity in zip(comparison_nodes, sparcitys)]
    models = []

    for s in settings:
        models.extend([s] * comparison_number)
    df["experiment"] = models * 2

    # Draw grouped boxplot
    custom_palette = ['#1f77b4', 'green']  # First is blue, second is orange
    sns.boxplot(x='experiment', y='values', hue='model', data=df, palette=custom_palette)
    plt.ylim(-0.01, 1)
    # Add title
    # plt.title('Comparison of Two Models Across Experiments')

    plt.legend(title='', loc='upper right', prop={'size': 16})
    # Set x-axis and y-axis labels, supporting LaTeX characters
    plt.xlabel(r'', fontsize=18)  # Set x-axis label
    plt.ylabel(r'', fontsize=18)  # Set y-axis label
    plt.xticks(fontsize=13)  # Set x-axis tick label font size
    plt.yticks(fontsize=13)  # Set y-axis tick label font size
    # Display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "BodikComparison.png"))
