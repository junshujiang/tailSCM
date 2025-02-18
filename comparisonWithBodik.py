import os
import pickle
from helper_simulation import *
from helper_util import *
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri,numpy2ri
from rpy2.robjects.conversion import localconverter
import numpy as np
import pandas as pd
from comparison import ComparisonGong, ComparisonBodik
import numpy as np
from helper_simulation import *



if __name__=="__main__":

    ## PARAMS for test
    comparison_number=50
    comparison_nodes=np.array([5,9,15,35,50])
    sparcitys=np.array([0.4,0.2,0.1,0.04,0.03])


    numeberOfData=5000



    ## PARAMS for this paper
    pc_alpha=0.005
    quantile=1
    ## PARAMS for Bodik


    max_id=get_max("exp_result")
    exp_str=""
    log_path=f"exp_result/{str(max_id)}.ComparisonBodik.log"

    logger=get_logger(log_path)
    logger.info(f"Start test for {comparison_nodes} nodes, Sparcity {sparcitys}")

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

            adjacency_matrix,ground_true_graph = generate_dag(nodes_number,edge_probability=sparcity,lagged_causal=True)
            N_data=simulation(numeberOfData+1,nodes_number).T 
            X_data=oplus(otimes(adjacency_matrix,N_data[:,:-1],False),N_data[:,1:],False).T


            resultsThisPaper,_=method_this_paper(pd.DataFrame(X_data),tau_max=1,tau_min=1)
            resultBodik=ComparisonBodik(pd.DataFrame(X_data),1)

            # 创建一个 [N, N] 的对角线掩码
            mask = np.eye(ground_true_graph.shape[0], dtype=bool)

            # 将掩码扩展为 [N, N, k] 的形状
            # 利用广播机制，将掩码沿着第 3 维度（k 维度）扩展
            mask_3d = np.repeat(mask[:, :, np.newaxis], 2, axis=2)

            ground_true_graph_with_selfmask=ground_true_graph.copy()
            ground_true_graph_with_selfmask[mask_3d]=""
            resultsThisPaper_with_selfmask=resultsThisPaper.copy()
            resultsThisPaper_with_selfmask[mask_3d]=""
            resultBodik_with_selfmask=resultBodik.copy()
            resultBodik_with_selfmask[mask_3d]=""
            test_number=test_number+1
            result_this.append(compare_graphs(ground_true_graph_with_selfmask,resultsThisPaper_with_selfmask,lagged=True))
            result_bodik.append(compare_graphs(ground_true_graph_with_selfmask,resultBodik_with_selfmask,lagged=True))
        results[config_i]["THIS"]=result_this
        results[config_i]["BODIK"]=result_bodik
        

    results_df=[]
    results_ori=[]
    for config_i in results:
        df_tmp_ori=pd.DataFrame({"THIS":[x[0] for x in results[config_i]["THIS"]],
                "BODIK":[x[0] for x in results[config_i]["BODIK"]]})
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

