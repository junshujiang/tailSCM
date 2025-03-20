




"""
This file implements a comparison with Gong's method on non-time series simulated data. 
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
import numpy as np
from helper_simulation import *



## PARAMS for test
comparison_number=50

comparison_nodes=np.array([5,9,15,35,50])
sparcitys=np.array([0.4,0.2,0.1,0.04,0.03])
numeberOfData=5000





## PARAMS for this paper
pc_alpha=0.005
quantile=1
## PARAMS for Gongs


max_id=get_max("exp_result")
exp_str=""
log_path=f"exp_result/{str(max_id)}.ComparisonWithMyself.log"

logger=get_logger(log_path)


results={}
for config_i, nodes_number in enumerate(comparison_nodes):
    sparcity=sparcitys[config_i]
    logger.info(f"Start test for {nodes_number} nodes, Sparcity {sparcity}")
    results[config_i]={"THIS_DIRECTION":[],"THIS_WITHOUT_DIRECTION":[]}
    result_this_direction=[]
    result_this_without_direction=[]
    
    test_number=0
    while (test_number<comparison_number):
        logger.info(f"Test {test_number}")
        adjacency_matrix,ground_true_graph = generate_dag(nodes_number,edge_probability=sparcity)
        IC_1=np.linalg.inv(np.eye(adjacency_matrix.shape[0])-adjacency_matrix)
        N_data=simulation(numeberOfData,nodes_number).T
        X_data=otimes(IC_1,N_data,False)
        data_df=pd.DataFrame(X_data.T)
        resultsthis_paper,_=method_this_paper(data_df,quantile=quantile,pc_alpha=pc_alpha,tau_max=0)
        test_number=test_number+1
        result_this_without_direction.append(compare_graphs(ground_true_graph,resultsthis_paper,True))
        result_this_direction.append(compare_graphs(ground_true_graph,resultsthis_paper,False))
    results[config_i]["THIS_WITHOUT_DIRECTION"]=result_this_without_direction
    results[config_i]["THIS_DIRECTION"]=result_this_direction
    
    
results_df=[]
results_ori=[]
for config_i in results:
    df_tmp_ori=pd.DataFrame({"THIS_WITHOUT_DIRECTION":[x[0] for x in results[config_i]["THIS_WITHOUT_DIRECTION"]],
            "THIS_DIRECTION":[x[0] for x in results[config_i]["THIS_DIRECTION"]]})
    results_ori.append(df_tmp_ori)
    mean=df_tmp_ori.mean()
    std=df_tmp_ori.std()
    df_tmp=pd.concat([mean,std],axis=0)
    df_tmp.index=["THIS_WITHOUT_DIRECTION_mean","THIS_WITHOUT_DIRECTION_std","THIS_DIRECTION_mean","THIS_DIRECTION_std"]
    df_tmp=pd.DataFrame(df_tmp).T    
    results_df.append(df_tmp)
with open(os.path.join(log_path,f"result_ori.pkl"),"wb") as f:
    pickle.dump(results_ori,f)
df_result=pd.concat(results_df,axis=0)
df_result.index=comparison_nodes
df_result.to_csv(os.path.join(log_path,"THISComparison.csv"))
logger.info(f"Results saved to {os.path.join(log_path,'THISComparison.csv')}")



## Draw
results = pd.concat(results_ori)

df = pd.DataFrame()
df["values"] = np.concatenate([results.values[:, 0], results.values[:, 1]])
df["model"] = (['$NED$'] * results.shape[0] + ['$UNED$'] * results.shape[0])
settings = [f"({node}, {sparcity})" for node, sparcity in zip(comparison_nodes, sparcitys)]
models = []

for s in settings:
    models.extend([s] * comparison_number)
df["experiment"] = models * 2

# Draw grouped boxplot
custom_palette = ['#1f77b4', 'orange']  # First is blue, second is orange
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
plt.savefig(os.path.join(log_path, "THISComparison.png"))

