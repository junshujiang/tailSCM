import seaborn as sns   
import os
import pickle
from helper_simulation import *
from helper_util import *

import numpy as np
import pandas as pd
from helper_simulation import *
from datetime import datetime
current_date = datetime.now().strftime("%Y%m%d")

## PARAMS for test
comparison_number=50


comparison_nodes=np.array([5,9,15,35,50])
sparcitys=np.array([0.4,0.2,0.1,0.04,0.03])#
numeberOfData=5000

## PARAMS for this paper
pc_alpha=0.005
quantile=1

max_id=get_max("exp_result")
exp_str=""
log_path=f"exp_result/{str(max_id)}.{current_date}.ComparisonBothTailNonTS.log"

logger=get_logger(log_path)

logger.info(f"comparison_number: {comparison_number}")
logger.info(f"comparison_nodes: {comparison_nodes}")
logger.info(f"sparcitys: {sparcitys}")
logger.info(f"numeberOfData: {numeberOfData}")
logger.info(f"pc_alpha: {pc_alpha}")
logger.info(f"quantile: {quantile}")



results={}
for config_i, nodes_number in enumerate(comparison_nodes):
    sparcity=sparcitys[config_i]
    logger.info(f"Start test for {nodes_number} nodes, Sparcity {sparcity}")
    results[config_i]={"THIS":[]}
    result_this=[]

    
    test_number=0
    while (test_number<comparison_number):
        logger.info(f"Test {test_number}")
        adjacency_matrix,edge_shape = generate_dag_two_tails(nodes_number,edge_probability=sparcity)

        data_df=simulation_both_tail_cross_section(numeberOfData,adjacency_matrix)

        data_df_bar=expand_data_df(data_df)

        


        resultsThisPaper,_=method_this_paper(data_df_bar,both_tail_variable=nodes_number,pc_alpha=pc_alpha,quantile=quantile)

        error_rate_this_paper,_=compare_graphs(resultsThisPaper,edge_shape)
        test_number=test_number+1
        result_this.append(error_rate_this_paper)
    results[config_i]["THIS"]=result_this




results_df=[]
results_ori=[]
for config_i in results:
    df_tmp_ori=pd.DataFrame({"THIS":[x for x in results[config_i]["THIS"]]})
    results_ori.append(df_tmp_ori)
    mean=df_tmp_ori.mean()
    std=df_tmp_ori.std()
    df_tmp=pd.concat([mean,std],axis=0)
    df_tmp.index=["THIS_mean","THIS_std"]
    df_tmp=pd.DataFrame(df_tmp).T    
    results_df.append(df_tmp)
with open(os.path.join(log_path,f"result_ori.pkl"),"wb") as f:
    pickle.dump(results_ori,f)
df_result=pd.concat(results_df,axis=0)
df_result.index=comparison_nodes
df_result.to_csv(os.path.join(log_path,"bothTailComparison.csv"))
logger.info(f"Results saved to {os.path.join(log_path,'bothTailComparison.csv')}")




## Draw
results = pd.concat(results_ori)

df = pd.DataFrame()
df["values"] = results.values[:, 0]
df["model"] = (['This work'] * results.shape[0] )
settings = [f"({node}, {sparcity})" for node, sparcity in zip(comparison_nodes, sparcitys)]
models = []

for s in settings:
    models.extend([s] * comparison_number)
df["experiment"] = models 

# Draw grouped boxplot
custom_palette = ['#1f77b4']  # First is blue, second is orange
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
plt.savefig(os.path.join(log_path, "comparisonBothTailNonTS.png"))
