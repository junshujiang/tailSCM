import pandas as pd
from datetime import datetime
from helper_util import *
import seaborn as sns
from helper_simulation import expand_data_df
current_date = datetime.now().strftime("%Y%m%d")    
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


if __name__=="__main__":

    INTERVAL=600
    pc_alpha=0.01
    quantile=1


    max_id=get_max("exp_result")
    exp_str=""
    log_path=f"exp_result/{str(max_id)}.{current_date}.applicationChinaDerivatives_I{INTERVAL}"

    df_price_volume=pd.read_parquet("/home/jianj0c/project/STP/dataset/price_volume.parquet")

    logger=get_logger(log_path)


    logger.info(f"INTERVAL: {INTERVAL}")
    logger.info(f"pc_alpha: {pc_alpha}")
    logger.info(f"quantile: {quantile}")



    # ------------------------LOAD DATA--------------------------------





    data_path=f"/home/jianj0c/project/STP/dataset/price_volume{INTERVAL}.parquet"
    if os.path.exists(data_path):
        df_in_one=pd.read_parquet(data_path)
    else:


        price_series=df_price_volume[[x for x in df_price_volume.columns if x.endswith(".price")]]
        return_df=pd.concat([price_series.pct_change(periods=INTERVAL),df_price_volume[["info.segment_index"]]],axis=1).iloc[::INTERVAL,].copy()
        return_df.loc[~return_df["info.segment_index"].diff().eq(0).fillna(False),[x for x in return_df.columns if not x.startswith("info.")]] = 0
        # remove the return that span two sections

        volumes=df_price_volume[[x for x in df_price_volume.columns if x.endswith(".volume")]+["info.segment_index","info.tradingday"]].copy()
        daily_volumes = volumes.groupby("info.tradingday").apply(
            lambda x: x[[c for c in x.columns if c.endswith(".volume")]].iloc[-1] 
        )



        incremental=volumes[[x for x in volumes.columns if x.endswith(".volume")]].diff(INTERVAL)
        incremental["info.tradingday"]=volumes["info.tradingday"]

        incremental=incremental.iloc[::INTERVAL,].copy()

        results=(
            incremental.groupby("info.tradingday", group_keys=False)  
            .apply(lambda x: x[[col for col in x if col.endswith(".volume")]] 
                .div(daily_volumes.loc[x.name], axis=1)
                ) 
            .reset_index() 
        )


        volume_percentage=results.fillna(0)
        volume_percentage.set_index("raw.timeindex",inplace=True,drop=True)
        volume_percentage["info.segment_index"]=return_df["info.segment_index"]
        volume_percentage.loc[~volume_percentage["info.segment_index"].diff().eq(0).fillna(False),[x for x in volume_percentage.columns if not x.startswith("info.")]] = 0

        df_in_one=pd.concat([volume_percentage[[x for x in volume_percentage.columns if x.endswith(".volume")]],return_df[[x for x in return_df.columns if  x.endswith(".price")]]],axis=1)
        df_in_one.to_parquet(data_path)


    df_in_one=df_in_one.reset_index(drop=True).copy()

    df_in_one.fillna(0,inplace=True)



    # # ------------------------VOLUME GRAPH --------------------------------


    # base_path_volume_pairplot=os.path.join(log_path,"volume_pairplot")
    # if not os.path.exists(base_path_volume_pairplot):
    #     os.mkdir(base_path_volume_pairplot)

    # base_path_volume_graph=os.path.join(log_path,"volume_graph")
    # if not os.path.exists(base_path_volume_graph):
    #     os.mkdir(base_path_volume_graph)


    # for category in categories:
    #     print(category)
    #     asset_list=categories[category]
        
    #     sub_df=df_in_one[[f"{x}_0.volume" for x in asset_list]]
    #     if sub_df.shape[1]==0:
    #         continue
    #     g=sns.pairplot(tranform_frechet_df(sub_df),corner=False)

    #     save_path=os.path.join(base_path_volume_pairplot,f"{category}_volume_pairplot.png")
    #     g.savefig(save_path,dpi=200,  pad_inches=0)

    #     if len(asset_list)==1:
    #         continue
    #     sub_df=df_in_one[[f"{x}_0.volume" for x in asset_list]]

    #     resultsThisPaper,results_tail=method_this_paper(sub_df,pc_alpha=pc_alpha,quantile=quantile,tau_max=1)


    #     save_path=os.path.join(base_path_volume_graph,f"{category}_volume_timeseries_graph.png")
    #     draw_graph_timeseries(vmin_edges=0,vmax_edges=1,show_colorbar=False,edge_shape=resultsThisPaper,save_path=save_path,var_names=sub_df.columns)
    #     save_path=os.path.join(base_path_volume_graph,f"{category}_volume_graph.png")
    #     figsize=(10,10)
    #     draw_graph(arrow_linewidth=3,arrowhead_size=5,label_fontsize=20,figsize=figsize,edge_shape=resultsThisPaper,save_path=save_path,var_names=sub_df.columns)



    # categories_series=dict()
    # for category in categories:
    #     print(category)
    #     asset_list=categories[category]
    #     sub_df=df_in_one[[f"{x}_0.volume" for x in asset_list]]
    #     categories_series[category]=sub_df.mean(axis=1)
        
    # categories_volumes=pd.DataFrame(categories_series)
    # g=sns.pairplot(tranform_frechet_df(categories_volumes),corner=False)
    # save_path=os.path.join(base_path_volume_pairplot,f"sectors_volume_pairplot.png")
    # g.savefig(save_path,dpi=200,  pad_inches=0)



    # resultsThisPaper,results_tail=method_this_paper(categories_volumes,pc_alpha=pc_alpha,quantile=quantile,tau_max=1)
    # save_path=os.path.join(base_path_volume_graph,f"sectors_volume_timeseries_graph.png")
    # draw_graph_timeseries(vmin_edges=0,vmax_edges=1,show_colorbar=False,var_names=categories_volumes.columns,edge_shape=resultsThisPaper,save_path=save_path)
    # save_path=os.path.join(base_path_volume_graph,f"sectors_volume_graph.png")
    # draw_graph(arrow_linewidth=3,arrowhead_size=5,label_fontsize=10,node_size=0.2,figsize=(10,10),var_names=categories_volumes.columns,edge_shape=resultsThisPaper,save_path=save_path)


    # base_path_volume_pairplot=os.path.join(log_path,"volume_pairplot")
    # if not os.path.exists(base_path_volume_pairplot):
    #     os.mkdir(base_path_volume_pairplot)

    # base_path_volume_graph=os.path.join(log_path,"volume_graph")
    # if not os.path.exists(base_path_volume_graph):
    #     os.mkdir(base_path_volume_graph)



    # # ------------------------VOLUME GRAPH (individual products ) --------------------------------

    # all_assets=[]
    # for category in categories:
    #     all_assets.extend(categories[category])

    # asset_list =all_assets
    # sub_df=df_in_one[[f"{x}_0.volume" for x in asset_list]]


    # resultsThisPaper,results_tail=method_this_paper(sub_df,pc_alpha=pc_alpha,quantile=quantile,tau_max=1)

    # edge_number= ((resultsThisPaper[:,:,0]!="").sum()/2+(resultsThisPaper[:,:,1]!="").sum())


    # spacity= ((resultsThisPaper[:,:,0]!="").sum()/2+(resultsThisPaper[:,:,1]!="").sum())/(len(all_assets)**2+len(all_assets)*len(all_assets)-1)
    # logger.info(f"edge number: {edge_number}")
    # logger.info(f"spacity: {spacity}")


    # save_path=os.path.join(base_path_volume_graph,f"all_volume_timeseries_graphp{pc_alpha}_q{quantile}.png")
    # draw_graph_timeseries(vmin_edges=0,vmax_edges=1,node_size=0.02,show_colorbar=False,edge_shape=resultsThisPaper,save_path=save_path,var_names=[x for x in all_assets])
    # save_path=os.path.join(base_path_volume_graph,f"all_volume_graphp{pc_alpha}_q{quantile}.png")
    # figsize=(10,10)
    # draw_graph(arrow_linewidth=3,arrowhead_size=5,node_size=0.04,label_fontsize=25,figsize=figsize,edge_shape=resultsThisPaper,save_path=save_path,var_names=[x for x in all_assets])




    # # ------------------------PRICE GRAPH --------------------------------

    # base_path_price_pairplot=os.path.join(log_path,"price_pairplot")
    # if not os.path.exists(base_path_price_pairplot):
    #     os.mkdir(base_path_price_pairplot)

    # base_path_price_graph=os.path.join(log_path,"price_graph")
    # if not os.path.exists(base_path_price_graph):
    #     os.mkdir(base_path_price_graph)



    # categories_series={}
    # for category in categories:

    #     asset_list=categories[category]
    #     sub_df=df_in_one[[f"{x}_0.price" for x in asset_list]]
    #     g=sns.pairplot(transform_regular_varing_df(sub_df),corner=False)
    #     save_path=os.path.join(base_path_price_pairplot,f"{category}_fut_pairplot.png")
    #     g.savefig(save_path,dpi=200,  pad_inches=0)

    #     data_df_bar=expand_data_df(sub_df)
    #     resultsThisPaper,results_tail=method_this_paper(data_df_bar,pc_alpha=pc_alpha,quantile=quantile,tau_max=1,both_tail_variable=sub_df.shape[1])
    #     var_names=np.array([f"${i}^{{u}}$" for i in asset_list]+[f"${i}^{{l}}$" for i in asset_list])

        
    #     if category=="Chemicals":
    #         figsize=(10,10)
    #     else:
    #         figsize=(5,5)
    #     save_path=os.path.join(base_path_price_graph,f"{category}_fut_timeseries_graph.png")
    #     draw_graph_timeseries(vmin_edges=0,vmax_edges=1,show_colorbar=False,**sort_name_and_edge(resultsThisPaper,var_names),save_path=save_path)
    #     save_path=os.path.join(base_path_price_graph,f"{category}_fut_graph.png")
    #     draw_graph(arrow_linewidth=3,arrowhead_size=5,label_fontsize=20,figsize=figsize,**sort_name_and_edge(resultsThisPaper,var_names),save_path=save_path)




    # categories_series={}
    # for category in categories:
    #     sub_df=df_in_one[[x for x in df_in_one.columns if x.split(".")[0][:-2] in categories[category] and x.endswith(".price")]]
    #     categories_series[category]=sub_df.mean(axis=1)
    # categories_returns=pd.DataFrame(categories_series)
    # g=sns.pairplot(transform_regular_varing_df(categories_returns),corner=False)
    # save_path=os.path.join(base_path_price_pairplot,f"sectors_fut_pairplot.png")
    # g.savefig(save_path,dpi=200,  pad_inches=0)
    # data_df_bar=expand_data_df(categories_returns)
    # resultsThisPaper,results_tail=method_this_paper(data_df_bar,pc_alpha=pc_alpha,quantile=quantile,tau_max=1,both_tail_variable=categories_returns.shape[1])
    # var_names=np.array([f"${i}^{{u}}$" for i in categories_returns.columns]+[f"${i}^{{l}}$" for i in categories_returns.columns])
    # save_path=os.path.join(base_path_price_graph,f"sectors_fut_timeseries_graph.png")
    # draw_graph_timeseries(vmin_edges=0,vmax_edges=1,show_colorbar=False,**sort_name_and_edge(resultsThisPaper,var_names),save_path=save_path)
    # save_path=os.path.join(base_path_price_graph,f"sectors_fut_graph.png")
    # draw_graph(arrow_linewidth=3,arrowhead_size=5,label_fontsize=10,node_size=0.2,figsize=(10,10),**sort_name_and_edge(resultsThisPaper,var_names),save_path=save_path)




    # # ------------------------Hypergraph--------------------------------

    # base_path_hypergraph=os.path.join(log_path,"hypergraph")
    # if not os.path.exists(base_path_hypergraph):
    #     os.mkdir(base_path_hypergraph)


    # for category in categories:
    #     asset_list=categories[category]
    #     sub_df=df_in_one[[f"{x}_0.price" for x in asset_list]]


    #     data_df_bar=expand_data_df(sub_df)

    #     sub_df_vol=df_in_one[[f"{x}_0.volume" for x in asset_list]]

    #     data_df_bar_vol=pd.concat([data_df_bar,sub_df_vol],axis=1)

    #     resultsThisPaper,results_tail=method_this_paper(data_df_bar_vol,pc_alpha=pc_alpha,quantile=quantile,tau_max=1,both_tail_variable=sub_df.shape[1])
    #     var_names=np.array([f"${i}^{{u}}$" for i in asset_list]+[f"${i}^{{l}}$" for i in asset_list]+[f"${i}.vol$" for i in asset_list])


    #     if category=="Chemicals":
    #         figsize=(10,10)
    #     else:
    #         figsize=(5,5)
    #     save_path=os.path.join(base_path_hypergraph,f"{category}_olut_timeseries_graph.png")

    #     draw_graph_timeseries(vmin_edges=0,vmax_edges=1,show_colorbar=False,save_path=save_path,**sort_name_and_edge_price_volume(resultsThisPaper,var_names))
    #     save_path=os.path.join(base_path_hypergraph,f"{category}_olut_graph.png")
    #     draw_graph(arrow_linewidth=3,arrowhead_size=5,node_size=0.2,label_fontsize=15,figsize=figsize,**sort_name_and_edge_price_volume(resultsThisPaper,var_names),save_path=save_path)

    # categories_series={}
    # for category in categories:
    #     asset_list=categories[category]
    #     sub_df=df_in_one[[x for x in df_in_one.columns if x.split(".")[0][:-2] in asset_list and x.endswith(".price")]]
    #     price_series=sub_df.mean(axis=1)

    #     sub_df_vol=df_in_one[[x for x in df_in_one.columns if x.split(".")[0][:-2] in asset_list and x.endswith(".volume")]]
    #     volume_series=sub_df_vol.mean(axis=1)

    #     categories_series[f"{category}.price"]=price_series
    #     categories_series[f"{category}.vol"]=volume_series


    # categories_df=pd.DataFrame(categories_series)



    # categori_ordered=sorted(list(categories.keys()))
    # return_df=categories_df[[f"{x}.price" for x in categori_ordered]]
    # volume_df=categories_df[[f"{x}.vol" for x in categori_ordered]]
    # data_df_bar=expand_data_df(return_df)
    # data_df_bar_vol=pd.concat([data_df_bar,volume_df],axis=1)

    # resultsThisPaper,results_tail=method_this_paper(data_df_bar_vol,pc_alpha=pc_alpha,quantile=quantile,tau_max=1,both_tail_variable=return_df.shape[1])

    # var_names=np.array([f"${i}^{{u}}$" for i in categori_ordered]+[f"${i}^{{l}}$" for i in categori_ordered]+[f"${i}.vol$" for i in categori_ordered])
    # save_path=os.path.join(base_path_hypergraph,f"sectors_price_volume_timeseries_graph.png")

    # draw_graph_timeseries(vmin_edges=0,vmax_edges=1,show_colorbar=False,save_path=save_path,**sort_name_and_edge_price_volume(resultsThisPaper,var_names))

    # save_path=os.path.join(base_path_hypergraph,"sectors_price_volume_graph.png")
    # draw_graph(arrow_linewidth=3,arrowhead_size=5,node_size=0.1,label_fontsize=15,figsize=(15,15),**sort_name_and_edge_price_volume(resultsThisPaper,var_names),save_path=save_path)




    # categori_ordered=sorted(list(categories.keys()))
    # return_df=categories_df[[f"{x}.price" for x in categori_ordered]]
    # volume_df=categories_df[[f"{x}.vol" for x in categori_ordered]]
    # data_df_bar=expand_data_df(return_df)
    # data_df_bar_vol=pd.concat([data_df_bar,volume_df],axis=1)

    # resultsThisPaper,results_tail=method_this_paper(data_df_bar_vol,pc_alpha=pc_alpha,quantile=quantile,tau_max=1,both_tail_variable=return_df.shape[1])

    # var_names=np.array([f"${i}^{{u}}$" for i in categori_ordered]+[f"${i}^{{l}}$" for i in categori_ordered]+[f"${i}.vol$" for i in categori_ordered])
    # save_path=os.path.join(base_path_hypergraph,f"sectors_price_volume_timeseries_graph.png")

    # draw_graph_timeseries(vmin_edges=0,vmax_edges=1,show_colorbar=False,save_path=save_path,**sort_name_and_edge_price_volume(resultsThisPaper,var_names))

    # save_path=os.path.join(base_path_hypergraph,"sectors_price_volume_graph.png")
    # draw_graph(arrow_linewidth=3,arrowhead_size=5,node_size=0.1,label_fontsize=15,figsize=(15,15),**sort_name_and_edge_price_volume(resultsThisPaper,var_names),save_path=save_path)





    # ------------------------WHOLE MARKET--------------------------------


    all_assets=[]
    for category in categories:
        all_assets.extend(categories[category])
    all_assets=sorted(all_assets)
    ## [TODO] check if the sorted all_assets is required.

    len(all_assets)

    sub_df=df_in_one[[f"{x}_0.price" for x in all_assets]]


    data_df_bar=expand_data_df(sub_df)

    sub_df_vol=df_in_one[[f"{x}_0.volume" for x in all_assets]]

    data_df_bar_vol=pd.concat([data_df_bar,sub_df_vol],axis=1)




    resultsThisPaper,results_tail=method_this_paper(data_df_bar_vol,pc_alpha=pc_alpha,quantile=quantile,tau_max=1,both_tail_variable=sub_df.shape[1])
    #l0.01_q1.pkl"
    import pickle
    pickle.dump(resultsThisPaper,open(os.path.join(log_path,"whole_market_resultsThisPaper.pkl"),"wb"))
    pickle.dump(results_tail,open(os.path.join(log_path,f"whole_market_results_detail{pc_alpha}_q{quantile}.pkl"),"wb"))
    pickle.dump(all_assets,open(os.path.join(log_path,"all_assets.pkl"),"wb"))
    edge_number=((resultsThisPaper[:,:,0]!="").sum()/2+(resultsThisPaper[:,:,1]!="").sum())


    possible_edge_number=((len(all_assets)*3)**2)
    print(f"edge number: {edge_number}, possible edge number: {possible_edge_number}, sparsity ratio: {edge_number/possible_edge_number:.2%}")
    logger.info(f"edge number: {edge_number}, possible edge number: {possible_edge_number}, sparsity ratio: {edge_number/possible_edge_number:.2%}")
    var_names=np.array([f"${i}^{{u}}$" for i in all_assets]+[f"${i}^{{l}}$" for i in all_assets]+[f"${i}.vol$" for i in all_assets])



    save_path=os.path.join(log_path,f"all_assets_olut_timeseries_graph{pc_alpha}_q{quantile}.png")

    draw_graph_timeseries(vmin_edges=0,vmax_edges=1,show_colorbar=False,save_path=save_path,**sort_name_and_edge_price_volume(resultsThisPaper,var_names),figsize=(50,50))

    save_path=os.path.join(log_path,f"all_assets_olut_graph{pc_alpha}_q{quantile}.png")
    logger.info(f"save path: {save_path}")

    figsize=(50,50)
    draw_graph(arrow_linewidth=3,arrowhead_size=5,node_size=0.02,label_fontsize=15,figsize=figsize,**sort_name_and_edge_price_volume(resultsThisPaper,var_names),save_path=save_path)