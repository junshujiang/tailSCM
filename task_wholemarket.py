import pandas as pd
from datetime import datetime
from helper_util import *
import seaborn as sns
from helper_simulation import expand_data_df
current_date = datetime.now().strftime("%Y%m%d")    
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


if __name__=="__main__":

    INTERVAL=60
    pc_alpha=0.001
    quantile=0.05


    max_id=get_max("exp_result")
    exp_str=""
    log_path=f"exp_result/{str(max_id)}.{current_date}.applicationChinaDerivatives_I{INTERVAL}"

    df_price_volume=pd.read_parquet("/home/jianj0c/project/STP/dataset/price_volume.parquet")

    logger=get_logger(log_path)


    logger.info(f"INTERVAL: {INTERVAL}")
    logger.info(f"pc_alpha: {pc_alpha}")
    logger.info(f"quantile: {quantile}")


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




    all_assets=[]
    for category in categories:
        all_assets.extend(categories[category])



    len(all_assets)

    sub_df=df_in_one[[x for x in df_in_one.columns if x.split(".")[0][:-2] in all_assets and x.endswith(".price")]]


    data_df_bar=expand_data_df(sub_df)

    sub_df_vol=df_in_one[[x for x in df_in_one.columns if x.split(".")[0][:-2] in all_assets and x.endswith(".volume")]]

    data_df_bar_vol=pd.concat([data_df_bar,sub_df_vol],axis=1)



    resultsThisPaper,results_tail=method_this_paper(data_df_bar_vol,pc_alpha=pc_alpha,quantile=quantile,tau_max=1,both_tail_variable=sub_df.shape[1])

    import pickle
    pickle.dump(resultsThisPaper,open(os.path.join(log_path,"whole_market_resultsThisPaper.pkl"),"wb"))
    pickle.dump(results_tail,open(os.path.join(log_path,"whole_market_results_detail.pkl"),"wb"))
    pickle.dump(all_assets,open(os.path.join(log_path,"all_assets.pkl"),"wb"))


    var_names=np.array([f"${i}^{{u}}$" for i in all_assets]+[f"${i}^{{l}}$" for i in all_assets]+[f"${i}.vol$" for i in all_assets])



    save_path=os.path.join(log_path,f"all_assets_olut_timeseries_graph.png")

    draw_graph_timeseries(vmin_edges=0,vmax_edges=1,show_colorbar=False,save_path=save_path,**sort_name_and_edge_price_volume(resultsThisPaper,var_names),figsize=(50,50))

    save_path=os.path.join(log_path,f"all_assets_olut_graph.png")

    figsize=(50,50)
    draw_graph(arrow_linewidth=3,arrowhead_size=5,node_size=0.02,label_fontsize=15,figsize=figsize,**sort_name_and_edge_price_volume(resultsThisPaper,var_names),save_path=save_path)