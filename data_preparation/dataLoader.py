# %%
# this code is unrunnable in reader's environment as it requires the data/software from the server. 
# The code is only for the purpose of documentation and to reproduce the dataset for authors' own sake.


from xvector import Database
from xvector import Cacher
from xvector.dataLoader import ChinaDerivateDataLoader
from xvector.util_helper import Helper_researcher, Helper_nameparser
from xvector.util_helper import Helper_nameparser
from xvector.const_xvector import *
import py_vollib_vectorized
import traceback
import numpy as np 
import pandas as pd
import pickle
import traceback
import os
import time 
import io
import contextlib
import pandas as pd
pd.set_option('display.max_rows', None)

if __name__=="__main__":

    # %% [markdown]
    # - Start from 20240101 to 20250113.
    # - fetch all options daily data 
    # - it takes 3 hours to get the data.
    # %%
    start_date,end_date=20240102,20250113#20230801


    db=Database(json_path="../Config/db.json")


    tablename=db.prefix[PREFIX_FOR_GENERAL]+"Daily"

    sql=f"""
        SELECT * from {tablename} WHERE trading_day between {start_date} and {end_date}; 
    """

    log_str,logger=Helper_researcher.get_exp_logger(exp_name=f".OptionDataLoader")
    db.cur.execute(sql)
    results=db.cur.fetchall()
    logger.info(f"Get {len(results)} records from {tablename} between {start_date} and {end_date}")



    options_daily=[x for x in results if Helper_nameparser.parse_option_code(x["instrument_id"])!=None]

    logger.info(f"Get {len(options_daily)} options records from {tablename} between {start_date} and {end_date}")


    # %% [markdown]
    # get the products that have options since the start date. At the start date, there are 39 products that have options.

    # %%
    option_number={}
    for product in db.get_product():
        option_number[product]=len(db.get_instrument_list_v1(start_date,product,True))

    options_ins_number_at_start_date=pd.Series(option_number).to_frame().sort_values(0,ascending=False)
    products=options_ins_number_at_start_date[options_ins_number_at_start_date.values>0].index.tolist()

    logger.info(f"Get {len(products)} products that have options since {start_date}")
    logger.info("\n"+options_ins_number_at_start_date[options_ins_number_at_start_date.values>0].to_string())

    # %% [markdown]
    # Aggregation informantions:
    # - the information from the code:  a2403-C-4150
    # - obtain all underlying futures information from the code

    # %%

    [options_daily[i].update(Helper_nameparser.parse_option_code(options_daily[i]["instrument_id"])) for i in range(len(options_daily))] 


    for i in range(len(options_daily)):
        options_daily[i]["future"]=f"""{options_daily[i][PRODUCT_NAME]}{options_daily[i][MATURITY]}""" 
        
        
        
    unique_instrument=list(set([x["future"] for x in options_daily]))



    instrument_info={}
    for ins in unique_instrument:
        instrument_info[ins]=db.get_instrument(ins)
        
        
        
    for i in range(len(options_daily)):
        if options_daily[i][PRODUCT_NAME] in ["HO","MO","IO"]: ## Not include the financial option
            continue
        options_daily[i].update(instrument_info[options_daily[i]["future"]])
        

    # %% [markdown]
    # For each product (with options), for each trading day, find the most active option.

    # %%


    most_active_options={}
    for product in products:
        try:
            name_of_options=[]
            data_sub=[x for x in options_daily if x[PRODUCT_NAME]==product]
            for trading_day in db.range(start_date,end_date):
                data_sub_sub=[x for x in data_sub if str(x["trading_day"])==trading_day ]


                if len(data_sub_sub)==0:
                    name_of_options.append({"trading_day":trading_day})
                    continue

                turnovers=np.array([x["turnover"] for x in data_sub_sub])
                max_index=np.array(turnovers).argmax()
                max_turnover=turnovers[max_index]
                max_daily=data_sub_sub[max_index]
                name_of_options.append(max_daily)
            
            daily_product_df=pd.DataFrame(name_of_options)

            daily_product_df["trading_day"]=daily_product_df["trading_day"].astype(int).astype(str)
            daily_product_df.set_index("trading_day",drop=True)["turnover"]
            most_active_options[product]=daily_product_df
        except Exception as e:
            logger.error(str(e))
            logger.error(traceback.print_exc())
    cacher=Cacher("/quant/stp/cacher")
    memory_id=cacher.helper_BigRandomInt()
    cacher.save(most_active_options,memory_id,"pickle")
    logger.info(f"save daily info for most liqudate option into memory {memory_id}" )

    open(os.path.join(log_str,"daily.pkl"),"wb").write(pickle.dumps(most_active_options,protocol=4))




    # %%
    most_active_options=pickle.loads(open(os.path.join(log_str,"daily.pkl"),"rb").read())

    # %% [markdown]
    # Load the high frequency data for both the option and the underlying futures.

    # %%

    data_loader=ChinaDerivateDataLoader(logger)


    data_loader.db=Database(json_path="../Config/db.json")
    start_mask=10
    end_mask=20
    close_tick=10
    force_close= FORCE_CLOSE_ON_DAY 
    for key in most_active_options:
        most_active_options[key]=most_active_options[key].set_index("trading_day",drop=True)
        most_active_options[key]=most_active_options[key].add_prefix(f"{key}_")
    mata_info=pd.concat(list(most_active_options.values()),axis=1)
    opt_ins_tables=mata_info[[f"{key}_instrument_id" for key in most_active_options]]
    opt_ins_tables.columns=[x.split("_")[0] for x in opt_ins_tables.columns]

    ## opt_ins_tables store the instrument_id of the option and the underlying futures.

    # %%
    futures_ins_tables=mata_info[[f"{key}_future" for key in most_active_options]].copy()
    futures_ins_tables.columns=[x.split("_")[0] for x in futures_ins_tables.columns]

    mata_info.to_csv(os.path.join(log_str,"meta_info.csv"))


    # %% [markdown]
    # Load option high-frequency data based on the instrument_id table

    # %%



    def load_data(table,is_option):
        mainforce_table=table
        mainid_data_dict=dict()
        for ins_main_id in mainforce_table.columns:
            product=Helper_nameparser.get_product_out_of_ins(ins_main_id)
            sessions=data_loader.db.get_product(product= product)
            files=[]
            for date in mainforce_table.index:
                ins=mainforce_table.loc[date,ins_main_id]
                if type(ins)!=str: ## if no ins, 
                    continue
                pathh=data_loader.db.get_fullpath(date,ins,market_type="MARKET_MAINLAND_DERIVATIVE",is_option=is_option)
                                
                if not os.path.exists(pathh):
                    continue
                files.append((pathh,sessions,date))
            logger.info(f"Get {len(files)} files for {ins_main_id}")
            datas=[]
            for file in files:
                float32_cols = {c: np.float32 for c in ['last_price', 'bid_price1',
                        'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5', 'ask_price1',
                        'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5']}
                df=pd.read_csv(file[0],sep="\t",parse_dates=["timeindex"],index_col="timeindex",dtype=float32_cols)
                datas.append(df)
            try:
                data=pd.concat(datas)
            except Exception as e:
                logger.error(str(e))
                logger.error(traceback.print_exc())
                print(ins_main_id)
                continue
            mainid_data_dict[ins_main_id]=data




        db=data_loader.db
        dates = db.range(table.index[0],table.index[-1],include_right=True)


        index_dict=ChinaDerivateDataLoader.get_index("09:00,10:15|10:30,11:30|13:30,15:00",dates,start_mask,end_mask,close_tick,force_close,False,db.config)


        stand_index=index_dict[STANDARD_INDEX]
        trading_date=index_dict[TRADING_DAY_MASK]
        segment_id=index_dict[SEGMENT_MASK]



        big_table=pd.DataFrame(index=stand_index)
        big_table.index.name = 'timeindex'
        for main_id in mainid_data_dict:
            df_for_ins=mainid_data_dict[main_id]
            df_for_ins=df_for_ins.add_prefix("raw.{ins}.".format(ins=main_id))
            df_for_ins=df_for_ins.reset_index()
            df_for_ins=df_for_ins.set_index("timeindex",drop=True)
            df_for_ins=df_for_ins[~df_for_ins.index.duplicated(keep='first')]
            df_for_ins.index = pd.to_datetime(df_for_ins.index)
            
            exchange_id=db.get_product(Helper_nameparser.get_product_out_of_ins(main_id))["exchange"]
            #if exchange_id in ['CZCE','DCE','GFEX','CFFEX']:
            df_for_ins=ChinaDerivateDataLoader._standardize_index(df_for_ins,exchange_id,False)
            big_table=big_table.join(df_for_ins) ## 到此处19分钟。
            
        data_loader.write_log("19minutes",LOG_INFO)


        big_table["info.segment_index"]=segment_id
        big_table["info.tradingday"]=trading_date
        big_table["info.segment_index"]=big_table["info.segment_index"].astype(int)
        big_table["raw.timeindex"]=big_table.index


            
        midPrice={}
        for product in mainid_data_dict:
            midPrice[f"{product}.opt" if is_option else f"{product}.fut"]=(big_table[f"raw.{product}.ask_price1"]+big_table[f"raw.{product}.bid_price1"])/2
        prices_series=pd.DataFrame(midPrice)
        info_columns=big_table.columns[big_table.columns.str.startswith("info")].tolist()
        prices_series[info_columns]=big_table[info_columns]
        prices_series=data_loader._padding(prices_series)
        return prices_series



    # %%
    prices_series_opt=load_data(opt_ins_tables,True)
    prices_series_futures=load_data(futures_ins_tables,False)


    # %%
    opt_fut_prices=pd.concat([prices_series_opt.iloc[:,:-2],prices_series_futures],axis=1)
    opt_fut_prices["info.tradingday"]=opt_fut_prices["info.tradingday"].astype(str)
    opt_fut_prices.to_parquet(os.path.join(log_str,"opt_fut_prices.parquet"))


    # %% [markdown]
    # calculate the implied volatility


    # %%


    # py_vollib.black_scholes_merton.implied_volatility.implied_volatility(price, S, K, t, r, flag, q=0, return_as='numpy')

    # py_vollib_vectorized.vectorized_implied_volatility(price, S, K, t, r, flag, q=0, model='black_scholes_merton',return_as='numpy')  # equivalent

    # products=[x.split("_")[0] for x in mata_info.columns if x.endswith('future')]






    implied_volatility={}

    risk_free_rate=0.02 # as reference in  https://www.ceicdata.com/en/indicator/china/short-term-interest-rate


    for product in products:
        try:

            price = opt_fut_prices[f"{product}.opt"].values

            ## 求行权价

            tmp_df = mata_info[[f"{product}_STRIKE_PRICE",f"{product}_ExpireDate",f"{product}_OPTION_TYPE"]].reset_index()
            tmp_df.columns = ['info.tradingday', 'strike_price','expire_price','option_type']
            future_data = opt_fut_prices[[f"{product}.fut",'info.tradingday']].merge(tmp_df, on='info.tradingday', how='left')
            future_data.index=future_data.index

            S=future_data[f"{product}.fut"].values
            S[S<=0]=np.nan
            K=future_data["strike_price"].values.astype(float)  
            K[K<=0]=np.nan


            expire_date=future_data["expire_price"]
            t=(pd.to_datetime( expire_date, format='%Y%m%d')-pd.to_datetime(future_data["info.tradingday"], format='%Y%m%d') ).dt.days.values/365.25 

            flag=future_data["option_type"].str.lower().values.copy()
            r = np.array([risk_free_rate]*len(S))


            wrong_index_flag = ~np.isin(flag, ['c', 'p'])
            flag[wrong_index_flag]="c"
            with io.StringIO() as f, io.StringIO() as ferr:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(ferr):
                    iv_s=py_vollib_vectorized.vectorized_implied_volatility(price, S, K, t, r, flag, q=0, return_as='numpy',model='black_scholes_merton')
                    iv_s[wrong_index_flag]=np.nan
                    implied_volatility[product]=pd.Series(iv_s,index=opt_fut_prices.index,name=f"{product}.iv")
        except Exception as e:
            print(product)
            print(e)
            print(traceback.format_exc())

    # %% [markdown]
    # Save the aggregated data

    iv_df=pd.concat(list(implied_volatility.values()),axis=1)
    opt_fut_prices[iv_df.columns]=iv_df
    columns=[x for x in opt_fut_prices.columns if x not in ['info.segment_index', 'info.tradingday'] ] + ['info.segment_index', 'info.tradingday']
    opt_fut_prices=opt_fut_prices[columns]

    opt_fut_prices.to_parquet(os.path.join(log_str,"opt_fut_iv_prices.parquet"))




