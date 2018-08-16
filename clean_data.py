import argparse
import pandas as pd
import numpy as np

def count_users(df):
    return len(df["user_id"].unique())

def log_users_that(df,s):
    try:
        ids=df.index.get_level_values("user_id").unique()
    except:
        ids=df["user_id"].unique()
    finally:
        print("The following "+str(len(ids))+" users "+s+"\n"+",".join(ids))

def select_negate_intersection(df1,df2):
    return pd.merge(df1,df2,how='outer',on=list(df1.columns),indicator=True).query('_merge != "both"')

def remove_preferences(df1,df2):
    ret=select_negate_intersection(df1,df2)
    ret.drop(["_merge"],axis=1,inplace=True)
    assert(df1.shape[0]-df2.shape[0]==ret.shape[0])
    return ret

def remove_users_that(s,ids,df):
    print("removing "+str(len(ids[ids]))+" users that "+s+"\n"+",".join(list(ids[ids].index)))
    ids=ids.ix[df["user_id"]].fillna(False)
    assert(df.shape[0]==len(ids))
    ret=df.loc[list(~ids)]
    print("Removed "+str(count_users(df)-count_users(ret))+" users, "+str(count_users(ret))+" users remain")
    return ret

def compute_group_measure(df,mode,newname="newcol",fct=lambda x: x["privacy_level"]):
    cols=["user_id","day_no"]+(["round"] if mode=="round_based" else [])
    def build_lists(x):
        y=x.sort_values(["appeared_before","timestamp","QID"])
        return pd.Series({"QID":list(y["QID"]), # repeat QID
                          "timestamp":list(y["timestamp"]),
                          newname:list(fct(y)) # compute cumulative sum
        })
    subset=df.groupby(cols).apply(lambda x: build_lists(x) # apply function to each user at each round
    ).reset_index()
    def stack_cols(df,lst_cols):  # expand a column containing a list
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in df.columns.difference(lst_cols)
        }).assign(**{k:v for k,v in zip(lst_cols,[np.concatenate(df[l].values) for l in lst_cols])})[df.columns.tolist()]
    subset=stack_cols(subset,["QID","timestamp",newname]) # expand two columns
    return pd.merge(df,subset,on=cols+["QID","timestamp"],how="outer") # merge and return a dataframe with an updated column


""" define parser """
parser = argparse.ArgumentParser(description='Clean dataset of Privacy Preferences Experiment.')
parser.add_argument('--data_dirty', metavar='data_dirty', nargs=1,default="data",help='the raw data: UserResponse file')
parser.add_argument('--data_clean', metavar='data_clean', nargs=1,default="data",help='the name of the output file')

args = parser.parse_args()

df=pd.read_csv(args.data_dirty[0])
df.drop(labels=['unique_ID', 'QID_counting', '_id', 'contexts', 'credit_can_be', 'credit_question', #'appeared_before',
                'data_collectors', 'improve', 'privacy_can_be', 'sensors', 'Unnamed: 21', '_acl', '_kmd'],axis=1,inplace=True)
print("Total of "+str(count_users(df))+" users")
## clean up data
# remove day4>
log_users_that(df[df["day_no"]==4],"have a day 4")
df=df[df["day_no"]<4]
# remove users that do not answer every day
df=remove_users_that("do not answer every day",df.groupby(["user_id"])["day_no"].nunique()<3,df)
## remove users whose dataset is damaged (errors in recording)
broken=['5804dd81fc9ba43959b64692','58209bcf53186b2e611de8f2','58209c84a8ae36b04fa5f169','5829d5f1334c6fa53de01ab7','5833133d8c29db400453af38','583313791b0b7435025ed30b','583c4fe6d8fd4c9017032e42','57f2694ba0f31b4069b937bd','5817675238d89e4447b98701','5845861700a5907e7d00a2ab']
df=remove_users_that("have collection errors in the dataset",pd.Series(index=broken,data=True).rename_axis('user_id'),df)
## remove duplicate rows
dups=df.duplicated(subset=["credit_gain","credit","QID","user_id","privacy_level","day_no","timestamp"],keep=False) # mark all dups as true
dups=df[dups]
dups.sort_values(["user_id","day_no","QID","appeared_before"]) # order them by order of appearance
dups=dups[dups.duplicated(subset=["QID","user_id","privacy_level","day_no","timestamp"],keep="first")] # keep the older ones
## add more duplicates manually
f=lambda x,user_id,day_no,QID,appeared,plevel: x.loc[(x["QID"]==QID) &
                                                (x["user_id"]==user_id) &
                                                (x["day_no"]==day_no) &
                                                (x["appeared_before"]==appeared) &
                                                (x["privacy_level"]==plevel)]
dups=dups.append(f(df,'57f269290e08884b6f7733fc',3,11,7,1))
dups=dups.append(f(df,'5804ddcdbaca3fa57e5000a5',2,12,5,2))
dups=dups.append(f(df,'5804ddbe167a759064652ebd',3,56,2,1))
dups=dups.append(f(df,'58176252df9ede672bd5b8ac',2,38,1,3))
dups=dups.append(f(df,'5817675238d89e4447b98701',3,42,1,1))
dups=dups.append(f(df,'58209cc3d25a962a7a83e085',2,44,2,3))
dups=dups.append(f(df,'5829d606c53e0b383f65aec9',2,54,1,1))
dups=dups.append(f(df,'5829d606c53e0b383f65aec9',2,54,2,1)) #the credit depends on this value, but it should be the last one so it should be safe to remove
## remove duplicates from data
df=remove_preferences(df,dups)
### update all other questions with same QID in day and reduce 'appeared_before' by 1
for i,row in dups[["QID","user_id","day_no"]].drop_duplicates().iterrows():
    appearances=dups.loc[(dups["QID"]==row["QID"]) &
                      (dups["user_id"]==row["user_id"]) &
                      (dups["day_no"]==row["day_no"])]["appeared_before"]
    df.loc[(df["QID"]==row["QID"]) &
           (df["user_id"]==row["user_id"]) &
           (df["day_no"]==row["day_no"]),"appeared_before"]=df.loc[
               (df["QID"]==row["QID"]) &
               (df["user_id"]==row["user_id"]) &
               (df["day_no"]==row["day_no"]),"appeared_before"].transform(lambda x:
                                                                          x-sum(appearances<x) if x>appearances.min() else x)
log_users_that(dups,"have duplicated answers")
print("Removing "+str(dups.shape[0])+" duplicated answers")

## debug
asd=compute_group_measure(df,newname="credit2",mode="default",fct= lambda x: x["credit_gain"].cumsum())
tmp=asd.groupby(["user_id","day_no"]).apply(lambda x: abs(x["credit"]-x["credit2"])>0.0001) # where the newly computed credit differs from the old one
dsa=tmp.groupby("user_id").any() # group by user
assert(~any(dsa[dsa]))           # all users are ok

## create column with order of answers, by user and day
df['count']=1
df=compute_group_measure(df,newname="order",mode="default",fct= lambda x: x["count"].cumsum())
df.drop("count",axis=1,inplace=True)

# ## remove users where the plan cost would be negative (sum of credit gain is larger than max_credit_gain)
# tmp=round(lastprefs.groupby(["user_id","day_no"]).agg({"credit_gain":np.sum})["credit_gain"],2)
# log_users_that(tmp[tmp>max_credit_gain],"have credit gains above "+str(max_credit_gain))
# lastprefs=remove_users_that2("have negative plan costs",(tmp<=max_credit_gain).groupby("user_id").agg({"count":all}),lastprefs)

df.to_csv(args.data_clean[0],index=False)
