from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import List
from sklearn import preprocessing


app=FastAPI()


class scoringItem(BaseModel):
    age: int
    famsize:int	
    Fedu:int	
    reason:int	
    guardian:int	
    traveltime:int	
    studytime:int	
    failures:int	
    famrel:int	
    freetime:int		
    Walc:int	
    health:int
    G1:int	
    G2:int
    gender:int	
    absences_skew_r:float	
    address_encoded:int	
    schoolsup_encoded:int	
    famsup_encoded:int	
    paid_encoded:int	
    activities_encoded:int	
    nursery_encoded:int	
    higher_encoded:int	
    internet_encoded:int	
    romantic_encoded:int

with open("Sperformance.sav", "rb") as f:
    model = pickle.load(f)

@app.post("/")

# async def scoring_endpoint(item: List[scoringItem]):
#     # print(item)
#     # d = list(map(list, item))
#     print(d)
#     # df=pd.DataFrame([item.dict().values()],columns=item.dict().keys())
#     # yhat=model.predict(df)
#     # return {"prediction":float(yhat)}
async def scoring_endpoint(item: List[scoringItem]):
    df = pd.DataFrame([t.__dict__ for t in item])
    sc = preprocessing.StandardScaler()
    x_train = sc.fit_transform(df)

    # l = df.values.tolist()
    print(df)

    # l = list(map(dict,item))
    # data = []
    # for i in l:
    #     data.append([i[j] for j in i])
    # # print(data)
    # df = pd.DataFrame(data,columns=item[0].dict().keys())
    yhat=model.predict(x_train)
    print(type(yhat))
    return (yhat.tolist())

    # print(item)
    