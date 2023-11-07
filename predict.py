from sklearn.feature_extraction import DictVectorizer
from flask import Flask,jsonify,request
import xgboost as xgb
import pickle
import ast

app=Flask("credict_risk_app")

with open("model.pickle","rb") as f:
    model=pickle.load(f)

with open("dict_vectorizer.pickle","rb") as f:
    dv=pickle.load(f)

# dv=DictVectorizer(sparse=False)

@app.route("/predict",methods=["GET","POST"])
def predict():
    data=request.args.get('seed')
    data=data.strip('"')
    data=ast.literal_eval(data)
    x_test=dv.transform(data)
    feature_names=model.feature_names
    dval=xgb.DMatrix(x_test,feature_names=feature_names)
    y_pred=model.predict(dval)
    res=0
    if y_pred[0]>0.5:
        res=1
    return jsonify({"result":int(res)})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8790)

    

