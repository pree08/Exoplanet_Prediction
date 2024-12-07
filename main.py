import io
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile , HTTPException , Response,Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import StandardScaler

classifier = joblib.load('/home/kubuntu/Desktop/my_pc/Exoplanet_Prediction/xgb.joblib')

template = Jinja2Templates(directory="templates")

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def index(req: Request):
    return template.TemplateResponse(
        name="index.html",
        context = {"request":req}
    )



@app.post("/predict-body")
async def predict_body(file: UploadFile = File(...)):
   
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    sc = StandardScaler()
    scaled = sc.fit_transform(df)
    final_df = pd.DataFrame(scaled,columns=df.columns)
   

    predictions = classifier.predict(final_df)

    df['predictions'] = np.where(predictions == 1, 'confirmed', 'falsepositive')
    df['percentage'] = ''  # Add an empty column named 'percentage'

    output = io.StringIO()
    df.to_csv(output,index=False)
    output.seek(0)

    return Response(content=output.getvalue(),media_type="text/csv",headers={"Content-Disposition": "attachment; filename=predictions.csv"})


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='127.0.0.1')





    
