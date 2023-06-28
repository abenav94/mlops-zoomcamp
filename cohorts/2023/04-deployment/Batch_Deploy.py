import pickle
import pandas as pd
import numpy as np
import sys

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv,model

def get_pred(df,dv,model):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
    '/content/drive/MyDrive/MLOPs Zoomcamp/Data/' + output_file,
    engine='pyarrow',
    compression=None,
    index=False)

    df_result.to_parquet(output_file, index=False)

def run():
    year = int(sys.argv[0])
    month = int(sys.argv[1])
    input_file = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet'
    output_file = f'{year:04d}-{month:02d}.parquet'
    model_path = '/content/drive/MyDrive/MLOPs Zoomcamp/Data/model.bin'
    df = read_data(input_file)
    dv,model = load_model(model_path)
    y_pred = get_pred(df,dv,model)
    save_results(df, y_pred, output_file)

if __name__ == '__main__':
    run()