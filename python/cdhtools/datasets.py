from .ADMDatamart import ADMDatamart

def CDHSample(plotting_engine='plotly', query=None):
    path = 'https://raw.githubusercontent.com/pegasystems/cdh-datascientist-tools/master/data'
    models = 'Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip'
    predictors = 'Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip'
    return ADMDatamart(path = path, model_filename=models, predictor_filename=predictors, plotting_engine = plotting_engine, query=query)