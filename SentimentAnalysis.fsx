#r "nuget: Microsoft.ML, 1.3.1"

open Microsoft.ML
open Microsoft.ML.Data

[<CLIMutable>]
type SentimentIssue = {
    [<LoadColumn(0)>] Sentiment: int
    [<LoadColumn(1)>] SentimentText: string
    [<LoadColumn(2)>] LoggedIn: bool
}

let mlContext = MLContext()
let dataPath = "wikipedia-detox-250-line-data.tsv"
let testPath = "wikipedia-detox-250-line-test.tsv"
let dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(dataPath, hasHeader=true)
let testDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(testPath, hasHeader=true)
let dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Sentiment", "SentimentText")

let trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName = "Sentiment", featureColumnName = "SentimentText")
let trainingPipeline = dataProcessPipeline.Append(trainer)
let trainedModel = trainingPipeline.Fit(dataView)
