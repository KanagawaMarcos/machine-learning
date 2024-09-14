#r "nuget: Microsoft.ML"

open Microsoft.ML
open Microsoft.ML.Data

[<CLIMutable>]
type SentimentIssue = {
    [<LoadColumn(0)>] Sentiment: int
    [<LoadColumn(1)>] SentimentText: string
    [<LoadColumn(2)>] LoggedIn: bool
}

// Load dataset
let mlContext = MLContext()
let dataPath = "wikipedia-detox-250-line-data.tsv"
let testPath = "wikipedia-detox-250-line-test.tsv"
let dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(dataPath, hasHeader=true)
let testDataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(testPath, hasHeader=true)
let trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction = 0.2)