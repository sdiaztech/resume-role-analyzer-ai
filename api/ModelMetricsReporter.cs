using System.Text.Json;

internal static class ModelMetricsReporter
{
    public static void Log(ILogger logger, string contentRoot)
    {
        var modelsPath = Path.Combine(ProjectPaths.RepositoryRoot(contentRoot), "models");
        LogRankingMetrics(logger, Path.Combine(modelsPath, "career_corpus_metrics.json"));
        LogClassifierMetrics(logger, Path.Combine(modelsPath, "metrics.json"));
    }

    private static void LogRankingMetrics(ILogger logger, string path)
    {
        if (!File.Exists(path))
        {
            logger.LogWarning("Career recommendation benchmark not found at {MetricsPath}", path);
            return;
        }

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            var overall = document.RootElement.GetProperty("overall");
            logger.LogInformation(
                "Career recommendation benchmark ({Samples} resumes): " +
                "Top-1 {Top1:P1} | Top-5 {Top5:P1} | Median expected-role rank {MedianRank}. " +
                "These are ranking metrics, not hiring probability.",
                overall.GetProperty("samples").GetInt32(),
                overall.GetProperty("hit_at_1").GetDouble(),
                overall.GetProperty("hit_at_5").GetDouble(),
                overall.GetProperty("median_rank").GetInt32());
        }
        catch (JsonException error)
        {
            logger.LogWarning(error, "Could not read career recommendation benchmark metrics");
        }
    }

    private static void LogClassifierMetrics(ILogger logger, string path)
    {
        if (!File.Exists(path)) return;
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            var test = document.RootElement.GetProperty("test");
            logger.LogInformation(
                "Pair-classifier test metrics: accuracy {Accuracy:P1} | ROC-AUC {RocAuc:P1} | " +
                "average precision {AveragePrecision:P1}. Pair accuracy does not equal " +
                "Top-1 career accuracy.",
                test.GetProperty("accuracy").GetDouble(),
                test.GetProperty("roc_auc").GetDouble(),
                test.GetProperty("average_precision").GetDouble());
        }
        catch (JsonException error)
        {
            logger.LogWarning(error, "Could not read pair-classifier metrics");
        }
    }
}
