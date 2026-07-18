using System.Diagnostics;
using System.Text.Json;

internal static class AnalyzerRunner
{
    public static async Task<IResult> RunAsync(
        string contentRoot,
        IConfiguration configuration,
        ILogger logger,
        object? payload,
        string? resumeFile,
        string? resumeId,
        bool includeErrorDetails,
        CancellationToken requestCancellation)
    {
        var repositoryRoot = ProjectPaths.RepositoryRoot(contentRoot);
        var python = configuration["Ai:PythonExecutable"]
            ?? Path.Combine(repositoryRoot, "ai", ".venv", "bin", "python");
        var jobs = configuration["Ai:JobsPath"]
            ?? Path.Combine(repositoryRoot, "datasets", "raw", "job_positions.csv");
        var timeoutSeconds = configuration.GetValue("Ai:TimeoutSeconds", 30);
        using var process = StartProcess(repositoryRoot, python, jobs, resumeFile, resumeId);
        if (process is null) return Results.Problem("Could not start the AI process.");

        if (payload is not null)
            await process.StandardInput.WriteAsync(
                JsonSerializer.Serialize(payload, JsonSerializerOptions.Web));
        process.StandardInput.Close();

        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(requestCancellation);
        timeout.CancelAfter(TimeSpan.FromSeconds(timeoutSeconds));
        try
        {
            await process.WaitForExitAsync(timeout.Token);
        }
        catch (OperationCanceledException) when (!requestCancellation.IsCancellationRequested)
        {
            process.Kill(entireProcessTree: true);
            return Results.Problem(
                "Resume analysis timed out.", statusCode: StatusCodes.Status504GatewayTimeout);
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited) process.Kill(entireProcessTree: true);
            throw;
        }

        var output = await outputTask;
        var error = await errorTask;
        if (process.ExitCode == 0) return Results.Text(output, "application/json");
        logger.LogWarning(
            "Analyzer exited with code {ExitCode}: {AnalyzerError}", process.ExitCode, error);
        return Results.Problem(
            detail: includeErrorDetails ? error : "The resume could not be analyzed.",
            statusCode: StatusCodes.Status422UnprocessableEntity);
    }

    private static Process? StartProcess(
        string repositoryRoot,
        string python,
        string jobs,
        string? resumeFile,
        string? resumeId)
    {
        var startInfo = new ProcessStartInfo(python)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            WorkingDirectory = repositoryRoot,
        };
        startInfo.ArgumentList.Add("-m");
        startInfo.ArgumentList.Add("resume_role_ai.cli");
        startInfo.ArgumentList.Add("--jobs");
        startInfo.ArgumentList.Add(jobs);
        if (resumeFile is not null)
        {
            startInfo.ArgumentList.Add("--file");
            startInfo.ArgumentList.Add(resumeFile);
            startInfo.ArgumentList.Add("--id");
            startInfo.ArgumentList.Add(resumeId ?? "uploaded-resume");
        }
        startInfo.Environment["PYTHONPATH"] = Path.Combine(repositoryRoot, "ai", "src");
        return Process.Start(startInfo);
    }
}
