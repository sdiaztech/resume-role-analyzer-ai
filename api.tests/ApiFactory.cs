using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;

namespace ResumeRoleAnalyzer.Api.Tests;

public sealed class ApiFactory : WebApplicationFactory<Program>
{
    public string RepositoryRoot { get; } = FindRepositoryRoot();

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");
        builder.ConfigureAppConfiguration((_, configuration) =>
        {
            configuration.AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["Ai:PythonExecutable"] = Path.Combine(
                    RepositoryRoot, "ai", ".venv", "bin", "python"),
                ["Ai:JobsPath"] = Path.Combine(
                    RepositoryRoot, "datasets", "raw", "job_positions.csv"),
                ["Ai:TimeoutSeconds"] = "30",
            });
        });
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "ai")) &&
                Directory.Exists(Path.Combine(directory.FullName, "datasets")))
                return directory.FullName;
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException("Could not find the repository root.");
    }
}
