using System.Net;
using System.Net.Http.Json;
using System.Text.Json;

namespace ResumeRoleAnalyzer.Api.Tests;

public sealed class ApiTests(ApiFactory factory) : IClassFixture<ApiFactory>
{
    private readonly HttpClient _client = factory.CreateClient();

    [Fact]
    public async Task HealthEndpointReportsHealthy()
    {
        var response = await _client.GetAsync("/health");
        var payload = await response.Content.ReadFromJsonAsync<JsonElement>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("healthy", payload.GetProperty("status").GetString());
        Assert.Equal("nosniff", response.Headers.GetValues("X-Content-Type-Options").Single());
        Assert.Equal("DENY", response.Headers.GetValues("X-Frame-Options").Single());
        Assert.True(response.Headers.Contains("Content-Security-Policy"));
    }

    [Fact]
    public async Task JsonAnalysisReturnsFiveRankedCareers()
    {
        var response = await _client.PostAsJsonAsync("/api/resumes/analyze", new
        {
            id = "integration-json",
            rawText = "Built machine learning models using Python, SQL, pandas, and scikit-learn.",
            skills = new[] { "Python", "SQL", "pandas", "scikit-learn", "Machine Learning" },
            education = Array.Empty<string>(),
            experience = Array.Empty<string>(),
            certifications = Array.Empty<string>(),
        });
        var payload = await response.Content.ReadFromJsonAsync<JsonElement>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("integration-json", payload.GetProperty("resume_id").GetString());
        Assert.Equal(5, payload.GetProperty("matches").GetArrayLength());
        Assert.Equal("Data Scientist", payload.GetProperty("matches")[0].GetProperty("title").GetString());
    }

    [Fact]
    public async Task TextUploadReturnsFiveRankedCareers()
    {
        using var form = new MultipartFormDataContent();
        using var resume = new StringContent(
            "Data analyst experienced with SQL, Excel, Python, statistics, dashboards, and reports.");
        form.Add(resume, "file", "resume.txt");

        var response = await _client.PostAsync("/api/resumes/upload", form);
        var payload = await response.Content.ReadFromJsonAsync<JsonElement>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(5, payload.GetProperty("matches").GetArrayLength());
    }

    [Theory]
    [InlineData("resume.rtf")]
    [InlineData("resume.exe")]
    public async Task UploadRejectsUnsupportedFileTypes(string filename)
    {
        using var form = new MultipartFormDataContent();
        form.Add(new StringContent("resume"), "file", filename);

        var response = await _client.PostAsync("/api/resumes/upload", form);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task UploadRejectsEmptyFiles()
    {
        using var form = new MultipartFormDataContent();
        form.Add(new ByteArrayContent([]), "file", "resume.txt");

        var response = await _client.PostAsync("/api/resumes/upload", form);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task AnalyzerFailuresDoNotExposeInternalDiagnostics()
    {
        using var form = new MultipartFormDataContent();
        form.Add(new StringContent("   "), "file", "resume.txt");

        var response = await _client.PostAsync("/api/resumes/upload", form);
        var payload = await response.Content.ReadFromJsonAsync<JsonElement>();

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        Assert.Equal("The resume could not be analyzed.", payload.GetProperty("detail").GetString());
    }

    [Fact]
    public async Task UploadRejectsFilesLargerThanFiveMegabytes()
    {
        using var form = new MultipartFormDataContent();
        form.Add(new ByteArrayContent(new byte[5 * 1024 * 1024 + 1]), "file", "resume.txt");

        var response = await _client.PostAsync("/api/resumes/upload", form);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}

public sealed class RateLimitTests
{
    [Fact]
    public async Task AnalysisRateLimitRejectsExcessRequests()
    {
        await using var factory = new ApiFactory();
        using var client = factory.CreateClient();
        HttpResponseMessage? response = null;

        for (var index = 0; index < 21; index++)
        {
            using var form = new MultipartFormDataContent();
            form.Add(new StringContent("resume"), "file", "resume.rtf");
            response?.Dispose();
            response = await client.PostAsync("/api/resumes/upload", form);
        }

        Assert.NotNull(response);
        Assert.Equal(HttpStatusCode.TooManyRequests, response.StatusCode);
        response.Dispose();
    }
}
