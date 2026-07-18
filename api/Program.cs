using System.Threading.RateLimiting;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.Extensions.FileProviders;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddProblemDetails();
builder.Services.AddRateLimiter(options =>
{
    options.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
    options.AddPolicy("analysis", context =>
        RateLimitPartition.GetFixedWindowLimiter(
            context.Connection.RemoteIpAddress?.ToString() ?? "unknown",
            _ => new FixedWindowRateLimiterOptions
            {
                PermitLimit = 20,
                Window = TimeSpan.FromMinutes(1),
                QueueLimit = 0,
            }));
});
builder.WebHost.ConfigureKestrel(options =>
    options.Limits.MaxRequestBodySize = 6 * 1024 * 1024);
builder.Environment.WebRootPath = Path.Combine(
    ProjectPaths.RepositoryRoot(builder.Environment.ContentRootPath), "web");

var app = builder.Build();
ModelMetricsReporter.Log(app.Logger, app.Environment.ContentRootPath);

if (!app.Environment.IsDevelopment()) app.UseExceptionHandler();
if (app.Environment.IsProduction()) app.UseHsts();
app.Use(async (context, next) =>
{
    context.Response.Headers.XContentTypeOptions = "nosniff";
    context.Response.Headers["Referrer-Policy"] = "strict-origin-when-cross-origin";
    if (!app.Environment.IsDevelopment())
    {
        context.Response.Headers.XFrameOptions = "DENY";
        context.Response.Headers.ContentSecurityPolicy =
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' " +
            "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; " +
            "img-src 'self' data:; connect-src 'self'; base-uri 'self'; " +
            "frame-ancestors 'none'";
    }
    if (context.Request.Path == "/" ||
        context.Request.Path.StartsWithSegments("/app.js") ||
        context.Request.Path.StartsWithSegments("/styles.css"))
        context.Response.Headers.CacheControl = "no-store, max-age=0";
    await next();
});
app.UseRateLimiter();
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(app.Environment.WebRootPath),
});

app.MapGet("/", () => Results.File(
    Path.Combine(app.Environment.WebRootPath, "index.html"),
    "text/html; charset=utf-8"));
app.MapGet("/health", () => Results.Ok(new { status = "healthy" }));
app.MapPost("/api/resumes/analyze", async (
    Resume resume,
    IConfiguration configuration,
    ILogger<Program> logger,
    CancellationToken cancellationToken) =>
{
    var payload = new
    {
        resume.Id,
        resume.RawText,
        Skills = resume.Skills ?? [],
        Education = resume.Education ?? [],
        Experience = resume.Experience ?? [],
        Certifications = resume.Certifications ?? [],
    };
    return await AnalyzerRunner.RunAsync(
        app.Environment.ContentRootPath, configuration, logger, payload, null, null,
        app.Environment.IsDevelopment(), cancellationToken);
}).RequireRateLimiting("analysis");

app.MapPost("/api/resumes/upload", async (
    IFormFile file,
    IConfiguration configuration,
    ILogger<Program> logger,
    CancellationToken cancellationToken) =>
{
    const long maxFileSize = 5 * 1024 * 1024;
    var extension = Path.GetExtension(file.FileName).ToLowerInvariant();
    if (file.Length == 0)
        return Results.BadRequest(new { error = "The uploaded resume is empty." });
    if (file.Length > maxFileSize)
        return Results.BadRequest(new { error = "The resume must be 5 MB or smaller." });
    if (extension is not (".pdf" or ".docx" or ".txt"))
        return Results.BadRequest(new { error = "Only PDF, DOCX, and TXT resumes are supported." });

    var temporaryFile = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}{extension}");
    try
    {
        await using (var stream = File.Create(temporaryFile))
            await file.CopyToAsync(stream, cancellationToken);

        return await AnalyzerRunner.RunAsync(
            app.Environment.ContentRootPath,
            configuration,
            logger,
            null,
            temporaryFile,
            Path.GetFileNameWithoutExtension(file.FileName),
            app.Environment.IsDevelopment(),
            cancellationToken);
    }
    finally
    {
        if (File.Exists(temporaryFile)) File.Delete(temporaryFile);
    }
}).DisableAntiforgery().RequireRateLimiting("analysis");

app.Run();

public partial class Program;
