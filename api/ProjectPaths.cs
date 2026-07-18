internal static class ProjectPaths
{
    public static string RepositoryRoot(string contentRoot)
    {
        var fullContentRoot = Path.GetFullPath(contentRoot);
        return Directory.Exists(Path.Combine(fullContentRoot, "ai"))
            ? fullContentRoot
            : Path.GetFullPath(Path.Combine(fullContentRoot, ".."));
    }
}
