internal sealed record Resume(
    string Id,
    string RawText,
    List<string>? Skills,
    List<string>? Education,
    List<string>? Experience,
    List<string>? Certifications);
