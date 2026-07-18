# Resume Role Analyzer AI

An end-to-end platform that recommends five career paths for an uploaded resume. Python
combines MiniLM sentence embeddings, O*NET career-family filtering, and normalized skill
overlap; an ASP.NET Core API exposes the analyzer to the browser client.

## Run it

The sample database is `datasets/raw/job_positions.csv`. Add or replace rows
there to teach the matcher about more positions.

```bash
cd ai
source .venv/bin/activate
pip install .
pytest

cd ../api
dotnet run
```

Run the API integration tests from the repository root:

```bash
dotnet test api.tests/ResumeRoleAnalyzer.Api.Tests.csproj
```

For a production-style container build and hosting requirements, see
[`DEPLOYMENT.md`](DEPLOYMENT.md).

Analyze a resume at `POST http://localhost:5000/api/resumes/analyze` (use the URL printed
by `dotnet run`). Responses contain the five highest ranked careers and evidence such as
matched and missing skills:

```json
{
  "id": "resume-1",
  "rawText": "Built predictive models and production APIs.",
  "skills": ["Python", "pandas", "scikit-learn", "SQL"],
  "education": [],
  "experience": [],
  "certifications": []
}
```

Or upload a PDF, DOCX, or TXT resume (maximum 5 MB):

```bash
curl -X POST http://localhost:5000/api/resumes/upload \
  -F "file=@/path/to/resume.pdf"
```

PDF parsing extracts embedded text; scanned image-only PDFs require an OCR step that is
not included yet. Skills found in the job database are detected automatically before
the roles are ranked.

The browser presents qualitative skill-overlap labels rather than treating raw embedding
similarity as a probability. Recommendations are career-exploration aids, not hiring
probabilities or qualification decisions.

## Prepare training data

The project can download the MIT-licensed
[Candidate-Job Matching Synthetic Dataset](https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic)
and turn its resumes, jobs, and relevance labels into deterministic train, validation,
and test pairs:

```bash
cd ai
source .venv/bin/activate
pip install .
prepare-resume-training-data
```

This writes `datasets/processed/training_pairs.parquet` with `job_text`, `resume_text`,
`label`, and `split` columns. Source and processed datasets are ignored by Git. The
split is grouped by job ID to prevent the same job leaking across evaluation splits.

## Train the matcher

Train the supervised pair classifier and evaluate its ranking performance:

```bash
cd ai
source .venv/bin/activate
train-resume-matcher
```

The command writes `models/resume_matcher.joblib` and `models/metrics.json`. This pair
classifier is retained for evaluation and comparison. It is not blended into live ranking
because it reduced ranking quality on the independent validation benchmark.

Build the sentence-transformer catalog artifact after importing or changing job profiles.
The first run downloads the MiniLM model:

```bash
build-catalog-ranker
```

Live ranking uses sentence embeddings, career-family filtering, and rarity-weighted skill
overlap. The supervised pair model remains in the benchmark for comparison but is not
blended into live ranking because it reduced validation quality.

## Refresh the job catalog

Import the versioned O*NET occupation, essential-skill, and software-skill data while
preserving hand-authored roles:

```bash
cd ai
source .venv/bin/activate
import-onet-catalog
```

The generated `datasets/raw/job_positions.csv` contains modified information from the
O*NET 30.3 Database by the U.S. Department of Labor, Employment and Training
Administration (USDOL/ETA), used under the
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). O*NET® is a trademark
of USDOL/ETA. USDOL/ETA has not approved, endorsed, or tested these modifications.

## Evaluate with independently annotated resumes

Run the catalog benchmark against all 302 anonymized CareerCorpus resumes and their six
expert-provided domain labels:

```bash
cd ai
source .venv/bin/activate
evaluate-career-corpus
```

The command verifies the published workbook checksum and writes aggregate-only results
to `models/career_corpus_metrics.json`; it does not copy resume text into the report.
CareerCorpus by Adiba et al. (2025), DOI `10.17632/wzzwn37gmd.1`, is used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This project transforms its
published domain labels into deterministic occupation-title groups for evaluation.

## Model results

The latest CareerCorpus run evaluates 302 resumes against 1,022 occupation profiles. The
metrics most representative of the website are ranking metrics:

| Ranking metric | Result |
| --- | ---: |
| Expected career ranked first (Top-1) | 18.5% |
| Expected career included in five recommendations (Top-5) | 54.3% |
| Median expected-career rank | 4 |
| Mean reciprocal rank | 33.1% |

The separately trained resume/job pair classifier reports stronger binary-classification
results:

| Pair-classification metric | Result |
| --- | ---: |
| Accuracy | 90.4% |
| ROC-AUC | 94.5% |
| Average precision | 91.0% |

These two sets of results measure different tasks. Pair classification asks whether one
given resume and one given job are related. Career recommendation must order 1,022
occupations and distinguish between many closely related roles. Most randomly paired
resume/job examples are easy negatives, so high pair accuracy does not imply equally high
Top-1 recommendation accuracy.

Compared with the previous TF-IDF catalog ranker, the sentence-embedding pipeline improved
Top-5 from 43.7% to 54.3% and improved the median expected-career rank from 7 to 4. Top-1
decreased from 20.2% to 18.5%, so the current model is more useful as a five-role shortlist
than as a single definitive career prediction.

## Limitations and future work

- CareerCorpus contains only 302 resumes across six domains, so it does not represent the
  full range of 1,022 O*NET occupations.
- Synthetic training pairs are not perfectly aligned with the final O*NET catalog.
- Similar occupations are difficult to order without verified resume-to-occupation labels
  and hard negative examples.
- Skill detection uses deterministic taxonomy and alias matching. It can miss uncommon
  wording or incorrectly equate ambiguous terms.
- PDF extraction supports embedded text but not scanned, image-only resumes.
- Recommendations do not estimate employability, hiring probability, or candidate quality.

The most valuable next steps are collecting consented resume-to-O*NET labels, training on
closely related hard negatives, adding a cross-encoder reranker, and extracting structured
job titles, seniority, and years of experience.
