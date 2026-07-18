const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#resume-file");
const dropZone = document.querySelector("#drop-zone");
const selectedFile = document.querySelector("#selected-file");
const fileName = document.querySelector("#file-name");
const fileSize = document.querySelector("#file-size");
const removeFile = document.querySelector("#remove-file");
const analyzeButton = document.querySelector("#analyze-button");
const formMessage = document.querySelector("#form-message");
const results = document.querySelector("#results");
const resultList = document.querySelector("#result-list");
const analyzeAnother = document.querySelector("#analyze-another");
const fileBadge = document.querySelector(".file-badge");
const hero = document.querySelector(".hero");

const allowedExtensions = ["pdf", "docx", "txt"];
const maxFileSize = 5 * 1024 * 1024;

function humanFileSize(bytes) {
  return bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function setFile(file, syncInput = false) {
  formMessage.textContent = "";
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!allowedExtensions.includes(extension)) {
    clearFile();
    formMessage.textContent = "Choose a PDF, DOCX, or TXT file.";
    return;
  }
  if (file.size > maxFileSize) {
    clearFile();
    formMessage.textContent = "Your resume must be 5 MB or smaller.";
    return;
  }
  if (syncInput) {
    try {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      fileInput.files = transfer.files;
    } catch {
      clearFile();
      formMessage.textContent = "Drag-and-drop is not supported by this browser. Choose the PDF using the file picker instead.";
      return;
    }
  }
  fileName.textContent = file.name;
  fileSize.textContent = humanFileSize(file.size);
  fileBadge.textContent = extension.toUpperCase();
  dropZone.hidden = true;
  selectedFile.hidden = false;
  analyzeButton.disabled = false;
}

function clearFile() {
  fileInput.value = "";
  dropZone.hidden = false;
  selectedFile.hidden = true;
  analyzeButton.disabled = true;
}

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});
removeFile.addEventListener("click", clearFile);

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files[0]) setFile(event.dataTransfer.files[0], true);
});

function skillMarkup(skills, kind) {
  if (!skills?.length) return "";
  const label = kind === "matched" ? "Skills found" : "Skills to develop";
  return `<span class="skill-label">${label}</span>${skills
    .map((skill) => `<span class="skill ${kind}">${escapeHtml(skill)}</span>`)
    .join("")}`;
}

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = value;
  return element.innerHTML;
}

function matchPresentation(match) {
  const matchedCount = match.matched_skills?.length ?? 0;
  const missingCount = match.missing_skills?.length ?? 0;
  const totalSkills = matchedCount + missingCount;
  const coverage = totalSkills ? matchedCount / totalSkills : 0;
  const label = !totalSkills
    ? "Experience match"
    : coverage >= 0.75
    ? "High skill overlap"
    : coverage >= 0.5
    ? "Moderate skill overlap"
    : matchedCount > 0
    ? "Some skill overlap"
    : "Semantic match";
  return {
    label,
    evidence: totalSkills
      ? `${matchedCount} of ${totalSkills} listed skills found`
      : "Ranked from resume experience",
  };
}

function renderResults(data) {
  resultList.innerHTML = data.matches.map((match, index) => {
    const { label, evidence } = matchPresentation(match);
    return `<article class="result-card">
      <div class="score-block">
        <div class="match-badge" aria-label="${label}"><strong>${label}</strong></div>
        <small>${evidence}</small>
      </div>
      <div>
        <div class="result-top"><div><span class="rank">Match ${index + 1}</span><h3>${escapeHtml(match.title)}</h3></div></div>
        <p class="explanation">${escapeHtml(match.explanation)}</p>
        <div class="skills">
          ${skillMarkup(match.matched_skills, "matched")}
          ${skillMarkup(match.missing_skills, "missing")}
        </div>
      </div>
    </article>`;
  }).join("");
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (window.location.protocol === "file:") {
    formMessage.textContent = "This page was opened as a local file. Start the .NET API and open the http://localhost URL printed in the terminal.";
    return;
  }
  const file = fileInput.files[0];
  if (!file) return;
  analyzeButton.disabled = true;
  analyzeButton.classList.add("loading");
  formMessage.textContent = "";
  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch("/api/resumes/upload", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error || "Analysis failed.");
    renderResults(data);
  } catch (error) {
    formMessage.textContent = window.location.protocol === "file:"
      ? "Start the .NET API and open the http://localhost URL printed in the terminal."
      : error.message === "Failed to fetch"
      ? "Could not reach the analyzer. Make sure the C# API is running."
      : error.message;
  } finally {
    analyzeButton.classList.remove("loading");
    analyzeButton.disabled = false;
  }
});

analyzeAnother.addEventListener("click", () => {
  clearFile();
  results.hidden = true;
  formMessage.textContent = "";
  hero.scrollIntoView({ behavior: "smooth" });
});
