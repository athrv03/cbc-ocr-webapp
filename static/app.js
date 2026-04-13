const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("pdfFile");
const statusEl = document.getElementById("status");
const resultsSection = document.getElementById("results");
const extractedJsonEl = document.getElementById("extractedJson");
const predictedEl = document.getElementById("predictedDiseases");
const tableBody = document.querySelector("#resultTable tbody");
const chartCanvas = document.getElementById("chartCanvas");

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b91c1c" : "#0f766e";
}

function renderPredictedDiseases(labels) {
  predictedEl.innerHTML = "";
  labels.forEach((label) => {
    const span = document.createElement("span");
    span.className = "badge good";
    span.textContent = label;
    predictedEl.appendChild(span);
  });
}

function renderTable(rows) {
  tableBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.Disease}</td>
      <td>${row.Predicted ? "Yes" : "No"}</td>
      <td>${Number(row.Probability).toFixed(2)}</td>
    `;
    tableBody.appendChild(tr);
  });
}

function renderChart(rows) {
  const ctx = chartCanvas.getContext("2d");
  const W = chartCanvas.width;
  const H = chartCanvas.height;
  const padLeft = 140;
  const padRight = 40;
  const top = 24;
  const bottom = 26;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W, H);

  const usableW = W - padLeft - padRight;
  const rowH = (H - top - bottom) / rows.length;

  ctx.strokeStyle = "#d1d5db";
  ctx.beginPath();
  ctx.moveTo(padLeft, top - 4);
  ctx.lineTo(padLeft, H - bottom + 4);
  ctx.stroke();

  rows.slice().reverse().forEach((row, i) => {
    const y = top + i * rowH + rowH * 0.2;
    const h = rowH * 0.6;
    const p = Math.max(0, Math.min(100, Number(row.Probability)));
    const barW = usableW * (p / 100);

    ctx.fillStyle = row.Predicted ? "#166534" : "#0f766e";
    ctx.fillRect(padLeft, y, barW, h);

    ctx.fillStyle = "#111827";
    ctx.font = "14px sans-serif";
    ctx.fillText(row.Disease, 10, y + h * 0.75);
    ctx.fillText(`${p.toFixed(2)}%`, padLeft + barW + 8, y + h * 0.75);
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) {
    setStatus("Please choose a PDF file.", true);
    return;
  }

  const btn = form.querySelector("button");
  btn.disabled = true;
  setStatus("Running OCR and prediction. This may take a few seconds...");

  try {
    const fd = new FormData();
    fd.append("file", fileInput.files[0]);

    const response = await fetch("/api/predict-from-pdf", {
      method: "POST",
      body: fd,
    });

    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Request failed");
    }

    extractedJsonEl.textContent = JSON.stringify(body.extracted_values, null, 2);
    renderPredictedDiseases(body.predicted_diseases || []);
    renderTable(body.results || []);
    renderChart(body.results || []);

    resultsSection.classList.remove("hidden");
    setStatus(`Done: ${body.file_name}`);
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
  } finally {
    btn.disabled = false;
  }
});
