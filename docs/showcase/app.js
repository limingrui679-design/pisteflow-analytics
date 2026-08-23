const repositoryBase = "https://github.com/limingrui679-design/pisteflow-analytics/blob/main/";

const lensLabels = {
  "decision-analytics": "Decision analytics",
  "reproducible-ml": "Reproducible ML",
  "statistical-inference": "Statistical inference",
  "data-systems-governance": "Data systems & governance",
  "risk-finance": "Risk & finance",
  "behavioral-science": "Behavioral science",
  "population-health": "Population health",
  "spatial-policy": "Spatial policy",
};

const statusLabels = {
  lift: "Lift",
  "no-lift": "No lift",
  mixed: "Mixed",
  replication: "Replication",
  diagnostic: "Diagnostic",
  descriptive: "Descriptive",
};

const statusOrder = ["lift", "replication", "no-lift", "mixed", "diagnostic", "descriptive"];

const elements = {
  caseList: document.querySelector("#case-list"),
  detailPanel: document.querySelector("#detail-panel"),
  empty: document.querySelector("#empty"),
  error: document.querySelector("#error"),
  form: document.querySelector("#filters"),
  search: document.querySelector("#search"),
  lens: document.querySelector("#lens"),
  status: document.querySelector("#status"),
  weapon: document.querySelector("#weapon"),
  resultCount: document.querySelector("#result-count"),
  caseCount: document.querySelector("#case-count"),
  signalCaseCount: document.querySelector("#signal-case-count"),
  lensCount: document.querySelector("#lens-count"),
  evidenceCount: document.querySelector("#evidence-count"),
  negativeCount: document.querySelector("#negative-count"),
  statusSpectrum: document.querySelector("#status-spectrum"),
  statusLegend: document.querySelector("#status-legend"),
  capabilityBars: document.querySelector("#capability-bars"),
  coverageLabel: document.querySelector("#coverage-label"),
};

let catalog = null;
let selectedCaseId = null;

function appendText(parent, tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  parent.append(element);
  return element;
}

function statusBadge(status) {
  return appendText(
    document.createDocumentFragment(),
    "span",
    statusLabels[status] || status,
    `status status-${status}`,
  );
}

function renderOverview() {
  const cases = catalog.cases;
  const lensCounts = new Map(Object.keys(lensLabels).map((lens) => [lens, 0]));
  const statusCounts = new Map(statusOrder.map((status) => [status, 0]));

  cases.forEach((item) => {
    item.capabilities.forEach((lens) => lensCounts.set(lens, (lensCounts.get(lens) || 0) + 1));
    statusCounts.set(item.result_status, (statusCounts.get(item.result_status) || 0) + 1);
  });

  const negativeCount = statusCounts.get("no-lift") || 0;
  const evidenceCount = Object.keys(catalog.evidence_sha256 || {}).length;
  elements.caseCount.textContent = cases.length;
  elements.signalCaseCount.textContent = cases.length;
  elements.lensCount.textContent = lensCounts.size;
  elements.evidenceCount.textContent = evidenceCount;
  elements.negativeCount.textContent = negativeCount;
  elements.coverageLabel.textContent = `${lensCounts.size} lenses`;

  const spectrum = [];
  const legend = [];
  const statusSummary = [];
  statusOrder.forEach((status) => {
    const count = statusCounts.get(status) || 0;
    if (!count) return;

    const segment = document.createElement("span");
    segment.className = `spectrum-segment status-${status}`;
    segment.style.flexGrow = String(count);
    segment.title = `${statusLabels[status]}: ${count}`;
    spectrum.push(segment);

    const legendItem = document.createElement("span");
    legendItem.className = `legend-item status-${status}`;
    appendText(legendItem, "i", "", "legend-dot");
    appendText(legendItem, "span", statusLabels[status]);
    appendText(legendItem, "strong", String(count));
    legend.push(legendItem);
    statusSummary.push(`${statusLabels[status]} ${count}`);
  });
  elements.statusSpectrum.replaceChildren(...spectrum);
  elements.statusSpectrum.setAttribute("aria-label", `Result mix: ${statusSummary.join(", ")}`);
  elements.statusLegend.replaceChildren(...legend);

  const maxLensCount = Math.max(...lensCounts.values());
  const capabilityRows = [...lensCounts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([lens, count]) => {
      const row = document.createElement("div");
      row.className = "capability-row";
      appendText(row, "span", lensLabels[lens] || lens, "capability-name");
      appendText(row, "span", String(count).padStart(2, "0"), "capability-value");
      const track = document.createElement("span");
      track.className = "capability-track";
      const fill = document.createElement("span");
      fill.className = "capability-fill";
      fill.style.width = `${(count / maxLensCount) * 100}%`;
      track.append(fill);
      row.append(track);
      return row;
    });
  elements.capabilityBars.replaceChildren(...capabilityRows);
  elements.capabilityBars.setAttribute(
    "aria-label",
    `Capability coverage: ${[...lensCounts.entries()]
      .map(([lens, count]) => `${lensLabels[lens] || lens} ${count}`)
      .join(", ")}`,
  );
}

function caseRow(item) {
  const button = document.createElement("button");
  const index = catalog.cases.findIndex((candidate) => candidate.id === item.id) + 1;
  button.type = "button";
  button.className = "case-row";
  button.dataset.caseId = item.id;
  button.setAttribute("aria-pressed", String(item.id === selectedCaseId));
  button.setAttribute("aria-label", `Inspect case ${index}: ${item.title}, ${statusLabels[item.result_status]}`);

  appendText(button, "span", String(index).padStart(2, "0"), "case-number");
  const copy = document.createElement("span");
  copy.className = "case-row-copy";
  appendText(copy, "span", item.title, "case-row-title");
  appendText(copy, "span", item.headline, "case-row-result");
  button.append(copy, statusBadge(item.result_status));
  return button;
}

function addOutcomeCell(parent, label, value) {
  const cell = document.createElement("div");
  cell.className = "outcome-cell";
  appendText(cell, "span", label, "outcome-label");
  appendText(cell, "strong", value, "outcome-value");
  parent.append(cell);
}

function addContractItem(parent, label, value, wide = false) {
  const item = document.createElement("div");
  item.className = `contract-item${wide ? " contract-item-wide" : ""}`;
  appendText(item, "dt", label, "contract-label");
  appendText(item, "dd", value, "contract-value");
  parent.append(item);
}

function renderDetail(item) {
  if (!item) {
    elements.detailPanel.hidden = true;
    elements.detailPanel.replaceChildren();
    return;
  }

  const index = catalog.cases.findIndex((candidate) => candidate.id === item.id) + 1;
  const inner = document.createElement("div");
  inner.className = "detail-inner";

  const topline = document.createElement("div");
  topline.className = "detail-topline";
  appendText(
    topline,
    "span",
    `Case ${String(index).padStart(2, "0")} / ${item.evidence_kind.replaceAll("-", " ")}`,
    "detail-index",
  );
  topline.append(statusBadge(item.result_status));
  inner.append(topline);

  appendText(inner, "h3", item.title);
  appendText(inner, "p", item.summary, "detail-summary");

  const tags = document.createElement("div");
  tags.className = "detail-tags";
  item.capabilities.forEach((lens) => appendText(tags, "span", lensLabels[lens] || lens, "detail-tag"));
  item.weapons.forEach((weapon) => appendText(tags, "span", weapon, "detail-tag"));
  inner.append(tags);

  const outcome = document.createElement("div");
  outcome.className = "outcome-grid";
  addOutcomeCell(outcome, "Observed result", item.headline);
  addOutcomeCell(
    outcome,
    "Credible baseline",
    item.baseline || "Not applicable for this descriptive evidence",
  );
  inner.append(outcome);

  const contract = document.createElement("dl");
  contract.className = "detail-contract";
  addContractItem(contract, "Research question", item.question);
  addContractItem(contract, "Unit & sample", `${item.unit}; ${item.sample}`);
  addContractItem(contract, "Method", item.method);
  addContractItem(contract, "Validation", item.validation);
  addContractItem(contract, "Decision use", item.decision_use, true);
  inner.append(contract);

  const boundary = document.createElement("section");
  boundary.className = "boundary-card";
  appendText(boundary, "span", "Claim boundary", "contract-label");
  appendText(boundary, "p", item.claim_boundary);
  inner.append(boundary);

  const evidence = document.createElement("section");
  evidence.className = "evidence-section";
  appendText(evidence, "span", `Evidence paths · ${item.evidence_paths.length}`, "contract-label");
  const evidenceList = document.createElement("ul");
  evidenceList.className = "evidence-list";
  item.evidence_paths.forEach((path) => {
    const listItem = document.createElement("li");
    const link = document.createElement("a");
    link.href = `${repositoryBase}${path}`;
    link.rel = "noreferrer";
    link.target = "_blank";
    appendText(link, "span", path);
    appendText(link, "span", "↗");
    listItem.append(link);
    evidenceList.append(listItem);
  });
  evidence.append(evidenceList);
  inner.append(evidence);

  elements.detailPanel.hidden = false;
  elements.detailPanel.setAttribute("aria-label", `Selected case: ${item.title}`);
  elements.detailPanel.replaceChildren(inner);
  elements.detailPanel.scrollTop = 0;
}

function currentFilters() {
  return {
    q: elements.search.value.trim().toLocaleLowerCase(),
    lens: elements.lens.value,
    status: elements.status.value,
    weapon: elements.weapon.value,
  };
}

function matchingCases(filters) {
  return catalog.cases.filter((item) => {
    const haystack = [
      item.title,
      item.question,
      item.summary,
      item.headline,
      item.method,
      item.validation,
      item.decision_use,
      item.claim_boundary,
      ...item.capabilities,
      ...item.weapons,
    ]
      .join(" ")
      .toLocaleLowerCase();
    return (
      (!filters.q || haystack.includes(filters.q)) &&
      (!filters.lens || item.capabilities.includes(filters.lens)) &&
      (!filters.status || item.result_status === filters.status) &&
      (!filters.weapon || item.weapons.includes(filters.weapon))
    );
  });
}

function updateUrl(filters) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value) params.set(key, value);
  });
  if (selectedCaseId) params.set("case", selectedCaseId);
  const query = params.toString();
  window.history.replaceState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}`);
}

function render() {
  if (!catalog) return;
  const filters = currentFilters();
  const matches = matchingCases(filters);

  if (!matches.some((item) => item.id === selectedCaseId)) {
    selectedCaseId = matches[0]?.id || null;
  }

  elements.caseList.replaceChildren(...matches.map(caseRow));
  elements.caseList.setAttribute("aria-busy", "false");
  elements.empty.hidden = matches.length !== 0;
  elements.resultCount.textContent = `${matches.length} / ${catalog.cases.length} cases visible`;
  renderDetail(matches.find((item) => item.id === selectedCaseId));
  updateUrl(filters);
}

function applyUrlState() {
  const params = new URLSearchParams(window.location.search);
  elements.search.value = params.get("q") || "";
  elements.lens.value = params.get("lens") || "";
  elements.status.value = params.get("status") || "";
  elements.weapon.value = params.get("weapon") || "";
  selectedCaseId = params.get("case");
}

async function loadCatalog() {
  try {
    const response = await fetch("cases.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Catalog request failed with status ${response.status}`);
    catalog = await response.json();
    applyUrlState();
    renderOverview();
    render();
  } catch (error) {
    elements.caseList.setAttribute("aria-busy", "false");
    elements.resultCount.textContent = "Catalog unavailable";
    elements.detailPanel.hidden = true;
    elements.error.hidden = false;
    elements.error.textContent = `The case catalog could not be loaded. ${error.message}`;
  }
}

elements.form.addEventListener("input", render);
elements.form.addEventListener("reset", () => {
  selectedCaseId = catalog?.cases[0]?.id || null;
  window.setTimeout(render, 0);
});

elements.caseList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-case-id]");
  if (!button) return;
  selectedCaseId = button.dataset.caseId;
  render();

  if (window.matchMedia("(max-width: 900px)").matches) {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.requestAnimationFrame(() => {
      elements.detailPanel.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    });
  }
});

loadCatalog();
