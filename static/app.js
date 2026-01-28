const state = {
  curve: [],
  histogram: null,
  aepKwh: 0,
};

const qs = (id) => document.getElementById(id);

const fetchJson = async (url, options = {}) => {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    throw new Error(`Request failed: ${resp.status}`);
  }
  return resp.json();
};

const downloadCsv = (filename, rows) => {
  const blob = new Blob([rows], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
};

const tableFromObject = (tableEl, obj) => {
  tableEl.innerHTML = "";
  const tbody = document.createElement("tbody");
  Object.entries(obj).forEach(([key, value]) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${key}</td><td>${Number(value).toFixed(4)}</td>`;
    tbody.appendChild(row);
  });
  tableEl.appendChild(tbody);
};

const tableFromRows = (tableEl, rows) => {
  tableEl.innerHTML = "";
  if (!rows.length) {
    return;
  }
  const thead = document.createElement("thead");
  const headers = Object.keys(rows[0]);
  thead.innerHTML = `<tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr>`;
  tableEl.appendChild(thead);
  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = headers.map((h) => `<td>${row[h]}</td>`).join("");
    tbody.appendChild(tr);
  });
  tableEl.appendChild(tbody);
};

const bindTabs = () => {
  document.querySelectorAll(".tab-button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((section) => section.classList.remove("active"));
      btn.classList.add("active");
      qs(btn.dataset.tab).classList.add("active");
    });
  });
};

const loadDefaults = async () => {
  const data = await fetchJson("/api/scenarios");
  const defaults = data.default;
  qs("sweptArea").value = defaults.swept_area_m2;
  qs("cp").value = defaults.cp;
  qs("rho").value = defaults.rho;
  qs("drivetrainEff").value = defaults.drivetrain_eff;
  qs("availability").value = defaults.availability;
  qs("inverterRating").value = defaults.inverter_rating_kw;
  qs("weibullK").value = defaults.weibull_k;
  qs("weibullC").value = defaults.weibull_c;
  qs("l1").value = defaults.l1;
  qs("l2").value = defaults.l2;
  qs("mUpper").value = defaults.m_upper_arm;
  qs("mMiddle").value = defaults.m_middle;
  qs("mLower").value = defaults.m_lower_arm;
  qs("mTip").value = defaults.m_tip;
  qs("nPendulums").value = defaults.n_pendulums;
  qs("capexMech").value = 800000;
  qs("capexElec").value = 220000;
  qs("capexCivil").value = 150000;
  qs("capexSoft").value = 50000;
  qs("capexCont").value = 8;
  qs("projectLife").value = 20;
  qs("wacc").value = 8;
  qs("fixedOm").value = 10000;
  qs("variableOm").value = 0.0;
  qs("smoothModules").value = 4;
  qs("phaseOffset").value = 30;
  qs("smoothInverter").value = defaults.inverter_rating_kw;
  qs("smoothBasePower").value = 120;
  qs("siteModules").value = 12;
  qs("moduleSpacing").value = 200;
  qs("moduleDensity").value = "";
  qs("landCost").value = 8;

  const presetSelect = qs("weibullPreset");
  presetSelect.innerHTML = data.weibull_presets
    .map((preset, idx) => `<option value="${idx}">${preset.name}</option>`)
    .join("");
  presetSelect.dataset.presets = JSON.stringify(data.weibull_presets);
};

const getAeroParams = () => ({
  swept_area_m2: Number(qs("sweptArea").value),
  cp: Number(qs("cp").value),
  rho: Number(qs("rho").value),
  drivetrain_eff: Number(qs("drivetrainEff").value),
  availability: Number(qs("availability").value),
  inverter_rating_kw: Number(qs("inverterRating").value),
});

const getSimParams = () => ({
  l1: Number(qs("l1").value),
  l2: Number(qs("l2").value),
  m_upper_arm: Number(qs("mUpper").value),
  m_middle: Number(qs("mMiddle").value),
  m_lower_arm: Number(qs("mLower").value),
  m_tip: Number(qs("mTip").value),
  n_pendulums: Number(qs("nPendulums").value),
});

const renderCurve = () => {
  const speeds = state.curve.map((row) => row[0]);
  const power = state.curve.map((row) => row[1]);
  Plotly.newPlot(
    "curvePlot",
    [{ x: speeds, y: power, type: "scatter", mode: "lines+markers", name: "Power (kW)" }],
    { title: "Power Curve", xaxis: { title: "Wind speed (m/s)" }, yaxis: { title: "Power (kW)" } }
  );
};

const runSim = async () => {
  const payload = { wind_speed: Number(qs("simWind").value), ...getSimParams() };
  const data = await fetchJson("/api/simulate", { method: "POST", body: JSON.stringify(payload) });
  const times = data.frames.map((f) => f.t);
  const power = data.frames.map((f) => f.power_kw);
  Plotly.newPlot(
    "simPlot",
    [{ x: times, y: power, type: "scatter", mode: "lines", name: "Power (kW)" }],
    { title: "Simulated Power", xaxis: { title: "Time (s)" }, yaxis: { title: "Power (kW)" } }
  );
};

const generateCurve = async () => {
  const payload = {
    mode: qs("energyModel").value === "sim" ? "sim" : "aero",
    speeds: Array.from({ length: 26 }, (_, i) => i),
    aero_params: getAeroParams(),
    sim_params: getSimParams(),
  };
  const data = await fetchJson("/api/power_curve", { method: "POST", body: JSON.stringify(payload) });
  state.curve = data.curve;
  renderCurve();
};

const calcAep = async () => {
  if (!state.curve.length) {
    await generateCurve();
  }
  const payload = {
    curve: state.curve,
    wind_mode: state.histogram ? "histogram" : "weibull",
    histogram: state.histogram,
    weibull_k: Number(qs("weibullK").value),
    weibull_c: Number(qs("weibullC").value),
  };
  const data = await fetchJson("/api/aep", { method: "POST", body: JSON.stringify(payload) });
  state.aepKwh = data.aep_kwh;
  qs("aepResult").innerHTML = `<strong>AEP:</strong> ${data.aep_kwh.toFixed(0)} kWh/year`;
};

const calcLcoe = async () => {
  if (!state.aepKwh) {
    await calcAep();
  }
  const payload = {
    mechanical_usd: Number(qs("capexMech").value),
    electrical_usd: Number(qs("capexElec").value),
    civil_bos_usd: Number(qs("capexCivil").value),
    soft_costs_usd: Number(qs("capexSoft").value),
    contingency_pct: Number(qs("capexCont").value),
    aep_kwh: state.aepKwh,
    project_life_years: Number(qs("projectLife").value),
    wacc_pct: Number(qs("wacc").value),
    fixed_om_usd_per_year: Number(qs("fixedOm").value),
    variable_om_usd_per_kwh: Number(qs("variableOm").value),
    land_area_m2: Number(qs("moduleSpacing").value) * Number(qs("siteModules").value),
  };
  const data = await fetchJson("/api/lcoe", { method: "POST", body: JSON.stringify(payload) });
  tableFromObject(qs("lcoeTable"), data.lcoe);
  tableFromRows(qs("investorTable"), data.investor_scenarios);
};

const calcSmoothing = async () => {
  const payload = {
    number_of_modules: Number(qs("smoothModules").value),
    phase_offset_deg: Number(qs("phaseOffset").value),
    inverter_rating_kw: Number(qs("smoothInverter").value),
    base_power_kw: Number(qs("smoothBasePower").value),
  };
  const data = await fetchJson("/api/smoothing", { method: "POST", body: JSON.stringify(payload) });
  Plotly.newPlot(
    "smoothingPlot",
    [
      {
        x: data.clipping_loss_pct,
        y: data.smoothing_index,
        type: "scatter",
        mode: "lines+markers",
      },
    ],
    {
      title: "Smoothing index vs clipping loss (%)",
      xaxis: { title: "Clipping loss (%)" },
      yaxis: { title: "Smoothing index" },
    }
  );
};

const calcSite = async () => {
  if (!state.aepKwh) {
    await calcAep();
  }
  const payload = {
    modules_count: Number(qs("siteModules").value),
    module_spacing_m2: qs("moduleSpacing").value ? Number(qs("moduleSpacing").value) : null,
    density_modules_per_ha: qs("moduleDensity").value ? Number(qs("moduleDensity").value) : null,
    land_cost_usd_per_m2: Number(qs("landCost").value),
    module_aep_kwh: state.aepKwh,
    capex_inputs: {
      mechanical_usd: Number(qs("capexMech").value),
      electrical_usd: Number(qs("capexElec").value),
      civil_bos_usd: Number(qs("capexCivil").value),
      soft_costs_usd: Number(qs("capexSoft").value),
      contingency_pct: Number(qs("capexCont").value),
    },
    project_life_years: Number(qs("projectLife").value),
    wacc_pct: Number(qs("wacc").value),
    fixed_om_usd_per_year: Number(qs("fixedOm").value),
    variable_om_usd_per_kwh: Number(qs("variableOm").value),
  };
  const data = await fetchJson("/api/site_rollup", { method: "POST", body: JSON.stringify(payload) });
  qs("siteResults").innerHTML = `
    <ul>
      <li>Total capex: $${data.total_capex_usd.toFixed(0)}</li>
      <li>Total AEP: ${data.total_aep_kwh.toFixed(0)} kWh</li>
      <li>Total avg kW: ${data.total_avg_kw.toFixed(2)} kW</li>
      <li>Land area: ${data.land_area_m2.toFixed(1)} m² (${data.land_area_ha.toFixed(2)} ha)</li>
      <li>LCOE: $${data.lcoe.lcoe.toFixed(4)} / kWh</li>
    </ul>
  `;
};

const bindActions = () => {
  qs("runSim").addEventListener("click", runSim);
  qs("generateCurve").addEventListener("click", generateCurve);
  qs("calcAep").addEventListener("click", calcAep);
  qs("calcLcoe").addEventListener("click", calcLcoe);
  qs("calcSmoothing").addEventListener("click", calcSmoothing);
  qs("calcSite").addEventListener("click", calcSite);

  qs("exportCurveCsv").addEventListener("click", () => {
    const rows = ["speed_mps,power_kw", ...state.curve.map((row) => row.join(","))].join("\n");
    downloadCsv("power_curve.csv", rows);
  });

  qs("exportLcoeCsv").addEventListener("click", () => {
    const table = qs("lcoeTable");
    const rows = [...table.querySelectorAll("tr")].map((tr) =>
      [...tr.querySelectorAll("td,th")].map((td) => td.textContent).join(",")
    );
    downloadCsv("lcoe_table.csv", rows.join("\n"));
  });

  qs("applyPreset").addEventListener("click", () => {
    const presets = JSON.parse(qs("weibullPreset").dataset.presets || "[]");
    const preset = presets[Number(qs("weibullPreset").value)];
    if (preset) {
      qs("weibullK").value = preset.k;
      qs("weibullC").value = preset.c;
    }
  });

  qs("parseHistogram").addEventListener("click", async () => {
    const csvText = qs("histogramCsv").value;
    const data = await fetchJson("/api/histogram_parse", {
      method: "POST",
      body: JSON.stringify({ csv_text: csvText }),
    });
    state.histogram = data.histogram;
    qs("histogramPreview").textContent = JSON.stringify(data.histogram, null, 2);
  });
};

document.addEventListener("DOMContentLoaded", async () => {
  bindTabs();
  await loadDefaults();
  bindActions();
  await generateCurve();
});
