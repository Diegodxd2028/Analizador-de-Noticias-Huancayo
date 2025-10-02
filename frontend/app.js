// Atajo para querySelector
const $ = (q) => document.querySelector(q);

const API = "http://127.0.0.1:8000"; // Backend FastAPI

const btn = $("#btn");
const card = $("#card");
const raw = $("#raw");
const scoreEl = $("#score");
const badge = $("#badge");
const warn = $("#warn");
const terms = $("#terms");

// Función para mostrar/ocultar
const show = (el) => el.classList.remove("hidden");
const hide = (el) => el.classList.add("hidden");

// Al hacer click en "Analizar"
btn.onclick = async () => {
  const text = $("#txt").value.trim();
  const url = $("#url").value.trim() || null;

  hide(card);
  terms.textContent = "";
  hide(warn);

  btn.disabled = true;
  btn.textContent = "Analizando…";

  try {
    if (!url && (!text || text.length < 30)) {
      throw new Error("Escribe al menos 30 caracteres o ingresa una URL.");
    }

    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text || null, url })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    console.log("Respuesta /predict:", data);

    raw.textContent = JSON.stringify(data, null, 2);
    scoreEl.textContent =
      `Confianza: ${(data.score * 100).toFixed(1)}%` +
      (data.pattern ? ` • Patrón: ${data.pattern}` : "");

    badge.textContent = data.label;
    badge.className =
      data.label.toLowerCase() === "fake"
        ? "text-sm px-2 py-1 rounded bg-red-100 text-red-700"
        : "text-sm px-2 py-1 rounded bg-green-100 text-green-700";

    if (data.abstain) {
      show(warn);
    }

    // Explicación con /explain
    try {
      const ex = await fetch(`${API}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text || null, url })
      });
      if (ex.ok) {
        const ejson = await ex.json();
        console.log("Respuesta /explain:", ejson);
        const list = (ejson.top_terms || [])
          .map((t) => t.term)
          .slice(0, 8)
          .join(", ");
        terms.textContent = list
          ? `Términos influyentes: ${list}`
          : "Sin términos destacados.";
      }
    } catch (exErr) {
      console.warn("Error en /explain:", exErr);
    }

    show(card);
  } catch (e) {
    alert("Error: " + e.message);
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analizar";
  }
};

// Test rápido al cargar la página
fetch(`${API}/`)
  .then((r) => r.json())
  .then((j) => console.log("API OK:", j))
  .catch((err) => console.error("API no accesible:", err));

  // al finalizar con éxito:
window.postMessage("analysis:done", "*");
// en catch:
window.postMessage("analysis:error", "*");
// en finally no hace falta, ya mandaste uno u otro
