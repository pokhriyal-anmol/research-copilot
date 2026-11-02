const out = document.getElementById("answer"), sources = document.getElementById("sources");
document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  out.textContent = "Searching...";
  sources.replaceChildren();
  try {
    const res = await fetch("/answer", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: document.getElementById("q").value}),
    });
    const data = await res.json();
    if (!res.ok) { out.textContent = typeof data.detail === "string" ? data.detail : "Please check your question."; return; }
    out.textContent = data.text;
    for (const c of data.citations) {
      const a = Object.assign(document.createElement("a"), {href: c.url, textContent: `[${c.number}] ${c.title}, page ${c.page}`});
      const li = document.createElement("li");
      li.append(a);
      sources.append(li);
    }
  } catch { out.textContent = "The request failed. Please retry."; }
});
