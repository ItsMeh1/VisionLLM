const status = document.querySelector("#copy-status");

function announce(message) {
  if (status) status.textContent = message;
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  const copied = document.execCommand("copy");
  area.remove();
  if (!copied) throw new Error("Copy command was unavailable");
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copy);
    if (!target) return;

    const text = target.innerText.trim();
    const original = button.textContent;

    try {
      await copyText(text);
      button.textContent = "Copied";
      announce("Quickstart commands copied to the clipboard.");
    } catch {
      button.textContent = "Select it";
      announce("Copy failed. Select the commands in the terminal block.");
    }

    window.setTimeout(() => {
      button.textContent = original;
    }, 1800);
  });
});
