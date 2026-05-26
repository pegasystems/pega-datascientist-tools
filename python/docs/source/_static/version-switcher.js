// Populate the sidebar version switcher from a sibling versions.json.
//
// versions.json lives at the gh-pages root; from any built version the
// relative path is "../versions.json".  When absent (e.g. local builds),
// the dropdown silently degrades to a single static entry.
(function () {
  "use strict";

  function init() {
    const container = document.querySelector(".version-switcher");
    if (!container) return;
    const select = container.querySelector("select");
    if (!select) return;

    const current = container.dataset.currentVersion || "";

    fetch("../versions.json", { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((versions) => {
        if (!Array.isArray(versions) || versions.length === 0) return;
        select.innerHTML = "";
        for (const entry of versions) {
          const opt = document.createElement("option");
          opt.value = entry.url;
          opt.textContent = entry.preferred
            ? `${entry.version} (stable)`
            : entry.version;
          if (entry.version === current) opt.selected = true;
          select.appendChild(opt);
        }
        select.addEventListener("change", (e) => {
          const url = e.target.value;
          if (url) window.location.href = url;
        });
      })
      .catch(() => {
        // No manifest — leave the static placeholder in place.
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
