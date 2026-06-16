// Populate the sidebar version switcher from versions.json at the site root.
(function () {
  "use strict";

  function getManifestUrl(currentVersion) {
    if (!currentVersion) return "versions.json";

    const marker = `/${currentVersion}/`;
    const { origin, pathname } = window.location;
    const index = pathname.indexOf(marker);

    if (index === -1) {
      return "versions.json";
    }

    const siteRoot = pathname.slice(0, index + 1);
    return `${origin}${siteRoot}versions.json`;
  }

  function init() {
    const container = document.querySelector(".version-switcher");
    if (!container) return;

    const select = container.querySelector("select");
    if (!select) return;

    const currentVersion = container.dataset.currentVersion || "";
    const manifestUrl = getManifestUrl(currentVersion);

    fetch(manifestUrl, { cache: "no-cache" })
      .then((response) => (response.ok ? response.json() : Promise.reject(response.status)))
      .then((versions) => {
        if (!Array.isArray(versions) || versions.length === 0) return;

        select.innerHTML = "";
        for (const entry of versions) {
          const option = document.createElement("option");
          option.value = entry.url;
          option.textContent = entry.preferred
            ? `${entry.version} (stable)`
            : entry.version;
          option.selected = entry.version === currentVersion;
          select.appendChild(option);
        }

        if (select.selectedIndex === -1) {
          select.selectedIndex = 0;
        }

        select.addEventListener("change", (event) => {
          const url = event.target.value;
          if (url) {
            window.location.href = url;
          }
        });
      })
      .catch(() => {
        // No manifest available locally - leave the static placeholder.
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
