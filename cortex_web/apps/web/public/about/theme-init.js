(function applyStoredTheme() {
  try {
    var stored = localStorage.getItem("cortex-theme");
    if (stored === "light" || stored === "dark") {
      document.documentElement.setAttribute("data-theme", stored);
      return;
    }
  } catch (_) {
    // Storage can be unavailable in private contexts; keep the light default.
  }
  document.documentElement.setAttribute("data-theme", "light");
}());
