/* Set the color theme before first paint to avoid a flash of the wrong theme.
   Kept as a standalone same-origin script (NOT inline) so the Content-
   Security-Policy can forbid inline scripts entirely — script-src 'self'
   with no 'unsafe-inline'. Loaded render-blocking from <head> so it still
   runs before the body paints. */
(function () {
  try {
    var t = localStorage.getItem("cortex-theme");
    if (t === "dark" || t === "light") {
      document.documentElement.setAttribute("data-theme", t);
    }
  } catch (e) { /* private mode / storage disabled — keep the default theme */ }
})();
