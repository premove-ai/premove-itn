const isItnRoute = () =>
  window.location.pathname === "/itn" || window.location.pathname.startsWith("/itn/");

const updateBaseLinks = () => {
  if (isItnRoute()) return;

  document.querySelectorAll('#content a[href^="/itn"]').forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
};

const updateNavbarBrand = () => {
  const isItn = isItnRoute();
  const label = isItn ? "Premove ITN" : "Premove";
  const githubHref = isItn
    ? "https://github.com/premove-ai/premove-itn"
    : "https://github.com/premove-ai";

  document.body.classList.toggle("base-route", !isItn);

  document.querySelectorAll("#navbar .nav-logo").forEach((logo) => {
    if (logo.textContent !== label) logo.textContent = label;
  });

  document.querySelectorAll('#navbar a[href="/"] .sr-only').forEach((labelNode) => {
    const accessibleLabel = `${label} home page`;
    if (labelNode.textContent !== accessibleLabel) labelNode.textContent = accessibleLabel;
  });

  document.querySelectorAll('#navbar a[href^="https://github.com/"]').forEach((link) => {
    if (link.href !== githubHref) link.href = githubHref;
  });

  updateBaseLinks();
  updateThemeToggle();
};

const updateThemeToggle = () => {
  let trigger = document.querySelector("#theme-preference-menu-trigger");
  if (!trigger) return;

  if (trigger.dataset.premoveThemeToggle !== "true") {
    const replacement = trigger.cloneNode(false);
    replacement.removeAttribute("aria-haspopup");
    replacement.removeAttribute("aria-expanded");
    replacement.removeAttribute("data-state");
    replacement.dataset.premoveThemeToggle = "true";
    trigger.replaceWith(replacement);
    trigger = replacement;
    trigger.addEventListener("click", () => {
      const root = document.documentElement;
      const nextTheme = root.classList.contains("dark") ? "light" : "dark";
      root.classList.remove("light", "dark");
      root.classList.add(nextTheme);
      root.setAttribute("data-theme-preference", nextTheme);
      root.style.colorScheme = nextTheme;

      try {
        localStorage.setItem("isDarkMode", nextTheme);
      } catch {}

      updateThemeToggle();
    });
  }

  const isDark = document.documentElement.classList.contains("dark");
  trigger.setAttribute("aria-label", isDark ? "Switch to light theme" : "Switch to dark theme");
};

updateNavbarBrand();
window.addEventListener("popstate", updateNavbarBrand);
new MutationObserver(updateNavbarBrand).observe(document.body, {
  childList: true,
  subtree: true,
});
