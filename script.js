document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }

  const body = document.body;
  const menuButton = document.querySelector(".menu-toggle");
  const nav = document.querySelector(".site-nav");

  const setMenuOpen = (open) => {
    body.classList.toggle("menu-open", open);
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    menuButton.innerHTML = open ? '<i data-lucide="x"></i>' : '<i data-lucide="menu"></i>';
    if (window.lucide) {
      window.lucide.createIcons();
    }
  };

  menuButton.addEventListener("click", () => {
    setMenuOpen(!body.classList.contains("menu-open"));
  });

  nav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => setMenuOpen(false));
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 820 && body.classList.contains("menu-open")) {
      setMenuOpen(false);
    }
  });

  const revealElements = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const revealObserver = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            const comparisonBars = entry.target.querySelector(".comparison-bars");
            if (comparisonBars) comparisonBars.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.14 }
    );
    revealElements.forEach((element) => revealObserver.observe(element));
  } else {
    revealElements.forEach((element) => element.classList.add("is-visible"));
  }

  const sections = Array.from(document.querySelectorAll("main section[id]"));
  const navLinks = Array.from(nav.querySelectorAll("a"));
  if ("IntersectionObserver" in window) {
    const sectionObserver = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        navLinks.forEach((link) => {
          link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
        });
      },
      { rootMargin: "-25% 0px -60% 0px", threshold: [0.05, 0.2, 0.5] }
    );
    sections.forEach((section) => sectionObserver.observe(section));
  }

  const copyButton = document.querySelector(".copy-button");
  const bibtex = document.querySelector("#bibtex");

  copyButton.addEventListener("click", async () => {
    const text = bibtex.textContent.trim();
    try {
      await navigator.clipboard.writeText(text);
    } catch (_error) {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }

    copyButton.innerHTML = '<i data-lucide="check"></i><span>Copied</span>';
    if (window.lucide) window.lucide.createIcons();
    window.setTimeout(() => {
      copyButton.innerHTML = '<i data-lucide="copy"></i><span>Copy</span>';
      if (window.lucide) window.lucide.createIcons();
    }, 1800);
  });

  document.querySelector("#year").textContent = String(new Date().getFullYear());
});
