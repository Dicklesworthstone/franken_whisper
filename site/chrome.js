// Minimal site chrome (header glass + scroll reveals), from the reference site's main.js.
const header = document.getElementById("site-header");
if (header) {
  const apply = () => {
    const scrolled = window.scrollY > 20;
    header.classList.toggle("glass-modern", scrolled);
    header.classList.toggle("shadow-2xl", scrolled);
    header.classList.toggle("border-emerald-500/20", scrolled);
    header.classList.toggle("border-transparent", !scrolled);
  };
  window.addEventListener("scroll", () => requestAnimationFrame(apply), { passive: true });
  apply();
}
const revealables = document.querySelectorAll(".reveal");
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      entry.target.classList.add("revealed");
      observer.unobserve(entry.target);
    }
  }
}, { threshold: 0.08 });
for (const el of revealables) observer.observe(el);
// Instant jumps (scrollbar drags, the End key, anchor links on browsers that skip
// smooth scrolling) teleport past sections without ever intersecting them, which
// left every skipped block at opacity 0. Reveal anything at or above the viewport.
let revealSweepQueued = false;
const revealSweep = () => {
  revealSweepQueued = false;
  for (const el of revealables) {
    if (!el.classList.contains("revealed") && el.getBoundingClientRect().top < window.innerHeight + 200) {
      el.classList.add("revealed");
      observer.unobserve(el);
    }
  }
};
window.addEventListener("scroll", () => {
  if (!revealSweepQueued) { revealSweepQueued = true; requestAnimationFrame(revealSweep); }
}, { passive: true });
// Deep links (#wasm etc.) position the page before this script runs, without any
// scroll event afterwards; sweep once at load to cover that case.
revealSweep();
