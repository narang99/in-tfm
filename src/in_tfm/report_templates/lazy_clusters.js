// Cluster bodies ship inside <template>, so the browser parses them without building their DOM.
// A big report holds hundreds of clusters of token spans, and building them all is what makes it slow.
// Each body is stamped into the page once its section comes near the viewport.
function stampCluster(section) {
  const template = section.querySelector("template.lazy-body");
  if (!template) return;
  template.parentElement.classList.remove("pending");
  template.replaceWith(template.content);
  nearView.unobserve(section);
}

const nearView = new IntersectionObserver(
  (entries) => entries.filter((e) => e.isIntersecting).forEach((e) => stampCluster(e.target)),
  { rootMargin: "1500px 0px" }
);
document.querySelectorAll("section.cluster").forEach((section) => nearView.observe(section));

// A link to a cluster lands on its unstamped placeholder, so stamp it and scroll again.
function showLinkedCluster() {
  const target = location.hash && document.getElementById(location.hash.slice(1));
  if (!target || !target.matches("section.cluster")) return;
  stampCluster(target);
  target.scrollIntoView();
}
window.addEventListener("hashchange", showLinkedCluster);
showLinkedCluster();
