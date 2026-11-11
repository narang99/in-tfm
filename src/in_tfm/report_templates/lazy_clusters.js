// Cluster bodies ship inside <template>, so the browser parses them without building their DOM.
// A big report holds hundreds of clusters of token spans, and building them all is what makes it slow.
// A hosted report keeps each body in its own file instead, named by the template's data-src.
// The page then downloads only the clusters someone opens. See `in_tfm.hosted_report`.
//
// Clusters start collapsed, except the strongest: the OPEN_PER_SIGN highest median activations on each side.
// A report without medians (data-median) opens its first OPEN_PER_SIGN clusters, which are its largest.
// An open cluster's body is stamped once it comes near the viewport. A header click opens or closes it.
const OPEN_PER_SIGN = 10;

async function bodyOf(template) {
  if (!template.dataset.src) return template.content;
  const response = await fetch(template.dataset.src);
  if (!response.ok) throw new Error(`${response.status} for ${template.dataset.src}`);
  const holder = document.createElement("template");
  holder.innerHTML = await response.text();
  return holder.content;
}

async function stampCluster(section) {
  const template = section.querySelector("template.lazy-body");
  if (!template || template.dataset.loading) return;
  template.dataset.loading = "true";
  nearView.unobserve(section);
  const body = template.parentElement;
  try {
    template.replaceWith(await bodyOf(template));
  } catch (error) {
    body.textContent = `Could not load this cluster: ${error.message}`;
  }
  body.classList.remove("pending");
}

const nearView = new IntersectionObserver(
  (entries) => entries.filter((e) => e.isIntersecting).forEach((e) => stampCluster(e.target)),
  { rootMargin: "1500px 0px" }
);

function strongestClusters(sections) {
  const withMedian = sections.filter((s) => s.dataset.median !== undefined);
  if (withMedian.length === 0) return sections.slice(0, OPEN_PER_SIGN);
  const median = (s) => Number(s.dataset.median);
  const positive = withMedian.filter((s) => median(s) > 0).sort((a, b) => median(b) - median(a));
  const negative = withMedian.filter((s) => median(s) < 0).sort((a, b) => median(a) - median(b));
  return [...positive.slice(0, OPEN_PER_SIGN), ...negative.slice(0, OPEN_PER_SIGN)];
}

function openCluster(section) {
  section.classList.add("open");
  nearView.observe(section);
}

function toggleCluster(section) {
  if (section.classList.toggle("open")) stampCluster(section);
}

// A link to a cluster opens it, waits for its body, then scrolls to it.
// The scroll is instant, since the page's smooth scrolling would still be moving when bodies above it land.
async function showLinkedCluster() {
  const target = location.hash && document.getElementById(location.hash.slice(1));
  if (!target || !target.matches("section.cluster")) return;
  target.classList.add("open");
  await stampCluster(target);
  target.scrollIntoView({ behavior: "instant" });
}

async function start() {
  const sections = [...document.querySelectorAll("section.cluster")];
  sections.forEach((section) =>
    section.querySelector(".cluster-head").addEventListener("click", () => toggleCluster(section))
  );
  await showLinkedCluster();
  strongestClusters(sections).forEach(openCluster);
}
window.addEventListener("hashchange", showLinkedCluster);
start();
