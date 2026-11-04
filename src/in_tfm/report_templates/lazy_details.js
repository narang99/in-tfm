// A cluster bar's body ships inside <template>, so a page of many bars builds no hit DOM up front.
// The body is stamped in the first time its bar is opened.
document.addEventListener(
  "toggle",
  (event) => {
    const bar = event.target;
    if (!bar.open || !bar.matches("details.fire-row")) return;
    const template = bar.querySelector(":scope > .fire-body > template.lazy-body");
    if (template) template.replaceWith(template.content);
  },
  true
);
