const hideSink = document.getElementById("hide-sink");
const syncSink = () => document.body.classList.toggle("hide-sink", hideSink.checked);
hideSink.addEventListener("change", syncSink);
// A browser can restore the checkbox state on reload without firing `change`.
window.addEventListener("pageshow", syncSink);
syncSink();
