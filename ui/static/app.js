// Mission control: the rig live, and a run from upload to verdict. Reads the server's JSON and event stream only.
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined) el.append(kid.nodeType ? kid : String(kid));
    return el;
  };
  const ICON = {
    check: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    cross: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    eye: '<svg viewBox="0 0 24 24"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
    layers: '<svg viewBox="0 0 24 24"><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
    route: '<svg viewBox="0 0 24 24"><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="6" r="2.5"/><path d="M8.5 18H15a3 3 0 000-6H9a3 3 0 010-6h6.5"/></svg>',
    arm: '<svg viewBox="0 0 24 24"><path d="M4 20h7M7 20v-5l6-6M13 9l3-3M16 6l3 1M16 6l1-3"/></svg>',
    gavel: '<svg viewBox="0 0 24 24"><path d="M14 4l6 6M11 7l6 6M8.5 9.5l6 6M3 21l7-7"/></svg>',
    alert: '<svg viewBox="0 0 24 24"><path d="M12 8v5M12 16.5v.5"/><path d="M10.3 3.9L2.4 18a2 2 0 001.7 3h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/></svg>',
    dot: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/></svg>',
  };
  const STAGES = [
    { id: "watch", name: "Watch", idle: "Muse reads the video", icon: ICON.eye },
    { id: "segment", name: "Segment", idle: "SAM 3.1 masks", icon: ICON.layers },
    { id: "plan", name: "Plan", idle: "The arms' task", icon: ICON.route },
    { id: "act", name: "Act", idle: "Muse drives the arms", icon: ICON.arm },
    { id: "judge", name: "Judge", idle: "Did it work?", icon: ICON.gavel },
  ];

  // ---- small helpers -------------------------------------------------------------------------------------------------
  const lottieIn = (el, small = false) => {
    if (window.lottie) {
      try {
        const anim = window.lottie.loadAnimation({ container: el, renderer: "svg", loop: true, autoplay: true,
          path: "/static/scan.json" });
        return { destroy() { anim.destroy(); el.replaceChildren(); } };
      } catch (e) { /* fall through to the CSS rings */ }
    }
    const rings = h("div", { class: small ? "css-pulse small" : "css-pulse" });  // the same rings, without Lottie
    el.append(rings);
    return { destroy() { rings.remove(); } };
  };
  const fmtBytes = (n) => n > 1e9 ? (n / 1e9).toFixed(1) + " GB" : n > 1e6 ? (n / 1e6).toFixed(1) + " MB" : Math.round(n / 1e3) + " KB";
  const fmtTime = (s) => s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${(+s).toFixed(1)}s`;
  const ago = (t) => {
    const s = Date.now() / 1000 - t;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${Math.round(s / 3600)} h ago`;
    return new Date(t * 1000).toLocaleDateString();
  };
  const callLabel = (name, a) => {
    if (!a || typeof a !== "object") return name;
    const xy = (v) => Array.isArray(v) ? `(${v.map((n) => Math.round(n * 10) / 10).join(", ")})` : v;
    if (name === "handover") return `handover · ${a.giver} → ${a.taker}`;
    const bits = [a.arm, a.target, a.xy && xy(a.xy), a.on && `on ${a.on}`, a.arms, a.cams].filter(Boolean);
    return [name, ...bits].join(" · ");
  };
  const typeInto = (el, text, live, done) => {
    if (!live || !text) { el.textContent = text || ""; if (done) done(); return; }
    el.textContent = "";
    const caret = h("span", { class: "caret" });
    const node = document.createTextNode("");
    el.append(node, caret);
    let i = 0;
    const step = Math.max(1, Math.round(text.length / 90));
    const tick = () => {
      i = Math.min(text.length, i + step);
      node.data = text.slice(0, i);
      if (i < text.length) setTimeout(tick, 14);
      else { caret.remove(); if (done) done(); }
    };
    tick();
  };

  // ---- views, panel ----------------------------------------------------------------------------------------------------
  const setView = (v) => {
    document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
    document.querySelectorAll(".view").forEach((s) => s.classList.toggle("on", s.id === `view-${v}`));
    if (v === "new") loadRecent();
  };
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  const setPanel = (open) => {
    document.body.classList.toggle("panel-open", open);
    $("#panelToggle").classList.toggle("active", open);
  };
  $("#panelToggle").addEventListener("click", () => setPanel(!document.body.classList.contains("panel-open")));
  $("#panelClose").addEventListener("click", () => setPanel(false));
  $("#themeToggle").addEventListener("click", () => {
    const light = document.documentElement.dataset.theme !== "light";
    if (light) document.documentElement.dataset.theme = "light";
    else delete document.documentElement.dataset.theme;
    try { localStorage.setItem("theme", light ? "light" : "dark"); } catch (e) { /* private window: not kept */ }
  });
  const lightbox = $("#lightbox");
  const zoom = (src) => { $("img", lightbox).src = src; lightbox.hidden = false; };
  lightbox.addEventListener("click", () => { lightbox.hidden = true; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") lightbox.hidden = true; });

  // ---- the stepper -----------------------------------------------------------------------------------------------------
  const steps = $("#steps");
  const stepEls = {};
  for (const s of STAGES) {
    const li = h("li", { class: "step", "data-stage": s.id },
      h("div", { class: "step-dot", html: s.icon }),
      h("div", { class: "step-text" }, h("div", { class: "step-name" }, s.name), h("div", { class: "step-sub" }, s.idle)));
    steps.append(li);
    stepEls[s.id] = li;
  }
  const stepState = (id, state, sub) => {
    const li = stepEls[id];
    if (!li) return;
    li.classList.remove("running", "done", "failed");
    if (state) li.classList.add(state);
    const dot = $(".step-dot", li);
    const s = STAGES.find((x) => x.id === id);
    dot.innerHTML = state === "done" ? ICON.check : state === "failed" ? ICON.cross : s.icon;
    if (sub !== undefined) $(".step-sub", li).textContent = sub;
  };
  const resetSteps = () => STAGES.forEach((s) => stepState(s.id, null, s.idle));

  // ---- status, arms ----------------------------------------------------------------------------------------------------
  const pill = (cls, text) => h("span", { class: `pill ${cls}` }, text);
  let lastState = null;
  const renderState = (st) => {
    lastState = st;
    const bar = $("#status");
    bar.replaceChildren();
    if (!st.robotd) bar.append(pill("bad", "robotd offline"));
    else {
      bar.append(pill("", st.sim ? "Simulator" : "Real arms"));
      bar.append(pill(st.armed ? "ok" : "bad", st.armed ? "Armed" : "Torque off"));
      if (st.episode && st.episode.recording) bar.append(pill("rec", `Recording ${st.episode.episode}`));
      else if (st.skill) bar.append(pill("run", st.skill));
    }
    const arms = $("#arms");
    arms.replaceChildren();
    for (const side of ["left", "right"]) {
      const a = (st.arms || {})[side];
      const color = side === "left" ? "#ff2dfd" : "#00e5ff";
      const head = h("div", { class: "arm-head" }, h("span", { class: "arm-swatch", style: `background:${color}` }),
        `${side[0].toUpperCase()}${side.slice(1)} arm`, h("span", { class: "sub" }, a ? a.mode : "offline"));
      const xyz = a ? `x ${a.tip_cm[0].toFixed(1)}  y ${a.tip_cm[1].toFixed(1)}  z ${a.tip_cm[2].toFixed(1)} cm` : "—";
      const hold = a ? (a.holding ? `Holding ${a.object || "an object"}` : `Open · ${Math.round(a.width_mm)} mm`) : "";
      arms.append(h("div", { class: "arm" }, head, h("div", { class: "arm-xyz" }, xyz), h("div", { class: "arm-hold" }, hold)));
    }
    const hint = $("#hint");
    if (!st.robotd) { hint.textContent = "robotd isn't running: python -m harness.robotd --sim"; hint.classList.add("bad"); }
    else if (st.busy) { hint.textContent = "A run is going: wait for it, or stop it in the session panel."; hint.classList.add("bad"); }
    else { hint.textContent = st.sim ? "Runs on the simulator" : "Runs on the real arms"; hint.classList.remove("bad"); }
    updateStart();
  };
  const pollState = async () => {
    try { renderState(await (await fetch("/api/state")).json()); }
    catch (e) { renderState({ robotd: false }); }
    setTimeout(pollState, 1000);
  };

  // ---- the composer ----------------------------------------------------------------------------------------------------
  let file = null;
  const fileInput = $("#file"), drop = $("#drop"), task = $("#task"), startBtn = $("#startBtn");
  const updateStart = () => { startBtn.disabled = !file || !(lastState && lastState.robotd && !lastState.busy); };
  const pick = (f) => {
    if (!f) return;
    file = f;
    $(".drop-empty", drop).hidden = true;
    $(".drop-file", drop).hidden = false;
    $("#fileName").textContent = f.name;
    const thumb = $("#thumb");
    thumb.src = URL.createObjectURL(f);
    thumb.onloadedmetadata = () => {
      $("#fileMeta").textContent = `${fmtBytes(f.size)} · ${fmtTime(thumb.duration || 0)}`;
      thumb.currentTime = Math.min(1, (thumb.duration || 0) / 3);
    };
    $("#fileMeta").textContent = fmtBytes(f.size);
    updateStart();
  };
  fileInput.addEventListener("change", () => pick(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));
  task.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#composer").requestSubmit(); });

  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    if (startBtn.disabled) return;
    const bar = $("#upProgress");
    bar.hidden = false;
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/jobs?name=${encodeURIComponent(file.name)}&task=${encodeURIComponent(task.value.trim())}`);
    xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) $("div", bar).style.width = `${(ev.loaded / ev.total) * 100}%`; };
    xhr.onload = () => {
      bar.hidden = true;
      $("div", bar).style.width = "0";
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (err) { /* empty */ }
      if (xhr.status !== 201) { $("#hint").textContent = body.error || "Couldn't start the run"; $("#hint").classList.add("bad"); return; }
      startBtn.disabled = true;
      openJob(body.id, true);
      setView("live");
      setPanel(true);
    };
    xhr.onerror = () => { bar.hidden = true; $("#hint").textContent = "The upload failed"; };
    xhr.send(file);
  });

  const loadRecent = async () => {
    const box = $("#recent");
    let list = [];
    try { list = await (await fetch("/api/jobs")).json(); } catch (e) { /* offline */ }
    box.replaceChildren();
    if (!list.length) return;
    box.append(h("h2", {}, "Recent runs"));
    for (const j of list.slice(0, 6)) {
      const icon = { done: "✓", failed: "✕", stopped: "■", running: "●" }[j.status] || "·";
      box.append(h("button", { class: "recent-item", onclick: () => { openJob(j.id, false); setView("live"); setPanel(true); } },
        h("span", {}, icon), h("span", { class: "r-task" }, j.task || j.video), h("span", { class: "r-when" }, ago(j.created))));
    }
  };

  // ---- a run: its events into the panel and the main page -------------------------------------------------------------
  let source = null, run = null;
  const chat = $("#chat");
  const scrollChat = () => { chat.scrollTop = chat.scrollHeight; };
  const think = $("#think"), thinkText = $("#thinkText"), thinkState = $("#thinkState"), thinkCalls = $("#thinkCalls");
  const screen = $("#screen"), video = $("#video");

  const setVideo = (src, mode) => {
    if (!src) return;
    screen.classList.add("has-video");
    if (video.dataset.src === src) return;
    video.dataset.src = src;
    video.src = src;
    video.play().catch(() => {});
    document.querySelectorAll("#playerTabs button").forEach((b) => b.classList.toggle("on", b.dataset.src === mode));
  };
  document.querySelectorAll("#playerTabs button").forEach((b) => b.addEventListener("click", () => {
    if (!run || b.disabled) return;
    setVideo(b.dataset.src === "segmented" ? run.files.segmented : run.files.original, b.dataset.src);
  }));
  const scanning = (on) => {
    if (on && !run.screenAnim) run.screenAnim = lottieIn($("#screenLottie"));
    if (!on && run.screenAnim) { run.screenAnim.destroy(); run.screenAnim = null; }
    screen.classList.toggle("scanning", on);
  };

  const card = (icon, title, cls = "") => {
    const badge = h("span", { class: `badge-icon ${cls}`, html: icon });
    const c = h("div", { class: "card" }, h("div", { class: "card-head" }, badge, h("span", { class: "title" }, title)));
    chat.append(c);
    scrollChat();
    return c;
  };
  const stageCard = (stage, title) => {
    const c = card(STAGES.find((s) => s.id === stage)?.icon || ICON.dot, title);
    c.classList.add("running");
    const badge = $(".badge-icon", c);
    const mini = h("span", { class: "mini-lottie" });
    badge.replaceWith(mini);
    c.anim = lottieIn(mini, true);
    c.badge = badge;
    c.stage = stage;
    return c;
  };
  const settle = (c, cls = "") => {
    if (!c || !c.classList.contains("running")) return;
    c.classList.remove("running");
    if (c.anim) c.anim.destroy();
    const mini = $(".mini-lottie", c);
    if (mini) { c.badge.className = `badge-icon ${cls}`; mini.replaceWith(c.badge); }
  };
  const logInto = (c, text) => {
    if (!c) return;
    let d = $("details.log", c);
    if (!d) { d = h("details", { class: "log" }, h("summary", {}, "Details"), h("pre", {})); c.append(d); }
    $("pre", d).textContent += text + "\n";
  };

  const thought = (text, live) => {
    think.classList.add("active");
    thinkState.textContent = "Thinking";
    thinkText.classList.remove("idle");
    typeInto(thinkText, text || "…", live);
  };

  const handle = (e, live) => {
    if (!run || e.i <= run.last) return;
    run.last = e.i;
    const k = e.kind;
    if (k === "start") {
      chat.replaceChildren();
      $("#sessionTitle").textContent = e.task || e.name;
      run.files.original = e.video;
      setVideo(e.video, "original");
      $("#playerTabs [data-src=segmented]").disabled = true;
      chat.append(h("div", { class: "msg-user" },
        h("div", { class: "attach", html: `${ICON.eye.replace("<svg", '<svg width="13" height="13"')} ${e.name}` }),
        e.task || "Do what the video shows."));
      resetSteps();
      thinkText.textContent = "Waiting for the video to be read…";
      thinkText.classList.add("idle");
      thinkCalls.replaceChildren();
      think.classList.remove("active");
      thinkState.textContent = "Waiting";
    } else if (k === "stage") {
      STAGES.forEach((s) => {
        if (s.id === e.stage) stepState(s.id, "running", e.text);
        else if (STAGES.findIndex((x) => x.id === s.id) < STAGES.findIndex((x) => x.id === e.stage) &&
          !stepEls[s.id].classList.contains("failed")) stepState(s.id, "done");
      });
      if (run.card && run.card.stage !== e.stage) settle(run.card, "green");
      if (e.stage === "watch" || e.stage === "segment") {
        run.card = stageCard(e.stage, e.text);
        if (live) scanning(true);
      } else {
        scanning(false);
        if (e.stage === "act") { run.card = null; thought("Looking at the table…", live); }
        if (e.stage === "judge") run.card = stageCard("judge", e.text);
      }
      $("#footState").textContent = e.text;
    } else if (k === "log") {
      logInto(run.card, e.text);
    } else if (k === "summary") {
      const body = h("div", { class: "card-body" });
      run.card.append(body);
      typeInto(body, e.text, live);
      stepState("watch", "running", "Muse described the video");
    } else if (k === "objects") {
      run.card.append(h("div", { class: "chips" }, e.names.map((n, i) => h("span", { class: "chip", style: `animation-delay:${i * 80}ms` }, n))));
    } else if (k === "segmenting") {
      let chips = $(".chips", run.card);
      if (!chips) { chips = h("div", { class: "chips" }); run.card.append(chips); }
      chips.append(h("span", { class: "chip" }, h("span", { class: "spin" }), e.concept));
      stepState("segment", "running", `Segmenting “${e.concept}”`);
    } else if (k === "segmented") {
      const c = run.card;
      $(".chips", c)?.remove();
      $(".title", c).textContent = "SAM 3.1 segmented the video";
      c.append(h("div", { class: "chips" }, e.objects.map((o, i) => {
        const bar = h("i", o.color ? { style: `background:${o.color}` } : {});
        setTimeout(() => { bar.style.width = `${Math.round((o.visible || 0) * 100)}%`; }, live ? 120 + i * 90 : 0);
        const tip = `${o.instances > 1 ? `${o.instances} of them, ` : ""}seen in ${Math.round((o.visible || 0) * 100)}% of frames` +
          (o.muse_name ? ` (Muse called it “${o.muse_name}”)` : "");
        return h("span", { class: "chip", style: `animation-delay:${i * 90}ms`, title: tip },
          o.color ? h("span", { class: "dot", style: `background:${o.color}` }) : "",
          o.instances > 1 ? h("b", {}, `${o.instances}×`) : "", o.muse_name ? `${o.muse_name}` : o.name,
          h("span", { class: "vis" }, bar));
      })));
      if (e.pass_seen) c.append(h("div", { class: "card-body muted" }, `Hand-to-hand pass: ${e.giver} → ${e.taker}`));
      settle(c, "green");
      scanning(false);
      run.files.segmented = e.files["preview.mp4"];
      if (e.files["clip.mp4"]) run.files.original = e.files["clip.mp4"];
      if (run.files.segmented) {
        $("#playerTabs [data-src=segmented]").disabled = false;
        setVideo(run.files.segmented, "segmented");
        if (live) { screen.classList.remove("reveal"); void screen.offsetWidth; screen.classList.add("reveal"); }
      }
      stepState("segment", "done", `${e.objects.length} things masked`);
    } else if (k === "plan") {
      settle(run.card, "green");
      const c = card(ICON.route, "The task for the arms", "");
      c.append(h("div", { class: "quote" }, e.task));
      if (e.video_summary) c.append(h("div", { class: "card-body muted" }, `From the video: ${e.video_summary}`));
      if (e.target) c.append(h("div", { class: "chips" }, h("span", { class: "chip" }, `Target (${e.target[0]}, ${e.target[1]}) cm`)));
      run.card = null;
      stepState("plan", "done", e.target ? `Target (${e.target[0]}, ${e.target[1]})` : "From your description");
    } else if (k === "recording") {
      stepState("act", "running", `Recording episode ${e.episode}`);
    } else if (k === "thought") {
      const t = h("div", { class: "turn" }, h("div", { class: "turn-label" }, `Turn ${e.turn}`,
        h("span", { class: "turn-time" }, ` · ${fmtTime(e.seconds)}`)));
      if (e.reasoning) t.append(h("details", { class: "reasoning" }, h("summary", {}, "Reasoning"), h("p", {}, e.reasoning)));
      const txt = h("div", { class: "turn-text" });
      t.append(txt);
      typeInto(txt, e.text || (e.calls.length ? "" : "…"), live);
      t.calls = h("div", { class: "calls" });
      t.append(t.calls);
      chat.append(t);
      run.turn = t;
      scrollChat();
      thought(e.reasoning ? `${e.text}` : e.text, live);
      if (e.calls.includes("done")) thinkState.textContent = "Wrapping up";
    } else if (k === "call") {
      const label = callLabel(e.name, e.args);
      const row = h("div", { class: "call" }, h("span", { class: "state spin" }), h("span", {}, label));
      (run.turn ? run.turn.calls : chat).append(row);
      run.call = row;
      const mini = h("div", { class: "call" }, h("span", { class: "state spin" }), h("span", {}, label));
      thinkCalls.prepend(mini);
      while (thinkCalls.children.length > 4) thinkCalls.lastChild.remove();
      run.miniCall = mini;
      scrollChat();
    } else if (k === "result") {
      for (const row of [run.call, run.miniCall]) {
        if (!row) continue;
        const ok = !e.outcome || e.outcome === "done";
        row.classList.add(ok ? "done" : "fail");
        const s = $(".state", row);
        s.classList.remove("spin");
        s.innerHTML = ok ? ICON.check : ICON.cross;
        if (!ok && e.why) row.append(h("span", { class: "why", title: e.why }, e.why));
      }
      run.call = run.miniCall = null;
    } else if (k === "seen") {
      if (run.turn && e.turn >= 1) {
        let th = $(".thumbs", run.turn);
        if (!th) { th = h("div", { class: "thumbs" }); run.turn.append(th); }
        th.append(h("img", { src: e.url, alt: e.camera, title: e.camera.replace("_", " "), onclick: () => zoom(e.url) }));
      }
    } else if (k === "acted") {
      think.classList.remove("active");
      thinkState.textContent = e.success ? "Done" : "Stopped";
      const c = card(e.success ? ICON.check : ICON.cross, e.success ? "Muse says the task is done" : "Muse couldn't finish",
        e.success ? "green" : "red");
      c.append(h("div", { class: "card-body muted" }, e.summary));
      c.append(h("div", { class: "stats" }, e.steps != null ? h("span", {}, h("b", {}, e.steps), " steps") : "",
        h("span", {}, h("b", {}, e.turns), e.steps != null ? " Muse calls" : " turns"),
        h("span", {}, h("b", {}, fmtTime(e.seconds)), " total"), h("span", {}, h("b", {}, fmtTime(e.llm_seconds)), " thinking"),
        h("span", {}, h("b", {}, `$${(+e.cost).toFixed(3)}`), " Muse")));
      stepState("act", e.success ? "done" : "failed", e.success ? (e.steps != null ? `${e.steps} steps` : `${e.turns} turns`) : "Not finished");
    } else if (k === "recorded") {
      stepState("act", stepEls.act.classList.contains("failed") ? "failed" : "done", `Episode ${e.episode} · ${fmtTime(e.duration_s || 0)}`);
    } else if (k === "verdict") {
      settle(run.card, e.success ? "green" : "red");
      const c = run.card || card(ICON.gavel, "The judge");
      $(".title", c).textContent = e.success ? "The judge: it worked" : "The judge: it didn't work";
      const pct = Math.round(e.confidence * 100);
      const ring = h("div", { class: "ring", style: `--p:${pct};--c:${e.success ? "var(--green)" : "var(--red)"}` },
        h("span", {}, `${pct}%`));
      c.append(h("div", { class: "verdict" }, ring, h("div", { class: "card-body", style: "margin:0" }, e.reason)));
      c.append(h("div", { class: "frames" }, e.frames.map((f) => h("figure", {},
        h("img", { src: f.url, alt: f.why, onclick: () => zoom(f.url) }), h("figcaption", {}, f.why)))));
      if (e.final) {
        const cls = e.final.train ? "train" : e.final.review ? "review" : "no";
        const text = e.final.train ? "Goes into training" : e.final.review ? "Needs a person to review" : "Not used for training";
        c.append(h("span", { class: `decision ${cls}` }, text));
      }
      run.card = null;
      stepState("judge", e.success ? "done" : "failed", `${e.success ? "Success" : "Failure"} · ${pct}% sure`);
      scrollChat();
    } else if (k === "warning" || k === "note") {
      card(ICON.alert, e.text, k === "warning" ? "amber" : "").classList.add(k === "warning" ? "warn" : "x");
      if (/nothing to judge/.test(e.text)) stepState("judge", null, "Not recorded");
    } else if (k === "error") {
      settle(run.card, "red");
      if (run.card) stepState(run.card.stage, "failed", "Stopped here");
      else if (e.stage) stepState(e.stage, "failed", "Stopped here");
      card(ICON.alert, e.text, "red").classList.add("error");
      run.card = null;
      scanning(false);
    } else if (k === "end") {
      settle(run.card, e.status === "done" ? "green" : "red");
      scanning(false);
      think.classList.remove("active");
      if (thinkState.textContent === "Thinking") thinkState.textContent = e.status === "done" ? "Done" : "Stopped";
      $("#panelFoot").hidden = true;
      if (source) { source.close(); source = null; }
      run.ended = true;
    }
  };

  const openJob = async (id, fresh) => {
    if (source) { source.close(); source = null; }
    run = { id, last: -1, files: {}, card: null, turn: null };
    chat.replaceChildren();
    resetSteps();
    $("#panelFoot").hidden = false;
    $("#footState").textContent = "Starting…";
    const opened = Date.now() / 1000;
    source = new EventSource(`/api/jobs/${id}/events?since=-1`);
    source.onmessage = (msg) => {
      const e = JSON.parse(msg.data);
      handle(e, fresh || e.t > opened - 3);
    };
  };
  $("#stopBtn").addEventListener("click", async () => {
    if (!run) return;
    $("#footState").textContent = "Stopping…";
    await fetch(`/api/jobs/${run.id}/stop`, { method: "POST" });
  });

  // ---- start -------------------------------------------------------------------------------------------------------------
  resetSteps();
  pollState();
  (async () => {
    try {
      const st = await (await fetch("/api/state")).json();
      if (st.job) { openJob(st.job.id, false); if (st.job.status === "running") setPanel(true); }
    } catch (e) { /* offline */ }
  })();
})();
