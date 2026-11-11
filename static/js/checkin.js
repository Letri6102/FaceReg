(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const FRAME_INTERVAL_MS = 120;
  const CAPTURE_WIDTH = 640;
  const REVIEW_POLL_MS = 3000;
  const REVIEW_POLL_MAX_MS = 15 * 60 * 1000;

  let stream = null;
  let running = false;
  let sessionId = null;

  if (params.get("code")) $("code").value = params.get("code");
  if (params.get("exam")) {
    $("exam").value = params.get("exam");
    $("examRow").classList.add("hidden");
  }

  async function api(path, body) {
    const res = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    let data = null;
    try { data = await res.json(); } catch (_) { data = {}; }
    if (!res.ok) {
      const err = new Error((data && data.detail) || `Lỗi ${res.status}`);
      err.status = res.status;
      err.code = data && data.code;
      throw err;
    }
    return data;
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function show(section) {
    for (const id of ["formCard", "camCard", "resultCard"]) $(id).classList.toggle("hidden", id !== section);
  }

  function showFormError(msg) {
    $("formError").textContent = msg;
    $("formError").classList.toggle("hidden", !msg);
  }

  async function openCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Trình duyệt không hỗ trợ camera, hoặc trang không chạy qua HTTPS/localhost.");
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    const v = $("video");
    v.srcObject = stream;
    await new Promise((resolve) => {
      if (v.readyState >= 2) return resolve();
      v.onloadeddata = () => resolve();
    });
  }

  function stopCamera() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  // Chụp khung hình GỐC (không lật gương) để server xác định đúng hướng quay mặt.
  function captureFrame() {
    const v = $("video");
    const c = $("canvas");
    const scale = Math.min(1, CAPTURE_WIDTH / v.videoWidth);
    c.width = Math.round(v.videoWidth * scale);
    c.height = Math.round(v.videoHeight * scale);
    c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.85);
  }

  function renderSteps(total, idx) {
    const el = $("steps");
    el.innerHTML = "";
    for (let i = 0; i < total; i++) {
      const s = document.createElement("span");
      if (i < idx) s.className = "done";
      else if (i === idx) s.className = "cur";
      el.appendChild(s);
    }
  }

  function showResult(result) {
    stopCamera();
    running = false;
    show("resultCard");
    const st = result.status;
    $("resultBadge").innerHTML = `<span class="badge ${st}">${st === "ACCEPT" ? "ĐẠT" : st === "REVIEW" ? "CHỜ DUYỆT" : "KHÔNG ĐẠT"}</span>`;
    $("resultText").textContent = result.message;
    $("resultSub").textContent = "";
    $("retryBtn").classList.toggle("hidden", st !== "REJECT");
    if (st === "ACCEPT") {
      if (result.full_name) $("resultSub").textContent = `Thí sinh: ${result.full_name} (${result.candidate_code})`;
      if (result.redirect_url) {
        $("resultSub").textContent += " · Đang chuyển tới phòng thi...";
        setTimeout(() => { location.href = result.redirect_url; }, 2000);
      }
    } else if (st === "REVIEW") {
      pollReview();
    }
  }

  async function pollReview() {
    const started = Date.now();
    while (Date.now() - started < REVIEW_POLL_MAX_MS) {
      await sleep(REVIEW_POLL_MS);
      try {
        const r = await api(`/api/checkin/${encodeURIComponent(sessionId)}/status`);
        if (r.status !== "REVIEW") return showResult(r);
      } catch (_) { /* thử lại ở vòng sau */ }
    }
  }

  function showError(msg, canRetry = true) {
    stopCamera();
    running = false;
    show("resultCard");
    $("resultBadge").innerHTML = `<span class="badge REJECT">LỖI</span>`;
    $("resultText").textContent = msg;
    $("resultSub").textContent = "";
    $("retryBtn").classList.toggle("hidden", !canRetry);
  }

  async function run() {
    showFormError("");
    const code = $("code").value.trim();
    if (!code) return showFormError("Vui lòng nhập số báo danh.");
    $("startBtn").disabled = true;
    try {
      const start = await api("/api/checkin/start", { candidate_code: code, exam_code: $("exam").value.trim() });
      sessionId = start.session_id;
      show("camCard");
      await openCamera();

      running = true;
      const total = start.steps.length;
      let idx = 0;
      $("instruction").textContent = start.steps[0].instruction;
      renderSteps(total, 0);

      while (running) {
        const res = await api(`/api/checkin/${encodeURIComponent(sessionId)}/frame`, { image: captureFrame() });
        $("timer").textContent = `Thời gian còn lại: ${res.expires_in}s`;
        $("feedback").textContent = res.feedback || "";
        $("oval").classList.toggle("good", res.step_passed || res.hold > 0);
        if (res.step_index !== idx) {
          idx = res.step_index;
          renderSteps(total, idx);
        }
        $("instruction").textContent = res.instruction;
        if (res.done) break;
        await sleep(FRAME_INTERVAL_MS);
      }

      $("instruction").textContent = "Đang xử lý...";
      const result = await api(`/api/checkin/${encodeURIComponent(sessionId)}/finish`, {});
      showResult(result);
    } catch (e) {
      if (e.name === "NotAllowedError") showError("Bạn cần cho phép trình duyệt truy cập camera để xác thực.");
      else if (e.status === 404 && e.code === "not_found") { stopCamera(); show("formCard"); showFormError(e.message); }
      else showError(e.message || "Có lỗi xảy ra.");
    } finally {
      $("startBtn").disabled = false;
    }
  }

  $("startBtn").addEventListener("click", run);
  $("code").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  $("retryBtn").addEventListener("click", () => { show("formCard"); });
  window.addEventListener("beforeunload", stopCamera);
})();
