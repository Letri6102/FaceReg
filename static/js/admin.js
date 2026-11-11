(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const TOKEN_KEY = "facereg_admin_token";
  let token = "";
  let reviewTimer = null;

  function loadToken() {
    try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch (_) { return token; }
  }
  function saveToken(t) {
    token = t;
    try { if (t) sessionStorage.setItem(TOKEN_KEY, t); else sessionStorage.removeItem(TOKEN_KEY); } catch (_) {}
  }

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtTime = (t) => (t ? new Date(t * 1000).toLocaleString("vi-VN") : "");
  const fmtNum = (x) => (x === null || x === undefined ? "–" : Number(x).toFixed(3));
  const fmtScores = (sc) => Object.entries(sc || {}).map(([m, v]) => `${esc(m)}: ${fmtNum(v)}`).join(" · ");

  async function api(path, opts = {}) {
    const headers = Object.assign({ "X-Admin-Token": token }, opts.headers || {});
    const res = await fetch(path, Object.assign({}, opts, { headers }));
    if (res.status === 401) { logout(); throw new Error("Phiên đăng nhập không hợp lệ."); }
    let data = {};
    try { data = await res.json(); } catch (_) {}
    if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);
    return data;
  }

  // Ảnh cần header X-Admin-Token nên tải bằng fetch rồi gán blob URL.
  async function loadImg(img, url) {
    try {
      const res = await fetch(url, { headers: { "X-Admin-Token": token } });
      if (!res.ok) return;
      const blob = await res.blob();
      img.src = URL.createObjectURL(blob);
    } catch (_) {}
  }
  function hydrateImages(root) {
    root.querySelectorAll("img[data-src]").forEach((img) => loadImg(img, img.dataset.src));
  }

  // ---------------- Đăng nhập ----------------
  async function login(t) {
    saveToken(t);
    try {
      await api("/api/admin/candidates?limit=1");
      $("loginCard").classList.add("hidden");
      $("app").classList.remove("hidden");
      $("logoutLink").classList.remove("hidden");
      switchTab("review");
    } catch (e) {
      saveToken("");
      $("loginError").textContent = e.message;
      $("loginError").classList.remove("hidden");
    }
  }
  function logout() {
    saveToken("");
    clearInterval(reviewTimer);
    $("app").classList.add("hidden");
    $("logoutLink").classList.add("hidden");
    $("loginCard").classList.remove("hidden");
  }

  // ---------------- Tabs ----------------
  function switchTab(name) {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    document.querySelectorAll("[data-panel]").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== name));
    clearInterval(reviewTimer);
    if (name === "review") { loadReview(); reviewTimer = setInterval(loadReview, 10000); }
    if (name === "candidates") loadCandidates();
    if (name === "history") loadHistory();
  }

  // ---------------- Duyệt ----------------
  async function loadReview() {
    const rows = await api("/api/admin/checkins?status=REVIEW&limit=200");
    $("reviewCount").textContent = rows.length ? `(${rows.length})` : "";
    const list = $("reviewList");
    if (!rows.length) { list.innerHTML = '<p class="muted">Không có trường hợp nào cần duyệt.</p>'; return; }
    list.innerHTML = rows.map((r) => `
      <div class="review-item" data-id="${r.id}">
        <div><b>${esc(r.full_name)}</b> · ${esc(r.candidate_code)} ${r.exam_code ? "· " + esc(r.exam_code) : ""}</div>
        <div class="muted">${fmtTime(r.created_at)} · Điểm gộp: <b>${fmtNum(r.score)}</b> · Liveness: ${fmtNum(r.spoof_score)}</div>
        <div class="muted" style="font-size:12px">${fmtScores(r.scores)}</div>
        <div class="pair">
          <div><img data-src="/api/admin/candidates/${encodeURIComponent(r.candidate_code)}/face" alt=""><div class="cap">Ảnh đăng ký</div></div>
          <div><img data-src="/api/admin/checkins/${r.id}/capture" alt=""><div class="cap">Ảnh check-in</div></div>
        </div>
        <ul>${(r.reasons || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
        <input type="text" placeholder="Ghi chú (tùy chọn)" class="note">
        <p class="row" style="margin-top:8px">
          <button class="success" data-act="APPROVE">Đúng người – cho thi</button>
          <button class="danger" data-act="REJECT">Không đúng – từ chối</button>
        </p>
      </div>`).join("");
    hydrateImages(list);
  }

  $("reviewList").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const item = btn.closest(".review-item");
    const decision = btn.dataset.act;
    if (decision === "REJECT" && !confirm("Xác nhận TỪ CHỐI thí sinh này?")) return;
    item.querySelectorAll("button").forEach((b) => (b.disabled = true));
    try {
      await api(`/api/admin/checkins/${item.dataset.id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, note: item.querySelector(".note").value }),
      });
      loadReview();
    } catch (err) {
      alert(err.message);
      item.querySelectorAll("button").forEach((b) => (b.disabled = false));
    }
  });

  // ---------------- Đăng ký ----------------
  async function enroll() {
    const out = $("enrollResult");
    const file = $("enPhoto").files[0];
    if (!$("enCode").value.trim() || !file) { out.innerHTML = '<div class="alert bad">Cần nhập mã thí sinh và chọn ảnh.</div>'; return; }
    const fd = new FormData();
    fd.append("code", $("enCode").value.trim());
    fd.append("full_name", $("enName").value.trim());
    fd.append("replace", $("enReplace").checked ? "true" : "false");
    fd.append("photo", file);
    $("enrollBtn").disabled = true;
    out.innerHTML = '<div class="alert warn">Đang xử lý...</div>';
    try {
      const r = await api("/api/admin/candidates", { method: "POST", body: fd });
      const m = (r.quality && r.quality.metrics) || {};
      const metrics = Object.keys(m).length
        ? `<div class="muted">Kích thước mặt ${m.face_px}px · nghiêng ${m.yaw_deg}° · ngẩng/cúi ${m.pitch_deg}° · độ nét ${m.sharpness} · độ sáng ${m.brightness}</div>` : "";
      if (r.ok) {
        out.innerHTML = `<div class="alert ok">${r.replaced ? "Đã cập nhật" : "Đã đăng ký"} thí sinh ${esc(r.code)}.</div>${metrics}`;
        $("enCode").value = ""; $("enName").value = ""; $("enPhoto").value = ""; $("enReplace").checked = false;
      } else {
        out.innerHTML = `<div class="alert bad">Ảnh không đạt yêu cầu:<ul>${r.quality.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>${metrics}`;
      }
    } catch (e) {
      out.innerHTML = `<div class="alert bad">${esc(e.message)}</div>`;
    } finally {
      $("enrollBtn").disabled = false;
    }
  }

  // ---------------- Danh sách thí sinh ----------------
  async function loadCandidates() {
    const rows = await api(`/api/admin/candidates?q=${encodeURIComponent($("candQ").value.trim())}`);
    const body = $("candBody");
    body.innerHTML = rows.map((r) => `
      <tr>
        <td><img class="thumb" data-src="/api/admin/candidates/${encodeURIComponent(r.code)}/face" alt=""></td>
        <td>${esc(r.code)}</td><td>${esc(r.full_name)}</td>
        <td>${r.live_templates}</td><td>${r.checkins}</td><td>${fmtTime(r.updated_at)}</td>
        <td><button class="danger" data-del="${esc(r.code)}">Xóa</button></td>
      </tr>`).join("") || '<tr><td colspan="7" class="muted">Chưa có thí sinh.</td></tr>';
    hydrateImages(body);
  }
  $("candBody").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-del]");
    if (!btn) return;
    if (!confirm(`Xóa thí sinh ${btn.dataset.del} và toàn bộ dữ liệu khuôn mặt, lịch sử check-in?`)) return;
    try { await api(`/api/admin/candidates/${encodeURIComponent(btn.dataset.del)}`, { method: "DELETE" }); loadCandidates(); }
    catch (err) { alert(err.message); }
  });

  // ---------------- Lịch sử ----------------
  async function loadHistory() {
    const q = new URLSearchParams({ status: $("hStatus").value, exam_code: $("hExam").value.trim(), candidate_code: $("hCand").value.trim(), limit: "300" });
    const rows = await api(`/api/admin/checkins?${q}`);
    $("hBody").innerHTML = rows.map((r) => `
      <tr>
        <td>${r.id}</td><td>${fmtTime(r.created_at)}</td>
        <td>${esc(r.full_name)}<br><span class="muted">${esc(r.candidate_code)}</span></td>
        <td>${esc(r.exam_code)}</td><td title="${fmtScores(r.scores)}">${fmtNum(r.score)}<br><span class="muted" style="font-size:11px">${fmtScores(r.scores)}</span></td><td>${fmtNum(r.spoof_score)}</td>
        <td><span class="badge ${r.auto_status}">${r.auto_status}</span></td>
        <td><span class="badge ${r.final_status}">${r.final_status}</span>${r.reviewer_note ? `<br><span class="muted">${esc(r.reviewer_note)}</span>` : ""}</td>
        <td>${r.monitor_alerts || 0}</td>
      </tr>`).join("") || '<tr><td colspan="9" class="muted">Không có dữ liệu.</td></tr>';
  }

  // ---------------- Sự kiện ----------------
  $("loginBtn").addEventListener("click", () => login($("token").value.trim()));
  $("token").addEventListener("keydown", (e) => { if (e.key === "Enter") login($("token").value.trim()); });
  $("logoutLink").addEventListener("click", (e) => { e.preventDefault(); logout(); });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  $("refreshReview").addEventListener("click", loadReview);
  $("enrollBtn").addEventListener("click", enroll);
  $("candSearch").addEventListener("click", loadCandidates);
  $("candQ").addEventListener("keydown", (e) => { if (e.key === "Enter") loadCandidates(); });
  $("hSearch").addEventListener("click", loadHistory);

  const saved = loadToken();
  if (saved) login(saved);
})();
