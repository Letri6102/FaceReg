/*
 * FaceReg Monitor - nhúng vào trang làm bài thi để kiểm tra định kỳ người ngồi trước máy.
 *
 * Cách dùng (trong trang thi của hệ thống MOOC):
 *   <script src="https://<facereg-server>/static/js/monitor.js"></script>
 *   <script>
 *     const mon = FaceRegMonitor.start({
 *       server: "https://<facereg-server>",   // bỏ trống nếu cùng domain
 *       token: "<monitor_token nhận được sau khi check-in ACCEPT>",
 *       intervalSec: 30,                       // chu kỳ trung bình, có dao động ngẫu nhiên ±50%
 *       onResult: (status) => {},              // "OK" | "NO_FACE" | "MULTI_FACE" | "MISMATCH"
 *       onError: (err) => {},
 *     });
 *     // mon.stop() khi nộp bài.
 *   </script>
 *
 * Nếu trang thi khác domain với FaceReg, khai báo domain đó trong CORS_ORIGINS ở file .env.
 */
(function (global) {
  "use strict";

  function start(opts) {
    const server = (opts.server || "").replace(/\/$/, "");
    const base = Math.max(Number(opts.intervalSec) || 30, 5) * 1000;
    const onResult = opts.onResult || function () {};
    const onError = opts.onError || function () {};
    let stream = null;
    let timer = null;
    let stopped = false;

    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.style.cssText = "position:fixed;width:1px;height:1px;opacity:0;pointer-events:none;";
    const canvas = document.createElement("canvas");

    async function ensureCamera() {
      if (stream && stream.active) return;
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, facingMode: "user" }, audio: false });
      video.srcObject = stream;
      if (!video.isConnected) document.body.appendChild(video);
      await video.play();
    }

    async function tick() {
      if (stopped) return;
      try {
        await ensureCamera();
        const w = Math.min(640, video.videoWidth || 640);
        const h = Math.round(w * (video.videoHeight || 480) / (video.videoWidth || 640));
        canvas.width = w;
        canvas.height = h;
        canvas.getContext("2d").drawImage(video, 0, 0, w, h);
        const res = await fetch(server + "/api/monitor", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token: opts.token, image: canvas.toDataURL("image/jpeg", 0.8) }),
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) onResult(data.status);
        else if (res.status !== 429) onError(new Error(data.detail || "HTTP " + res.status));
      } catch (e) {
        onError(e);
      }
      schedule();
    }

    function schedule() {
      if (stopped) return;
      const jitter = base * (0.5 + Math.random());
      timer = setTimeout(tick, jitter);
    }

    schedule();
    return {
      stop() {
        stopped = true;
        clearTimeout(timer);
        if (stream) stream.getTracks().forEach((t) => t.stop());
        video.remove();
      },
      checkNow() { clearTimeout(timer); return tick(); },
    };
  }

  global.FaceRegMonitor = { start: start };
})(window);
