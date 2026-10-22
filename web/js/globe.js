// 3D views: Earth globe in the Earth-fixed frame, and the spacecraft attitude in the
// local orbital frame. three.js r160 (vendored).
import * as THREE from "/static/vendor/three.module.min.js";

const RE_KM = 6378.137;
const D2R = Math.PI / 180;

// ECEF (x, y, z) -> three.js (x, z, -y): north is up.
export const ecefToThree = (v) => new THREE.Vector3(v[0], v[2], -v[1]);
function llToThree(lat, lon, r = 1) {
  const la = lat * D2R, lo = lon * D2R;
  return new THREE.Vector3(r * Math.cos(la) * Math.cos(lo), r * Math.sin(la), -r * Math.cos(la) * Math.sin(lo));
}

function earthTexture(land, aois) {
  const W = 4096, H = 2048;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, "#1b3150"); grad.addColorStop(0.5, "#1f3a63"); grad.addColorStop(1, "#1b3150");
  g.fillStyle = grad; g.fillRect(0, 0, W, H);
  const X = (lon) => (lon + 180) / 360 * W, Y = (lat) => (90 - lat) / 180 * H;
  g.strokeStyle = "rgba(160,190,230,0.10)"; g.lineWidth = 2;
  for (let lon = -180; lon <= 180; lon += 30) { g.beginPath(); g.moveTo(X(lon), 0); g.lineTo(X(lon), H); g.stroke(); }
  for (let lat = -60; lat <= 60; lat += 30) { g.beginPath(); g.moveTo(0, Y(lat)); g.lineTo(W, Y(lat)); g.stroke(); }
  g.fillStyle = "#3c5a4a"; g.strokeStyle = "#7f9f8c"; g.lineWidth = 2;
  for (const ring of land) {
    g.beginPath();
    ring.forEach(([lon, lat], i) => (i ? g.lineTo(X(lon), Y(lat)) : g.moveTo(X(lon), Y(lat))));
    g.closePath(); g.fill(); g.stroke();
  }
  g.strokeStyle = "#d4a84b"; g.lineWidth = 5; g.setLineDash([18, 10]);
  for (const a of aois) g.strokeRect(X(a.lon[0]), Y(a.lat[1]), X(a.lon[1]) - X(a.lon[0]), Y(a.lat[0]) - Y(a.lat[1]));
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

export class Globe {
  constructor(el, info, land) {
    this.el = el;
    this.follow = true;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    this.camDist = 4.4; this.camLat = 30; this.camLon = 15;

    const tex = earthTexture(land, info.aois);
    this.earth = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 64),
      new THREE.MeshPhongMaterial({ map: tex, shininess: 8, specular: 0x223344 }));
    this.scene.add(this.earth);
    // thin atmosphere rim
    const atm = new THREE.Mesh(new THREE.SphereGeometry(1.035, 64, 32), new THREE.MeshBasicMaterial({
      color: 0x5b8def, transparent: true, opacity: 0.07, side: THREE.BackSide }));
    this.scene.add(atm);
    this.sun = new THREE.DirectionalLight(0xffffff, 2.6);
    this.scene.add(this.sun);
    this.scene.add(new THREE.AmbientLight(0x6f86b0, 0.55));

    // ground stations + visibility circles
    for (const gs of info.stations) {
      const p = llToThree(gs.lat, gs.lon, 1.003);
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), new THREE.MeshBasicMaterial({ color: 0x4fc3d9 }));
      m.position.copy(p); this.scene.add(m);
      const h = (info.elements.a_km - RE_KM) / RE_KM;
      const el = 5 * D2R;
      const lam = Math.acos(Math.cos(el) / (1 + h)) - el;
      this.scene.add(this.smallCircle(gs.lat, gs.lon, lam, 0x4fc3d9, 0.55));
    }
    // dynamic objects
    this.sat = new THREE.Mesh(new THREE.OctahedronGeometry(0.022), new THREE.MeshBasicMaterial({ color: 0xd4a84b }));
    this.scene.add(this.sat);
    this.halo = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.036, 32), new THREE.MeshBasicMaterial({ color: 0xd4a84b, side: THREE.DoubleSide, transparent: true, opacity: 0.6 }));
    this.scene.add(this.halo);
    this.nadirLine = this.line([], 0xd4a84b, 0.45);
    this.pastTrack = this.line([], 0xd4a84b, 0.55);
    this.futureTrack = this.line([], 0x9aa9c2, 0.45, true);
    this.firePts = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
      size: 0.018, vertexColors: true, sizeAttenuation: true, transparent: true, depthWrite: false }));
    this.scene.add(this.firePts);

    this.bindControls();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
  }

  smallCircle(lat, lon, lam, color, opacity) {
    const c = llToThree(lat, lon, 1).normalize();
    const ref = Math.abs(c.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const u = new THREE.Vector3().crossVectors(c, ref).normalize();
    const w = new THREE.Vector3().crossVectors(c, u);
    const pts = [];
    for (let k = 0; k <= 96; k++) {
      const a = (k / 96) * 2 * Math.PI;
      pts.push(c.clone().multiplyScalar(Math.cos(lam)).add(u.clone().multiplyScalar(Math.sin(lam) * Math.cos(a)))
        .add(w.clone().multiplyScalar(Math.sin(lam) * Math.sin(a))).multiplyScalar(1.004));
    }
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  }

  line(pts, color, opacity, dashed = false) {
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = dashed
      ? new THREE.LineDashedMaterial({ color, dashSize: 0.02, gapSize: 0.015, transparent: true, opacity })
      : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
    const l = new THREE.Line(g, mat);
    this.scene.add(l);
    return l;
  }

  setLine(l, pts) {
    l.geometry.dispose();
    l.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    if (l.material.isLineDashedMaterial) l.computeLineDistances();
  }

  bindControls() {
    const el = this.renderer.domElement;
    let drag = null;
    el.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.follow = false; this.onFollow && this.onFollow(false);
      this.camLon -= (e.clientX - drag.x) * 0.3;
      this.camLat = Math.max(-85, Math.min(85, this.camLat + (e.clientY - drag.y) * 0.3));
      drag = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener("pointerup", () => (drag = null));
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.camDist = Math.max(1.6, Math.min(8, this.camDist * (1 + Math.sign(e.deltaY) * 0.08)));
    }, { passive: false });
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + "px";
    this.renderer.domElement.style.height = h + "px";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  update(s) {
    const o = s.orbit;
    const sun = ecefToThree(o.sun_ecef);
    this.sun.position.copy(sun.multiplyScalar(10));
    const r = ecefToThree(o.r_ecef).multiplyScalar(1 / (RE_KM * 1000));
    this.sat.position.copy(r);
    this.halo.position.copy(r);
    this.halo.lookAt(this.camera.position);
    this.sat.material.color.set(s.payload.imaging ? 0xffd27a : 0xd4a84b);
    this.setLine(this.nadirLine, [r, r.clone().normalize().multiplyScalar(1.002)]);
    const alt = 1 + o.alt_km / RE_KM;
    this.setLine(this.pastTrack, o.track_hist.map(([la, lo]) => llToThree(la, lo, alt)).concat([r]));
    if (o.prediction) {
      const n = Math.min(o.prediction.lat.length, 80);
      const pts = [];
      for (let i = 0; i < n; i++) pts.push(llToThree(o.prediction.lat[i], o.prediction.lon[i], alt));
      this.setLine(this.futureTrack, pts);
    }
    const pos = [], col = [];
    for (const [lat, lon, st] of s.fires) {
      const p = llToThree(lat, lon, 1.006);
      pos.push(p.x, p.y, p.z);
      col.push(...(st === 2 ? [1.0, 0.25, 0.3] : st === 1 ? [1.0, 0.72, 0.3] : [0.75, 0.42, 0.22]));
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    this.firePts.geometry.dispose();
    this.firePts.geometry = g;
    if (this.follow) { this.camLat += (o.lat * 0.75 - this.camLat) * 0.15; this.camLon = lerpAngle(this.camLon, o.lon, 0.15); }
  }

  render() {
    const p = llToThree(this.camLat, this.camLon, this.camDist);
    this.camera.position.copy(p);
    this.camera.lookAt(0, 0, 0);
    this.renderer.render(this.scene, this.camera);
  }
}

function lerpAngle(a, b, t) {
  let d = ((b - a + 540) % 360) - 180;
  return a + d * t;
}

// ----------------------------------------------------------------------------------------------
// Spacecraft attitude view: body frame drawn in the local orbital frame (LVLH):
// screen-up = zenith, screen-right = velocity direction.
export class AttitudeView {
  constructor(el, dims) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
    this.camera.position.set(1.15, 0.75, 1.6);
    this.camera.lookAt(0, 0, 0);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(2, 3, 2);
    this.scene.add(key);

    const [dx, dy, dz] = dims.map((d) => d * 1.5);
    this.body = new THREE.Group();
    // body: gold MLI, white radiator on -X, dark payload face on +Z
    const mats = [
      new THREE.MeshStandardMaterial({ color: 0xc9a043, metalness: 0.6, roughness: 0.35 }), // +X
      new THREE.MeshStandardMaterial({ color: 0xe8edf3, metalness: 0.1, roughness: 0.6 }),  // -X radiator
      new THREE.MeshStandardMaterial({ color: 0xc9a043, metalness: 0.6, roughness: 0.35 }), // +Y
      new THREE.MeshStandardMaterial({ color: 0xb8913a, metalness: 0.6, roughness: 0.4 }),  // -Y
      new THREE.MeshStandardMaterial({ color: 0x2b2f38, metalness: 0.3, roughness: 0.5 }),  // +Z payload
      new THREE.MeshStandardMaterial({ color: 0xb8913a, metalness: 0.6, roughness: 0.4 }),  // -Z
    ];
    // three box axes: (x, y, z) -> body (X, Y, Z)
    this.body.add(new THREE.Mesh(new THREE.BoxGeometry(dx, dy, dz), mats));
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(dy * 0.35, dy * 0.35, 0.03, 24),
      new THREE.MeshStandardMaterial({ color: 0x111418, metalness: 0.8, roughness: 0.2 }));
    lens.rotation.x = Math.PI / 2; lens.position.z = dz / 2 + 0.015; this.body.add(lens);
    const cellMat = new THREE.MeshStandardMaterial({ color: 0x2d4f9e, metalness: 0.5, roughness: 0.3, emissive: 0x0b1a3a });
    for (const sx of [1, -1]) {
      const w = new THREE.Mesh(new THREE.BoxGeometry(dx, 0.006, dz), cellMat);
      w.position.set(sx * dx, dy / 2, 0); this.body.add(w);
    }
    const axis = (v, color) => {
      const a = new THREE.ArrowHelper(v, new THREE.Vector3(0, 0, 0), 0.42, color, 0.06, 0.035);
      this.body.add(a);
    };
    axis(new THREE.Vector3(1, 0, 0), 0xec4f55);
    axis(new THREE.Vector3(0, 1, 0), 0x3dbb8c);
    axis(new THREE.Vector3(0, 0, 1), 0x5b8def);
    this.scene.add(this.body);
    this.sunArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 0.75, 0xffd27a, 0.08, 0.05);
    this.nadirArrow = new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 0), 0.75, 0x4fc3d9, 0.08, 0.05);
    this.velArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 0.6, 0x9aa9c2, 0.06, 0.04);
    this.scene.add(this.sunArrow, this.nadirArrow, this.velArrow);
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + "px";
    this.renderer.domElement.style.height = h + "px";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  update(s) {
    const r = s.orbit.r_eci, v = s.orbit.v_eci, A = s.adcs.dcm;
    if (!A) return;
    const norm = (a) => { const n = Math.hypot(...a); return a.map((x) => x / n); };
    const crossv = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const zen = norm(r);                    // screen up
    const h = norm(crossv(r, v));           // orbit normal
    const along = crossv(h, zen);           // ~velocity, screen right
    // LVLH components of an inertial vector -> three (right=along, up=zenith, towards viewer=-h)
    const toThree = (u) => new THREE.Vector3(dot(u, along), dot(u, zen), -dot(u, h));
    const m = new THREE.Matrix4();
    const bx = toThree(A[0]), by = toThree(A[1]), bz = toThree(A[2]);
    m.makeBasis(bx, by, bz);
    this.body.setRotationFromMatrix(m);
    this.sunArrow.setDirection(toThree(s.orbit.sun_eci).normalize());
    this.velArrow.setDirection(toThree(norm(v)).normalize());
  }

  render() { this.renderer.render(this.scene, this.camera); }
}
