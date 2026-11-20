/* Interactive 360° architecture scene (three.js).
   Tiers are stacked like a building (storage → engine → backend → frontend); data packets travel
   along every connection; clicking a part raises it and reveals its labels; hovering a raised
   part reports what is under the pointer so the UI can show its details. */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { COMP, COMPS, FLOWS, FLOW_KINDS, LAYER, LAYERS, METHOD_IDS, type Flow, type LayerId, type TourStep } from "./archModel";
import { ICONS, type IconName } from "./icons";

export type Pick =
  | { kind: "comp"; id: string; layer: LayerId; elevated: true }
  | { kind: "layer"; id: LayerId; layer: LayerId; elevated: boolean }
  | { kind: "flow"; id: string };

export type SceneEvents = {
  onHover: (p: Pick | null) => void;
  onPointer: (x: number, y: number) => void;
  onSelect: (layer: LayerId | null) => void;
  onComp: (id: string | null) => void;
  onFlow: (id: string | null) => void;
  onInteract: () => void;
};
export type Theme = "light" | "dark";

const V3 = THREE.Vector3;
const UP = new V3(0, 1, 0);
const LIFT = 1.6;          // how far a clicked part rises
// …and how far it slides out of the tower toward the viewer, so nothing above can hide it
const PULL: Record<LayerId, number> = { frontend: 7, backend: 7.5, engine: 7.5, storage: 6, datasets: 4.5, metrics: 4.5, researcher: 3.5 };
// camera distance when a part is in focus (sized to the part)
const FOCUS_DIST: Record<LayerId, number> = { frontend: 19, backend: 19, engine: 18, storage: 27, datasets: 13, metrics: 14, researcher: 10 };
const DIM = 0.22;          // opacity of the parts that are not in focus
const HOME_TARGET = new V3(0, 4.3, 0);
const HOME_DIR = new V3(19.5, 12.5, 27.5).normalize();   // home viewing direction (front-right, from above)

// ── Layout ────────────────────────────────────────────────────────────────────
const LAYER_POS: Record<LayerId, THREE.Vector3> = {
  storage: new V3(0, 0, 0),
  engine: new V3(0, 3.3, 0),
  datasets: new V3(-9.4, 3.3, 0),
  metrics: new V3(9.4, 3.3, 0),
  backend: new V3(0, 6.6, 0),
  frontend: new V3(0, 9.9, 0),
  researcher: new V3(-9.8, 9.9, 0.6),
};
type Slab = { kind: "box"; w: number; d: number; h: number } | { kind: "cyl"; r: number; h: number };
const SLABS: Record<LayerId, Slab> = {
  frontend: { kind: "box", w: 10.4, d: 7.6, h: 0.28 },
  backend: { kind: "box", w: 10.6, d: 5.8, h: 0.28 },
  engine: { kind: "cyl", r: 4.95, h: 0.28 },
  datasets: { kind: "box", w: 3.9, d: 5.4, h: 0.28 },
  metrics: { kind: "box", w: 3.9, d: 6.9, h: 0.28 },
  storage: { kind: "box", w: 24, d: 7.4, h: 0.36 },
  researcher: { kind: "cyl", r: 1.35, h: 0.45 },
};
const MAIN_LABEL: Record<LayerId, [number, number, number]> = {
  frontend: [0, 2.25, 0], backend: [0, 1.8, 0], engine: [0, 2.45, 0], datasets: [0, 1.75, 0],
  metrics: [0, 1.35, 0], storage: [0, 2.2, 0], researcher: [0, 2.75, 0],
};
type Shape = "panel" | "cube" | "block" | "mlp" | "hub" | "chip" | "metric" | "dataset" | "cabinet" | "folder" | "db" | "gpu" | "person";
type Place = { x: number; z: number; shape: Shape; rot?: number };
const RING = 3.5;
const ring = (deg: number, shape: Shape): Place => {
  const a = (deg * Math.PI) / 180;
  return { x: RING * Math.cos(a), z: -RING * Math.sin(a), shape, rot: a + Math.PI / 2 };
};
const HUB = { x: 0, z: 0.9 };
const PLACE: Record<string, Place> = {
  researcher: { x: 0, z: 0, shape: "person" },
  fe_overview: { x: -3.3, z: -2.4, shape: "panel" }, fe_compare: { x: 0, z: -2.4, shape: "panel" }, fe_monitor: { x: 3.3, z: -2.4, shape: "panel" },
  fe_studio: { x: -3.3, z: 0, shape: "panel" }, fe_paper: { x: 0, z: 0, shape: "panel" }, fe_analysis: { x: 3.3, z: 0, shape: "panel" },
  fe_explorer: { x: -3.3, z: 2.4, shape: "panel" }, fe_docs: { x: 0, z: 2.4, shape: "panel" }, fe_geometry: { x: 3.3, z: 2.4, shape: "panel" },
  svc_experiment: { x: -3.8, z: -1.35, shape: "cube" }, svc_result: { x: -1.27, z: -1.35, shape: "cube" },
  svc_validation: { x: 1.27, z: -1.35, shape: "cube" }, svc_metrics: { x: 3.8, z: -1.35, shape: "cube" },
  svc_registry: { x: -3.8, z: 1.35, shape: "cube" }, svc_config: { x: -1.27, z: 1.35, shape: "cube" },
  svc_log: { x: 1.27, z: 1.35, shape: "cube" }, svc_resource: { x: 3.8, z: 1.35, shape: "cube" },
  ds_permuted: { x: 0, z: -1.3, shape: "dataset" }, ds_rotated: { x: 0, z: 1.3, shape: "dataset" },
  met_norm: { x: 0, z: -2.6, shape: "metric" }, met_decay: { x: 0, z: -1.3, shape: "metric" }, met_dormant: { x: 0, z: 0, shape: "metric" },
  met_rank: { x: 0, z: 1.3, shape: "metric" }, met_geometry: { x: 0, z: 2.6, shape: "metric" },
  st_registry: { x: -7.8, z: 0, shape: "cabinet" }, st_checkpoints: { x: -2.6, z: 0, shape: "folder" },
  st_results: { x: 2.6, z: 0, shape: "db" }, st_compute: { x: 7.8, z: 0, shape: "gpu" },
  eng_input: ring(180, "block"), eng_mlp: ring(135, "mlp"), eng_opt: ring(90, "block"),
  eng_train: ring(45, "block"), eng_eval: ring(0, "block"),
  eng_methods: { x: HUB.x, z: HUB.z, shape: "hub" },
};
METHOD_IDS.forEach((id, i) => {
  const a = Math.PI / 2 + (i * 2 * Math.PI) / METHOD_IDS.length;
  PLACE[id] = { x: HUB.x + 1.55 * Math.cos(a), z: HUB.z - 1.55 * Math.sin(a), shape: "chip", rot: a + Math.PI / 2 };
});
const SHORT: Record<string, string> = {
  m_orth: "Orthogonal", m_inward: "Inward", m_anr: "ANR", m_sp: "Shrink & Perturb", svc_resource: "Resource (GPU)",
};

// ── Palette ───────────────────────────────────────────────────────────────────
type Pal = {
  dark: boolean; tile: THREE.Color; side: THREE.Color; ground: THREE.Color; grid: [number, number];
  sky: number; gnd: number; hemi: number; sun: number; slabTo: THREE.Color; slabT: number; slabOpacity: number;
  faceTo: THREE.Color; faceT: number; fog: number;
};
const palette = (t: Theme): Pal =>
  t === "dark"
    ? { dark: true, tile: new THREE.Color("#2b3039"), side: new THREE.Color("#20242b"), ground: new THREE.Color("#101216"),
        grid: [0x2c323b, 0x1b1f25], sky: 0xb4c0da, gnd: 0x121417, hemi: 0.85, sun: 1.15,
        slabTo: new THREE.Color("#14171c"), slabT: 0.3, slabOpacity: 0.5, faceTo: new THREE.Color("#1b1e24"), faceT: 0.7, fog: 0x0e1013 }
    : { dark: false, tile: new THREE.Color("#fbfaf7"), side: new THREE.Color("#eeebe5"), ground: new THREE.Color("#ebe6de"),
        grid: [0xcdc5b8, 0xded7cc], sky: 0xffffff, gnd: 0xd6cdbf, hemi: 1.05, sun: 1.45,
        slabTo: new THREE.Color("#ffffff"), slabT: 0.55, slabOpacity: 0.55, faceTo: new THREE.Color("#ffffff"), faceT: 0.86, fog: 0xe9e4dc };
const mix = (c: THREE.ColorRepresentation, to: THREE.Color, t: number) => new THREE.Color(c).lerp(to, t);
const css = (c: THREE.Color) => `#${c.getHexString()}`;

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath(); g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

// ── Scene graph records ───────────────────────────────────────────────────────
type MatRec = { m: THREE.Material; base: number; fixedDepth?: boolean };
type CompNode = {
  id: string; layer: LayerId; group: THREE.Group; mats: THREE.MeshStandardMaterial[]; top: THREE.Vector3;
  label: CSS2DObject | null; hover: number; hoverTarget: number; baseY: number;
};
type LabelRec = { obj: CSS2DObject; inner: HTMLDivElement; prio: number; w: number; h: number; dy: number };
type LayerNode = {
  id: LayerId; group: THREE.Group; base: THREE.Vector3; off: THREE.Vector3; offTarget: THREE.Vector3; dim: number; dimTarget: number;
  glow: number; glowTarget: number; mats: MatRec[]; slabMat: THREE.MeshStandardMaterial; edge: THREE.LineBasicMaterial;
  main: CSS2DObject; mainEl: HTMLDivElement; mainRec: LabelRec; comps: CompNode[];
};
type FlowNode = {
  def: Flow; a: CompNode; b: CompNode; curve: THREE.CubicBezierCurve3; len: number;
  tube: THREE.Mesh; hit: THREE.Mesh; cone: THREE.Mesh; points: THREE.Points; label: CSS2DObject;
  tubeMat: THREE.MeshBasicMaterial; coneMat: THREE.MeshBasicMaterial; pointMat: THREE.PointsMaterial;
  vis: number; visTarget: number; hi: boolean; phase: number;
};
type PickData = { layer: LayerId; comp?: string; flow?: string };

export class ArchScene {
  private renderer: THREE.WebGLRenderer;
  private labels: CSS2DRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private ray = new THREE.Raycaster();
  private ptr = new THREE.Vector2();
  private inside = false;
  private world = new THREE.Group();
  private layers = new Map<LayerId, LayerNode>();
  private comps = new Map<string, CompNode>();
  private flows: FlowNode[] = [];
  private pickables: THREE.Object3D[] = [];
  private tex = new Map<string, THREE.Texture>();
  private neurons: { m: THREE.MeshStandardMaterial; layer: number }[] = [];
  private dies: THREE.MeshStandardMaterial[] = [];
  private animated = new Set<THREE.Material>();   // materials whose glow is driven by an animation
  private spinners: THREE.Object3D[] = [];
  private dot: THREE.Texture;
  private pal: Pal;
  private theme: Theme;
  private selected: LayerId | null = null;
  private selectedComp: string | null = null;
  private tourFlows: Set<string> | null = null;
  private tourLayer: LayerId | null = null;
  private pinned: string | null = null;
  private flowLabels = false;
  private labelRecs: LabelRec[] = [];
  private userMoved = false;
  private hover: Pick | null = null;
  private extHover: string | null = null;
  private targetGoal: THREE.Vector3 | null = null;
  private pullDir: THREE.Vector3 | null = null;
  private cameraGoal: THREE.Vector3 | null = null;
  private clock = new THREE.Clock();
  private raf = 0;
  private active = true;
  private disposed = false;
  private down = { x: 0, y: 0, t: 0 };
  private reduced: boolean;
  private snap = true;

  constructor(private host: HTMLElement, theme: Theme, private ev: SceneEvents) {
    this.theme = theme;
    this.pal = palette(theme);
    this.reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const w = Math.max(host.clientWidth, 1), h = Math.max(host.clientHeight, 1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.className = "a3-canvas";
    host.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    this.labels.setSize(w, h);
    Object.assign(this.labels.domElement.style, { position: "absolute", left: "0", top: "0", pointerEvents: "none" });
    this.labels.domElement.className = "a3-labels";
    host.appendChild(this.labels.domElement);

    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 400);
    this.camera.position.copy(this.homeCamera());
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(HOME_TARGET);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 75;
    this.controls.minPolarAngle = 0.12;
    this.controls.maxPolarAngle = 1.5;
    this.controls.autoRotate = !this.reduced;
    this.controls.autoRotateSpeed = 0.55;
    this.controls.addEventListener("start", this.onControlStart);

    const el = this.renderer.domElement;
    el.addEventListener("pointermove", this.onMove);
    el.addEventListener("pointerdown", this.onDown);
    el.addEventListener("pointerup", this.onUp);
    el.addEventListener("pointerleave", this.onLeave);
    window.addEventListener("keydown", this.onKey);

    this.dot = this.dotTexture();
    this.build();
    this.raf = requestAnimationFrame(this.frame);
  }

  // ── Public API ──────────────────────────────────────────────────────────────
  select(layer: LayerId | null) {
    if (layer === this.selected && !this.tourFlows && !this.pinned && !this.selectedComp) return;
    this.selected = layer;
    this.selectedComp = null;
    this.tourFlows = null; this.tourLayer = null; this.pinned = null;
    this.focusLayer(layer);
    this.applyState();
  }
  selectComp(id: string | null) {
    this.selectedComp = id;
    this.applyState();
  }
  setTour(step: TourStep | null) {
    if (!step && !this.tourFlows) return;     // not touring: leave the user's own selection alone
    this.selected = null; this.selectedComp = null; this.pinned = null;
    this.tourFlows = step ? new Set(step.flows) : null;
    this.tourLayer = step ? step.layer : null;
    this.focusLayer(this.tourLayer);
    this.applyState();
  }
  pinFlow(id: string | null) {
    this.pinned = id;
    this.applyState();
  }
  setFlowLabels(on: boolean) { this.flowLabels = on; this.applyState(); }
  setAutoRotate(on: boolean) { this.controls.autoRotate = on; }
  setExternalHover(id: string | null) { this.extHover = id; }
  resetView() {
    this.select(null);
    this.targetGoal = HOME_TARGET.clone();
    this.cameraGoal = this.homeCamera();
    this.userMoved = false;
  }
  setTheme(theme: Theme) {
    if (theme === this.theme) return;
    this.theme = theme;
    this.pal = palette(theme);
    this.clearWorld();
    this.build();
  }
  setActive(on: boolean) { this.active = on; if (on) this.clock.getDelta(); }
  resize() {
    const w = Math.max(this.host.clientWidth, 1), h = Math.max(this.host.clientHeight, 1);
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (!this.userMoved) this.camera.position.copy(this.homeCamera());
  }
  /** Home viewpoint, pulled back on narrow stages so the whole tower stays in frame. */
  private homeCamera() {
    return HOME_TARGET.clone().addScaledVector(HOME_DIR, 31 * Math.max(1, 1.55 / this.camera.aspect));
  }
  private makeLabel(cls: string, color: string, prio: number, fill: (el: HTMLDivElement) => void) {
    const outer = document.createElement("div");          // positioned by CSS2DRenderer
    outer.className = "a3-anchor";
    const inner = document.createElement("div");          // nudged by the declutter pass
    inner.className = `a3-label ${cls}`;
    inner.style.setProperty("--c", color);
    fill(inner);
    outer.appendChild(inner);
    const obj = new CSS2DObject(outer);
    const rec: LabelRec = { obj, inner, prio, w: 0, h: 0, dy: 0 };
    this.labelRecs.push(rec);
    return { obj, inner, rec };
  }
  /** Screen-space separation: a label that would cover another one is lifted just above it. */
  private declutter() {
    const W = this.renderer.domElement.clientWidth, H = this.renderer.domElement.clientHeight;
    const v = new V3();
    const items: { r: LabelRec; x: number; y: number; z: number }[] = [];
    for (const r of this.labelRecs) {
      if (!r.obj.visible || (r.obj.element as HTMLElement).style.display === "none") continue;
      let o: THREE.Object3D | null = r.obj, shown = true;
      while (o) { if (!o.visible) { shown = false; break; } o = o.parent; }
      if (!shown) continue;
      if (!r.w) { r.w = r.inner.offsetWidth; r.h = r.inner.offsetHeight; }
      r.obj.getWorldPosition(v).project(this.camera);
      items.push({ r, x: ((v.x + 1) / 2) * W, y: ((1 - v.y) / 2) * H, z: v.z });
    }
    items.sort((a, b) => a.r.prio - b.r.prio || a.z - b.z);
    const placed: [number, number, number, number][] = [];
    const pad = 3;
    for (const it of items) {
      const { r } = it;
      const x0 = it.x - r.w / 2 - pad, x1 = it.x + r.w / 2 + pad;
      const half = r.h / 2 + pad;
      // search upward and downward for the nearest free slot; take the smaller move
      const search = (dir: 1 | -1) => {
        let dy = 0;
        for (let guard = 0; guard < 24; guard++) {
          const y0 = it.y - half - dy, y1 = it.y + half - dy;
          const hit = placed.find((p) => x0 < p[2] && x1 > p[0] && y0 < p[3] && y1 > p[1]);
          if (!hit) return dy;
          dy = dir === 1 ? it.y + half - hit[1] + 1 : it.y - half - hit[3] - 1;
        }
        return dir * 1e4;
      };
      if (r.prio >= 3) {
        const y0 = it.y - half, y1 = it.y + half;
        const clash = placed.some((p) => x0 < p[2] && x1 > p[0] && y0 < p[3] && y1 > p[1]);
        r.inner.style.visibility = clash ? "hidden" : "";
        r.inner.style.transform = "";
        r.dy = 0;
        if (!clash) placed.push([x0, y0, x1, y1]);
        continue;
      }
      r.inner.style.visibility = "";
      const up = search(1), down = search(-1);
      const dy = Math.abs(up) <= Math.abs(down) ? up : down;
      placed.push([x0, it.y - half - dy, x1, it.y + half - dy]);
      r.dy += (dy - r.dy) * 0.45;
      r.inner.style.transform = Math.abs(r.dy) < 0.5 ? "" : `translateY(${-r.dy.toFixed(1)}px)`;
    }
  }
  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    const el = this.renderer.domElement;
    el.removeEventListener("pointermove", this.onMove);
    el.removeEventListener("pointerdown", this.onDown);
    el.removeEventListener("pointerup", this.onUp);
    el.removeEventListener("pointerleave", this.onLeave);
    window.removeEventListener("keydown", this.onKey);
    this.controls.removeEventListener("start", this.onControlStart);
    this.controls.dispose();
    this.clearWorld();
    this.dot.dispose();
    this.renderer.dispose();
    el.remove();
    this.labels.domElement.remove();
  }

  // ── Build ───────────────────────────────────────────────────────────────────
  private build() {
    const p = this.pal;
    this.world = new THREE.Group();
    this.scene.add(this.world);
    this.scene.fog = new THREE.Fog(p.fog, 62, 130);

    this.world.add(new THREE.HemisphereLight(p.sky, p.gnd, p.hemi));
    const sun = new THREE.DirectionalLight(0xffffff, p.sun);
    sun.position.set(14, 30, 18);
    sun.target.position.copy(HOME_TARGET);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -21, right: 21, top: 21, bottom: -21, near: 1, far: 90 });
    sun.shadow.camera.updateProjectionMatrix();
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    this.world.add(sun, sun.target);

    const ground = new THREE.Mesh(new THREE.CircleGeometry(38, 96),
      new THREE.MeshStandardMaterial({ color: p.ground, roughness: 1, metalness: 0 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.24;
    ground.receiveShadow = true;
    this.world.add(ground);
    const grid = new THREE.PolarGridHelper(32, 24, 10, 96, p.grid[0], p.grid[1]);
    grid.position.y = -0.22;
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.55;
    this.world.add(grid);

    for (const l of LAYERS) this.buildLayer(l.id);
    this.scene.updateMatrixWorld(true);
    for (const f of FLOWS) this.buildFlow(f);
    this.snap = true;
    this.applyState();
  }

  private clearWorld() {
    this.world.traverse((o) => {
      if (o instanceof CSS2DObject) o.element.remove();
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
      const mat = m.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
      else if (mat) mat.dispose();
    });
    this.scene.remove(this.world);
    this.tex.forEach((t) => t.dispose());
    this.tex.clear();
    this.layers.clear(); this.comps.clear();
    this.flows = []; this.pickables = []; this.neurons = []; this.dies = []; this.spinners = [];
    this.animated.clear();
    this.hover = null;
    this.labelRecs = [];
  }

  private std(L: LayerNode, C: CompNode | null, color: THREE.ColorRepresentation,
              o: THREE.MeshStandardMaterialParameters = {}, base = 1, fixedDepth = false) {
    const m = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.04, transparent: base < 1, opacity: base, ...o });
    L.mats.push({ m, base, fixedDepth });
    if (C) C.mats.push(m);
    return m;
  }
  private basic(L: LayerNode, color: THREE.ColorRepresentation, base = 1, o: THREE.MeshBasicMaterialParameters = {}) {
    const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: base, ...o });
    L.mats.push({ m, base, fixedDepth: true });
    return m;
  }
  private part(L: LayerNode, C: CompNode, geo: THREE.BufferGeometry, mat: THREE.Material,
               pos: [number, number, number] = [0, 0, 0], parent: THREE.Object3D = C.group, cast = true) {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(...pos);
    m.castShadow = cast;
    m.receiveShadow = true;
    m.userData.pick = { layer: L.id, comp: C.id } satisfies PickData;
    this.pickables.push(m);
    parent.add(m);
    return m;
  }
  private iconTex(icon: IconName, stroke: string, bg: string | null, glyph = 0.6, aspect = 1) {
    const key = `${icon}|${stroke}|${bg}|${glyph}|${aspect}`;
    const hit = this.tex.get(key);
    if (hit) return hit;
    const H = 256, W = Math.round(H * aspect);
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const g = c.getContext("2d")!;
    if (bg) { g.fillStyle = bg; roundRect(g, 6, 6, W - 12, H - 12, 34); g.fill(); }
    const k = (H * glyph) / 24;
    g.save();
    g.translate(W / 2 - 12 * k, H / 2 - 12 * k);
    g.scale(k, k);
    g.strokeStyle = stroke; g.lineWidth = 1.9; g.lineCap = "round"; g.lineJoin = "round";
    g.stroke(new Path2D(ICONS[icon]));
    g.restore();
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 8;
    this.tex.set(key, t);
    return t;
  }
  private decal(L: LayerNode, C: CompNode, icon: IconName, w: number, h: number, y: number,
                opts: { bg?: string | null; glyph?: number; stroke?: string; x?: number; z?: number; parent?: THREE.Object3D; front?: boolean } = {}) {
    const color = opts.stroke ?? LAYER[L.id].color;
    const map = this.iconTex(icon, color, opts.bg ?? null, opts.glyph ?? 0.62, w / h);
    const mat = this.basic(L, 0xffffff, 1, { map, depthWrite: false });
    const m = this.part(L, C, new THREE.PlaneGeometry(w, h), mat, [opts.x ?? 0, y, opts.z ?? 0], opts.parent ?? C.group, false);
    if (!opts.front) m.rotation.x = -Math.PI / 2;
    m.receiveShadow = false;
    return m;
  }
  private dotTexture() {
    const S = 64, c = document.createElement("canvas");
    c.width = c.height = S;
    const g = c.getContext("2d")!;
    const grd = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
    grd.addColorStop(0, "rgba(255,255,255,1)");
    grd.addColorStop(0.32, "rgba(255,255,255,0.95)");
    grd.addColorStop(0.62, "rgba(255,255,255,0.32)");
    grd.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grd; g.fillRect(0, 0, S, S);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  private buildLayer(id: LayerId) {
    const meta = LAYER[id], s = SLABS[id], p = this.pal;
    const color = new THREE.Color(meta.color);
    const L = {
      id, group: new THREE.Group(), base: LAYER_POS[id].clone(), off: new V3(), offTarget: new V3(), dim: 1, dimTarget: 1,
      glow: 0, glowTarget: 0, mats: [], comps: [],
    } as unknown as LayerNode;
    L.group.position.copy(L.base);
    this.world.add(L.group);
    this.layers.set(id, L);

    L.slabMat = this.std(L, null, mix(meta.color, p.slabTo, p.slabT),
      { roughness: 0.32, metalness: 0.08, depthWrite: false, emissive: color, emissiveIntensity: 0 }, p.slabOpacity, true);
    const slabGeo = s.kind === "cyl" ? new THREE.CylinderGeometry(s.r, s.r, s.h, 72) : new RoundedBoxGeometry(s.w, s.h, s.d, 3, 0.12);
    const slab = new THREE.Mesh(slabGeo, L.slabMat);
    slab.receiveShadow = true;
    slab.userData.pick = { layer: id } satisfies PickData;
    this.pickables.push(slab);
    L.group.add(slab);
    const edgeSrc = s.kind === "cyl" ? new THREE.CylinderGeometry(s.r, s.r, s.h, 72) : new THREE.BoxGeometry(s.w - 0.03, s.h, s.d - 0.03);
    L.edge = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 });
    L.mats.push({ m: L.edge, base: 0.8, fixedDepth: true });
    L.group.add(new THREE.LineSegments(new THREE.EdgesGeometry(edgeSrc, 30), L.edge));   // outline only, not every facet
    edgeSrc.dispose();

    const main = this.makeLabel("a3-main", meta.color, 0, (el) => {
      const b = document.createElement("b");
      b.textContent = meta.label;
      el.appendChild(b);
      if (meta.tag) { const sp = document.createElement("span"); sp.textContent = meta.tag; el.appendChild(sp); }
    });
    L.main = main.obj;
    L.mainEl = main.inner;
    L.mainRec = main.rec;
    L.main.position.set(...MAIN_LABEL[id]);
    L.group.add(L.main);

    for (const c of COMPS) if (c.layer === id) this.buildComp(L, c.id);
  }

  private buildComp(L: LayerNode, id: string) {
    const def = COMP[id], pl = PLACE[id], p = this.pal;
    const color = LAYER[L.id].color;
    const slabTop = SLABS[L.id].h / 2;
    const C: CompNode = { id, layer: L.id, group: new THREE.Group(), mats: [], top: new V3(0, 0.5, 0), label: null, hover: 0, hoverTarget: 0, baseY: slabTop };
    C.group.position.set(pl.x, slabTop, pl.z);
    if (pl.rot) C.group.rotation.y = pl.rot;
    L.group.add(C.group);
    L.comps.push(C);
    this.comps.set(id, C);
    const tile = (o: THREE.MeshStandardMaterialParameters = {}) => this.std(L, C, p.tile, { emissive: new THREE.Color(color), emissiveIntensity: 0, ...o });
    const tint = (t: number) => mix(color, p.dark ? p.faceTo : new THREE.Color("#ffffff"), t);
    const faceBg = css(mix(color, p.faceTo, p.faceT));
    let labelY = 1.2;

    switch (pl.shape) {
      case "panel": {                                   // frontend page: a screen on a stand
        const W = 2.55, H = 1.5, D = 0.12, t = 0.62;
        const pivot = new THREE.Group();
        pivot.position.set(0, 0.06, 0.45);
        pivot.rotation.x = -t;
        C.group.add(pivot);
        this.part(L, C, new RoundedBoxGeometry(W, H, D, 3, 0.05), tile(), [0, H / 2, 0], pivot);
        this.decal(L, C, def.icon, W - 0.16, H - 0.16, H / 2, { bg: faceBg, glyph: 0.5, z: D / 2 + 0.004, parent: pivot, front: true });
        this.part(L, C, new RoundedBoxGeometry(1.15, 0.08, 0.6, 2, 0.03), this.std(L, C, p.side), [0, 0.04, 0.42]);
        C.top.set(0, (H * Math.cos(t)) / 2 + 0.06, 0.45 - (H * Math.sin(t)) / 2);
        labelY = H * Math.cos(t) + 0.5;
        break;
      }
      case "cube": {                                    // backend service: a server block
        this.part(L, C, new RoundedBoxGeometry(2.1, 0.95, 1.9, 4, 0.13), tile(), [0, 0.475, 0]);
        this.decal(L, C, def.icon, 0.95, 0.95, 0.956, { glyph: 0.7 });
        const led = this.basic(L, color, 0.9);
        this.part(L, C, new THREE.BoxGeometry(1.3, 0.05, 0.02), led, [0, 0.36, 0.955], C.group, false);
        this.part(L, C, new THREE.BoxGeometry(0.75, 0.05, 0.02), led, [-0.27, 0.22, 0.955], C.group, false);
        C.top.set(0, 0.95, 0);
        labelY = 1.5;
        break;
      }
      case "block": {                                   // engine pipeline stage
        this.part(L, C, new RoundedBoxGeometry(1.8, 0.7, 1.25, 4, 0.14), tile({ color: tint(0.78) }), [0, 0.35, 0]);
        this.decal(L, C, def.icon, 0.85, 0.85, 0.706, { glyph: 0.72 });
        C.top.set(0, 0.7, 0);
        labelY = 1.25;
        break;
      }
      case "mlp": {                                     // the network itself, drawn as a small MLP
        this.part(L, C, new RoundedBoxGeometry(2.3, 0.4, 1.7, 4, 0.12), tile({ color: tint(0.78) }), [0, 0.2, 0]);
        const counts = [4, 3, 3, 2], xs = [0.85, 0.28, -0.28, -0.85], y = 0.92;
        const cols = [p.dark ? 0xa3a9b4 : 0xd9d5ce, 0x3b82f6, 0x3b82f6, 0xa855f7];
        const pos: THREE.Vector3[][] = counts.map((n, li) =>
          Array.from({ length: n }, (_, k) => new V3(xs[li], y, (k - (n - 1) / 2) * 0.4)));
        const sphere = new THREE.SphereGeometry(0.09, 16, 12);
        pos.forEach((layer, li) => layer.forEach((v) => {
          const m = this.std(L, C, cols[li], { emissive: new THREE.Color(cols[li]), emissiveIntensity: 0.2, roughness: 0.35 });
          this.neurons.push({ m, layer: li });
          this.animated.add(m);
          this.part(L, C, sphere, m, [v.x, v.y, v.z]);
        }));
        const seg: number[] = [];
        for (let li = 0; li < pos.length - 1; li++)
          for (const a of pos[li]) for (const b of pos[li + 1]) seg.push(a.x, a.y, a.z, b.x, b.y, b.z);
        const lg = new THREE.BufferGeometry();
        lg.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
        const lm = new THREE.LineBasicMaterial({ color: 0x5b8fd8, transparent: true, opacity: 0.5 });
        L.mats.push({ m: lm, base: 0.5, fixedDepth: true });
        C.group.add(new THREE.LineSegments(lg, lm));
        for (const v of [[0.6, 0.4], [-0.6, 0.4]] as const) {          // posts holding the net
          this.part(L, C, new THREE.CylinderGeometry(0.025, 0.025, 0.55, 6), this.std(L, C, tint(0.4)), [v[0], 0.67, -0.75]);
          this.part(L, C, new THREE.CylinderGeometry(0.025, 0.025, 0.55, 6), this.std(L, C, tint(0.4)), [v[0], 0.67, 0.75]);
        }
        C.top.set(0, 1.12, 0);
        labelY = 1.6;
        break;
      }
      case "hub": {                                     // update methods, the chips sit around it
        const mc = new THREE.Color("#7c5cbf");
        this.part(L, C, new THREE.CylinderGeometry(0.85, 0.95, 0.5, 48), tile({ color: mix(mc, p.dark ? p.faceTo : new THREE.Color("#ffffff"), 0.72) }), [0, 0.25, 0]);
        this.decal(L, C, def.icon, 0.9, 0.9, 0.506, { glyph: 0.7, stroke: "#7c5cbf" });
        const ring = new THREE.Mesh(new THREE.TorusGeometry(1.55, 0.022, 8, 120), this.basic(L, mc, 0.55));
        ring.rotation.x = Math.PI / 2;
        ring.position.y = 0.06;
        C.group.add(ring);
        const tick = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 8), this.basic(L, mc, 0.9));
        tick.position.set(1.55, 0, 0);
        ring.add(tick);
        this.spinners.push(ring);
        C.top.set(0, 0.5, 0);
        labelY = 1.0;
        break;
      }
      case "chip": {                                    // one update method
        const c = def.accent ? new THREE.Color("#a78bfa") : p.tile;
        this.part(L, C, new RoundedBoxGeometry(0.66, 0.24, 0.42, 3, 0.07), tile({ color: c }), [0, 0.12, 0]);
        this.part(L, C, new THREE.BoxGeometry(0.4, 0.012, 0.06), this.basic(L, "#7c5cbf", 0.9), [0, 0.247, 0.12], C.group, false);
        C.top.set(0, 0.24, 0);
        labelY = 0.62;
        break;
      }
      case "metric": {
        this.part(L, C, new RoundedBoxGeometry(3.1, 0.3, 1.0, 3, 0.08), tile(), [0, 0.15, 0]);
        this.decal(L, C, def.icon, 0.72, 0.72, 0.306, { glyph: 0.72, x: -1.05 });
        this.part(L, C, new THREE.BoxGeometry(1.7, 0.012, 0.05), this.basic(L, color, 0.55), [0.4, 0.306, 0.18], C.group, false);
        this.part(L, C, new THREE.BoxGeometry(1.2, 0.012, 0.05), this.basic(L, color, 0.35), [0.15, 0.306, -0.12], C.group, false);
        C.top.set(0, 0.3, 0);
        labelY = 0.75;
        break;
      }
      case "dataset": {                                 // a stack of task cards: shuffled or rotated
        this.part(L, C, new RoundedBoxGeometry(3.1, 0.42, 2.0, 3, 0.1), tile(), [0, 0.21, 0]);
        this.decal(L, C, def.icon, 0.95, 0.95, 0.426, { glyph: 0.72, x: -0.72 });
        const card = new THREE.BoxGeometry(0.78, 0.025, 0.78);
        for (let i = 0; i < 5; i++) {
          const m = this.part(L, C, card, this.std(L, C, mix(color, new THREE.Color("#ffffff"), 0.15 + i * 0.12), { emissive: new THREE.Color(color), emissiveIntensity: 0 }),
            [0.78 + (id === "ds_permuted" ? (i % 2 ? 0.07 : -0.07) : 0), 0.5 + i * 0.095, 0]);
          if (id === "ds_rotated") m.rotation.y = i * 0.28;
        }
        C.top.set(0.4, 0.95, 0);
        labelY = 1.35;
        break;
      }
      case "cabinet": {                                 // experiment registry: a filing cabinet
        this.part(L, C, new RoundedBoxGeometry(2.6, 1.3, 1.8, 4, 0.12), tile(), [0, 0.65, 0]);
        const line = this.basic(L, color, 0.85);
        for (const y of [0.36, 0.68, 1.0]) {
          this.part(L, C, new THREE.BoxGeometry(2.25, 0.025, 0.02), line, [0, y - 0.16, 0.905], C.group, false);
          this.part(L, C, new THREE.BoxGeometry(0.45, 0.06, 0.04), line, [0, y, 0.915], C.group, false);
        }
        this.decal(L, C, def.icon, 0.95, 0.95, 1.306, { glyph: 0.7 });
        C.top.set(0, 1.3, 0);
        labelY = 1.85;
        break;
      }
      case "folder": {                                  // checkpoints
        const fm = tile({ color: tint(0.8) });
        this.part(L, C, new RoundedBoxGeometry(2.8, 0.85, 1.9, 4, 0.1), fm, [0, 0.425, 0]);
        this.part(L, C, new RoundedBoxGeometry(1.1, 0.16, 0.55, 2, 0.05), fm, [-0.75, 0.9, -0.6]);
        this.decal(L, C, def.icon, 0.95, 0.95, 0.856, { glyph: 0.7, x: 0.35 });
        C.top.set(0, 0.85, 0);
        labelY = 1.4;
        break;
      }
      case "db": {                                      // results & logs: a database stack
        const dm = tile({ color: tint(0.75) });
        const cyl = new THREE.CylinderGeometry(0.95, 0.95, 0.34, 48);
        for (const y of [0.17, 0.59, 1.01]) this.part(L, C, cyl, dm, [0, y, 0]);
        const band = this.basic(L, color, 0.7);
        for (const y of [0.38, 0.8]) {
          const r = this.part(L, C, new THREE.TorusGeometry(0.93, 0.02, 6, 64), band, [0, y, 0], C.group, false);
          r.rotation.x = Math.PI / 2;
        }
        this.decal(L, C, def.icon, 0.95, 0.95, 1.186, { glyph: 0.7 });
        C.top.set(0, 1.18, 0);
        labelY = 1.7;
        break;
      }
      case "gpu": {                                     // compute: a GPU package on a board
        this.part(L, C, new RoundedBoxGeometry(2.8, 0.16, 2.4, 2, 0.05), this.std(L, C, p.dark ? 0x1f4a33 : 0x2f6b47), [0, 0.08, 0]);
        this.part(L, C, new RoundedBoxGeometry(1.6, 0.2, 1.6, 2, 0.05), this.std(L, C, 0x30343b), [0, 0.26, 0]);
        const die = this.std(L, C, 0x16a34a, { emissive: new THREE.Color(0x22c55e), emissiveIntensity: 0.6, roughness: 0.3 });
        this.dies.push(die);
        this.animated.add(die);
        this.part(L, C, new THREE.BoxGeometry(0.95, 0.05, 0.95), die, [0, 0.385, 0]);
        const pin = new THREE.BoxGeometry(0.08, 0.04, 0.24), gold = this.std(L, C, 0xd4a94a, { metalness: 0.6, roughness: 0.35 });
        for (let i = 0; i < 7; i++) {
          const o = -0.6 + i * 0.2;
          this.part(L, C, pin, gold, [o, 0.18, 0.95], C.group, false);
          this.part(L, C, pin, gold, [o, 0.18, -0.95], C.group, false);
          this.part(L, C, pin, gold, [0.95, 0.18, o], C.group, false).rotation.y = Math.PI / 2;
          this.part(L, C, pin, gold, [-0.95, 0.18, o], C.group, false).rotation.y = Math.PI / 2;
        }
        this.decal(L, C, "chip", 0.8, 0.8, 0.413, { glyph: 0.72, stroke: "#eafff1" });
        C.top.set(0, 0.42, 0);
        labelY = 0.95;
        break;
      }
      case "person": {                                  // the researcher behind a laptop
        const body = this.std(L, C, "#3b6fd6", { emissive: new THREE.Color("#3b6fd6"), emissiveIntensity: 0 });
        this.part(L, C, new THREE.CapsuleGeometry(0.34, 0.5, 6, 18), body, [0, 0.59, -0.1]);
        this.part(L, C, new THREE.SphereGeometry(0.28, 24, 18), this.std(L, C, "#7aa2ea", { emissive: new THREE.Color("#3b6fd6"), emissiveIntensity: 0 }), [0, 1.48, -0.1]);
        this.part(L, C, new RoundedBoxGeometry(1.25, 0.62, 0.55, 3, 0.06), tile(), [0, 0.31, 0.62]);
        this.part(L, C, new THREE.BoxGeometry(0.95, 0.04, 0.55), this.std(L, C, 0x3a3f47), [0, 0.64, 0.62]);
        const scr = this.part(L, C, new THREE.BoxGeometry(0.95, 0.58, 0.03), this.std(L, C, 0x3a3f47), [0, 0.93, 0.9]);
        scr.rotation.x = 0.22;
        C.top.set(0, 1.8, 0);
        labelY = 2.2;
        break;
      }
    }

    if (id !== "researcher") {
      C.label = this.makeLabel("a3-comp", color, 1, (el) => { el.textContent = SHORT[id] ?? def.label; }).obj;
      C.label.position.set(0, labelY, 0);
      C.label.visible = false;
      C.group.add(C.label);
    }
  }

  private buildFlow(def: Flow) {
    const a = this.comps.get(def.from), b = this.comps.get(def.to);
    if (!a || !b) return;
    const kc = new THREE.Color(FLOW_KINDS[def.kind].color);
    const tubeMat = new THREE.MeshBasicMaterial({ color: kc, transparent: true, opacity: 0.3, depthWrite: false });
    const coneMat = new THREE.MeshBasicMaterial({ color: kc, transparent: true, opacity: 0.6, depthWrite: false });
    const tube = new THREE.Mesh(new THREE.BufferGeometry(), tubeMat);
    const hit = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ visible: false }));
    hit.userData.pick = { layer: a.layer, flow: def.id } satisfies PickData;
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.1, 0.26, 12), coneMat);
    const n = def.main ? 5 : 4;
    const pg = new THREE.BufferGeometry();
    pg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const pointMat = new THREE.PointsMaterial({
      color: kc, map: this.dot, size: 0.42, sizeAttenuation: true, transparent: true, depthWrite: false,
      blending: this.pal.dark ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    const points = new THREE.Points(pg, pointMat);
    points.frustumCulled = false;
    const label = this.makeLabel("a3-flow", FLOW_KINDS[def.kind].color, 2, (el) => { el.textContent = def.label; }).obj;
    label.visible = false;
    this.world.add(tube, hit, cone, points, label);
    const F: FlowNode = {
      def, a, b, curve: new THREE.CubicBezierCurve3(), len: 1, tube, hit, cone, points, label, tubeMat, coneMat, pointMat,
      vis: 0, visTarget: 0, hi: false, phase: Math.random(),
    };
    this.updateFlowGeometry(F);
    this.flows.push(F);
  }

  private anchor(c: CompNode, which: "top" | "bottom") {
    return c.group.localToWorld(which === "top" ? c.top.clone() : new V3(0, 0, 0));
  }
  private updateFlowGeometry(F: FlowNode) {
    const at = this.anchor(F.a, "top"), bt = this.anchor(F.b, "top");
    const horiz0 = Math.hypot(bt.x - at.x, bt.z - at.z);
    const dy0 = bt.y - at.y;
    const vertical = Math.abs(dy0) > 2 || Math.abs(dy0) > 0.5 * horiz0;
    let p0 = at, p3 = bt;
    if (vertical && dy0 < 0) { p0 = this.anchor(F.a, "bottom"); p3 = bt; }
    else if (vertical && dy0 > 0) { p0 = at; p3 = this.anchor(F.b, "bottom"); }
    const d = p3.clone().sub(p0);
    let c1: THREE.Vector3, c2: THREE.Vector3;
    if (vertical) {
      c1 = new V3(p0.x, p0.y + d.y * 0.5, p0.z);
      c2 = new V3(p3.x, p3.y - d.y * 0.5, p3.z);
    } else {
      const lift = Math.min(0.28 * Math.hypot(d.x, d.z) + 0.3, 2.2);
      c1 = p0.clone().lerp(p3, 0.3).add(new V3(0, lift, 0));
      c2 = p0.clone().lerp(p3, 0.7).add(new V3(0, lift, 0));
    }
    if (F.def.bend) { const bnd = new V3(...F.def.bend); c1.add(bnd); c2.add(bnd); }
    F.curve = new THREE.CubicBezierCurve3(p0, c1, c2, p3);
    F.len = Math.max(F.curve.getLength(), 0.1);
    F.tube.geometry.dispose();
    F.tube.geometry = new THREE.TubeGeometry(F.curve, 48, 0.032, 6, false);
    F.hit.geometry.dispose();
    F.hit.geometry = new THREE.TubeGeometry(F.curve, 18, 0.2, 5, false);
    F.cone.position.copy(F.curve.getPointAt(0.965));
    F.cone.quaternion.setFromUnitVectors(UP, F.curve.getTangentAt(0.965).normalize());
    F.label.position.copy(F.curve.getPointAt(0.5));
  }

  // ── State → targets ─────────────────────────────────────────────────────────
  /** Fly the camera to a part (keeping the current viewing direction) or back to the whole tower. */
  private focusLayer(layer: LayerId | null) {
    const dir = this.camera.position.clone().sub(this.controls.target);
    const sph = new THREE.Spherical().setFromVector3(dir);
    if (!layer) {
      this.pullDir = null;
      this.targetGoal = HOME_TARGET.clone();
      sph.radius = this.homeCamera().distanceTo(HOME_TARGET);
      sph.phi = THREE.MathUtils.clamp(sph.phi, 0.7, 1.25);
      this.cameraGoal = HOME_TARGET.clone().add(new V3().setFromSpherical(sph));
      return;
    }
    const h = new V3(dir.x, 0, dir.z);
    if (h.lengthSq() < 1e-6) h.set(0, 0, 1);
    this.pullDir = h.normalize();
    const center = LAYER_POS[layer].clone().addScaledVector(this.pullDir, PULL[layer]).add(new V3(0, LIFT + 0.4, 0));
    this.targetGoal = center;
    sph.radius = FOCUS_DIST[layer];
    sph.phi = THREE.MathUtils.clamp(sph.phi, 0.78, 1.05);   // look down onto the raised part
    this.cameraGoal = center.clone().add(new V3().setFromSpherical(sph));
  }
  private applyState() {
    const focus = this.tourLayer ?? this.selected;
    for (const L of this.layers.values()) {
      L.offTarget = focus === L.id && this.pullDir
        ? this.pullDir.clone().multiplyScalar(PULL[L.id]).add(new V3(0, LIFT, 0)) : new V3();
      L.mainRec.prio = focus === null || focus === L.id ? 0 : 3;   // dimmed tiers yield their label space
      L.dimTarget = focus === null || focus === L.id ? 1 : DIM;
      for (const c of L.comps) if (c.label) c.label.visible = focus === L.id;
      L.mainEl.classList.toggle("is-focus", focus === L.id);
      L.mainEl.classList.toggle("is-dim", focus !== null && focus !== L.id);
      if (this.snap) { L.off.copy(L.offTarget); L.dim = L.dimTarget; L.group.position.copy(L.base).add(L.off); this.applyDim(L); }
    }
    for (const r of this.labelRecs) r.w = 0;
    const hiSet = this.pinned ? new Set([this.pinned]) : this.tourFlows;
    for (const F of this.flows) {
      const touches = !!this.selected && (F.a.layer === this.selected || F.b.layer === this.selected);
      if (hiSet) { F.hi = hiSet.has(F.def.id); F.visTarget = F.hi ? 1 : F.def.main ? 0.1 : 0; }
      else if (this.selected) { F.hi = touches; F.visTarget = touches ? 1 : F.def.main ? 0.14 : 0; }
      else { F.hi = false; F.visTarget = F.def.main ? 1 : 0; }
      F.label.visible = (!!hiSet && F.hi) || (this.flowLabels && F.visTarget > 0.5);
      if (this.snap) F.vis = F.visTarget;
    }
    if (this.snap) { this.scene.updateMatrixWorld(true); this.flows.forEach((F) => this.updateFlowGeometry(F)); }
    this.snap = false;
  }
  private applyDim(L: LayerNode) {
    for (const r of L.mats) {
      const o = r.base * L.dim;
      r.m.opacity = o;
      r.m.transparent = o < 0.999;
      if (!r.fixedDepth) r.m.depthWrite = L.dim > 0.9;
    }
    L.mainEl.style.opacity = String(0.35 + 0.65 * L.dim);
  }

  // ── Events ──────────────────────────────────────────────────────────────────
  private onControlStart = () => {
    this.userMoved = true;
    this.controls.autoRotate = false;
    this.targetGoal = null;
    this.cameraGoal = null;
    this.ev.onInteract();
  };
  private onMove = (e: PointerEvent) => {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.inside = true;
    this.ev.onPointer(e.clientX, e.clientY);
  };
  private onLeave = () => { this.inside = false; };
  private onDown = (e: PointerEvent) => { this.down = { x: e.clientX, y: e.clientY, t: performance.now() }; };
  private onUp = (e: PointerEvent) => {
    if (e.button !== 0) return;
    if (Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) > 6 || performance.now() - this.down.t > 700) return;
    this.onMove(e);
    const p = this.pick();
    this.ev.onInteract();
    if (!p) {
      if (this.selected || this.pinned || this.tourLayer) { this.select(null); this.ev.onSelect(null); this.ev.onFlow(null); }
      return;
    }
    if (p.kind === "flow") {
      this.pinned = this.pinned === p.id ? null : p.id;
      this.applyState();
      this.ev.onFlow(this.pinned);
      return;
    }
    if (p.layer !== this.selected) { this.select(p.layer); this.ev.onSelect(p.layer); return; }
    if (p.kind === "comp") { this.selectComp(p.id); this.ev.onComp(p.id); }
  };
  private onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape" && (this.selected || this.pinned || this.tourLayer)) {
      this.select(null); this.ev.onSelect(null); this.ev.onFlow(null);
    }
  };

  private pick(): Pick | null {
    if (!this.inside) return null;
    this.ray.setFromCamera(this.ptr, this.camera);
    const objs = this.pickables.concat(this.flows.filter((f) => f.vis > 0.3).map((f) => f.hit));
    const hit = this.ray.intersectObjects(objs, false)[0];
    if (!hit) return null;
    const d = hit.object.userData.pick as PickData;
    if (d.flow) return { kind: "flow", id: d.flow };
    const elevated = this.selected === d.layer || this.tourLayer === d.layer;
    if (d.comp && elevated) return { kind: "comp", id: d.comp, layer: d.layer, elevated: true };
    return { kind: "layer", id: d.layer, layer: d.layer, elevated };
  }

  // ── Frame loop ──────────────────────────────────────────────────────────────
  private frame = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.frame);
    if (!this.active) return;
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;
    const k = 1 - Math.pow(0.002, dt);                // frame-rate independent easing

    // hover
    const p = this.pick();
    const same = (a: Pick | null, b: Pick | null) =>
      a === b || (!!a && !!b && a.kind === b.kind && a.id === b.id && ("elevated" in a ? a.elevated : 0) === ("elevated" in b ? b.elevated : 0));
    if (!same(p, this.hover)) { this.hover = p; this.ev.onHover(p); }
    this.renderer.domElement.style.cursor = p ? "pointer" : "grab";
    const hoverComp = p?.kind === "comp" ? p.id : this.extHover;
    const hoverLayer = p && p.kind === "layer" && !p.elevated ? p.layer : null;

    // layers: lift, dim, glow
    const moved = new Set<LayerId>();
    for (const L of this.layers.values()) {
      const before = L.off.clone();
      L.off.lerp(L.offTarget, k);
      if (L.off.distanceTo(L.offTarget) < 1e-3) L.off.copy(L.offTarget);
      if (!L.off.equals(before)) { L.group.position.copy(L.base).add(L.off); moved.add(L.id); }
      const pd = L.dim;
      L.dim += (L.dimTarget - L.dim) * k;
      if (Math.abs(L.dimTarget - L.dim) < 1e-3) L.dim = L.dimTarget;
      if (L.dim !== pd) this.applyDim(L);
      const focus = (this.tourLayer ?? this.selected) === L.id;
      L.glowTarget = focus ? 0.22 : hoverLayer === L.id ? 0.35 : 0;
      L.glow += (L.glowTarget - L.glow) * k;
      L.slabMat.emissiveIntensity = L.glow;
      L.edge.color.set(LAYER[L.id].color).offsetHSL(0, 0, focus || hoverLayer === L.id ? 0.08 : 0);
      for (const c of L.comps) {
        c.hoverTarget = hoverComp === c.id ? 1 : 0;
        const ph = c.hover;
        c.hover += (c.hoverTarget - c.hover) * k;
        if (Math.abs(c.hoverTarget - c.hover) < 1e-3) c.hover = c.hoverTarget;
        if (c.hover !== ph) { c.group.position.y = c.baseY + c.hover * 0.28; moved.add(L.id); }
        const em = c.hover * 0.3 + (this.selectedComp === c.id ? 0.18 : 0);
        for (const m of c.mats) if (!this.animated.has(m)) m.emissiveIntensity = em;
      }
    }
    if (moved.size) {
      this.scene.updateMatrixWorld();
      for (const F of this.flows) if (moved.has(F.a.layer) || moved.has(F.b.layer)) this.updateFlowGeometry(F);
    }

    // flows: fade + travelling packets
    const speed = this.reduced ? 1.1 : 2.3;
    for (const F of this.flows) {
      F.vis += (F.visTarget - F.vis) * k;
      if (Math.abs(F.visTarget - F.vis) < 1e-3) F.vis = F.visTarget;
      const show = F.vis > 0.01;
      F.tube.visible = F.cone.visible = F.points.visible = show;
      if (!show) continue;
      F.tubeMat.opacity = F.vis * (F.hi ? 0.85 : 0.32);
      F.coneMat.opacity = F.vis * (F.hi ? 0.95 : 0.55);
      F.pointMat.opacity = F.vis;
      F.pointMat.size = F.hi ? 0.6 : 0.42;
      const arr = (F.points.geometry.getAttribute("position") as THREE.BufferAttribute);
      const n = arr.count, v = (speed * (F.hi ? 1.35 : 1)) / F.len;
      for (let i = 0; i < n; i++) {
        const u = (t * v + i / n + F.phase) % 1;
        const q = F.curve.getPointAt(u);
        arr.setXYZ(i, q.x, q.y, q.z);
      }
      arr.needsUpdate = true;
    }

    // living details
    for (const nrn of this.neurons) {
      const s = Math.max(0, Math.sin(t * 2.6 - nrn.layer * 0.95));
      nrn.m.emissiveIntensity = 0.18 + 0.8 * s * s;
    }
    for (const d of this.dies) d.emissiveIntensity = 0.55 + 0.35 * Math.sin(t * 1.6);
    for (const s of this.spinners) s.rotation.z += dt * 0.35;

    // camera focus
    if (this.targetGoal) {
      const before = this.controls.target.clone();
      this.controls.target.lerp(this.targetGoal, k * 0.6);
      if (!this.cameraGoal) this.camera.position.add(this.controls.target.clone().sub(before));
      if (this.controls.target.distanceTo(this.targetGoal) < 0.01) this.targetGoal = null;
    }
    if (this.cameraGoal) {
      this.camera.position.lerp(this.cameraGoal, k * 0.6);
      if (this.camera.position.distanceTo(this.cameraGoal) < 0.02) this.cameraGoal = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
    this.declutter();
  };
}
