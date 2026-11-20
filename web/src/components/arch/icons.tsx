/* Simple stroke icons (24×24 grid) shared by the 2D diagram (SVG) and the 3D scene
   (drawn onto canvas textures with Path2D). Hand-drawn geometric shapes. */

const circle = (cx: number, cy: number, r: number) =>
  `M${cx + r} ${cy} A${r} ${r} 0 1 1 ${cx - r} ${cy} A${r} ${r} 0 1 1 ${cx + r} ${cy} Z`;

export const ICONS = {
  home: "M3 11 L12 4 L21 11 M5.5 9.5 V20 H18.5 V9.5 M10 20 V14.5 H14 V20",
  flask: "M9 3 H15 M10 3 V9.5 L4.6 18.6 A1.6 1.6 0 0 0 6 21 H18 A1.6 1.6 0 0 0 19.4 18.6 L14 9.5 V3 M7.2 15 H16.8",
  play: "M7 4.5 L19 12 L7 19.5 Z",
  bars: "M6 20 V14 M12 20 V9 M18 20 V4 M3 20.5 H21",
  scale: "M12 3.5 V20.5 M7.5 20.5 H16.5 M4 7 H20 M6 7 L3 13.5 A3 3 0 0 0 9 13.5 Z M18 7 L15 13.5 A3 3 0 0 0 21 13.5 Z",
  cube: "M12 2.5 L20.5 7.2 V16.8 L12 21.5 L3.5 16.8 V7.2 Z M3.5 7.2 L12 12 L20.5 7.2 M12 12 V21.5",
  layers: "M12 3 L21 8 L12 13 L3 8 Z M3 12 L12 17 L21 12 M3 16 L12 21 L21 16",
  file: "M6 2.5 H14 L19.5 8 V21.5 H6 Z M14 2.5 V8 H19.5 M9 13 H16.5 M9 17 H16.5",
  book: "M2.5 5 C5.5 4 9 4 12 6 C15 4 18.5 4 21.5 5 V19 C18.5 18 15 18 12 20 C9 18 5.5 18 2.5 19 Z M12 6 V20",
  gear: `${circle(12, 12, 3.6)} M12 2.5 V5.5 M12 18.5 V21.5 M2.5 12 H5.5 M18.5 12 H21.5 M5.3 5.3 L7.4 7.4 M16.6 16.6 L18.7 18.7 M5.3 18.7 L7.4 16.6 M16.6 7.4 L18.7 5.3`,
  database: "M4 6 A8 3 0 1 0 20 6 A8 3 0 1 0 4 6 M4 6 V18 A8 3 0 0 0 20 18 V6 M4 12 A8 3 0 0 0 20 12",
  trend: "M3 17.5 L9 11.5 L13 15.5 L21 7.5 M15 7.5 H21 V13.5",
  chip: "M7 7 H17 V17 H7 Z M10 10 H14 V14 H10 Z M9.5 3 V7 M14.5 3 V7 M9.5 17 V21 M14.5 17 V21 M3 9.5 H7 M3 14.5 H7 M17 9.5 H21 M17 14.5 H21",
  shield: "M12 2.5 L20 5.5 V11 C20 16 16.5 19.8 12 21.5 C7.5 19.8 4 16 4 11 V5.5 Z M8.5 12 L11 14.5 L15.5 9.5",
  image: `M3 5 H21 V19 H3 Z M3 16 L8 11 L12 15 L15 12 L21 17 ${circle(15.5, 8.8, 1.5)}`,
  refresh: "M19.5 10.5 A7.8 7.8 0 0 0 5.6 7 M3.5 3.5 V8.5 H8.5 M4.5 13.5 A7.8 7.8 0 0 0 18.4 17 M20.5 20.5 V15.5 H15.5",
  sigma: "M18 4 H6 L12.5 12 L6 20 H18",
  network: `${circle(5, 7, 2)} ${circle(5, 17, 2)} ${circle(19, 12, 2)} ${circle(12, 12, 2)} M7 7.8 L10.2 11 M7 16.2 L10.2 13 M14 12 H17`,
  geometry: "M4 20 L20 4 M13.5 4 H20 V10.5 M4 13.5 V20 H10.5",
  user: `${circle(12, 7.5, 3.8)} M4 21 C4 15.8 7.6 13.5 12 13.5 C16.4 13.5 20 15.8 20 21`,
  folder: "M3 6 H9.5 L11.5 8.5 H21 V19 H3 Z",
  input: "M3 12 H14 M10.5 8.5 L14 12 L10.5 15.5 M14 4 H21 V20 H14",
  target: `${circle(12, 12, 8.5)} ${circle(12, 12, 4.5)} ${circle(12, 12, 0.8)}`,
  flame: "M12 2.5 C13 6 17 8 17 13.5 A5 5 0 0 1 7 13.5 C7 10.5 9 9 9.5 6.5 C11 8 11.5 9.5 11.5 11 C13 9.5 12.8 6 12 2.5 Z",
  api: "M8 6 L3 12 L8 18 M16 6 L21 12 L16 18 M13.5 4.5 L10.5 19.5",
} as const;

export type IconName = keyof typeof ICONS;

export function Icon({ name, size = 20, stroke = 1.8 }: { name: IconName; size?: number; stroke?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor"
         strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={ICONS[name]} />
    </svg>
  );
}
