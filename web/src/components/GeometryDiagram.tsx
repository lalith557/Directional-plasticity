import { C } from "../lib/colors";

/**
 * Schematic of the fixed-length update decomposition (matched step).
 * gamma in [0,1] rotates the unit update from tangential (task descent)
 * toward the inward radial direction (-theta). Purely illustrative;
 * measured accuracy/norm live in the results panel.
 */
export function GeometryDiagram({ gamma }: { gamma: number }) {
  const O = { x: 58, y: 152 };
  const T = { x: 198, y: 152 };
  const L = 80;
  const phi = gamma * (Math.PI / 2);
  const A = { x: T.x, y: T.y - L * Math.cos(phi) }; // tangential leg (up)
  const U = { x: T.x - L * Math.sin(phi), y: A.y }; // + radial leg (inward/left)

  const arrow = (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    color: string,
    w = 1.8,
    dash?: string
  ) => (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={color}
      strokeWidth={w}
      strokeDasharray={dash}
      markerEnd={`url(#ah-${color.replace("#", "")})`}
    />
  );

  const colors = [C.blue, C.amber, C.red, C.ink3];

  return (
    <svg viewBox="0 0 300 210" width="100%" style={{ maxWidth: 320 }} role="img"
         aria-label="Direction decomposition schematic">
      <defs>
        {colors.map((col) => (
          <marker
            key={col}
            id={`ah-${col.replace("#", "")}`}
            markerWidth="7"
            markerHeight="7"
            refX="5.5"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L6,3 L0,6 Z" fill={col} />
          </marker>
        ))}
      </defs>

      {/* norm sphere (quarter arc through T around O) */}
      <path
        d={`M ${O.x + (T.x - O.x)} ${O.y} A ${T.x - O.x} ${T.x - O.x} 0 0 0 ${O.x} ${
          O.y - (T.x - O.x)
        }`}
        fill="none"
        stroke={C.border}
        strokeWidth="1"
        strokeDasharray="2 3"
      />

      {/* theta */}
      {arrow(O.x, O.y, T.x, T.y, C.blue, 2)}
      <text x={(O.x + T.x) / 2 - 4} y={O.y + 16} fontSize="12" fill={C.blue} fontStyle="italic">
        &#952;
      </text>
      <circle cx={O.x} cy={O.y} r="2.5" fill={C.ink2} />

      {/* tangential leg */}
      {arrow(T.x, T.y, A.x, A.y, C.blue, 1.6, "4 3")}
      {/* radial leg (inward) */}
      {arrow(A.x, A.y, U.x, U.y, C.amber, 1.6, "4 3")}
      {/* resultant fixed-length update */}
      {arrow(T.x, T.y, U.x, U.y, C.red, 2.2)}

      {/* labels */}
      <text x={A.x + 6} y={(T.y + A.y) / 2} fontSize="9.5" fill={C.blue}>
        (1&minus;&#947;) tangential
      </text>
      <text x={(A.x + U.x) / 2 - 30} y={A.y - 6} fontSize="9.5" fill={C.amber}>
        &#947; radial (inward)
      </text>
      <text x={U.x - 8} y={U.y - 8} fontSize="9.5" fill={C.red} textAnchor="end">
        fixed-length update
      </text>

      <text x={6} y={16} fontSize="9" fill={C.ink3}>
        Radial (inward) &#8594; &minus;&#952;
      </text>
    </svg>
  );
}
