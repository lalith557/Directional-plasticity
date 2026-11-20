import { ReactNode } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ErrorBar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ComposedChart,
  Line,
  Scatter,
  ScatterChart,
  ZAxis,
  Legend,
  ReferenceLine,
} from "recharts";
import { C } from "../lib/colors";

/* ---------- frame ---------- */
export function ChartFrame({
  title,
  children,
  height = 220,
  right,
}: {
  title?: string;
  children: ReactNode;
  height?: number;
  right?: ReactNode;
}) {
  return (
    <div>
      {(title || right) && (
        <div className="row between aic mb3">
          {title && <div className="card-title">{title}</div>}
          {right}
        </div>
      )}
      <div style={{ width: "100%", height }}>{children}</div>
    </div>
  );
}

// Getter object so axis style re-reads CSS variables on each spread,
// keeping chart ink/grid colors in sync with the active theme.
const axis = {
  get stroke() { return C.border; },
  get tick() { return { fill: C.ink3, fontSize: 11 }; },
} as any;

/* ---------- generic tooltip ---------- */
export function SciTooltip({
  active,
  payload,
  title,
  rows,
}: any & {
  title?: (p: any) => string;
  rows?: (p: any) => { label: string; value: ReactNode }[];
}) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tt">
      {title && <div className="tt-title">{title(p)}</div>}
      {rows &&
        rows(p).map((r: any, i: number) => (
          <div className="tt-row" key={i}>
            <span>{r.label}</span>
            <b>{r.value}</b>
          </div>
        ))}
    </div>
  );
}

/* ---------- bar chart (single metric, error bars, per-bar color) ---------- */
export function BarSci({
  data,
  xKey,
  yKey,
  stdKey,
  color = C.blue,
  colorFor,
  height = 210,
  yDomain,
  yTickFmt,
  valueFmt,
  seedN,
  metricLabel,
  onBarClick,
  reference0,
}: {
  data: any[];
  xKey: string;
  yKey: string;
  stdKey?: string;
  color?: string;
  colorFor?: (d: any) => string;
  height?: number;
  yDomain?: [number, number];
  yTickFmt?: (v: number) => string;
  valueFmt?: (v: number) => string;
  seedN?: number;
  metricLabel?: string;
  onBarClick?: (d: any) => void;
  reference0?: boolean;
}) {
  return (
    <ChartFrame height={height}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -8 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <XAxis dataKey={xKey} {...axis} interval={0} />
          <YAxis
            {...axis}
            domain={yDomain}
            tickFormatter={yTickFmt ?? ((v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2)))}
            width={44}
          />
          <Tooltip
            cursor={{ fill: "rgba(23,23,23,0.04)" }}
            content={(p: any) => (
              <SciTooltip
                {...p}
                title={(d: any) => d.name ?? d[xKey]}
                rows={(d: any) => [
                  {
                    label: metricLabel ?? "Mean",
                    value: valueFmt ? valueFmt(d[yKey]) : d[yKey],
                  },
                  ...(stdKey && d[stdKey] != null
                    ? [{ label: "Std", value: valueFmt ? valueFmt(d[stdKey]) : d[stdKey] }]
                    : []),
                  ...(seedN ? [{ label: "Seeds", value: seedN }] : []),
                ]}
              />
            )}
          />
          {reference0 && (
            <ReferenceLine y={0} stroke={C.ink3} strokeWidth={1} />
          )}
          <Bar
            dataKey={yKey}
            radius={[2, 2, 0, 0]}
            maxBarSize={46}
            onClick={onBarClick}
            cursor={onBarClick ? "pointer" : "default"}
            isAnimationActive={false}
          >
            {data.map((d, i) => (
              <Cell key={i} fill={colorFor ? colorFor(d) : color} />
            ))}
            {stdKey && (
              <ErrorBar dataKey={stdKey} width={3} strokeWidth={1} stroke={C.ink2} direction="y" />
            )}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

/* ---------- line chart (sweeps; optional dual axis + log right) ---------- */
export interface LineSpec {
  key: string;
  name: string;
  color: string;
  stdKey?: string;
  axis?: "left" | "right";
}
export function LineSci({
  data,
  xKey,
  xTickFmt,
  lines,
  height = 220,
  leftDomain,
  rightLog = false,
  leftFmt,
  rightFmt,
  xLabel,
  legend = false,
  highlightX,
  tooltip,
}: {
  data: any[];
  xKey: string;
  xTickFmt?: (v: any) => string;
  lines: LineSpec[];
  height?: number;
  leftDomain?: [number, number];
  rightLog?: boolean;
  leftFmt?: (v: number) => string;
  rightFmt?: (v: number) => string;
  xLabel?: string;
  legend?: boolean;
  highlightX?: any;
  tooltip?: (p: any) => { label: string; value: ReactNode }[];
}) {
  const hasRight = lines.some((l) => l.axis === "right");
  return (
    <ChartFrame height={height}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 10, right: hasRight ? 14 : 22, bottom: xLabel ? 22 : 8, left: 10 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          {highlightX != null && (
            <ReferenceLine x={highlightX} yAxisId="left" stroke={C.blue} strokeDasharray="3 3" strokeWidth={1.4} />
          )}
          <XAxis
            dataKey={xKey}
            {...axis}
            tickFormatter={xTickFmt}
            interval={0}
            padding={{ left: 10, right: 14 }}
            label={
              xLabel
                ? { value: xLabel, position: "insideBottom", dy: 16, fill: C.ink3, fontSize: 11 }
                : undefined
            }
          />
          <YAxis yAxisId="left" {...axis} domain={leftDomain} tickFormatter={leftFmt} width={54} />
          {hasRight && (
            <YAxis
              yAxisId="right"
              orientation="right"
              {...axis}
              scale={rightLog ? "log" : "auto"}
              domain={rightLog ? [0.05, "auto"] : undefined}
              allowDataOverflow={false}
              tickFormatter={rightFmt}
              width={52}
            />
          )}
          <Tooltip
            content={(p: any) => (
              <SciTooltip
                {...p}
                title={(d: any) => (xTickFmt ? xTickFmt(d[xKey]) : `${xKey} = ${d[xKey]}`)}
                rows={tooltip}
              />
            )}
          />
          {legend && <Legend verticalAlign="top" height={26} iconType="plainline" />}
          {lines.map((l) => (
            <Line
              key={l.key}
              yAxisId={l.axis ?? "left"}
              type="monotone"
              dataKey={l.key}
              name={l.name}
              stroke={l.color}
              strokeWidth={1.8}
              dot={{ r: 2.6, fill: l.color, strokeWidth: 0 }}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            >
              {l.stdKey && (
                <ErrorBar dataKey={l.stdKey} width={3} strokeWidth={1} stroke={l.color} direction="y" />
              )}
            </Line>
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

/* ---------- scatter (multi-series, optional log x, optional trend) ---------- */
export interface ScatterSeries {
  name: string;
  color: string;
  data: any[];
  trend?: { x: number; y: number }[];
}
export function ScatterSci({
  series,
  xKey = "x",
  yKey = "y",
  xLog = false,
  xLabel,
  yLabel,
  height = 260,
  legend = true,
  xFmt,
  tooltip,
  onPointClick,
  xDomain,
  yDomain,
}: {
  series: ScatterSeries[];
  xKey?: string;
  yKey?: string;
  xLog?: boolean;
  xLabel?: string;
  yLabel?: string;
  height?: number;
  legend?: boolean;
  xFmt?: (v: number) => string;
  tooltip?: (p: any) => { title: string; rows: { label: string; value: ReactNode }[] };
  onPointClick?: (p: any) => void;
  xDomain?: [any, any];
  yDomain?: [any, any];
}) {
  return (
    <ChartFrame height={height}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 10, right: 40, bottom: xLabel ? 28 : 10, left: 18 }}>
          <CartesianGrid stroke={C.grid} />
          <XAxis
            type="number"
            dataKey={xKey}
            {...axis}
            scale={xLog ? "log" : "auto"}
            domain={xDomain ?? (xLog ? [0.1, "auto"] : ["dataMin", "dataMax"])}
            allowDataOverflow={false}
            padding={{ left: 10, right: 20 }}
            tickFormatter={xFmt}
            label={
              xLabel
                ? { value: xLabel, position: "insideBottom", dy: 20, fill: C.ink3, fontSize: 11 }
                : undefined
            }
          />
          <YAxis
            type="number"
            dataKey={yKey}
            {...axis}
            domain={yDomain ?? ["auto", "auto"]}
            width={64}
            label={
              yLabel
                ? { value: yLabel, angle: -90, position: "insideLeft", dx: -4, dy: 40, fill: C.ink3, fontSize: 11 }
                : undefined
            }
          />
          <ZAxis range={[46, 46]} />
          <Tooltip
            cursor={{ stroke: C.border }}
            content={(p: any) => {
              if (!p.active || !p.payload || !p.payload.length) return null;
              const d = p.payload[0].payload;
              const t = tooltip?.(d);
              if (!t) return null;
              return (
                <div className="tt">
                  <div className="tt-title">{t.title}</div>
                  {t.rows.map((r, i) => (
                    <div className="tt-row" key={i}>
                      <span>{r.label}</span>
                      <b>{r.value}</b>
                    </div>
                  ))}
                </div>
              );
            }}
          />
          {legend && <Legend verticalAlign="top" height={26} iconType="circle" />}
          {series.map((s) => (
            <Scatter
              key={s.name}
              name={s.name}
              data={s.data}
              fill={s.color}
              onClick={onPointClick}
              cursor={onPointClick ? "pointer" : "default"}
              isAnimationActive={false}
            />
          ))}
          {series
            .filter((s) => s.trend)
            .map((s) => (
              <Scatter
                key={s.name + "-trend"}
                data={s.trend}
                line={{ stroke: s.color, strokeWidth: 1.4, strokeDasharray: "5 4" }}
                shape={() => <g />}
                legendType="none"
                isAnimationActive={false}
              />
            ))}
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

