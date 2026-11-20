export function SweepSlider({
  values,
  index,
  onChange,
  label,
  format,
}: {
  values: number[];
  index: number;
  onChange: (i: number) => void;
  label: string;
  format: (v: number) => string;
}) {
  return (
    <div>
      <div className="row between aic mb2">
        <span className="field-label" style={{ margin: 0 }}>
          {label}
        </span>
        <span className="mono" style={{ fontSize: 15, fontWeight: 600, color: "var(--blue)" }}>
          {format(values[index])}
        </span>
      </div>
      <input
        type="range"
        className="slider"
        min={0}
        max={values.length - 1}
        step={1}
        value={index}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={label}
        aria-valuetext={format(values[index])}
      />
      <div className="slider-ticks">
        {values.map((v, i) => (
          <span
            key={i}
            className={`slider-tick ${i === index ? "active" : ""}`}
            onClick={() => onChange(i)}
            role="button"
          >
            {format(v)}
          </span>
        ))}
      </div>
      <div className="small muted mt2">
        Snaps to measured points only &mdash; no values are interpolated.
      </div>
    </div>
  );
}
