import { useState } from "react";

export function ConfigPanel({
  title = "Configuration (excerpt)",
  lines,
}: {
  title?: string;
  lines: [string, string | number][];
}) {
  const [copied, setCopied] = useState(false);
  const text = lines.map(([k, v]) => `${k}: ${v}`).join("\n");
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard blocked; ignore */
    }
  };
  return (
    <div className="config">
      <div className="config-head">
        <span className="ct">{title}</span>
        <button className="copy-btn" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        {lines.map(([k, v], i) => (
          <div key={i}>
            <span className="k">{k}:</span> {String(v)}
          </div>
        ))}
      </pre>
    </div>
  );
}
