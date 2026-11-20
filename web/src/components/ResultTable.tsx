import { ReactNode, useMemo, useState } from "react";

export interface Column<T> {
  key: string;
  header: ReactNode;
  numeric?: boolean;
  sortable?: boolean;
  /** value used for sorting; falls back to render output */
  value?: (row: T) => number | string | null;
  render: (row: T) => ReactNode;
}

export function ResultTable<T>({
  columns,
  rows,
  getRowId,
  onRowClick,
  selectable = false,
  selected,
  onToggleSelect,
  initialSort,
}: {
  columns: Column<T>[];
  rows: T[];
  getRowId: (row: T) => string;
  onRowClick?: (row: T) => void;
  selectable?: boolean;
  selected?: Set<string>;
  onToggleSelect?: (id: string) => void;
  initialSort?: { key: string; dir: "asc" | "desc" };
}) {
  const [sort, setSort] = useState<{ key: string; dir: "asc" | "desc" } | null>(
    initialSort ?? null
  );

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col) return rows;
    const val = (r: T) => (col.value ? col.value(r) : (col.render(r) as unknown as string));
    return [...rows].sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      let cmp: number;
      if (typeof va === "number" && typeof vb === "number") cmp = va - vb;
      else cmp = String(va).localeCompare(String(vb));
      return sort.dir === "asc" ? cmp : -cmp;
    });
  }, [rows, sort, columns]);

  const toggleSort = (key: string) => {
    setSort((s) => {
      if (!s || s.key !== key) return { key, dir: "desc" };
      if (s.dir === "desc") return { key, dir: "asc" };
      return null;
    });
  };

  return (
    <div className="tbl-wrap">
      <table className="tbl">
        <thead>
          <tr>
            {selectable && <th style={{ width: 34 }} />}
            {columns.map((c) => (
              <th
                key={c.key}
                className={`${c.numeric ? "num" : ""} ${c.sortable ? "sortable" : ""}`}
                onClick={c.sortable ? () => toggleSort(c.key) : undefined}
              >
                {c.header}
                {sort?.key === c.key && (
                  <span className="arrow">{sort.dir === "asc" ? "▲" : "▼"}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const id = getRowId(r);
            const isSel = selected?.has(id);
            return (
              <tr
                key={id}
                className={`${onRowClick ? "clickable" : ""} ${isSel ? "selected" : ""}`}
                onClick={onRowClick ? () => onRowClick(r) : undefined}
              >
                {selectable && (
                  <td onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      className="ck"
                      checked={!!isSel}
                      onChange={() => onToggleSelect?.(id)}
                      aria-label={`Select ${id}`}
                    />
                  </td>
                )}
                {columns.map((c) => (
                  <td key={c.key} className={c.numeric ? "num" : ""}>
                    {c.render(r)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
