interface MediaTypeTabsProps {
  value:  "video";
  onChange: (value: "video") => void;
}

export function MediaTypeTabs({ value, onChange }: MediaTypeTabsProps) {
  return (
    <div className="inline-flex rounded-lg border border-white/10 bg-surface-850 p-1">
      {(["video"] as const).map((item) => (
        <button
          key={item}
          type="button"
          onClick={() => onChange(item)}
          className={`h-9 rounded px-4 text-sm font-medium capitalize transition ${
            value === item ? "bg-white/10 text-white" : "text-zinc-400 hover:text-white"
          }`}
        >
          {item}
        </button>
      ))}
    </div>
  );
}
