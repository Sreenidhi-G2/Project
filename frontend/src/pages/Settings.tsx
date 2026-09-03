import { API_BASE_URL } from "../api/client";

export default function Settings() {
  return (
    <section className="panel p-5">
      <div className="muted-label">Configuration</div>
      <h2 className="mt-2 text-lg font-semibold text-white">Frontend settings</h2>
      <div className="mt-5 rounded-lg border border-white/10 bg-surface-850 p-4">
        <div className="muted-label">VITE_API_BASE_URL</div>
        <div className="mt-2 break-all text-sm text-zinc-300">{API_BASE_URL}</div>
      </div>
    </section>
  );
}
