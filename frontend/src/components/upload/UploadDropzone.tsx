import { UploadCloud } from "lucide-react";
import type { ChangeEvent } from "react";

interface UploadDropzoneProps {
  accept: string;
  disabled?: boolean;
  onFileSelected: (file: File) => void;
}

export function UploadDropzone({ accept, disabled, onFileSelected }: UploadDropzoneProps) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      onFileSelected(file);
    }
  }

  return (
    <label className="block cursor-pointer rounded-lg border border-dashed border-white/15 bg-surface-850 p-8 transition hover:border-signal-info/50 hover:bg-surface-800">
      <input
        className="sr-only"
        type="file"
        accept={accept}
        disabled={disabled}
        onChange={handleChange}
      />
      <div className="flex flex-col items-center text-center">
        <div className="grid h-12 w-12 place-items-center rounded-lg bg-signal-info/15 text-signal-info">
          <UploadCloud size={23} />
        </div>
        <div className="mt-4 text-sm font-semibold text-white">Select media for analysis</div>
        <div className="mt-1 text-sm text-zinc-400">
          The file is sent to the configured FastAPI backend.
        </div>
      </div>
    </label>
  );
}
