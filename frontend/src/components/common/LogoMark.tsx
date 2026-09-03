interface LogoMarkProps {
  compact?: boolean;
}

export function LogoMark({ compact = false }: LogoMarkProps) {
  return (
    <div className="flex items-center gap-3">
      <div className="relative grid h-10 w-10 place-items-center rounded-md border border-signal-info/30 bg-signal-info/10 shadow-insetline">
        <div className="h-5 w-5 rotate-45 rounded-[3px] border-2 border-signal-info" />
        <div className="absolute h-1.5 w-1.5 rounded-full bg-signal-cyan" />
      </div>
      {!compact ? (
        <div>
          <div className="text-[1.05rem] font-semibold leading-5 text-white">VeriFace</div>
          <div className="mt-0.5 text-xs text-slate-500">KYC fraud console</div>
        </div>
      ) : null}
    </div>
  );
}
