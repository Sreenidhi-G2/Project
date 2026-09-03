import { EmptyState } from "../components/common/EmptyState";
import { MediaTypeTabs } from "../components/upload/MediaTypeTabs";
import { UploadDropzone } from "../components/upload/UploadDropzone";
import { VerificationResult } from "../components/verification/VerificationResult";
import { useVerification } from "../hooks/useVerification";

export default function Verify() {
  const { mediaType, setMediaType, submit, isLoading, error, result } = useVerification();
  const accept = "video/mp4,video/quicktime,video/x-msvideo";

  return (
    <div className="grid gap-6 xl:grid-cols-[25rem_minmax(0,1fr)]">
      <section className="panel p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="muted-label">Verification</div>
            <h2 className="mt-2 text-lg font-semibold text-white">Submit media</h2>
          </div>
          <MediaTypeTabs
            value={mediaType}
            onChange={(value) => {
              if (value === "video") {
                setMediaType(value);
              }
            }}
          />
        </div>

        <div className="mt-5">
          <UploadDropzone accept={accept} disabled={isLoading} onFileSelected={submit} />
        </div>

        {isLoading ? (
          <div className="mt-4 rounded-lg border border-signal-info/20 bg-signal-info/10 p-3 text-sm text-signal-info">
            Running backend verification...
          </div>
        ) : null}

        {error ? (
          <div className="mt-4 rounded-lg border border-signal-fake/20 bg-signal-fake/10 p-3 text-sm text-signal-fake">
            {error}
          </div>
        ) : null}
      </section>

      {result ? (
        <VerificationResult result={result} />
      ) : (
        <EmptyState title="No verification result yet">
          Results will appear after the backend finishes processing the uploaded file.
        </EmptyState>
      )}
    </div>
  );
}
