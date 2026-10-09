"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import gsap from "gsap";
import clsx from "clsx";
import { GravityField } from "@/components/canvas/gravity-field";
import { Nav } from "@/components/nav";
import {
  ApiError,
  deleteDocument,
  deleteIndex,
  getStats,
  uploadDocument,
  type DocumentInfo,
  type IndexStats,
} from "@/lib/api";

type UploadState =
  | { status: "idle" }
  | { status: "uploading"; fileName: string }
  | { status: "done"; chunks: number; fileName: string }
  | { status: "error"; message: string };

type DeletingState = Record<string, boolean>; // doc_id → is deleting

export default function IngestPage() {
  const [index, setIndex] = useState("faq");
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [uploadState, setUploadState] = useState<UploadState>({ status: "idle" });
  const [dragActive, setDragActive] = useState(false);
  const [confirmWipe, setConfirmWipe] = useState(false);
  const [deleting, setDeleting] = useState<DeletingState>({});
  const [loadingStats, setLoadingStats] = useState(false);
  const listRef = useRef<HTMLUListElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);

  const refreshStats = useCallback(async (idx: string) => {
    setLoadingStats(true);
    try {
      const s = await getStats(idx);
      setStats(s);
    } catch {
      setStats(null);
    } finally {
      setLoadingStats(false);
    }
  }, []);

  useEffect(() => {
    refreshStats(index);
  }, [index, refreshStats]);

  // Animate stat tiles whenever stats refresh
  useEffect(() => {
    if (!statsRef.current) return;
    gsap.fromTo(
      statsRef.current.querySelectorAll("[data-tile]"),
      { opacity: 0, y: 16 },
      { opacity: 1, y: 0, duration: 0.55, ease: "power3.out", stagger: 0.06 }
    );
  }, [stats]);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      const file = files?.[0];
      if (!file) return;
      setUploadState({ status: "uploading", fileName: file.name });
      try {
        const result = await uploadDocument(file, index);
        setUploadState({ status: "done", chunks: result.chunk_count, fileName: file.name });
        await refreshStats(index);
      } catch (err) {
        setUploadState({
          status: "error",
          message: err instanceof ApiError ? err.message : "Upload failed.",
        });
      }
    },
    [index, refreshStats]
  );

  async function handleDeleteDoc(doc: DocumentInfo) {
    setDeleting((d) => ({ ...d, [doc.doc_id]: true }));
    try {
      await deleteDocument(doc.doc_id, index);
      await refreshStats(index);
    } finally {
      setDeleting((d) => { const n = { ...d }; delete n[doc.doc_id]; return n; });
    }
  }

  async function handleReupload(doc: DocumentInfo) {
    // Re-uploading the same doc_id replaces its old chunks in the vector store.
    // We trigger the native file picker pre-filtered to the same filename hint.
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/pdf";
    input.onchange = () => handleFiles(input.files);
    input.click();
  }

  async function handleWipe() {
    if (!confirmWipe) {
      setConfirmWipe(true);
      return;
    }
    await deleteIndex(index);
    setConfirmWipe(false);
    await refreshStats(index);
    setUploadState({ status: "idle" });
  }

  const docCount = stats?.document_count ?? 0;
  const chunkCount = stats?.chunk_count ?? 0;
  const avgChunks = docCount > 0 ? Math.round(chunkCount / docCount) : 0;

  return (
    <main className="relative min-h-screen">
      <GravityField intensity={0.85} />
      <Nav />

      {/* ---------------------------------------------------------------- full-height dashboard grid */}
      <div className="relative z-10 mx-auto flex min-h-screen max-w-7xl flex-col px-4 pt-24 pb-8 md:flex-row md:gap-6 md:pt-28">

        {/* ====== LEFT PANEL: index selector + upload ====== */}
        <aside className="flex w-full flex-col gap-4 md:w-80 md:shrink-0">
          {/* Index name */}
          <div className="rounded-2xl border border-ink/10 bg-bg-elevated/85 backdrop-blur-2xl">
            <div className="flex items-center gap-3 px-5 py-4">
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-ink-faint">Index</span>
              <input
                id="index-input"
                value={index}
                onChange={(e) => {
                  setIndex(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ""));
                  setConfirmWipe(false);
                }}
                className="flex-1 bg-transparent font-mono text-sm text-ink outline-none placeholder:text-ink-faint"
                placeholder="faq"
                maxLength={50}
                spellCheck={false}
              />
              <span className="font-mono text-[10px] text-ink-faint">{index.length}/50</span>
            </div>
          </div>

          {/* Upload drop zone */}
          <div
            className="flex-1"
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => { e.preventDefault(); setDragActive(false); handleFiles(e.dataTransfer.files); }}
          >
            <label
              htmlFor="file-upload"
              className={clsx(
                "flex h-full min-h-[260px] cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border bg-bg-elevated/85 p-6 text-center backdrop-blur-2xl transition-all duration-300",
                dragActive
                  ? "border-accent shadow-[0_0_50px_-10px_var(--accent)] scale-[1.01]"
                  : "border-ink/10 hover:border-ink/20"
              )}
            >
              <input
                id="file-upload"
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={(e) => handleFiles(e.target.files)}
              />

              {uploadState.status === "uploading" ? (
                <>
                  <SpinnerIcon />
                  <p className="text-sm text-ink">Embedding chunks…</p>
                  <p className="font-mono text-[10px] text-ink-faint">{uploadState.fileName}</p>
                </>
              ) : (
                <>
                  <UploadIcon />
                  <p className="text-sm font-medium text-ink">Drop a PDF or click to browse</p>
                  <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">
                    One Q&amp;A per chunk · 10% overlap
                  </p>
                  <p className="font-mono text-[10px] text-ink-faint">Re-uploading replaces the old version</p>
                </>
              )}
            </label>
          </div>

          {/* Upload status feedback */}
          {uploadState.status === "done" && (
            <div className="rounded-xl border border-accent/30 bg-accent/8 px-4 py-3">
              <p className="font-mono text-[11px] text-accent">
                ✓ Indexed — {uploadState.chunks} chunk{uploadState.chunks === 1 ? "" : "s"} stored
              </p>
              <p className="mt-0.5 font-mono text-[10px] text-ink-faint">{uploadState.fileName}</p>
            </div>
          )}
          {uploadState.status === "error" && (
            <div className="rounded-xl border border-red-400/30 bg-red-400/8 px-4 py-3">
              <p className="font-mono text-[11px] text-red-400">✗ {uploadState.message}</p>
            </div>
          )}
        </aside>

        {/* ====== RIGHT PANEL: stats + document manager ====== */}
        <div ref={statsRef} className="flex flex-1 flex-col gap-4 overflow-hidden">

          {/* Stat tiles row */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Documents", value: docCount },
              { label: "Total chunks", value: chunkCount },
              { label: "Avg / doc", value: avgChunks },
            ].map(({ label, value }) => (
              <div
                key={label}
                data-tile
                className="rounded-2xl border border-ink/10 bg-bg-elevated/85 px-5 py-5 backdrop-blur-2xl"
              >
                <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faint">{label}</p>
                <p className="mt-2 font-serif-italic text-3xl text-ink">
                  {loadingStats ? "—" : value.toLocaleString()}
                </p>
              </div>
            ))}
          </div>

          {/* Document list */}
          <div className="flex flex-1 flex-col overflow-hidden rounded-2xl border border-ink/10 bg-bg-elevated/85 backdrop-blur-2xl">
            {/* List header */}
            <div className="flex items-center justify-between border-b border-ink/8 px-6 py-4">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-faint">
                  Documents in <span className="text-ink">{index}</span>
                </p>
                {stats && (
                  <p className="mt-0.5 font-mono text-[10px] text-ink-faint">
                    {docCount} document{docCount === 1 ? "" : "s"} · {chunkCount} chunks indexed
                  </p>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => refreshStats(index)}
                  title="Refresh"
                  className="flex h-8 w-8 items-center justify-center rounded-full border border-ink/10 text-ink-faint transition-colors hover:border-ink/25 hover:text-ink"
                >
                  <RefreshIcon spinning={loadingStats} />
                </button>
                {confirmWipe ? (
                  <>
                    <button
                      onClick={() => setConfirmWipe(false)}
                      className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-faint hover:text-ink"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleWipe}
                      className="rounded-full border border-red-400/60 bg-red-400/10 px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] text-red-300 hover:bg-red-400/20"
                    >
                      Confirm wipe
                    </button>
                  </>
                ) : (
                  <button
                    onClick={handleWipe}
                    className="rounded-full border border-ink/10 px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] text-ink-faint transition-colors hover:border-ink/25 hover:text-ink"
                  >
                    Wipe index
                  </button>
                )}
              </div>
            </div>

            {/* Scrollable doc rows */}
            {!stats || stats.documents.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-3 py-16 text-center">
                <EmptyIcon />
                <p className="text-sm text-ink-dim">No documents in &ldquo;{index}&rdquo; yet.</p>
                <p className="font-mono text-[10px] text-ink-faint">Upload a PDF on the left to get started.</p>
              </div>
            ) : (
              <ul ref={listRef} className="flex-1 divide-y divide-ink/6 overflow-y-auto">
                {stats.documents.map((doc) => (
                  <li
                    key={doc.doc_id}
                    className="group flex items-center gap-4 px-6 py-4 transition-colors hover:bg-ink/3"
                  >
                    {/* File icon */}
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-ink/10 text-ink-faint">
                      <PdfIcon />
                    </div>

                    {/* Info */}
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-ink" title={doc.file_name}>
                        {doc.file_name}
                      </p>
                      <p className="mt-0.5 font-mono text-[10px] text-ink-faint">
                        {doc.chunk_count} chunks &middot;{" "}
                        {new Date(doc.uploaded_at).toLocaleString(undefined, {
                          dateStyle: "medium",
                          timeStyle: "short",
                        })}
                      </p>
                      <p className="mt-0.5 font-mono text-[9px] text-ink-faint/60">{doc.doc_id}</p>
                    </div>

                    {/* Actions — always visible, not hover-only */}
                    <div className="flex shrink-0 items-center gap-2">
                      <button
                        onClick={() => handleReupload(doc)}
                        title="Re-upload (replaces old chunks)"
                        className="rounded-lg border border-ink/10 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint transition-colors hover:border-ink/25 hover:text-ink"
                      >
                        Replace
                      </button>
                      <button
                        onClick={() => handleDeleteDoc(doc)}
                        disabled={deleting[doc.doc_id]}
                        title="Delete this document and all its chunks"
                        className={clsx(
                          "rounded-lg border px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.12em] transition-colors",
                          deleting[doc.doc_id]
                            ? "border-ink/5 text-ink-faint/40 cursor-wait"
                            : "border-red-400/20 text-red-400/70 hover:border-red-400/50 hover:text-red-400"
                        )}
                      >
                        {deleting[doc.doc_id] ? "…" : "Delete"}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

// ---------------------------------------------------------------- sub-components

function UploadIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-ink-faint">
      <path
        d="M12 16V4M12 4L7 9M12 4l5 5M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"
        stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="animate-spin text-accent">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" strokeOpacity="0.2" />
      <path d="M12 2a10 10 0 0110 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function RefreshIcon({ spinning }: { spinning: boolean }) {
  return (
    <svg
      width="14" height="14" viewBox="0 0 24 24" fill="none"
      className={clsx("transition-transform duration-500", spinning && "animate-spin")}
    >
      <path
        d="M1 4v6h6M23 20v-6h-6M20.49 9A9 9 0 005.64 5.64L1 10M23 14l-4.64 4.36A9 9 0 013.51 15"
        stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  );
}

function PdfIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path
        d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"
        stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"
      />
      <polyline points="14 2 14 8 20 8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="9" y1="15" x2="15" y2="15" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <line x1="9" y1="11" x2="11" y2="11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function EmptyIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" className="text-ink-faint/40">
      <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.2" strokeDasharray="4 2" />
      <path d="M9 12h6M12 9v6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}
