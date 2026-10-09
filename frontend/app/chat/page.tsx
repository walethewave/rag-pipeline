"use client";

import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { GravityField } from "@/components/canvas/gravity-field";
import { Nav } from "@/components/nav";
import { ApiError, askQuestion, type Source } from "@/lib/api";

type Message =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; sources: Source[]; answeredFromDocuments: boolean }
  | { role: "error"; text: string };

export default function ChatPage() {
  const [index, setIndex] = useState("faq");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [pending, setPending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || pending) return;

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setPending(true);
    try {
      const result = await askQuestion(question, index);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: result.answer,
          sources: result.sources,
          answeredFromDocuments: result.answered_from_documents,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "error", text: err instanceof ApiError ? err.message : "Something went wrong." },
      ]);
    } finally {
      setPending(false);
    }
  }

  const hasConversation = messages.length > 0;

  return (
    <main className="relative flex min-h-screen flex-col">
      <GravityField intensity={hasConversation ? 0.45 : 1} />
      <Nav />

      <div
        className={clsx(
          "relative z-10 mx-auto flex w-full max-w-2xl flex-1 flex-col px-6 transition-[padding] duration-500",
          hasConversation ? "pt-28 pb-44" : "items-center justify-center"
        )}
      >
        {!hasConversation && (
          <div className="mb-10 text-center">
            <p className="fade-up font-mono text-[11px] uppercase tracking-[0.3em] text-ink-dim">
              Index: {index}
            </p>
            <h1 className="fade-up mt-5 text-balance font-serif-italic text-[clamp(2.25rem,5.5vw,3.75rem)] leading-[1.05] text-ink">
              Ask me anything
              <br />
              you&rsquo;ve given me.
            </h1>
          </div>
        )}

        <div className="flex flex-col gap-6">
          {messages.map((message, i) => (
            <MessageBubble key={i} message={message} />
          ))}
          {pending && <ThinkingBubble />}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* floating prompt bar, Krea-style: pinned, glassy, glows on focus */}
      <div className="fixed inset-x-0 bottom-0 z-40 flex justify-center px-6 pb-8">
        <form onSubmit={handleAsk} className="w-full max-w-2xl">
          <div className="shadow-elevated flex items-center gap-3 rounded-2xl border border-ink/10 bg-bg-elevated/90 p-2.5 pl-5 backdrop-blur-2xl transition-colors focus-within:border-accent/50">
            {!hasConversation && (
              <input
                value={index}
                onChange={(e) => setIndex(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ""))}
                className="w-16 shrink-0 border-r border-line bg-transparent pr-3 font-mono text-[11px] uppercase tracking-wide text-ink-dim outline-none"
                maxLength={50}
              />
            )}
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me to do anything…"
              className="flex-1 bg-transparent py-2 text-sm text-ink outline-none placeholder:text-ink-faint"
            />
            <button
              type="submit"
              disabled={pending || !input.trim()}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-ink text-bg transition-opacity disabled:opacity-30"
            >
              <ArrowIcon />
            </button>
          </div>
        </form>
      </div>
    </main>
  );
}

function MessageBubble({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="fade-up flex justify-end">
        <p className="max-w-[80%] rounded-2xl rounded-tr-sm bg-ink px-4 py-3 text-sm text-bg">{message.text}</p>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="fade-up flex justify-start">
        <p className="max-w-[80%] rounded-2xl rounded-tl-sm border border-red-400/30 bg-red-400/10 px-4 py-3 text-sm text-red-300">
          {message.text}
        </p>
      </div>
    );
  }

  return (
    <div className="fade-up flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-tl-sm border border-line bg-bg-elevated/70 px-5 py-4 backdrop-blur-sm">
        <p className="text-sm leading-relaxed text-ink">{message.text}</p>
        {message.sources.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2 border-t border-line pt-3">
            {message.sources.map((source, i) => (
              <span
                key={i}
                title={source.question}
                className="rounded-full border border-line px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide text-ink-faint"
              >
                [{i + 1}] {source.source} · {Math.round(source.score * 100)}%
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ThinkingBubble() {
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl rounded-tl-sm border border-line bg-bg-elevated/70 px-5 py-4">
        <span className="font-mono text-sm text-ink-faint">
          thinking<span className="animate-blink">_</span>
        </span>
      </div>
    </div>
  );
}

function ArrowIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
      <path d="M12 19V5M12 5l-6 6M12 5l6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
