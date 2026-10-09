/**
 * Thin fetch wrapper around the FastAPI backend (src/main.py). Every call returns the inner
 * `data` from the {success, data, error} envelope, or throws ApiError with the server's message
 * on failure - so page components never touch the envelope shape directly.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    public code: string
  ) {
    super(message);
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  const body = await response.json();
  if (!body.success) {
    throw new ApiError(body.error?.message ?? "Request failed", body.error?.code ?? "unknown_error");
  }
  return body.data as T;
}

export type DocumentInfo = {
  doc_id: string;
  file_name: string;
  index: string;
  uploaded_at: string;
  chunk_count: number;
};

export type IndexStats = {
  index: string;
  document_count: number;
  chunk_count: number;
  documents: DocumentInfo[];
};

export type Source = {
  source: string;
  question: string;
  score: number;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  retrieved_count: number;
  answered_from_documents: boolean;
};

export async function uploadDocument(file: File, index: string): Promise<DocumentInfo & { chunk_count: number }> {
  const form = new FormData();
  form.append("file", file);
  form.append("index", index);
  const response = await fetch(`${API_URL}/api/v1/documents`, { method: "POST", body: form });
  return unwrap(response);
}

export async function listDocuments(index: string): Promise<{ index: string; documents: DocumentInfo[] }> {
  const response = await fetch(`${API_URL}/api/v1/documents?index=${encodeURIComponent(index)}`);
  return unwrap(response);
}

export async function getStats(index: string): Promise<IndexStats> {
  const response = await fetch(`${API_URL}/api/v1/stats?index=${encodeURIComponent(index)}`);
  return unwrap(response);
}

export async function deleteDocument(docId: string, index: string): Promise<void> {
  const response = await fetch(
    `${API_URL}/api/v1/documents/${encodeURIComponent(docId)}?index=${encodeURIComponent(index)}`,
    { method: "DELETE" }
  );
  await unwrap(response);
}

export async function deleteIndex(index: string): Promise<void> {
  const response = await fetch(`${API_URL}/api/v1/index/${encodeURIComponent(index)}`, { method: "DELETE" });
  await unwrap(response);
}

export async function askQuestion(question: string, index: string, k = 3): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, index, k }),
  });
  return unwrap(response);
}
