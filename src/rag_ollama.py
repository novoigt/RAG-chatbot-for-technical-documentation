"""
Lokales RAG mit Ollama, LangChain und Chroma.

Schritte:
1. Ollama-Verbindung prüfen (Embedding + Chat-Modell)
2. HTML-Dokument laden
3. Text in Chunks aufteilen
4. Embeddings erzeugen und in Chroma speichern
5. Retriever testen (mit Distanzen)
6. RAG-Chain aufbauen (mit Quellen-Rückgabe)
7. Hauptfrage stellen
8. Qualitätstest mit mehreren Fragen (mit Quellen und Distanzen)
9. Testzusammenfassung und JSON-Report
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader


# ----------------------------------------------------------------------
# Logging konfigurieren
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

# HTTP-Logmeldungen von httpx unterdrücken, echte Fehler bleiben sichtbar
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Zentrale Konfiguration
# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
HTML_PATH = SCRIPT_DIR / ".." / "data" / "mg-zs-warning-messages.html"
CHROMA_PATH = SCRIPT_DIR / ".." / "chroma_db"
REPORT_PATH = SCRIPT_DIR / ".." / "test_report" / "rag_test_report.json"

CHAT_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 3

# Timeout für alle Ollama-Calls (verhindert ewiges Hängen)
OLLAMA_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=120.0,
    write=15.0,
    pool=15.0,
)


# ----------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------
def print_header(title: str) -> None:
    """Gibt einen übersichtlichen Abschnitts-Titel aus."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def format_docs(docs: list[Document]) -> str:
    """Wandelt retrieved Documents in einen Kontext-String mit Quellen um."""
    return "\n\n---\n\n".join(
        f"[Quelle {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

def collection_count(vectorstore: Chroma) -> int:
    """Gibt die Anzahl der Dokumente in der aktiven Chroma-Collection zurück."""
    result: dict[str, Any] = vectorstore.get(include=[])
    ids = result.get("ids", [])
    return len(ids)

def preview_text(text: str, max_chars: int = 250) -> str:
    """Erzeugt eine einzeilige Textvorschau."""
    return text[:max_chars].replace("\n", " ").strip()

def retrieve_with_scores(
    question: str,
    k: int = RETRIEVER_K,
) -> list[tuple[Document, float]]:
    """
    Führt eine Similarity Search aus und gibt Dokumente mit Distanz zurück.
    Chroma liefert eine Distanz: kleinere Werte bedeuten größere Ähnlichkeit.
    """
    return vectorstore.similarity_search_with_score(question, k=k)

def answer_with_sources(
    question: str,
) -> tuple[str, list[Document], list[float]]:
    """
    Führt Retrieval und Antwortgenerierung aus.
    Gibt zurück:
    - Antwort des LLM
    - verwendete Dokumente
    - Chroma-Distanzen der Dokumente
    """
    scored_docs = retrieve_with_scores(question, k=RETRIEVER_K)

    docs = [doc for doc, _distance in scored_docs]
    distances = [distance for _doc, distance in scored_docs]
    context = format_docs(docs)

    answer: str = (prompt | llm | StrOutputParser()).invoke(
        {"question": question, "context": context}
    )

    return answer, docs, distances

def print_sources(
    docs: list[Document],
    distances: list[float] | None = None,
    max_chars: int = 150,
) -> None:
    """Gibt die verwendeten Quellen inklusive optionaler Distanz aus."""
    print(f"\nVerwendete Quellen: {len(docs)} Chunk(s)")

    for i, doc in enumerate(docs, start=1):
        preview = preview_text(doc.page_content, max_chars=max_chars)

        if distances is not None and i <= len(distances):
            print(f"  [{i}] Distanz={distances[i - 1]:.4f} | {preview}...")
        else:
            print(f"  [{i}] {preview}...")


# ----------------------------------------------------------------------
# 1. Ollama-Verbindung prüfen (Embedding + Chat-Modell)
# ----------------------------------------------------------------------
print_header("1. OLLAMA-VERBINDUNG PRÜFEN")

try:
    test_embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
        client_kwargs={"timeout": OLLAMA_TIMEOUT},
    ).embed_query("Verbindungstest")

    print(f"OK: Ollama erreichbar unter {OLLAMA_URL}")
    print(f"OK: Embedding-Modell '{EMBEDDING_MODEL}' antwortet.")
    print(f"Embedding-Dimension: {len(test_embedding)}")

except Exception as exc:
    raise RuntimeError(
        f"Ollama-Check fehlgeschlagen ({OLLAMA_URL}).\n"
        f"Läuft der Ollama-Server? Ist das Embedding-Modell "
        f"'{EMBEDDING_MODEL}' gepullt? (ollama pull {EMBEDDING_MODEL})\n"
        f"Originalfehler: {exc}"
    ) from exc

try:
    test_llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0,
        base_url=OLLAMA_URL,
        client_kwargs={"timeout": OLLAMA_TIMEOUT},
    )

    test_llm.invoke("ping")

    print(f"OK: Chat-Modell '{CHAT_MODEL}' antwortet.")

except Exception as exc:
    raise RuntimeError(
        f"Chat-Modell '{CHAT_MODEL}' ist nicht verfügbar. "
        f"Ist es gepullt? (ollama pull {CHAT_MODEL})\n"
        f"Originalfehler: {exc}"
    ) from exc


# ----------------------------------------------------------------------
# 2. HTML-Dokument laden
# ----------------------------------------------------------------------
print_header("2. HTML-DOKUMENT LADEN")

if not HTML_PATH.exists():
    raise FileNotFoundError(
        f"HTML-Datei nicht gefunden: {HTML_PATH.resolve()}"
    )

loader = UnstructuredLoader(file_path=str(HTML_PATH))
car_docs = loader.load()

print(f"OK: {len(car_docs)} Dokument(e) geladen.")
print(f"Quelldatei: {HTML_PATH.resolve()}")

for i, doc in enumerate(car_docs, start=1):
    print(f"\nDokument {i}: {len(doc.page_content)} Zeichen")
    print(f"Vorschau: {preview_text(doc.page_content, 300)}...")


# ----------------------------------------------------------------------
# 3. Text in Chunks aufteilen
# ----------------------------------------------------------------------
print_header("3. TEXT IN CHUNKS AUFTEILEN")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

splits = text_splitter.split_documents(car_docs)

print(f"OK: {len(splits)} Chunks erzeugt.")
print(
    f"Einstellungen: chunk_size={CHUNK_SIZE}, "
    f"chunk_overlap={CHUNK_OVERLAP}"
)

for i, chunk in enumerate(splits[:3], start=1):
    print(f"\nChunk {i}: {len(chunk.page_content)} Zeichen")
    print(f"Vorschau: {preview_text(chunk.page_content)}...")

if len(splits) > 3:
    print(f"\n... und {len(splits) - 3} weitere Chunks.")


# ----------------------------------------------------------------------
# 4. Chroma-Vektordatenbank aufbauen
# ----------------------------------------------------------------------
print_header("4. EMBEDDINGS ERZEUGEN UND IN CHROMA SPEICHERN")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_URL,
    client_kwargs={"timeout": OLLAMA_TIMEOUT},
)

# Chroma öffnet ein vorhandenes Verzeichnis oder legt ein neues an.
vectorstore = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings,
)

# Alte Collection leeren, falls bereits Daten vorhanden sind.
existing_count = collection_count(vectorstore)

if existing_count > 0:
    vectorstore.reset_collection()
    print(
        f"OK: Vorhandene Chroma-Collection zurückgesetzt "
        f"({existing_count} alte Dokumente entfernt)."
    )

vectorstore.add_documents(splits)

print(f"OK: {len(splits)} Chunks in Chroma gespeichert.")
print(f"Chroma-Verzeichnis: {CHROMA_PATH.resolve()}")
print(f"Embedding-Modell: {EMBEDDING_MODEL}")


# ----------------------------------------------------------------------
# 5. Retriever testen
# ----------------------------------------------------------------------
print_header("5. RETRIEVER TESTEN")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVER_K},
)

retrieval_test_query = "Gasoline Particular Filter Full warning"
scored_test_docs = retrieve_with_scores(
    retrieval_test_query,
    k=RETRIEVER_K,
)

print(f"Testanfrage: {retrieval_test_query!r}")
print(f"OK: {len(scored_test_docs)} relevante Chunks gefunden.")
print("Hinweis: Kleinere Chroma-Distanz bedeutet größere Ähnlichkeit.")

for i, (doc, distance) in enumerate(scored_test_docs, start=1):
    print(f"\nGefundener Chunk {i} | Distanz={distance:.4f}")
    print(preview_text(doc.page_content) + "...")


# ----------------------------------------------------------------------
# 6. RAG-Chain aufbauen
# ----------------------------------------------------------------------
print_header("6. RAG-CHAIN AUFBAUEN")

llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0,
    base_url=OLLAMA_URL,
    client_kwargs={"timeout": OLLAMA_TIMEOUT},
)

prompt = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks.\n"
    "Use only the following retrieved context to answer the question.\n"
    "Do not use any knowledge that is not present in the context.\n"
    "If the context does not contain enough information, reply exactly:\n"
    "\"I don't know based on the provided manual.\"\n\n"
    "Use at most 4 sentences. If the context lists concrete actions, "
    "reproduce them as a short bullet list.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)

print(f"OK: Chat-Modell '{CHAT_MODEL}' initialisiert.")
print("OK: RAG-Chain aufgebaut:")
print(
    "  Frage -> Retriever -> Kontext formatieren -> "
    "Prompt -> LLM -> Antwort"
)


# ----------------------------------------------------------------------
# 7. Hauptfrage stellen
# ----------------------------------------------------------------------
print_header("7. HAUPTFRAGE")

main_query = (
    "The Gasoline Particular Filter Full warning has appeared. "
    "What does this mean and what should I do about it?"
)

print(f"Frage:\n{main_query}\n")

main_answer, main_sources, main_distances = answer_with_sources(main_query)

print("Antwort:")
print(main_answer)

print_sources(main_sources, main_distances)


# ----------------------------------------------------------------------
# 8. Qualitätstest mit mehreren Fragen
# ----------------------------------------------------------------------
print_header("8. QUALITÄTSTEST")

test_questions = [
    "What does the ABS warning light mean and what should I do?",
    "What should I do when the tyre pressure warning system indicates a problem?",
    "What does the engine coolant temperature warning mean?",
    "What should I do if the electric power steering warning appears?"
    "What is the recommended action for the brake system warning?",
    "What does the airbag warning light indicate?",
    "What should I do if the charging system warning appears?",
]

results: list[dict[str, Any]] = []

for number, question in enumerate(test_questions, start=1):
    print(f"\n{'-' * 70}")
    print(f"Test {number}/{len(test_questions)}")
    print(f"{'-' * 70}")
    print(f"Frage:\n{question}\n")

    try:
        answer, docs, distances = answer_with_sources(question)

        print("Antwort:")
        print(answer)

        print_sources(docs, distances, max_chars=100)

        results.append(
            {
                "number": number,
                "question": question,
                "answer": answer,
                "sources": docs,
                "distances": distances,
                "status": "OK",
            }
        )

    except Exception as exc:
        error_message = f"Fehler: {exc}"
        print(error_message)
        logger.exception("Testfrage fehlgeschlagen")

        results.append(
            {
                "number": number,
                "question": question,
                "answer": "",
                "sources": [],
                "distances": [],
                "status": "FEHLER",
            }
        )


# ----------------------------------------------------------------------
# 9. Testzusammenfassung und JSON-Report
# ----------------------------------------------------------------------
print_header("9. TESTZUSAMMENFASSUNG")

successful = sum(1 for result in results if result["status"] == "OK")
failed = len(results) - successful

print(f"Gesamte Fragen: {len(results)}")
print(f"Erfolgreich: {successful}")
print(f"Fehler: {failed}")

for result in results:
    short_question = result["question"][:80].replace("\n", " ")
    word_count = len(result["answer"].split())

    print(f"\nTest {result['number']} [{result['status']}]: {short_question}...")
    print(
        f"Antwortlänge: {word_count} Wörter, "
        f"Quellen: {len(result['sources'])}"
    )

    if result["distances"]:
        best_distance = min(result["distances"])
        print(f"Beste Retrieval-Distanz: {best_distance:.4f}")

class RagTestReport(TypedDict):
    created_at: str
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    summary: dict[str, int]
    results: list[dict[str, Any]]

# JSON-Report für spätere Vergleiche speichern
report: RagTestReport = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "chat_model": CHAT_MODEL,
    "embedding_model": EMBEDDING_MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "retriever_k": RETRIEVER_K,
    "summary": {
        "total_questions": len(results),
        "successful": successful,
        "failed": failed,
    },
    "results": [
        {
            "number": result["number"],
            "question": result["question"],
            "answer": result["answer"],
            "status": result["status"],
            "distances": result["distances"],
            "source_previews": [
                preview_text(doc.page_content, 200)
                for doc in result["sources"]
            ],
        }
        for result in results
    ],
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nTestbericht gespeichert: {REPORT_PATH.resolve()}")
print("\nFertig.")