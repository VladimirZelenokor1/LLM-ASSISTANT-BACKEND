from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------- token length helper (tiktoken if available) ----------
try:
    import tiktoken  # type: ignore

    _HAS_TIKTOKEN = True
except Exception:
    tiktoken = None  # type: ignore
    _HAS_TIKTOKEN = False

# Кэш кодировок, чтобы не дергать tiktoken.get_encoding много раз
_ENC_CACHE: Dict[str, Any] = {}


def _get_enc(name: str = "cl100k_base"):
    if not _HAS_TIKTOKEN:
        return None
    enc = _ENC_CACHE.get(name)
    if enc is None:
        enc = tiktoken.get_encoding(name)  # type: ignore[attr-defined]
        _ENC_CACHE[name] = enc
    return enc


def _tok_len(text: str, model: Optional[str] = None) -> int:
    """
    Возвращает длину текста в токенах (tiktoken).
    Важно: разрешаем спец-токены через disallowed_special=() — не падаем на '<|endoftext|>'.
    """
    if not text:
        return 0
    if not _HAS_TIKTOKEN:
        # Грубая эвристика: 1 токен ≈ 4 символа
        return max(1, len(text) // 4)

    enc = _get_enc(model or "cl100k_base")
    if enc is None:
        return max(1, len(text) // 4)

    try:
        # В tiktoken корректный способ «разрешить всё» — disallowed_special=()
        return len(enc.encode(text, disallowed_special=()))
    except Exception as e:
        logger.debug("tiktoken.encode failed (%s); using heuristic fallback", e)
        return max(1, len(text) // 4)


# ---------- common regex ----------
_par_break = re.compile(r"(?:\n{2,}|\r\n{2,})")
_bullet_line = re.compile(r"^\s*(?:[-*•]\s+|\d+\.\s+)")
_heading_line = re.compile(r"^\s{0,3}#{2,6}\s+")

# Аббревиатуры, после которых точка НЕ означает конец предложения
_ABBR_LIST = ["Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Fig.", "Eq.", "vs.", "No.", "e.g.", "i.e.", "cf.", "etc."]
_ABBR_RE = re.compile("|".join(re.escape(a) for a in _ABBR_LIST))
_DEC_RE = re.compile(r"(?<=\d)\.(?=\d)")  # 3.14
_SAFE_DOT = "∯"  # временная замена точки


def _mget(obj: Any, key: str, default: Any = None) -> Any:
    """Safe getattr/get для объектов и словарей."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _strip_hashes(s: Optional[str]) -> str:
    """Убираем префикс '#', '##', ... и тримим пробелы."""
    return re.sub(r"^\s*#{1,6}\s*", "", (s or "")).strip()


# ---------- sentence splitting (EN, list-aware) ----------
def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Возвращает список (sentence, start, end) в пределах данного plain-текста.
    Линии со списками (начинаются с '• ' / '-' / '*' / '1. ') считаются отдельными предложениями.
    """
    if not text:
        return []

    # Защищаем точки в аббревиатурах и десятичных числах
    safe = _ABBR_RE.sub(lambda m: m.group(0).replace(".", _SAFE_DOT), text)
    safe = _DEC_RE.sub(_SAFE_DOT, safe)

    out: List[Tuple[str, int, int]] = []
    i = 0
    buf: List[str] = []
    start = 0

    while i < len(safe):
        ch = safe[i]

        # Новый пункт списка → закрываем предыдущее предложение
        if ch == "•" and (i == 0 or safe[i - 1] == "\n"):
            if buf:
                s = "".join(buf)
                out.append((s, start, start + len(s)))
                buf = []
            # пропускаем маркер "• " если есть
            i = i + 2 if i + 1 < len(safe) and safe[i + 1] == " " else i + 1
            start = i
            continue

        buf.append(ch)

        # Граница предложения: .!? + следом заглавная/кавычка/скобка
        if ch in ".!?":
            j = i + 1
            while j < len(safe) and safe[j] == " ":
                j += 1
            if j < len(safe) and re.match(r"[A-Z\"'(\[]", safe[j] or ""):
                s = "".join(buf)
                out.append((s, start, start + len(s)))
                buf = []
                start = j
                i = j
                continue

        # Разрыв абзаца
        if ch == "\n" and i + 1 < len(safe) and safe[i + 1] == "\n":
            s = "".join(buf)
            out.append((s, start, start + len(s)))
            buf = []
            while i < len(safe) and safe[i] == "\n":
                i += 1
            start = i
            continue

        i += 1

    if buf:
        s = "".join(buf)
        out.append((s, start, start + len(s)))

    res: List[Tuple[str, int, int]] = []
    for s, a, b in out:
        s2 = s.replace(_SAFE_DOT, ".").strip()
        if s2:
            res.append((s2, a, a + len(s2)))
    return res


# ---------- block segmentation ----------
@dataclass
class Block:
    type: str  # 'p' | 'code' | 'table' | 'h2' | 'h3' | 'empty'
    text: str
    start: int
    end: int
    lang: Optional[str] = None  # для код-блоков


_CODE_RE = re.compile(r"```([A-Za-z0-9_+\-]*)\n.*?```", re.DOTALL)
_TILDE_RE = re.compile(r"~~~([A-Za-z0-9_+\-]*)\n.*?~~~", re.DOTALL)
_TABLE_LINE_RE = re.compile(r"^\|.+\|$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)
_H2_RE = re.compile(r"^\s*##\s+.+$", re.MULTILINE)
_H3_RE = re.compile(r"^\s*###\s+.+$", re.MULTILINE)


def segment_blocks(text: str) -> List[Block]:
    blocks: List[Block] = []
    if not text:
        return blocks

    used = [False] * len(text)

    def mark(a: int, b: int):
        for i in range(max(0, a), min(len(text), b)):
            used[i] = True

    # fenced code (```)
    for m in _CODE_RE.finditer(text):
        a, b = m.span()
        lang = m.group(1).lower() if m.group(1) else None
        blocks.append(Block("code", text[a:b], a, b, lang))
        mark(a, b)

    # fenced code (~~~)
    for m in _TILDE_RE.finditer(text):
        a, b = m.span()
        lang = m.group(1).lower() if m.group(1) else None
        blocks.append(Block("code", text[a:b], a, b, lang))
        mark(a, b)

    # tables
    for m in _TABLE_SEP_RE.finditer(text):
        a = text.rfind("\n", 0, m.start())
        a = 0 if a == -1 else a
        # ascend to table start (line with '|')
        while a > 0:
            prev = text.rfind("\n", 0, a)
            row = text[prev + 1:a]
            if not _TABLE_LINE_RE.match(row):
                break
            a = prev
        b = m.end()
        while b < len(text):
            nxt = text.find("\n", b)
            if nxt == -1:
                nxt = len(text)
            row = text[b:nxt]
            if not _TABLE_LINE_RE.match(row):
                break
            b = nxt + 1
        blocks.append(Block("table", text[a:b], a, b))
        mark(a, b)

    # headers
    for m in _H2_RE.finditer(text):
        a, b = m.span()
        if not used[a]:
            blocks.append(Block("h2", text[a:b], a, b))
            mark(a, b)
    for m in _H3_RE.finditer(text):
        a, b = m.span()
        if not used[a]:
            blocks.append(Block("h3", text[a:b], a, b))
            mark(a, b)

    # remaining paragraphs / empties
    i = 0
    while i < len(text):
        if used[i]:
            i += 1
            continue
        j = text.find("\n\n", i)
        if j == -1:
            j = len(text)
        piece = text[i:j]
        typ = "empty" if piece.strip() == "" else "p"
        blocks.append(Block(typ, piece, i, j))
        i = j + 2

    blocks.sort(key=lambda b: b.start)
    return blocks


# ---------- chunks ----------
@dataclass
class ChunkMeta:
    seq: int
    unit: str
    chunk_size: int
    char_start: int
    char_end: int
    sent_start_idx: Optional[int] = None
    sent_end_idx: Optional[int] = None
    splitter_type: str = "rule_aware"
    h_path: List[str] = field(default_factory=list)
    has_code: bool = False
    has_table: bool = False
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    meta: ChunkMeta


class RuleAwareSplitter:
    """
    Block-aware (code/table/headers/paragraphs) splitter с упаковкой предложений,
    бюджетами токенов/символов, overlap, атомарностью кода/таблиц и мягким сплитом кода.
    Гарантирует стабильный h_path и дедуп чанков внутри документа.
    """

    def __init__(self, cfg: Any, meta: Dict[str, Any]):
        self.cfg = cfg
        self.meta = meta
        self.seq = 0
        self.curr_h2: Optional[str] = None
        self.curr_h3: Optional[str] = None
        self._seen: set[str] = set()  # предотвращаем точные дубликаты чанков внутри документа

    # ---- helpers ----
    def _build_h_path(self) -> List[str]:
        title = _mget(self.meta, "title")
        section = _mget(self.meta, "section")
        sec_tail = (section or "/").strip("/").split("/")[-1] if section else None
        base = [x for x in (sec_tail, title) if x]
        h2 = _strip_hashes(self.curr_h2)
        h3 = _strip_hashes(self.curr_h3)
        return (base + [x for x in (h2, h3) if x])[:3]

    def _emit(self, text: str, start: int, end: int,
              *, has_code: bool = False, has_table: bool = False, code_lang: Optional[str] = None) -> Chunk:
        self.seq += 1
        unit: str = _mget(self.cfg.chunker, "unit", "tokens")
        size = _tok_len(text) if unit == "tokens" else len(text)

        # normalize code language aliases
        if code_lang:
            alias = {"py": "python", "sh": "bash", "md": "markdown", "txt": "text"}
            code_lang = alias.get(code_lang.lower(), code_lang.lower())

        return Chunk(
            text=text,
            meta=ChunkMeta(
                seq=self.seq,
                unit=unit,
                chunk_size=size,
                char_start=start,
                char_end=end,
                splitter_type="rule_aware",
                h_path=self._build_h_path(),
                has_code=has_code,
                has_table=has_table,
                extra={"code_lang": code_lang} if code_lang else {},
            ),
        )

    def _maybe_add(self, chunks: List[Chunk], text: str, start: int, end: int,
                   *, has_code=False, has_table=False, code_lang: Optional[str] = None):
        """Добавить чанк, если он не слишком короткий и не дубликат."""
        if not text or len(text) < int(_mget(self.cfg.chunker, "min_chars", 180)):
            return
        h = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
        if h in self._seen:
            return
        self._seen.add(h)
        chunks.append(self._emit(text, start, end,
                                 has_code=has_code, has_table=has_table, code_lang=code_lang))

    def _split_code_block(self, body: str, budget_tok: int, budget_chr: int) -> List[str]:
        """
        Мягкий сплит больших код-блоков по пустым строкам, чтобы уложиться в бюджеты.
        """
        parts = re.split(r"\n{2,}", body.strip())
        out: List[str] = []
        acc = ""
        for p in parts:
            cand = (acc + ("\n\n" if acc else "") + p).strip()
            over_tokens = budget_tok and _tok_len(cand) > budget_tok
            over_chars = budget_chr and len(cand) > budget_chr
            if over_tokens or over_chars:
                if acc:
                    out.append(acc.strip())
                acc = p
            else:
                acc = cand
        if acc:
            out.append(acc.strip())
        return [x for x in out if x]

    def _clean_line(self, s: str) -> str:
        return re.sub(r"[ \t]+", " ", s.strip())

    def _split_paragraphs(self, text: str) -> List[str]:
        # 1) Сначала крупные абзацы по двойным переносам
        paras = [p for p in _par_break.split(text) if p.strip()]
        out: List[str] = []
        for p in paras:
            lines = p.splitlines()
            buf = []
            for ln in lines:
                if _heading_line.match(ln) or _bullet_line.match(ln):
                    if buf:
                        out.append(self._clean_line("\n".join(buf)))
                        buf = []
                    out.append(self._clean_line(ln))
                else:
                    buf.append(ln)
            if buf:
                out.append(self._clean_line("\n".join(buf)))
        return out

    # ---- main ----
    def split(self, text: str) -> List[Chunk]:
        """
        Rule-aware разбиение:
          H2/H3 → абзацы → предложения, без разрыва кода/таблиц,
          упаковка до max_tokens/max_chars и sentence_max, overlap по предложениям.
        """
        chunks: List[Chunk] = []
        blocks = segment_blocks(text)

        # буфер текущего чанка: [(sentence_text, abs_start, abs_end)]
        curr_txt: List[Tuple[str, int, int]] = []

        def flush_with_overlap() -> None:
            """Закрыть текущий чанк и оставить overlap последних N предложений в буфере."""
            nonlocal curr_txt
            if not curr_txt:
                return
            assembled = " ".join(s for s, _, _ in curr_txt).strip()
            a = curr_txt[0][1]
            b = curr_txt[-1][2]
            self._maybe_add(chunks, assembled, a, b)
            keep = max(0, int(_mget(self.cfg.chunker, "overlap_sentences", 2)))
            curr_txt = curr_txt[-keep:] if keep else []

        for b in blocks:
            # ---- заголовки: закрываем чанк и обновляем H2/H3 контекст
            if b.type == "h2":
                flush_with_overlap()
                self.curr_h2 = _strip_hashes(b.text)
                self.curr_h3 = None
                continue
            if b.type == "h3":
                flush_with_overlap()
                self.curr_h3 = _strip_hashes(b.text)
                continue

            # ---- атомарные блоки: код/таблица (возможен мягкий сплит, если очень длинные)
            if b.type in ("code", "table"):
                flush_with_overlap()
                budget_tok = int(_mget(self.cfg.chunker, "max_tokens_code", 300))
                budget_chr = int(_mget(self.cfg.chunker, "max_chars_code", 1600))
                body = (b.text or "").strip()

                is_code = (b.type == "code")
                is_table = (b.type == "table")

                if _tok_len(body) > budget_tok or len(body) > budget_chr:
                    for piece in self._split_code_block(body, budget_tok, budget_chr):
                        self._maybe_add(
                            chunks, piece, b.start, b.end,
                            has_code=is_code, has_table=is_table, code_lang=b.lang
                        )
                else:
                    self._maybe_add(
                        chunks, body, b.start, b.end,
                        has_code=is_code, has_table=is_table, code_lang=b.lang
                    )
                continue

            # ---- обычный абзац → предложения и упаковка с бюджетами
            if b.type == "p":
                sents = split_sentences(b.text)  # -> List[(sent_text, rel_start, rel_end)]
                unit = _mget(self.cfg.chunker, "unit", "tokens")
                sent_budget = int(_mget(self.cfg.chunker, "sentence_max", 8))
                max_tok = int(_mget(self.cfg.chunker, "max_tokens", 600))
                max_chars = int(_mget(self.cfg.chunker, "max_chars", 800))

                for s, rel_a, rel_b in sents:
                    # локальная санация спец-токенов для tiktoken
                    if s:
                        s = s.replace("<|endoftext|>", " ").strip()
                    if not s:
                        continue

                    a = b.start + rel_a
                    z = b.start + rel_b

                    # кандидат — текущий буфер + новое предложение
                    assembled = " ".join(x for x, _, _ in (curr_txt + [(s, a, z)]))
                    if unit == "tokens":
                        over_budget = _tok_len(assembled) > max_tok
                    else:
                        over_budget = len(assembled) > max_chars
                    over_sents = (sent_budget and len(curr_txt) + 1 > sent_budget)

                    # если выходим за бюджет — сбрасываем с overlap
                    if curr_txt and (over_budget or over_sents):
                        flush_with_overlap()
                        curr_txt = []

                        # если само предложение слишком длинное — мягко режем по ; / , / • / —
                        if (unit == "tokens" and _tok_len(s) > max_tok) or (unit == "chars" and len(s) > max_chars):
                            parts = [p.strip() for p in re.split(r"(?:[;•]|—|,)(?:\s+|$)", s) if p and p.strip()]
                            for p in parts:
                                if (unit == "tokens" and _tok_len(p) > max_tok) or (
                                        unit == "chars" and len(p) > max_chars
                                ):
                                    # совсем длинный кусок — отдельным чанком
                                    self._maybe_add(chunks, p, a, z)
                                else:
                                    curr_txt.append((p, a, z))
                                    assembled2 = " ".join(x for x, _, _ in curr_txt)
                                    too_big = (
                                                  (_tok_len(assembled2) > max_tok) if unit == "tokens"
                                                  else (len(assembled2) > max_chars)
                                              ) or (sent_budget and len(curr_txt) > sent_budget)
                                    if too_big:
                                        flush_with_overlap()
                                        curr_txt = []
                            continue  # следующее предложение

                    # обычный случай — добавляем в буфер
                    curr_txt.append((s, a, z))
                continue  # следующий блок

            # ---- пустые/прочие: просто сбрасываем буфер (закрываем чанк)
            if b.type == "empty":
                flush_with_overlap()
                continue

        # финальный сброс
        flush_with_overlap()

        # защитный глобальный потолок на документ
        max_chunks = int(_mget(self.cfg.chunker, "max_chunks_per_doc", 60))
        if len(chunks) > max_chunks:
            logger.warning("Truncating chunks from %s to %s", len(chunks), max_chunks)
            chunks = chunks[:max_chunks]

        return chunks
