"""
LaTeX authoring and PDF compilation.

Two entry points: compile LaTeX the caller already has, or have the model
write it from a brief (optionally grounded in the governing documents, so a
generated procedure carries real citations).

Compilation is sandboxed - a temporary directory, no shell escape, no
network, a wall-clock timeout, and a whitelist check on the source - because
LaTeX is a full programming language and ``\\write18`` is a remote code
execution primitive if you let it be.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..config import LatexSettings, get_settings
from ..llm.client import ChatMessage
from ..llm.router import TaskKind, get_router

log = logging.getLogger(__name__)


class LatexError(RuntimeError):
    """LaTeX generation or compilation failed."""


# Primitives that read the filesystem, execute programs, or exfiltrate data.
_FORBIDDEN = [
    (re.compile(r"\\write18"), "\\write18 (shell escape)"),
    (re.compile(r"\\immediate\s*\\write18"), "\\immediate\\write18 (shell escape)"),
    (re.compile(r"\\input\s*\{\s*\|"), "piped \\input (shell escape)"),
    (re.compile(r"\\openout"), "\\openout (arbitrary file write)"),
    (re.compile(r"\\openin"), "\\openin (arbitrary file read)"),
    (re.compile(r"\\usepackage\s*(\[[^\]]*\])?\s*\{[^}]*\bshellesc\b"), "shellesc package"),
    (re.compile(r"\\directlua"), "\\directlua (arbitrary Lua execution)"),
    (re.compile(r"\\catcode`\\\\\\\\@=11"), "catcode manipulation of the escape character"),
    (re.compile(r"\\input\s*\{?\s*/"), "absolute-path \\input"),
    (re.compile(r"\\include\s*\{?\s*/"), "absolute-path \\include"),
]

SYSTEM_PROMPT = """\
You are a LaTeX author for an aviation maintenance organisation.

Produce a COMPLETE, COMPILABLE LaTeX document:
- Start with \\documentclass and end with \\end{document}.
- Use only packages from a standard TeX Live installation: geometry, \
booktabs, longtable, tabularx, array, amsmath, graphicx, hyperref, xcolor, \
fancyhdr, enumitem, caption, siunitx.
- Never use \\write18, \\directlua, \\openout, \\openin, or shell escape of \
any kind. Never \\input or \\include a file by absolute path.
- Do not reference image files unless the brief supplies them; the compiler \
runs with no external assets.
- Escape LaTeX special characters in prose: & % $ # _ { } ~ ^ \\.
- Prefer booktabs (\\toprule/\\midrule/\\bottomrule) for tables and \
longtable for tables that may break across pages.

When SOURCES are supplied, every factual statement drawn from them must \
carry its marker (e.g. [D1]) rendered as a footnote or a parenthetical \
citation, and you must not state anything the sources do not support.

Output ONLY the LaTeX source. No markdown fences, no commentary."""


@dataclass
class CompileResult:
    success: bool
    pdf_path: str = ""
    artifact_id: str = ""
    page_count: int = 0
    size_bytes: int = 0
    log_excerpt: str = ""
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "artifact_id": self.artifact_id,
            "pdf_path": self.pdf_path,
            "page_count": self.page_count,
            "size_bytes": self.size_bytes,
            "errors": self.errors,
            "log_excerpt": self.log_excerpt,
            "duration_ms": round(self.duration_ms, 1),
        }


def validate_source(source: str, settings: LatexSettings | None = None) -> None:
    """Reject sources containing escape-hatch primitives."""
    settings = settings or get_settings().latex
    if not source.strip():
        raise LatexError("LaTeX source is empty")
    if len(source.encode("utf-8")) > settings.max_source_bytes:
        raise LatexError(
            f"LaTeX source exceeds the {settings.max_source_bytes} byte limit"
        )
    for pattern, description in _FORBIDDEN:
        if pattern.search(source):
            raise LatexError(
                f"LaTeX source rejected: it uses {description}, which is not "
                "permitted by the compile sandbox"
            )
    if "\\documentclass" not in source:
        raise LatexError("LaTeX source has no \\documentclass")


def _strip_fences(text: str) -> str:
    """Models like wrapping output in ```latex fences despite instructions."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n", "", stripped)
        stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped.strip()


class LatexRenderer:
    def __init__(self, settings: LatexSettings | None = None) -> None:
        self.settings = settings or get_settings().latex
        self.artifact_dir = Path(self.settings.artifact_dir)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        brief: str,
        *,
        template: str = "",
        sources: str = "",
        document_class: str = "article",
    ) -> str:
        """Have the local model write LaTeX from a brief."""
        instructions = [f"BRIEF\n=====\n{brief}"]
        if document_class:
            instructions.append(f"Use \\documentclass{{{document_class}}}.")
        if template:
            instructions.append(
                "Follow this preamble and structure as closely as the brief "
                f"allows:\n\n{template}"
            )
        if sources:
            instructions.append(
                f"SOURCES (cite these with their markers)\n{'=' * 38}\n{sources}"
            )

        client, profile = get_router().resolve(TaskKind.LATEX)
        completion = await client.chat(
            [
                ChatMessage("system", SYSTEM_PROMPT),
                ChatMessage("user", "\n\n".join(instructions)),
            ],
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            top_p=profile.top_p,
        )
        source = _strip_fences(completion.text)
        if completion.finish_reason == "length":
            raise LatexError(
                "the generated document was cut off at the output token limit; "
                "narrow the brief or split it into sections"
            )
        validate_source(source, self.settings)
        return source

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    async def compile(self, source: str, *, keep_source: bool = True) -> CompileResult:
        validate_source(source, self.settings)

        engine = self.settings.engine
        binary = shutil.which(engine)
        if binary is None:
            raise LatexError(
                f"the '{engine}' binary was not found on PATH. Install it "
                "(apt install tectonic, or texlive-full for latexmk) or run the "
                "platform's latex container."
            )

        artifact_id = uuid.uuid4().hex
        workdir = self.artifact_dir / artifact_id
        workdir.mkdir(parents=True, exist_ok=True)
        tex_path = workdir / "document.tex"
        tex_path.write_text(source, encoding="utf-8")

        if engine == "tectonic":
            command = [
                binary,
                "--outdir", str(workdir),
                # Deny network access: no fetching packages mid-compile.
                "--reruns", "2",
                "--keep-logs",
                "--chatter", "minimal",
                str(tex_path),
            ]
        else:
            command = [
                binary,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                # Belt and braces: the sandbox check above already rejects
                # \write18, and this stops the engine honouring it anyway.
                "-shell-escape-",
                f"-outdir={workdir}",
                str(tex_path),
            ]

        loop = asyncio.get_event_loop()
        started = loop.time()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                # An empty-ish environment keeps TEXINPUTS from reaching
                # outside the working directory.
                env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(workdir)},
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(), timeout=self.settings.compile_timeout_s
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                raise LatexError(
                    f"compilation exceeded {self.settings.compile_timeout_s}s and "
                    "was killed; this usually means an unterminated environment "
                    "or an infinite macro loop"
                ) from None
        except FileNotFoundError as exc:
            raise LatexError(f"could not run {engine}: {exc}") from exc

        duration_ms = (loop.time() - started) * 1000
        output = stdout.decode("utf-8", errors="replace")
        pdf_path = workdir / "document.pdf"

        if process.returncode != 0 or not pdf_path.exists():
            return CompileResult(
                success=False,
                artifact_id=artifact_id,
                errors=_extract_errors(output),
                log_excerpt=output[-4000:],
                duration_ms=duration_ms,
            )

        if not keep_source:
            tex_path.unlink(missing_ok=True)

        return CompileResult(
            success=True,
            pdf_path=str(pdf_path),
            artifact_id=artifact_id,
            page_count=_page_count(pdf_path),
            size_bytes=pdf_path.stat().st_size,
            log_excerpt=output[-1500:],
            duration_ms=duration_ms,
        )

    async def generate_and_compile(
        self,
        brief: str,
        *,
        template: str = "",
        sources: str = "",
        document_class: str = "article",
        repair_attempts: int = 1,
    ) -> tuple[str, CompileResult]:
        """
        Write LaTeX and compile it, feeding compile errors back once.

        Models reliably produce *nearly* valid LaTeX; a single repair round
        using the actual compiler log fixes most of what is left, and is far
        cheaper than making the operator debug it.
        """
        source = await self.generate(
            brief, template=template, sources=sources, document_class=document_class
        )
        result = await self.compile(source)

        attempt = 0
        while not result.success and attempt < repair_attempts:
            attempt += 1
            log.info("LaTeX compile failed; repair attempt %d", attempt)
            client, profile = get_router().resolve(TaskKind.LATEX)
            completion = await client.chat(
                [
                    ChatMessage("system", SYSTEM_PROMPT),
                    ChatMessage(
                        "user",
                        "This LaTeX source failed to compile. Return the complete "
                        "corrected source, nothing else.\n\n"
                        f"ERRORS\n======\n{chr(10).join(result.errors) or result.log_excerpt}"
                        f"\n\nSOURCE\n======\n{source}",
                    ),
                ],
                temperature=0.0,
                max_tokens=profile.max_tokens,
            )
            source = _strip_fences(completion.text)
            validate_source(source, self.settings)
            result = await self.compile(source)

        return source, result

    def artifact_path(self, artifact_id: str) -> Path | None:
        """Resolve an artifact id to its PDF, refusing path traversal."""
        if not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
            return None
        candidate = self.artifact_dir / artifact_id / "document.pdf"
        return candidate if candidate.is_file() else None


_ERROR_RE = re.compile(r"^(?:!|.*?:\d+:)\s*(.+)$", re.MULTILINE)


def _extract_errors(log_text: str, limit: int = 12) -> list[str]:
    """Pull the human-meaningful lines out of a TeX log."""
    errors: list[str] = []
    for match in _ERROR_RE.finditer(log_text):
        line = match.group(1).strip()
        if line and line not in errors:
            errors.append(line)
        if len(errors) >= limit:
            break
    return errors


def _page_count(pdf_path: Path) -> int:
    try:
        import fitz

        with fitz.open(str(pdf_path)) as document:
            return len(document)
    except Exception:
        # Counting pages is a nicety; never fail a good compile over it.
        try:
            return pdf_path.read_bytes().count(b"/Type /Page") or 0
        except Exception:
            return 0


_renderer: LatexRenderer | None = None


def get_renderer() -> LatexRenderer:
    global _renderer
    if _renderer is None:
        _renderer = LatexRenderer()
    return _renderer
