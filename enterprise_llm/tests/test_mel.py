"""
Minimum Equipment List dispatch logic.

These are airworthiness rules, so the tests are written around the ways
real operations get them wrong rather than around the happy path.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from elp.mel.catalog import _build_column_map, parse_category, parse_row, validate
from elp.mel.dispatch import (
    CATEGORY_INTERVAL_DAYS,
    DispatchError,
    MelSpec,
    OpenDeferral,
    Verdict,
    evaluate_airworthiness,
    evaluate_dispatch,
    extension_expiry,
    not_in_mel,
    parse_category_a_interval,
    rectification_expiry,
)

TODAY = date(2026, 8, 25)


def item(**overrides) -> MelSpec:
    defaults = dict(
        id="i1",
        item_number="24-11-01",
        title="AC Generator",
        category="C",
        ata_chapter="24",
        number_installed=2,
        number_required=1,
        source_document_key="MEL-001",
        source_revision="6",
    )
    defaults.update(overrides)
    return MelSpec(**defaults)


def carried(item_number: str, expires_on: date | None, **overrides) -> OpenDeferral:
    defaults = dict(
        id=f"d-{item_number}",
        item_number=item_number,
        category="C",
        defect_description="defect",
        discovered_on=TODAY - timedelta(days=1),
        expires_on=expires_on,
    )
    defaults.update(overrides)
    return OpenDeferral(**defaults)


# ----------------------------------------------------------------------
# Rectification intervals
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "category,expected",
    [("B", date(2026, 1, 4)), ("C", date(2026, 1, 11)), ("D", date(2026, 5, 1))],
)
def test_interval_excludes_the_day_of_discovery(category, expected):
    """
    A Category C item found on the 1st runs to the end of the 11th.

    The interval is counted in consecutive calendar days *excluding* the day
    of discovery. Being one day out here is a finding in an audit and, worse,
    an aircraft dispatched a day past its limit.
    """
    assert rectification_expiry(category, date(2026, 1, 1)) == expected


def test_standard_intervals_match_the_categories():
    assert CATEGORY_INTERVAL_DAYS == {"B": 3, "C": 10, "D": 120}


def test_intervals_run_across_month_and_year_boundaries():
    assert rectification_expiry("C", date(2026, 12, 28)) == date(2027, 1, 7)
    # 2028 is a leap year: February has 29 days.
    assert rectification_expiry("B", date(2028, 2, 27)) == date(2028, 3, 1)


def test_category_a_has_no_standard_interval():
    assert rectification_expiry("A", date(2026, 1, 1)) is None
    assert rectification_expiry("A", date(2026, 1, 1), category_a_days=30) == date(2026, 1, 31)


def test_unknown_category_is_refused():
    with pytest.raises(DispatchError, match="unknown MEL category"):
        rectification_expiry("Z", date(2026, 1, 1))


@pytest.mark.parametrize(
    "remarks,expected",
    [
        ("Rectify within 30 consecutive calendar days", 30),
        ("May be inoperative for 7 days", 7),
        ("Limited to 1 day", 1),
        ("Rectify within 50 flight hours", None),
        ("Rectify within 100 cycles", None),
        ("", None),
        ("No specific limit", None),
    ],
)
def test_category_a_interval_read_from_remarks(remarks, expected):
    assert parse_category_a_interval(remarks) == expected


def test_hour_based_category_a_is_not_silently_converted():
    """
    Converting flight hours to days would require assuming a utilisation
    rate. That is fine for planning a inspection and not fine for an
    airworthiness limit, so the engine refuses and says why.
    """
    spec = item(category="A", remarks="Rectify within 50 flight hours")
    decision = evaluate_dispatch(spec, discovered_on=TODAY, today=TODAY)

    assert decision.verdict is Verdict.NO_GO
    assert any("flight hours" in reason for reason in decision.blocking_reasons)


# ----------------------------------------------------------------------
# The fundamental rule
# ----------------------------------------------------------------------

def test_an_item_absent_from_the_mel_is_a_no_go():
    """
    The MEL grants relief; it does not restrict it. Anything not listed must
    be serviceable. A cautious "probably fine" here is the failure that
    matters.
    """
    decision = not_in_mel("windscreen wiper motor seized")

    assert decision.verdict is Verdict.NOT_IN_MEL
    assert decision.dispatchable is False
    assert "must be serviceable" in decision.summary
    assert decision.blocking_reasons


# ----------------------------------------------------------------------
# Quantity relief
# ----------------------------------------------------------------------

def test_relief_is_the_difference_between_installed_and_required():
    assert item(number_installed=2, number_required=1).relief_quantity == 1
    assert item(number_installed=1, number_required=1).relief_quantity == 0
    assert item(number_installed=4, number_required=2).relief_quantity == 2


def test_no_relief_when_all_installed_units_are_required():
    decision = evaluate_dispatch(
        item(number_installed=1, number_required=1), discovered_on=TODAY, today=TODAY
    )
    assert decision.verdict is Verdict.NO_GO
    assert "none may be inoperative" in decision.blocking_reasons[0]


def test_second_failure_of_the_same_item_exhausts_the_relief():
    """Two generators, one required: the first may be deferred, not the second."""
    spec = item(number_installed=2, number_required=1)
    already = [carried("24-11-01", TODAY + timedelta(days=5))]

    first = evaluate_dispatch(spec, discovered_on=TODAY, today=TODAY)
    second = evaluate_dispatch(
        spec, discovered_on=TODAY, today=TODAY, open_deferrals=already
    )

    assert first.dispatchable is True
    assert second.verdict is Verdict.NO_GO
    assert "already inoperative" in second.blocking_reasons[0]


def test_deferring_more_than_the_relief_at_once_is_refused():
    spec = item(number_installed=2, number_required=1)
    decision = evaluate_dispatch(
        spec, discovered_on=TODAY, today=TODAY, quantity_inoperative=2
    )
    assert decision.verdict is Verdict.NO_GO


# ----------------------------------------------------------------------
# Interactions between items
# ----------------------------------------------------------------------

def test_incompatible_items_may_not_be_carried_together():
    spec = item(incompatible_with=["34-21-01"])
    already = [carried("34-21-01", TODAY + timedelta(days=5))]

    decision = evaluate_dispatch(
        spec, discovered_on=TODAY, today=TODAY, open_deferrals=already
    )
    assert decision.verdict is Verdict.NO_GO
    assert "34-21-01" in decision.blocking_reasons[0]


def test_an_expired_item_blocks_any_further_dispatch():
    """One item past its limit makes the aircraft undispatchable, full stop."""
    already = [carried("33-40-01", TODAY - timedelta(days=2))]
    decision = evaluate_dispatch(
        item(), discovered_on=TODAY, today=TODAY, open_deferrals=already
    )

    assert decision.verdict is Verdict.NO_GO
    assert any("expired" in reason for reason in decision.blocking_reasons)


def test_prohibited_operations_block_a_flight_that_needs_them():
    spec = item(prohibited_operations=["IFR", "night"])

    day_vfr = evaluate_dispatch(
        spec, discovered_on=TODAY, today=TODAY, intended_operation=["VFR", "day"]
    )
    night_flight = evaluate_dispatch(
        spec, discovered_on=TODAY, today=TODAY, intended_operation=["night"]
    )

    assert day_vfr.dispatchable is True
    assert night_flight.verdict is Verdict.NO_GO
    assert night_flight.prohibited_operations == ["IFR", "night"]


# ----------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------

def test_procedures_and_placards_become_stated_conditions():
    spec = item(
        operational_procedure="Monitor remaining generator load",
        maintenance_procedure="Deactivate and secure the generator",
        placard_text="GEN 2 INOP",
        remarks="One may be inoperative provided the other is serviceable",
    )
    decision = evaluate_dispatch(spec, discovered_on=TODAY, today=TODAY)

    assert decision.verdict is Verdict.GO_WITH_CONDITIONS
    assert decision.requires_placard is True
    joined = " ".join(decision.conditions)
    assert "(o)" in joined and "(m)" in joined and "GEN 2 INOP" in joined


def test_an_item_with_no_conditions_is_a_plain_go():
    decision = evaluate_dispatch(item(), discovered_on=TODAY, today=TODAY)
    assert decision.verdict is Verdict.GO
    assert decision.conditions == []


def test_every_decision_carries_a_citation():
    """A dispatch decision nobody can trace to the approved MEL is worthless."""
    decision = evaluate_dispatch(item(), discovered_on=TODAY, today=TODAY)
    assert "MEL-001" in decision.citation
    assert "Rev 6" in decision.citation
    assert "24-11-01" in decision.citation


def test_expiry_is_computed_from_discovery_not_from_today():
    """A defect found three days ago has three days less to run."""
    discovered = TODAY - timedelta(days=3)
    decision = evaluate_dispatch(item(category="C"), discovered_on=discovered, today=TODAY)

    assert decision.expires_on == discovered + timedelta(days=10)
    assert decision.days_available == 7


# ----------------------------------------------------------------------
# Extensions
# ----------------------------------------------------------------------

def test_extension_adds_one_further_interval():
    spec = item(extension_permitted=True)
    deferral = carried("24-11-01", date(2026, 8, 11), discovered_on=date(2026, 8, 1))
    assert extension_expiry(deferral, spec) == date(2026, 8, 21)


def test_extension_is_one_time_only():
    spec = item(extension_permitted=True)
    deferral = carried("24-11-01", date(2026, 8, 11), extended=True)
    with pytest.raises(DispatchError, match="already been extended"):
        extension_expiry(deferral, spec)


def test_extension_refused_when_the_mel_does_not_permit_it():
    deferral = carried("24-11-01", date(2026, 8, 11))
    with pytest.raises(DispatchError, match="not marked as extendable"):
        extension_expiry(deferral, item(extension_permitted=False))


def test_category_a_may_not_be_extended():
    """Category A intervals are item-specific and already stated as the limit."""
    spec = item(category="A", extension_permitted=True, category_a_days=30)
    deferral = carried("24-11-01", date(2026, 8, 31), category="A")
    with pytest.raises(DispatchError, match="may not be extended"):
        extension_expiry(deferral, spec)


# ----------------------------------------------------------------------
# Aircraft status
# ----------------------------------------------------------------------

def test_one_expired_item_makes_the_aircraft_undispatchable():
    status = evaluate_airworthiness(
        "PP-ABC",
        [
            carried("24-11-01", TODAY + timedelta(days=8)),
            carried("33-40-01", TODAY - timedelta(days=1)),
        ],
        today=TODAY,
    )
    assert status.dispatchable is False
    assert len(status.expired) == 1
    assert "NOT dispatchable" in status.summary


def test_items_expiring_soon_are_flagged_without_blocking():
    status = evaluate_airworthiness(
        "PP-ABC",
        [
            carried("24-11-01", TODAY + timedelta(days=2)),
            carried("25-10-01", TODAY + timedelta(days=30)),
        ],
        today=TODAY,
        warning_days=3,
    )
    assert status.dispatchable is True
    assert [row["item_number"] for row in status.expiring_soon] == ["24-11-01"]


def test_open_items_are_ordered_by_urgency():
    status = evaluate_airworthiness(
        "PP-ABC",
        [
            carried("A", TODAY + timedelta(days=30)),
            carried("B", TODAY + timedelta(days=2)),
            carried("C", None),
        ],
        today=TODAY,
    )
    assert [row["item_number"] for row in status.open_items] == ["B", "A", "C"]


def test_a_clean_aircraft_reports_nothing_open():
    status = evaluate_airworthiness("PP-ABC", [], today=TODAY)
    assert status.dispatchable is True
    assert status.open_count == 0
    assert "no open MEL items" in status.summary


def test_prohibited_operations_accumulate_across_open_items():
    status = evaluate_airworthiness(
        "PP-ABC",
        [carried("24-11-01", TODAY + timedelta(days=5)), carried("34-21-01", TODAY + timedelta(days=5))],
        today=TODAY,
        prohibited_by_item={"24-11-01": ["night"], "34-21-01": ["IFR", "night"]},
    )
    assert status.prohibited_operations == ["IFR", "night"]


# ----------------------------------------------------------------------
# Catalogue import
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("C", "C"), ("Cat C", "C"), ("Category C", "C"), ("C (10 days)", "C"),
        ("cat. b", "B"), ("D", "D"), ("A", "A"), ("CDL", "CDL"), ("", ""), ("X", ""),
    ],
)
def test_category_is_read_from_operator_spellings(raw, expected):
    assert parse_category(raw) == expected


def test_an_unreadable_category_is_rejected_not_guessed():
    """
    Inventing a category would put a wrong airworthiness limit into the
    system, so the row is rejected and reported instead.
    """
    parsed = parse_row(
        {"Item": "24-11-01", "Description": "Generator", "Category": "???"},
        _build_column_map(["Item", "Description", "Category"]),
    )
    problem = validate(parsed)
    assert problem is not None and "category" in problem


def test_required_exceeding_installed_is_rejected():
    parsed = parse_row(
        {"Item": "X", "Description": "Y", "Category": "C",
         "Number Installed": "1", "Number Required": "2"},
        _build_column_map(["Item", "Description", "Category", "Number Installed", "Number Required"]),
    )
    assert "exceeds" in validate(parsed)


def test_column_aliases_cover_common_mel_layouts():
    for headers in (
        ["MEL Item", "Description", "Category", "No. 1", "No. 2", "Remarks or Exceptions"],
        ["Item No", "Equipment", "Cat", "Installed", "Required for Dispatch", "Provisos"],
        ["Reference", "Item Description", "Repair Category", "Fitted", "Required", "Conditions"],
    ):
        mapped = set(_build_column_map(headers).values())
        assert {"item_number", "title", "category"} <= mapped, headers


def test_category_a_days_are_parsed_at_import():
    parsed = parse_row(
        {"Item": "31-10-01", "Description": "Clock", "Category": "A",
         "Remarks": "May be inoperative for 15 consecutive calendar days"},
        _build_column_map(["Item", "Description", "Category", "Remarks"]),
    )
    assert parsed["category"] == "A"
    assert parsed["category_a_days"] == 15
    assert validate(parsed) is None


def test_missing_counts_default_to_single_installation_with_relief():
    parsed = parse_row(
        {"Item": "X", "Description": "Y", "Category": "C"},
        _build_column_map(["Item", "Description", "Category"]),
    )
    assert parsed["number_installed"] == 1
    assert parsed["number_required"] == 0
    assert validate(parsed) is None
