"""Deterministic, identity-based PE patch audit toolkit."""

from .audit import AuditError, build_audit_report, write_audit_report

__all__ = ["AuditError", "build_audit_report", "write_audit_report"]
__version__ = "1.0.0"
