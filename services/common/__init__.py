"""Shared building blocks for the SmartFlow domain services.

Copied into every service image so each container carries the same settings,
authentication, event publishing and read-only data access rather than five
slightly different versions of each.
"""
