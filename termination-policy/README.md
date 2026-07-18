# CORTEX termination-policy contract

This small package contains the shared Python interface used by the concrete
AD6 and PrecisionPolicy projects. It deliberately contains no stopping math,
certification cuts, selector logic, or engine dependencies.

The two policy implementations live in sibling projects:

- `../ad6-policy`
- `../precision-policy`

Keeping the contract separate prevents either concrete policy from depending
on the other and makes both projects independently importable and testable.
