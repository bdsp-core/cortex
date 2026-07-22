# AD6 policy

This project is the standalone Python implementation of the shipped AD6
termination and verdict policy. Production now selects PrecisionPolicy for new
certification sittings; AD6 remains the supported fail-closed rollback policy
and the interpretation rule for sessions already stamped `ad6`.

AD6 owns cut-aware PASS/FAIL/PENDING decisions and monotone verdict locks. The
application-specific loader for the K=7 cut vector remains in the Python
reference integration layer because configuration-file discovery is not
stopping-policy math.

## Test in isolation

From this directory:

```bash
python -m pytest
```

The sibling `cortex-termination-policy` project supplies the shared policy
contract. A normal editable development install is:

```bash
python -m pip install -e ../termination-policy -e .
```
