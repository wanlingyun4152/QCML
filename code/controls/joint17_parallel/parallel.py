#!/usr/bin/env python3
"""QCML representative LiH parallel benchmark. No Slurm submission without submit script."""
from qrev.parallel import main
if __name__=='__main__':
    import sys,traceback
    try:sys.exit(main() or 0)
    except Exception:traceback.print_exc();sys.exit(2)
