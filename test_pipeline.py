#!/usr/bin/env python3
"""Manual CPU baseline entrypoint; production logic lives in autovs.PipelineService."""
from autovs.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
