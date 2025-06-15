#!/bin/bash

BIGIN_TIME=$(date -d "now + 10 minutes" +%Y-%m-%dT%H:%M:%S)
SHARED_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Submitting job to start at $BIGIN_TIME"
echo "Shared timestamp: $SHARED_TIMESTAMP"

INtERMEDIATE_TIME=