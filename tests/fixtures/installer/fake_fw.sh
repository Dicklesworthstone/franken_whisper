#!/usr/bin/env bash
# Fake `fw` for tests/installer_contract.rs. The test links this tracked script
# into a bin directory and writes the outcome beside the link as DATA (never as
# an executable), so no test execs a file it has just written (bd-wijg):
#   <bin dir>/fake_fw_outcome = "<pull exit status> <doctor ready: true|false>"
read -r pull_status doctor_ready < "${0%/*}/fake_fw_outcome"
printf '%s\n' "$*" >> "$FW_INSTALLER_CALL_LOG"
printf '%s\n' "$HOME" >> "$FW_INSTALLER_HOME_LOG"
if [[ "$1 $2" == "pull all" ]]; then
    exit "$pull_status"
fi
if [[ "$1 $2" == "doctor --json" ]]; then
    printf '%s\n' "{\"ready\":${doctor_ready}}"
    exit 0
fi
exit 64
