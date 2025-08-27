#!/bin/bash
#SBATCH --account=project_2010376
#SBATCH --partition=small
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=18
#SBATCH --job-name=cell-train
#SBATCH --chdir=/scratch/project_2010376/JDs_Project/cell_analysis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=TERM@300

set -Eeuo pipefail

# ---------- CONFIG ----------
EMAIL="jyotidip.barman@helsinki.fi"
LINES=200
INTERVAL=3600
TZ_REGION="Europe/Helsinki"
START_HOUR=7
END_HOUR=23
TRAIN_CMD="python -u main.py --mode train"
# --------------------------------

mkdir -p logs
LOG="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
ERR="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

send_mail() {
  local subject="$1" body="$2"
  if command -v mailx >/dev/null 2>&1; then
    printf "%s\n" "$body" | mailx -s "$subject" "$EMAIL"
  elif command -v mail >/dev/null 2>&1; then
    printf "%s\n" "$body" | mail -s "$subject" "$EMAIL"
  elif command -v sendmail >/dev/null 2>&1; then
    { printf "To: %s\nSubject: %s\n\n%s\n" "$EMAIL" "$subject" "$body"; } | sendmail -t
  else
    echo "No mailer found (mailx/mail/sendmail); skipping email." >&2
  fi
}

report_hourly() {
  while kill -0 "$PAYLOAD_PID" 2>/dev/null; do
    HOUR=$(TZ=$TZ_REGION date +%H)
    if [ "$HOUR" -ge "$START_HOUR" ] && [ "$HOUR" -le "$END_HOUR" ]; then
      NOW=$(TZ=$TZ_REGION date)
      STATS=$(sstat -j ${SLURM_JOB_ID}.batch --format=JobID,Elapsed,AveCPU,MaxRSS,MaxVMSize 2>/dev/null || true)
      OUT=$(tail -n "$LINES" "$LOG" 2>/dev/null || echo "(stdout not yet)")
      ERRTAIL=$(tail -n "$LINES" "$ERR" 2>/dev/null || echo "(stderr not yet)")
      BODY=$(cat <<TXT
Job: $SLURM_JOB_ID ($SLURM_JOB_NAME)
When: $NOW ($TZ_REGION)

=== sstat ===
$STATS

=== Last $LINES lines of STDOUT ===
$OUT

=== Last $LINES lines of STDERR ===
$ERRTAIL
TXT
)
      send_mail "[SLURM $SLURM_JOB_ID] Hourly log snapshot" "$BODY"
    fi
    sleep "$INTERVAL" || true
  done
}

final_mail() {
  local status="$1"
  NOW=$(TZ=$TZ_REGION date)
  SUM=$(sacct -j "$SLURM_JOB_ID" --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS,MaxVMSize,NodeList 2>/dev/null || true)
  OUT=$(tail -n "$LINES" "$LOG" 2>/dev/null || echo "(no stdout)")
  ERRTAIL=$(tail -n "$LINES" "$ERR" 2>/dev/null || echo "(no stderr)")
  BODY=$(cat <<TXT
Job: $SLURM_JOB_ID ($SLURM_JOB_NAME)
Finished: $NOW ($TZ_REGION)
Exit status: $status

=== sacct summary ===
$SUM

=== Last $LINES lines of STDOUT ===
$OUT

=== Last $LINES lines of STDERR ===
$ERRTAIL
TXT
)
  send_mail "[SLURM $SLURM_JOB_ID] COMPLETED (exit $status)" "$BODY"
}

trap 'final_mail "$?"' EXIT

# --- Your workload ---
source .venv/bin/activate
export PYTHONUNBUFFERED=1
stdbuf -oL -eL srun -u $TRAIN_CMD & PAYLOAD_PID=$!
# ---------------------

report_hourly &
wait "$PAYLOAD_PID"
