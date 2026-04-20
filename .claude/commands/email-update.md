Compose and send a research progress email to jh1161@princeton.edu.

1. Read current state from:
   - `~/.claude/projects/-scratch-gpfs-EHAZAN-jh1161/memory/MEMORY.md`
   - `slurm_job_log.md`
   - `squeue -u jh1161`

2. Compose email with sections:
   - **Results since last update** (new MASE numbers, new seeds, new experiments)
   - **Currently running** (job IDs, what they test, ETA)
   - **Key findings** (statistical significance, confounds, implications)
   - **Next steps** (what experiments to run next)

3. Send via: `echo "<body>" | mail -s "<subject>" jh1161@princeton.edu`

Keep it concise — bullet points, tables, no filler.
