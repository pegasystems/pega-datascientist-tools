# Configuration Backup Script Design

**Date:** 2026-03-06
**Purpose:** Automated weekly backup of command-line profiles, SSH keys, and configuration files to OneDrive/SharePoint

## Overview

Create a simple, reliable bash script that backs up personal development environment configurations to OneDrive on a weekly schedule. The solution enables quick recovery when setting up a new laptop by having all customizations readily available.

## Architecture

The solution consists of two components:

1. **Backup script** (`backup-config.sh`): Bash script that performs the backup, manages versions, and cleans up old backups
2. **launchd plist** (`com.perdo.config-backup.plist`): macOS scheduler configuration that runs the script every Friday at 10am

The script creates timestamped backup folders in OneDrive, copies all specified config files/folders, includes itself in each backup, and automatically deletes backups older than 4 weeks.

## Files and Folders to Backup

The following items from the home directory will be backed up:

### Shell Configurations
- `.zshrc`
- `.bash_profile`
- `.local/bin/env` (sourced by .zshrc)

### Git Configuration
- `.gitconfig`

### SSH Directory (entire folder including private keys)
- `.ssh/` (config, keys, known_hosts)
- **Security Note:** Private keys are included for convenience. OneDrive provides some security, but this trades security for ease of recovery.

### Tool Configs from .config/
- `.config/gh/` (GitHub CLI)
- `.config/git/`
- `.config/fish/`
- `.config/uv/`

### Cloud CLI Configs
- `.aws/` (AWS credentials and config)
- `.azure/` (Azure CLI config)

### The Backup Script Itself
- `backup-config.sh` (copied into each backup for self-recovery)

All items are copied preserving file permissions. The script handles missing files gracefully (logs warning but continues).

## Backup Location and Versioning

**Target Directory:**
```
/Users/perdo/Library/CloudStorage/OneDrive-PegasystemsInc/config-backups/
```

**Backup Naming:**
Each backup is stored in a dated folder: `config-backup-YYYY-MM-DD/`
- Example: `config-backup-2026-03-07/`
- Makes it easy to identify when each backup was created

**Retention Policy:**
- Keep last 4 backups (approximately 4 weeks)
- On each run, script checks for backups older than the 4th most recent and deletes them
- Deletion happens after new backup succeeds, ensuring at least one backup always exists

**Structure Inside Each Backup Folder:**
```
config-backup-2026-03-07/
├── backup-config.sh          (the script itself)
├── .zshrc
├── .bash_profile
├── .local/
│   └── bin/
│       └── env
├── .gitconfig
├── .ssh/
│   ├── config
│   ├── id_ed25519
│   ├── id_ed25519.pub
│   └── ...
├── .config/
│   ├── gh/
│   ├── git/
│   ├── fish/
│   └── uv/
├── .aws/
└── .azure/
```

## Scheduling Mechanism

**macOS launchd Configuration:**
- Location: `~/Library/LaunchAgents/com.perdo.config-backup.plist`
- Schedule: Every Friday at 10:00 AM
- Runs as user account (not root) for home directory and OneDrive access

**Scheduling Details:**
- Uses `Weekday` key set to `5` (Friday) and `Hour` key set to `10`
- Only runs if user is logged in (launchd user agent)
- If Mac is asleep/off at 10am Friday, job runs when next woken/booted

**Installation Steps:**
1. Copy plist to `~/Library/LaunchAgents/`
2. Load with: `launchctl load ~/Library/LaunchAgents/com.perdo.config-backup.plist`
3. Verify with: `launchctl list | grep config-backup`

**Manual Trigger:**
- Via launchctl: `launchctl start com.perdo.config-backup`
- Direct execution: `~/backup-config.sh`

## Error Handling and Logging

**Log File:**
- Location: `~/Library/Logs/config-backup.log`
- Each run appends timestamped entries
- Logs: start time, files copied, cleanup actions, errors, completion time
- Log rotation: keep last 10 runs to prevent unlimited growth

**Error Handling:**
- Check if OneDrive target directory is accessible before starting
- Verify each file/folder exists before attempting to copy
- If source file doesn't exist, log warning but continue with other files
- If OneDrive directory is missing (e.g., OneDrive not running), abort and log error
- Exit with non-zero code on critical failures

**Desktop Notifications:**
- **Success:** Brief notification: "✓ Config backup completed - 15 files backed up"
- **Failure:** Alert notification: "⚠️ Config backup failed - OneDrive not accessible"
- Uses macOS `osascript` to trigger native notification center alerts

## Script Location and Self-Backup

**Primary Script Location:**
`~/backup-config.sh` (home directory for easy access)

**Self-Backup Mechanism:**
The script copies itself into each backup folder. When setting up a new laptop:
1. Access OneDrive from new machine
2. Open any recent `config-backup-YYYY-MM-DD/` folder
3. Find `backup-config.sh` with setup instructions in comments
4. Copy script and plist, reload schedule

**Setup Instructions in Script:**
The script includes detailed comments at the top explaining:
- What it backs up
- How to install the launchd schedule
- How to manually run it
- How to verify it's working

## Implementation Approach

**Technology Stack:**
- Bash script (standard on macOS)
- macOS launchd for scheduling
- Standard Unix tools: `cp`, `mkdir`, `rm`, `find`, `date`
- `osascript` for notifications

**Rationale for Simple Approach:**
For personal configuration backups, simplicity is a feature. The solution:
- Uses only built-in tools (no dependencies)
- Easy to understand and modify
- Reliable and straightforward
- Quick to set up on a new machine

## Testing Strategy

**Manual Testing:**
1. Run script directly and verify all files copied
2. Check log file for correct entries
3. Verify notifications appear
4. Run multiple times and confirm retention policy (keeps 4, deletes older)
5. Test with missing source files (should log warning, not fail)
6. Test with OneDrive unavailable (should abort gracefully)

**Schedule Testing:**
1. Load launchd plist
2. Verify it's loaded: `launchctl list | grep config-backup`
3. Manually trigger: `launchctl start com.perdo.config-backup`
4. Check log to confirm scheduled run

## Recovery Workflow

**Setting Up a New Laptop:**
1. Install OneDrive and sync SharePoint folder
2. Open `config-backups/` and select most recent backup
3. Review files and copy desired configurations to home directory
4. Copy `backup-config.sh` to `~/backup-config.sh`
5. Make executable: `chmod +x ~/backup-config.sh`
6. Copy launchd plist to `~/Library/LaunchAgents/`
7. Load schedule: `launchctl load ~/Library/LaunchAgents/com.perdo.config-backup.plist`
8. Verify: `launchctl list | grep config-backup`

The script's embedded comments provide step-by-step instructions for this process.
