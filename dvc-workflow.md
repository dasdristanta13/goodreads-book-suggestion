# DVC with Google Drive Workflow Guide

## Initial Setup (Do this once)

### Prerequisites
1. Install Git: https://git-scm.com/downloads
2. Install DVC and Google Drive support:
   ```
   pip install 'dvc[gdrive]'
   ```
3. Have a Google account

### Project Setup
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. If DVC isn't initialized, do so:
   ```
   dvc init
   git commit -m "Initialize DVC"
   ```

3. Create a folder in Google Drive for DVC storage

4. Get the folder ID from the URL (long string after `/folders/`)

5. Add Google Drive as a DVC remote:
   ```
   dvc remote add -d myremote gdrive://<folder_id>
   git add .dvc/config
   git commit -m "Configure Google Drive DVC remote"
   ```

6. Push existing data to DVC (if any):
   ```
   dvc push
   ```
   (Follow Google authentication prompts)

## Regular Workflow (Do this for each data update)

### Updating Data (Person making changes)
1. Pull latest changes:
   ```
   git pull
   dvc pull
   ```

2. Make necessary changes to the data

3. Add updated data to DVC:
   ```
   dvc add <path_to_data>
   ```

4. Commit changes:
   ```
   git add <path_to_data>.dvc
   git commit -m "Update data: <brief description>"
   ```

5. Push changes:
   ```
   git push
   dvc push
   ```

### Getting Latest Data (All team members)
1. Pull latest Git changes:
   ```
   git pull
   ```

2. Update data:
   ```
   dvc pull
   ```

## Troubleshooting

- If `dvc pull` fails, try:
  ```
  dvc fetch
  dvc checkout
  ```

- To see status of data files:
  ```
  dvc status
  ```

- If you encounter Google Drive authentication issues:
  1. Delete the `.dvc/tmp/gdrive-user-credentials.json` file
  2. Run `dvc push` or `dvc pull` again to re-authenticate

## Best Practices
- Always pull before making changes
- Use meaningful commit messages
- Regularly check `dvc status` to ensure all changes are tracked
- Communicate with team members when pushing significant data updates

## Notes
- The first time you push or pull, you'll need to authenticate with Google
- Be mindful of Google Drive storage limits
- For large datasets, pushes and pulls might take some time depending on your internet speed