from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def download_folder(drive, folder_id, dest):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    for file1 in file_list:
        print(f'Title: {file1["title"]}, ID: {file1["id"]}')
        if file1['mimeType'] == 'application/vnd.google-apps.folder':
            # Create folder on local if not exists
            local_folder = os.path.join(dest, file1['title'])
            if not os.path.exists(local_folder):
                os.makedirs(local_folder)
            # Recursively download folder content
            download_folder(drive, file1['id'], local_folder)
        else:
            # Download file
            file1.GetContentFile(os.path.join(dest, file1['title']))

# Authentication
gauth = GoogleAuth()
gauth.DEFAULT_SETTINGS['client_config_file'] = "credentials.json"
gauth.LocalWebserverAuth()  # Creates local webserver and automatically handles authentication.

drive = GoogleDrive(gauth)

# ID of the folder you want to download
folder_id = 'YOUR_FOLDER_ID'
# Local path to save the folder
dest = 'data'

if not os.path.exists(dest):
    os.makedirs(dest)

download_folder(drive, folder_id, dest)
print("Folder downloaded successfully.")
