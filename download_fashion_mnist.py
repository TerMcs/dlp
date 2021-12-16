import numpy as np
import matplotlib, time, copy, os, requests, zipfile, sys


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_fm(PATH):
    if not os.path.exists(PATH) or not os.path.exists(os.path.join(PATH, 'fashion_mnist_npy')):
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        file_id = '1DQ2Nf2rY467kyZKOf_CG3Kib5FLv0xQu'
        destination = os.path.join(PATH, 'fashion_mnist_npy.zip')
        download_file_from_google_drive(file_id, destination)

        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(PATH)

        print("Data downloaded and extracted!")

        os.remove(destination)

    else:
        print("Data was already downloaded and extracted!")
