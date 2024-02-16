# dropbox
import dropbox

# 画像処理系
import pandas as pd
from PIL import Image

# 時間系
import datetime

# データ処理系？
from io import BytesIO


def linkDropbox():
    """ Dropbox にリンク

    Returns:
        client (?): dropboxのクライアント情報

    """

    df = pd.read_csv('./static/ready/dropbox.csv', encoding="shift-jis", header=None)

    app_key = df.loc[0,1]
    app_secret = df.loc[1,1]
    refresh_token = df.loc[2,1]

    rdbx = dropbox.Dropbox(oauth2_refresh_token=refresh_token, app_key=app_key, app_secret=app_secret)

    rdbx.refresh_access_token()

    dropbox_access_token = rdbx._oauth2_access_token

    client = dropbox.Dropbox(dropbox_access_token, timeout=300)
    return client


def save_doropbox(img_dict:dict, client, flag:str):
    """ dropboxに保存

    Args:
        img (ndarray): 画像データ
        client (?): dropboxのクライアント情報
        flag (str): dropboxに保存するか否か (フラグ)

    """

    now = datetime.datetime.now()
    now_second = now.second
    if now_second % 10 == 0:
        for type, img in img_dict.items():
            filename = now.strftime('%Y%m%d_%H%M%S') + '_' + type + '.jpg'
            if (flag == "true"):
                img_bytes = BytesIO()    # メモリ上でバイナリデータを扱う
                img = Image.fromarray(img[:, :, ::-1])
                img.save(img_bytes, "JPEG")     # メモリ上に保存
                img_bytes = img_bytes.getvalue()  # バッファすべてを bytes として出力

                dropbox_path="/test/" + filename
                client.files_upload(img_bytes, dropbox_path, mode=dropbox.files.WriteMode('overwrite'))
            
                # print("upload dropbox")