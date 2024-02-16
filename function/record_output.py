# ファイル操作系
import os
import glob
import shutil

# データ処理系
import csv
import pickle

# 自作関数
from function import place_lis

# 時間系
import datetime


def record_time_csv(data:list, flag:str):
    """ csvに計測した時間を保存

    Args:
        data (list): 時間のリスト
        flag (str): csvに保存するか否か (フラグ)

    """

    if (flag == "true") :
        with open('./static/output/record.csv', mode='a', encoding="shift-jis") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(data)


def record_temp(obj_dic:list, now):
    """ 画像処理の結果を一時的に保存

    Args:
        obj_dic (list): 画像処理の結果の情報
        now (?): 現在の時間

    """

    temp_path = 'cash/' + now.strftime('%H%M%S') + '.pkl'
    if not os.path.isfile(temp_path):
        with open(temp_path, 'wb') as f:
            pickle.dump(obj_dic, f)
    else:
        data = []
        with open(temp_path, 'rb') as f:
            try:
                for _ in range(100):
                    data.append(pickle.load(f))
            except:		# エラー回避 (ごり押し)
                pass
        data.append(obj_dic)
        with open(temp_path, 'wb') as f:
            for value in data:
                pickle.dump(value, f)


def record_post_data(output:dict, status:dict, signal_red:list, signal_yellow:list, now:str):
    """ 最終の結果を保存

    Args:
        output (dict): 音声として出力する物体の情報
        status (dict): ステータス
        signal_red (list): 危険ゾーンの情報
        signal_yellow (list): 警告ゾーンの情報
        now (str): 現在の時間

    """
    for place in place_lis:
        if status['place'] != place:
            before_path = 'cash/output_' + place + '.pkl'
            if os.path.isfile(before_path):
                num = 0
                while True:
                    after_path = 'cash/save/output_' + str(num) + '.pkl'
                    if not os.path.isfile(after_path):
                        os.replace(before_path, after_path)
                        break
                    num += 1
        else:
            if output != None:
                output_path = 'cash/output_' + status['place'] + '.pkl'
                file_name = now.strftime('%H%M%S') + '_' + output['name'] + '_' + output['degree']
                output_pkl = [output, status, signal_red, signal_yellow, file_name]
                if not os.path.isfile(output_path):
                    with open(output_path, 'wb') as f:
                        pickle.dump(output_pkl, f)
                else:
                    data = []
                    with open(output_path, 'rb') as f:
                        try:
                            for _ in range(100):
                                data.append(pickle.load(f))
                        except:		# エラー回避 (ごり押し)
                            pass
                    data.append(output_pkl)
                    with open(output_path, 'wb') as f:
                        for value in reversed(data):	# 最新のデータが頭になるように
                            pickle.dump(value, f)


def record_results_csv():
    """ csvに計測した時間を保存 """

    data = []
    saved_file = '/output_' + '*.pkl'
    saved_dir = 'cash/save'
    saved_path = saved_dir + saved_file
    saved_path_lis = glob.glob(saved_path)
    for path in saved_path_lis:
        with open(path, 'rb') as f:
            try:
                for _ in range(100):
                    data.append(pickle.load(f))
            except:		# エラー回避 (ごり押し)
                pass

        today = str(datetime.date.today())
        output_dir = 'static/output/' + today
        shutil.copytree(saved_dir, output_dir, dirs_exist_ok=True)

    # output, status, signal_red, signal_yellow, file_name の順番で保存される
    with open(output_dir + '/results.csv', mode='a', encoding="shift-jis", newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(len(data)):
            csv_writer.writerow(data[i])