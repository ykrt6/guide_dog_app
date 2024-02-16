# -*- coding: utf-8 -*-

# ctrl-C を行うため
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from scipy.stats import zscore

# web app 系
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG, ERROR, INFO
import threading
from flask import Flask, render_template, request, make_response, jsonify

# 画像処理系
import cv2

# 自作関数 (ディープラーニング)
from function import service, save_dropbox, audio_output_dash, record_output, image_process
from function import yamasou, tetsu

# 時間系
import time
import datetime

# ファイル操作系
import glob

# flask 設定
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
werkzeug_logger = getLogger("werkzeug")
formatter = Formatter('[%(levelname)s %(asctime)s - %(message)s %(filename)s]')
werkzeug_handler = FileHandler('static/output/error.txt')
werkzeug_handler.setLevel(ERROR)
werkzeug_handler.setFormatter(formatter)
werkzeug_logger.setLevel(ERROR)
werkzeug_logger.addHandler(werkzeug_handler)


# loggin 設定	(midas_load_2.py yamasou.py)
formatter = Formatter('[%(levelname)s %(asctime)s - %(message)s %(filename)s]')
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(formatter)
logger.setLevel(INFO)
logger.addHandler(handler)

# よくわからないやつ
sem = threading.Semaphore()

# 画像処理サイズ
from function import size


# html の読み込み
@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

# キャプチャ
@app.route('/', methods=['POST'])
def capture_img():
	sem.acquire()
	process_time_list = []

	############## 受信＆前処理 ##############
	start = time.time()
	img = service.base64_to_ndarray(request.form["img"])
	receive_time = time.time() - start
	process_time_list.append(f"receive:{receive_time:.4f}")
	logger.debug(f"受信:{receive_time:.4f} seconds")


	############## 画像処理 ##############
	start = time.time()

	img = cv2.resize(img, size)
	# 物体検知
	object, object_dict = tetsu.obj_detect(model, predictor, is_cli, img)

	# 深度推定
	depth, prediction = yamasou.depth_predict(img, device, loaded_model)

	img_process_time = time.time() - start
	process_time_list.append(f"img_process:{img_process_time:.4f}")
	logger.debug(f"画像処理:{img_process_time:.4f} seconds")


	############## 後処理 ##############
	start = time.time()

	# 物体検知の結果の後処理
	img_obj_base64 = service.pil_to_base64(object)

	# 深度推定の結果の後処理
	img_dep_base64 = service.pil_to_base64(depth)

	send_time = time.time() - start
	process_time_list.append(f"send:{send_time:.4f}")
	logger.debug(f"送信:{send_time:.4f} seconds")


	############## 音声選定 ##############
	start = time.time()
	dt_now = datetime.datetime.now()
	before_second = request.form["before_second"]
	before_second = (dt_now.second-2) if before_second == '' else int(before_second)
	output, sound, do_predict = audio_output_dash.sound(object_dict, prediction, dt_now, before_second)
	second = dt_now.second if do_predict else before_second
	sound_time = time.time() - start
	process_time_list.append(f"sound:{sound_time:.4f}")
	logger.info(sound)
	logger.debug(f"音声選定:{sound_time:.4f} seconds")


	############## jpgとして保存 ##############
	start = time.time()
	image_process.save_image({'orig': img, 'obj': object, 'dep': depth}, output, dt_now, request.form["save"])
	save_time = time.time() - start
	process_time_list.append(f"save:{save_time:.4f}")
	logger.debug(f"保存:{save_time:.4f} seconds")


	############## 処理時間の算出 ##############
	all_time = receive_time + img_process_time + send_time + save_time + sound_time
	process_time_list.append(f"all:{all_time:.4f}")
	process_time_list.append(f"fps:{1 / all_time:.4f}")
	logger.info(f"計:{all_time:.4f} seconds、fps:{1 / all_time:.0f}")


	############## csvに記録 ##############
	record_output.record_time_csv(process_time_list, request.form["record"])


	sem.release()

	return make_response(jsonify({'object_result': "data:image/jpeg;base64," + img_obj_base64,'depth_result': "data:image/jpeg;base64," + img_dep_base64, 'sound': sound, 'second': second}))


if __name__ == '__main__':
	try:
		# 時間記録用の古いcsvファイルを削除
		path = 'static/output/record.csv'
		if os.path.isfile(path):
			os.remove(path)

		# cashファイルがなければ作成、あればpklファイルを削除
		cash_path = 'cash'
		if not os.path.exists(cash_path):
			os.makedirs(cash_path)
		else:
			pkl_lis = glob.glob(cash_path + '\\*.pkl')
			for pkl in pkl_lis:
				os.remove(pkl)

		# 深度推定のモデルの読み込み
		device, loaded_model = yamasou.init_dep_model()
		# 物体検知のモデルの読み込み
		model, predictor, is_cli = tetsu.init_obj_model()

		# テスト画像の準備
		test_img = cv2.imread('static/ready/sample_video_img_000.jpg')
		test_img = cv2.resize(test_img, size)

		# 物体検知
		object, object_dict = tetsu.obj_detect(model, predictor, is_cli, test_img)
		logger.info("=======物体検知 予備準備完了=======")

		# 深度推定
		depth, prediction = yamasou.depth_predict(test_img, device, loaded_model)
		logger.info("=======深度推定 予備準備完了=======")

		# # ローカルサーバの立ち上げ
		# logger.error('click url: http://localhost:5000')
		logger.error('click url: https://t7w9gc6r-5000.asse.devtunnels.ms/')
		app.debug = False
		app.run(host='localhost')

	finally:
		# saveファイルにあるやつを全部抽出してcsvに保存する (ファイル名はoutput_results.csv)
		results_save_csv = input("結果をcsvに保存する y/n : ")
		if results_save_csv == 'y':
			record_output.record_results_csv()
			before_path = 'static/output/record.csv'
			after_path = 'static/output/' + str(datetime.date.today()) + '/record.csv'
			if os.path.isfile(before_path):
				os.replace(before_path, after_path)
			print("saved")
		else:
			print("finish")