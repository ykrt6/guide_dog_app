# データ処理系
import csv
import pickle

# ファイル操作系
import os
import glob

# 画像処理系
import numpy as np
import cv2
from PIL import Image, ImageDraw

# 自作関数
from function import image_process, record_output, run_16_notplot

# フィルター画像
from function import img_danger, img_warn, img_right, img_left

# ステータス状況の種類
from function import place_lis

# ラベル
from function import place_class_lis, border_lis, road_lis, obstacle_lis, ground_lis

# 音声インターバル
from function import sound_interval

# 画像処理サイズ
from function import process_img_size


def check_results(result_path:str) -> list:
	""" 検出したものを厳選する (１秒間での結果から信憑性の高いものを厳選)

	Args:
		result_path (str): 一時的に保存したファイルのパス

	Returns:
		extra (list): 厳選後の情報

	"""

	comb_lis = []
	with open(result_path, 'rb') as f:
		try:
			for _ in range(100):
				data_lis = pickle.load(f)
				for data in data_lis:
					if comb_lis:
						for i, comb in enumerate(comb_lis):
							if data['name'] == comb['name']:
								if abs(data['dep_max'] - comb['dep_max']) < 3:	# 相対深度値が±3以内なら同じものとみなす
									comb['dupli_time'] += 1
									comb['dep_max'] = data['dep_max']
									comb['x'] = data['x']
									comb['y'] = data['y']
									comb['box'] = data['box']
									break

							if i == (len(comb_lis)-1):
								data['dupli_time'] = 0
								comb_lis.append(data)
								break
					else:
						data['dupli_time'] = 0
						comb_lis.append(data)
		except:
			pass

	#  フィルター後の結果から複数回検知ものだけを抽出
	extra = list(filter(lambda item: item['dupli_time'] != 0, comb_lis))

	# 一時保存したファイルの削除
	os.remove(result_path)

	return extra


def sort_risk_degree(extra:list):
	""" 検知した物体を危険ゾーン、警告ゾーン、安全ゾーンの3つに分別

	Args:
		extra (list): 厳選後の情報

	Returns:
		signal_red (list): 危険ゾーンの情報
		signal_yellow (list): 警告ゾーンの情報
		signal_blue (list): 安全ゾーンの情報

	"""

	signal_red = []
	signal_yellow = []
	signal_blue = []
	for data in extra:
		img_object = Image.new('L', (256, 256), 0)
		draw_object = ImageDraw.Draw(img_object)
		draw_object.rectangle((data['box']['x1'], data['box']['y1'], data['box']['x2'], data['box']['y2']), fill=127)
		np_img = np.array(img_object)

		mat_add_warn = img_warn + np_img

		if np.max(mat_add_warn)>=254:	# 127*2
			mat_add_danger = img_danger + np_img

			if np.max(mat_add_danger)>=254:	# 127*2
				signal_red.append(data.copy())
				signal_red[len(signal_red)-1].pop('box')
			else:
				signal_yellow.append(data.copy())
				signal_yellow[len(signal_yellow)-1].pop('box')
		else:
			signal_blue.append(data.copy())
			signal_blue[len(signal_blue)-1].pop('box')

	return signal_red, signal_yellow, signal_blue


def load_status(status):
	""" ステータスを過去データから読み込み

	Args:
		status (dict): ステータス

	Returns:
		status (dict): ステータス
		post_output (list): 過去に出力した物体に関する情報

	"""

	post_output_path = glob.glob('cash/output_' + '*.pkl')
	post_output = []
	if post_output_path:
		with open(post_output_path[0], 'rb') as f:
			try:
				for i in range(100):
					post_data = pickle.load(f)
					post_output.append(post_data[0])	# 特定のラベルを繰り返し出力しないようにするため
					if i == 0:
						status = post_data[1]
			except:		# エラー回避 (ごり押し)
				pass

	return status, post_output

def change_status_place(extra:list, status:dict):
	""" ステータス状況の変更 (現在の結果から)

	Args:
		extra (list): 厳選後の情報
		status (dict): ステータス

	Returns:
		status (dict): ステータス

	"""

	match_index_lis = []
	for data in extra:
		try:
			match_index_lis.append([place in data.values() for place in place_class_lis].index(True))
		except ValueError:
			pass
	if match_index_lis:
		head_num = sorted(match_index_lis)[0]
		if head_num == 0:	# crosswalk の場合
			status['place'] = place_lis[0]
		elif head_num == 5:	# stairs の場合
			status['place'] = place_lis[2]
		else:	# corner の場合
			status['place'] = place_lis[1]
	else:
		status['place'] = place_lis[3]

	return status


def change_status_is_common(extra:list, status:dict):
	""" ステータスの歩車共通か否かを調べる (ステータス状況がotherの時に左右を確認して調べる)

	Args:
		extra (list): 厳選後の情報
		status (dict): ステータス

	Returns:
		status (dict): ステータス

	"""

	if status['place'] == place_lis[3]:
		is_include_left_border = False
		is_include_right_border = False
		is_include_left_road = False
		is_include_right_road = False

		for data in extra:
			img_object = Image.new('L', (256, 256), 0)
			draw_object = ImageDraw.Draw(img_object)
			draw_object.rectangle((data['box']['x1'], data['box']['y1'], data['box']['x2'], data['box']['y2']), fill=127)
			np_img = np.array(img_object)

			mat_add_left = img_left + np_img
			mat_add_right = img_right + np_img

			if np.max(mat_add_left) >= 254:	# 127*2
				if data['name'] in border_lis:
					is_include_left_border = True
				if data['name'] in road_lis:
					is_include_left_road = True

			if np.max(mat_add_right) >= 254:	# 127*2
				if data['name'] in border_lis:
					is_include_right_border = True
				if data['name'] in road_lis:
					is_include_right_road = True


		if is_include_left_border and is_include_left_road:
			status['is_common'] = False
		elif is_include_right_border and is_include_right_road:
			status['is_common'] = False
		else:
			status['is_common'] = True

	return status


def make_status(extra:list):
	""" ステータスの作成および更新

	Args:
		extra (list): 厳選後の情報

	Returns:
		status (dict): ステータス
		post_output (list): 過去に出力した物体に関する情報

	"""

	# ステータスの初期化
	status = {'place': place_lis[3], 'is_common': True}	# 現在の状況＆歩車共通か否か

	# ステータスの読み込み
	status, post_output = load_status(status)

	# ステータス状況の変更
	status = change_status_place(extra, status)

	# ステータスの歩車共通か否か
	status = change_status_is_common(extra, status)

	return status, post_output


def decide_output(signal_red, signal_yellow, status, post_output):
	""" 出力する物体の決定

	Args:
		signal_red (list): 危険ゾーンの情報
		signal_yellow (list): 警告ゾーンの情報
		status (dict): ステータス
		post_output (list): 過去に出力した物体に関する情報

	Returns:
		output (dict): 音声として出力する物体の情報

	"""

	output = None

	place_index_red = []
	road_index_red = []
	obstacle_index_red = []
	border_index_red = []
	ground_index_red = []
	place_index_yellow = []
	road_index_yellow = []
	obstacle_index_yellow = []
	ground_index_yellow = []

	status_place = status['place']
	try:
		status_place_index = [status_place == place for place in place_lis].index(True)
	except ValueError:
		pass
	if status_place_index == 0:	# crosswalk の場合
		status_place_class = [place_class_lis[0]]
	elif status_place_index == 1:	# corner の場合
		status_place_class = place_class_lis[1:5]
	elif status_place_index == 2:	# stairs の場合
		status_place_class = [place_class_lis[5]]
	else:	# other の場合
		status_place_class = [None]

	# in_status = [out['name'] in status_place_class for out in post_output]
	in_status = [out['name'] in place_class_lis for out in post_output]
	if True in in_status:	# status['place'] を過去に出力していた場合
		if signal_red:
			for data_red in signal_red:
				if not (data_red['name'] in status_place_class):
					place_index_red.append(data_red['name'] in place_class_lis)
				else:
					place_index_red.append(False)

				road_index_red.append(data_red['name'] in road_lis)
				obstacle_index_red.append(data_red['name'] in obstacle_lis)
				border_index_red.append(data_red['name'] in border_lis)
				ground_index_red.append(data_red['name'] in ground_lis)

			if (output==None) and signal_yellow:
				for data_yellow in signal_yellow:
					if not (data_yellow['name'] in status_place_class):
						place_index_yellow.append(data_yellow['name'] in place_class_lis)
					else:
						place_index_yellow.append(False)

					road_index_yellow.append(data_yellow['name'] in road_lis)
					obstacle_index_yellow.append(data_yellow['name'] in obstacle_lis)
					ground_index_yellow.append(data_yellow['name'] in ground_lis)

			if output == None:
				try:
					if True in place_index_red:
						index_num = [place_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in road_index_red:
						index_num = [road_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in obstacle_index_red:
						index_num = [obstacle_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in border_index_red:
						index_num = [border_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in ground_index_red:
						index_num = [ground_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in place_index_yellow:
						index_num = [place_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif (True in road_index_yellow) and (status['is_common'] == True):
						index_num = [road_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif True in obstacle_index_yellow:
						index_num = [obstacle_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif True in ground_index_yellow:
						index_num = [ground_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
				except:
					pass

		elif signal_yellow:
			for data_yellow in signal_yellow:
				if not (data_yellow['name'] in status_place_class):
					place_index_yellow.append(data_yellow['name'] in place_class_lis)
				else:
					place_index_yellow.append(False)

				road_index_yellow.append(data_yellow['name'] in road_lis)
				obstacle_index_yellow.append(data_yellow['name'] in obstacle_lis)
				ground_index_yellow.append(data_yellow['name'] in ground_lis)

			if output == None:
				try:
					if True in place_index_yellow:
						index_num = [place_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif (True in road_index_yellow) and (status['is_common'] == True):
						index_num = [road_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif True in obstacle_index_yellow:
						index_num = [obstacle_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif True in ground_index_yellow:
						index_num = [ground_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
				except:
					pass

	else:	# status['place'] を過去に出力していなかった場合
		if signal_red:
			for data_red in signal_red:
				if data_red['name'] in status_place_class:
					output = data_red
					output['degree'] = 'red'
					break

				place_index_red.append(data_red['name'] in place_class_lis)
				road_index_red.append(data_red['name'] in road_lis)
				obstacle_index_red.append(data_red['name'] in obstacle_lis)
				border_index_red.append(data_red['name'] in border_lis)
				ground_index_red.append(data_red['name'] in ground_lis)

			if (output==None) and signal_yellow:
				for data_yellow in signal_yellow:
					if data_yellow['name'] in status_place_class:
						output = data_yellow
						output['degree'] = 'yellow'
						break

					place_index_yellow.append(data_yellow['name'] in place_class_lis)
					road_index_yellow.append(data_yellow['name'] in road_lis)
					obstacle_index_yellow.append(data_yellow['name'] in obstacle_lis)
					ground_index_yellow.append(data_yellow['name'] in ground_lis)

			if output == None:
				try:
					if True in place_index_red:
						index_num = [place_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in road_index_red:
						index_num = [road_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in obstacle_index_red:
						index_num = [obstacle_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in border_index_red:
						index_num = [border_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in ground_index_red:
						index_num = [ground_index_red.index(True)]
						output = signal_red[index_num[0]]
						output['degree'] = 'red'
					elif True in place_index_yellow:
						index_num = [place_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif (True in road_index_yellow) and (status['is_common'] == True):
						index_num = [road_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif True in obstacle_index_yellow:
						index_num = [obstacle_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif True in ground_index_yellow:
						index_num = [ground_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
				except:
					pass

		elif signal_yellow:
			for data_yellow in signal_yellow:
				if data_yellow['name'] in status_place_class:
					output = data_yellow
					output['degree'] = 'yellow'
					break

				place_index_yellow.append(data_yellow['name'] in place_class_lis)
				road_index_yellow.append(data_yellow['name'] in road_lis)
				obstacle_index_yellow.append(data_yellow['name'] in obstacle_lis)
				ground_index_yellow.append(data_yellow['name'] in ground_lis)

			if output == None:
				try:
					if True in place_index_yellow:
						index_num = [place_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif (True in road_index_yellow) and (status['is_common'] == True):
						index_num = [road_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif True in obstacle_index_yellow:
						index_num = [obstacle_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
					elif True in ground_index_yellow:
						index_num = [ground_index_yellow.index(True)]
						output = signal_yellow[index_num[0]]
						output['degree'] = 'yellow'
				except:
					pass

	return output


def get_absolute_value(output:dict, prediction):
	""" 絶対深度値取得

	Args:
		output (dict): 音声として出力する物体の情報
		prediction (ndarray): 深度推定結果

	Returns:
		output (dict): 音声として出力する物体の情報

	"""

	output = image_process.backConvAxis(output, after_size=16, before_size=process_img_size)
	depth_at_position = run_16_notplot.get_depth_at_position(prediction, output['x'], output['y'])
	
	# 距離定義
	fig_x = 4284
	fig_y = 5712
	width, stature = prediction.shape
	
	# 相対的な座標
	y_coords = [4241 ,4083 ,3954 ,3840 ,3746 ,3671 ,3606 ,3548 ,3498 ,3456]
	y_normalized = [int(coord / fig_y * stature) for coord in y_coords]
	
	distance = run_16_notplot.depth10(depth_at_position, prediction, y_normalized[0], y_normalized[1], y_normalized[2], y_normalized[3], y_normalized[4], y_normalized[5], y_normalized[6], y_normalized[7], y_normalized[8], y_normalized[9])
	output['distance'] = int(distance[0][2]) * 0.01		# cm -> m

	return output


def make_sound_str(name:str, degree:str, distance:int=None) -> str:
	""" 出力音声用の文字列生成

	Args:
		name (str): 検知したラベル
		degree (str): 検知したラベルの危険度
		distance (int): 検知したラベルまでの距離

	Returns:
		sound (str): 音声出力用文字列

	"""

	filename = './static/ready/vidvipo_yolov8.csv'
	name_ja = ''
	with open(filename, newline='', encoding="shift-jis") as f:
		csvreader = csv.reader(f)
		for row in csvreader:
			if row[0] == name:
				name_ja = row[1]
				end_word = row[2] if degree=='yellow' else row[3]
				if (distance == None) or (degree == 'red'):
					sound = name_ja + end_word
					return sound
				else:
					sound = str(distance) + 'メートル先、' + name_ja +  end_word
					return sound

		else:
			sound = None
			return sound


def sound(object_dict:list, prediction, dt_now:str, before_second:int):
	""" 出力音声用の文字列生成

	Args:
		object_dict (list): 物体検知結果
		prediction (ndarray): 深度推定結果
		dt_now (str): 現在の時間
		before_second (int): 前回の音声出力時の秒

	Returns:
		output (dict): 音声として出力する物体の情報
		sound (str): 音声出力用文字列
		do_predict (bool): 音声選定を行ったか否か

	"""
	do_predict = False
	dt_second = int(dt_now.second)
	if ((dt_second - before_second) == (sound_interval-1)) or ((60-before_second + dt_second) == (sound_interval-1)):
		# プーリング
		depth_pooling, depth_pooling_size, depth_info = image_process.pooling(prediction, 16)
		if object_dict:
			# 最も近い深度の座標を取得
			object_dict = sorted(object_dict, key=lambda x: x['confidence'], reverse=True)
			new_object_dict = image_process.getDepthAxis(object_dict, depth_pooling, depth_pooling_size, process_img_size, depth_info)

			# (1秒間の)結果を一時保存
			record_output.record_temp(new_object_dict, dt_now)

		output = None
		sound = None


	elif ((dt_second - before_second) == sound_interval) or ((60-before_second + dt_second) == sound_interval):
		do_predict = True
		# pklファイルの取得
		if dt_second == 0:	# 59s の場合
			result_path = 'cash/' + dt_now.strftime('%H') + str(dt_now.minute-1) + '59.pkl'
		elif dt_second < 10: # 3s 7s の場合
			result_path = 'cash/' + dt_now.strftime('%H%M') + '0' + str(dt_second-1) + '.pkl'
		else:
			result_path = 'cash/' + dt_now.strftime('%H%M') + str(dt_second-1) + '.pkl'

		if os.path.isfile(result_path):
			# 正しく検出したものかフィルターをかける (１秒間に複数回検出したか否か)
			extra = check_results(result_path)

			if extra:
				# 危険ゾーンor警告ゾーンに入っているものを探す
				signal_red, signal_yellow, signal_blue = sort_risk_degree(extra)

				# ステータスの作成および更新
				status, post_output = make_status(extra)

				# 最終的に出力するラベルを決定
					# ステータスを出力したか
					# 危険ゾーン (~とぶつかりそうです or ~が近くにあります)
					# 警告ゾーン (~があります or ~がいます)
					# 優先順位 : 優先 -> (道路) -> 障害物 -> (境界) -> 地面
					# 歩車分離の時は道路系のラベルを出力しない
				output = decide_output(signal_red, signal_yellow, status, post_output)
				

				if output:
					# 絶対値取得
					output = get_absolute_value(output, prediction)
					
					# 音声出力文字列生成
					sound = make_sound_str(name=output['name'], degree=output['degree'], distance=output['distance'])
				else:
					sound = None

				# 過去データの保存or更新
				# 繰り返し出力しないための対策としてoutput=Noneの時は過去データに保存しない
				record_output.record_post_data(output, status, signal_red, signal_yellow, dt_now)
			
			else :
				sound = None
				output = None

		else:
			sound = None
			output = None

	else:
		sound = None
		output = None

	return output, sound, do_predict