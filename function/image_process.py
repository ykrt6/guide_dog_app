# 画像処理系
import numpy as np
import cv2

# 時間系
import datetime


def pooling(img, k):
    """ プーリング（最も近い深度を選択する）

    Args:
        img (pillow): 画像データ
        k (int): 何ピクセル分を1つにまとめるか

    Returns:
        dst (pillow): プーリング済み画像
        new_img_width (int): 画像の幅
        dep_info (tuple): 最大深度値、最小深度値

    """

    w, h= img.shape
    new_img_width = w // k
    dst = np.zeros((new_img_width, new_img_width))

    # プーリング処理
    for x in range(0, new_img_width):
        for y in range(0, new_img_width):     # BGRの順
            dst[x,y] = np.min(img[x*k:(x+1)*k,y*k:(y+1)*k])

    dep_info = (dst.max(), dst.min())

    return dst, new_img_width, dep_info


def convAxis(val:int, after_size:int, before_size:int) -> int:
    """ 座標を座標変換

    Args:
        obj_dict (int): 変換座標の値
        after_size (int): 変換後の画像の幅
        before_size (int): 変換前の画像の幅

    Returns:
        result (int): 変換後の座標

    """

    result = int(val * (after_size/before_size))
    return result if result < after_size else (after_size-1)

def backConvAxis(output:dict, after_size:int, before_size:int) -> int:
    """ 座標を元に戻す

    Args:
        output (dict): 音声として出力する物体の情報
        after_size (int): 変換後の画像の幅
        before_size (int): 変換前の画像の幅

    Returns:
        output (dict): 音声として出力する物体の情報

    """

    output['x'] = int(output['x'] * (before_size/after_size))
    output['y'] = int(output['y'] * (before_size/after_size))
    return output


def getDepthAxis(obj_dict:list, depth_pooling, depth_pooling_size:int, process_img_size:int, depth_info:tuple) -> list:
    """ 最も近い深度の座標を取得

    Args:
        obj_dict (dict): 物体検知の情報
        depth_pooling (pillow): プーリング済み画像
        depth_pooling_size (int): プーリング後の画像の幅
        process_img_size (int): プーリング前の画像の幅
        depth_info (tuple): 最大深度値、最小深度値

    Returns:
        results (list): 検知したラベル & 最も近い深度の座標 のリスト

    """

    depth_pooling = depth_pooling.copy()
    results = []
    for i in range(10 if len(obj_dict) > 10 else len(obj_dict)):  # 10こまで処理する (数が多いと処理時間がかかるため)
        box = obj_dict[i]['box']
        x1 = convAxis(box['x1'] , depth_pooling_size, process_img_size)
        y1 = convAxis(box['y1'], depth_pooling_size, process_img_size)
        x2 = convAxis(box['x2'], depth_pooling_size, process_img_size)
        y2 = convAxis(box['y2'], depth_pooling_size, process_img_size)
        
        best_abs = depth_info[0] - depth_info[1]   # 絶対更新したいやつ
        best_abs_axis = [0, 0]
        best_relative_val = 0
        for j in (range(y1, y2) if y1!=y2 else [y1]) :
            for k in (range(x1, x2) if x1!=x2 else [x1]):
                abs_val = abs(depth_pooling[j, k] - depth_info[0])

                if abs_val < best_abs:
                    best_abs = abs_val
                    best_abs_axis[0] = k
                    best_abs_axis[1] = j
                    best_relative_val = depth_pooling[j, k]
        
                # print(f'k:{k}, j:{j}, depth_pooling:{depth_pooling[k, j]}')
        
        # print(f'best_abs:{best_abs}, best_abs_axis:{best_abs_axis}, best_relative_val:{best_relative_val}')
        # cv2.circle(depth_pooling, (best_abs_axis[0], best_abs_axis[1]), 1, (63*i, 63*i, 63*i), thickness=-1)
        trans_depMax = 100*(best_relative_val - depth_info[1]) / (depth_info[0] - depth_info[1])
        results.append({'name': obj_dict[i]['name'], 'x': best_abs_axis[0], 'y': best_abs_axis[1], 'dep_max': trans_depMax, 'box': box})

    return results


def save_image(img_dict:dict, output:dict, now:str, flag:str):
    """ ローカルに保存

    Args:
        img_dict (dict): 画像データ (元画像、物体検知、深度マップ)
        output (dict): 出力ラベルの情報
        flag (str): ローカルに保存するか否か (フラグ)
    
    """

    if (flag == "true"):
        for type, img in img_dict.items():
            if output != None:
                filename = './static/output/' + type + '/' + now.strftime('%H%M%S') + '_' + output['name'] + '_' + output['degree'] + '_' + type + '.jpg'
                cv2.imwrite(filename, img)
            else:
                filename = './static/output/' + type + '/' + now.strftime('%H%M%S') + '_' + type + '.jpg'
                cv2.imwrite(filename, img)