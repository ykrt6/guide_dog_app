import glob
import csv
import cv2
from matplotlib import pyplot as plt

def main():
    orig_lis = glob.glob("./static/output/obj/*.jpg")
    csv_path = "./static/output/****-**-**/results.csv"     # 「*」を書き換える

    with open(csv_path) as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        
        # print(data)

        for i, path in enumerate(orig_lis):
            imgCV = cv2.imread(path)

            # cv2.cvtColorを使う方法
            imgCV_RGB = cv2.cvtColor(imgCV,cv2.COLOR_BGR2RGB)

            # スライスを使う方法
            # imgCV_RGB = imgCV[:, :, ::-1]

            print(f"=============={i}/{len(orig_lis)}==============")
            for l in data[i]:
                print(l)
            print("")
             
            plt.imshow(imgCV_RGB)
            plt.show()

            

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()