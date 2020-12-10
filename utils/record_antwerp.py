import os
import time
from datetime import date
import cv2
import argparse


def get_args():
    parser = argparse.ArgumentParser('Record data from Antwerp Smart Zone')
    parser.add_argument('--camera', type=str, default="CAM11_1")
    parser.add_argument('--numday', type=int, default=1)
    parser.add_argument('--numhour', type=int, default=24)
    parser.add_argument('--outputdir', type=str, default='/home/jovyan/bo/dataset/')
    args = parser.parse_args()
    return args
    

def record(args, url):
    camera_name = args.camera
    for i in range(args.numday):        
        today = date.today()
        today = today.strftime("%b_%d_%Y")
        output = args.outputdir + "%s/%s/" % (args.camera, today)
        count = 60 * 60 * 25
        for j in range(args.numhour):
            hour_name = "Sequence_%04d/" % j
            path = output + hour_name
            if not os.path.exists(path):
                os.makedirs(path)
            vcap = cv2.VideoCapture(url)
            while vcap.isOpened():
                ret, frame = vcap.read()
                frameid = vcap.get(1)
                cv2.imwrite(path + "%010d.jpg" % frameid, frame)
                if frameid > count:
                    vcap.release()
                    pass
            print("recording %d frames for day %s and hour %s" % (frameid, today, hour_name))

# ---------------------------------------------- #
# Collection of the urls for recording frames    #
# ---------------------------------------------- #
urls = ["rtsp://143.169.169.3:1935/smartcity/CAM1.1-Groenplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM1.2-Groenplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM1.3-Groenplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM1.4-Groenplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM2.1-Hoogstraat-Reynderstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM2.2-Hoogstraat-Reynderstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM2.3-Hoogstraat-Reynderstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM2.4-Hoogstraat-Reynderstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM3.1-Hoogstraat-St.Jansvliet.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM3.2-Hoogstraat-St.Jansvliet.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM3.3-Hoogstraat-St.Jansvliet.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM3.4-Hoogstraat-St.Jansvliet.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM4.1-Kloosterstraat-Kromme-elleboogstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM4.2-Kloosterstraat-Kromme-elleboogstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM4.3-Kloosterstraat-Kromme-elleboogstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM4.4-Kloosterstraat-Kromme-elleboogstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM5.1-Kloosterstraat-KorteVlierstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM5.2-Kloosterstraat-KorteVlierstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM5.3-Kloosterstraat-KorteVlierstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM5.4-Kloosterstraat-KorteVlierstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM6.1-Andriesplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM6.2-Andriesplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM6.3-Andriesplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM6.4-Andriesplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM7.1-Kloosterstraat-Riemstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM7.2-Kloosterstraat-Riemstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM7.3-Kloosterstraat-Riemstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM7.4-Kloosterstraat-Riemstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM8.1-Kloosterstraat-Scheldestraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM8.2-Kloosterstraat-Scheldestraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM8.3-Kloosterstraat-Scheldestraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM8.4-Kloosterstraat-Scheldestraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM9.1-LeopoldDeWaelstraat-Schildersstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM9.2-LeopoldDeWaelstraat-Schildersstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM9.3-LeopoldDeWaelstraat-Schildersstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM9.4-LeopoldDeWaelstraat-Schildersstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM10.1-Nationalestraat-Kronenburgstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM10.2-Nationalestraat-Kronenburgstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM10.3-Nationalestraat-Kronenburgstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM10.4-Nationalestraat-Kronenburgstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM11.1-Nationalestraat-Prekerstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM11.2-Nationalestraat-Prekerstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM11.3-Nationalestraat-Prekerstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM11.4-Nationalestraat-Prekerstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM12.1-Nationalestraat-Ijzerwaag.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM12.2-Nationalestraat-Ijzerwaag.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM12.3-Nationalestraat-Ijzerwaag.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM12.4-Nationalestraat-Ijzerwaag.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM13.1-Nationalestraat-Steenhouwersvest.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM13.2-Nationalestraat-Steenhouwersvest.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM13.3-Nationalestraat-Steenhouwersvest.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM13.4-Nationalestraat-Steenhouwersvest.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM14.1-Kloosterstraat-WillemLepelstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM14.2-Kloosterstraat-WillemLepelstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM14.3-Kloosterstraat-WillemLepelstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM14.4-Kloosterstraat-WillemLepelstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM15.1-Kronenburgstraat-volkstraat.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM16.1-StAndriesplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM17.1-StAndriesplaats.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM18.1-StAndriesplaats-costa.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM18.2-StAndriesplaats-costa.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM18.3-StAndriesplaats-costa.stream",
        "rtsp://143.169.169.3:1935/smartcity/CAM18.4-StAndriesplaats-costa.stream"
        ]
       

if __name__ == '__main__':
    opt = get_args()
    url_name = '/' + '.'.join(opt.camera.split("_"))
    select_url = [v for v in urls if url_name in v][0]
    print("Recording frames from url ", select_url)
    record(opt, select_url)

    