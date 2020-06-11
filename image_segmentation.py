import sensor, image, time, lcd
import KPU as kpu
import ulab as np
import sys


lcd.init(freq=40000000)
lcd.rotation(2)  # Rotate the lcd 180deg
sensor.reset(dual_buff=True)

sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.B64X64)
window_width = 32
window_height = 32
classes = 5
Scale = 2
sensor.set_windowing((window_height, window_width))
sensor.skip_frames(100)
sensor.run(1)

print("init kpu")
lcd.draw_string(10, 10, "init kpu")
lcd.draw_string(170, 10, "Running")

lcd.draw_string(10, 30, "load kmodel")
kpu.memtest()
task = kpu.load(0x500000)
lcd.draw_string(170, 30, "Done")

lcd.draw_string(10, 50, "set outputs")
fmap = kpu.set_outputs(task, 0, window_height, window_width, classes)
kpu.memtest()
lcd.draw_string(170, 50, "Done")

print("Done")
time.sleep_ms(1000)
lcd.draw_string(170, 10, "Done     ")
time.sleep_ms(500)
lcd.draw_string(60, 70, "Setup Done! :)")
clock = time.clock()

seg_img = image.Image()
output_array = np.array([0, 0, 0, 0, 0])
color_dict = {0: (0, 0, 0),  # 0 in VOC2012 BG,VOid
              1: (192, 0, 0),  # 9 in VOC2012 CHAIR, SOFA
              2: (64, 128, 0),  # 11 in VOC2012 TABLE
              3: (192, 128, 128),  # 15 in VOC2012 PEOPLE
              4: (0, 64, 128)}  # 20 in VOC2012 TV

lcd.clear()
while True:
    try:
        print("take a snapshot")
        img = sensor.snapshot()         # Take a picture and return the image.
        #img = img.resize(window_height, window_width)
        #img.pix_to_ai()
        clock.tick()
        print("run kpu")
        fmap = kpu.forward(task, img)
        data_tuple = fmap[:]

        print("make image")
        for i in range(window_height * window_width):
            for c in range(classes):
                data = data_tuple[c * window_height * window_width + i]
                output_array[c] = data
            _w = i % window_width
            _h = int(i / window_height)
            _w = i % window_width
            _h = int(i / window_height)
            for _w_ in range(Scale):
                for _h_ in range(Scale):
                    a = seg_img.set_pixel(Scale * _w + _w_, Scale * _h + _h_,
                                             color_dict[np.argmax(output_array)])
        fps = clock.fps()
        fps_0 = int(fps)
        fps_1 = (int(fps * 10) - fps_0 * 10)
        fps_2 = (int(fps * 100) - fps_1 * 10 - fps_0 * 100)
        fps_str = str(fps_0) + "." + str(fps_1) + str(fps_2) + " FPS"

        #img.ai_to_pix()
        img = img.resize(64,64)
        seg_img = seg_img.resize(64,64)
        #lcd.clear()
        lcd.display(img, oft=(4,4))
        lcd.display(seg_img, oft=(72,4))
        lcd.draw_string(40, 80, fps_str)
    except Exception as inst:
        print(inst)
        #break

a = kpu.deinit(task)
#sys.exit()
lcd.clear((30, 111, 150))
