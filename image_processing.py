import sensor, image, time, lcd
import KPU as kpu
import ulab as np

lcd.init(freq=24000000)
lcd.rotation(2)  # Rotate the lcd 180deg
sensor.reset()

sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((128, 128))
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
fmap = kpu.set_outputs(task, 0, 128, 128, 5)
kpu.memtest()
lcd.draw_string(170, 50, "Done")

print("Done")
time.sleep_ms(1000)
lcd.draw_string(170, 10, "Done     ")
time.sleep_ms(500)
lcd.draw_string(60, 70, "Setup Done! :)")
clock = time.clock()

while True:
    try:
        img = sensor.snapshot()         # Take a picture and return the image.
        clock.tick()
        fmap = kpu.forward(task, img)
        fps = clock.fps()
        print("%f[fps]" % fps)
        plist = fmap[:21]
        print(type(plist))
        # print(img.get_pixel(0, 0))
        # print(max(plist), plist.index(max(plist)))
        # img.set_pixel(50, 50, (255, 0, 0))

        lcd.display(img)
    except Exception as inst:
        print(inst)

a = kpu.deinit(task)
lcd.clear((30, 111, 150))
