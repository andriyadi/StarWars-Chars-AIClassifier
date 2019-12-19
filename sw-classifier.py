import sensor, image, lcd, time
import KPU as kpu

THRESHOLD = 0.7

CAM_IMG_WIDTH = 224
CAM_IMG_HEIGHT = 224

SCREEN_WIDTH = 320
SCREEN_HEIGHT = 240

lcd.init(freq=15000000)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((CAM_IMG_WIDTH, CAM_IMG_HEIGHT))
sensor.set_vflip(0)
sensor.set_hmirror(True)
sensor.run(1)

lcd.clear()
lcd.draw_string(90,96,"Star Wars Classifier")
lcd.draw_string(100,115,"Loading Models...")

labels=['BB8', 'Kylo Ren', 'Yoda']

# Load Model, adjust the path of your kmodel file
task = kpu.load('/sd/models/starwars.kmodel')
clock = time.clock()
allImg = image.Image()

labelYOffset = 70

while(True):
    img = sensor.snapshot()
    clock.tick()
    fmap = kpu.forward(task, img)
    fps = clock.fps()
    plist = fmap[:]
    pmax = max(plist)	
    labelText = ""
    if pmax > THRESHOLD:
        max_index=plist.index(pmax)	
        labelText = labels[max_index].strip()
    else:
        labelText = "Unknown"

    print("0 -> {:.3f}, 1 -> {:.3f}, 2 -> {:.3f}".format(fmap[0], fmap[1], fmap[2]))
    
    allImg.draw_image(img, 0, int((240-CAM_IMG_HEIGHT)/2))

    allImg.draw_rectangle((CAM_IMG_WIDTH + 5), labelYOffset, SCREEN_WIDTH - (CAM_IMG_WIDTH + 5), (240-labelYOffset), (0,0,0), 1, True)
    lcd.display(allImg, oft=(0, 0))
    lcd.draw_string((CAM_IMG_WIDTH+10), labelYOffset + 5, labelText)
    lcd.draw_string((CAM_IMG_WIDTH+10), labelYOffset + 5 + 20, ("{:.1f}%".format(pmax*100)))

    lcd.draw_string((CAM_IMG_WIDTH + 10), (SCREEN_HEIGHT - 45), "FPS:")
    lcd.draw_string((CAM_IMG_WIDTH + 10), (SCREEN_HEIGHT - 25), "{0:.2f}".format(fps))
    
a = kpu.deinit(task)