import pygame as pg
import sys
import os
import json
save_file = "metadata.txt"
pixel_crop = 8
load=True
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))

def generate_link(image_index=0):
    return "resized/00000"[:-len(str(image_index))] + str(image_index) +".jpg"

pg.init()
pg.key.set_repeat(300, 25)
screen = pg.display.set_mode((1920, 1024))
COLOR_INACTIVE = pg.Color('black')
COLOR_ACTIVE = pg.Color('white')
FONT = pg.font.SysFont('arial', 20)


class InputBox:

    def __init__(self, x, y, w, h, text=''):
        self.rect = pg.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.active = False

    def draw(self, screen):
        # Blit the text.
        # Blit the rect.
        pg.draw.rect(screen, self.color, self.rect, 2)

    def getWidth(self):
        return self.rect.w


class Background(pg.sprite.Sprite):
    def __init__(self, image_file, location):
        pg.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pg.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

def blit_text(surface, text, pos, font, boxWidth, color=pg.Color('black')):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    x = boxWidth+20
    for line in words:
        for word in line:
            word_surface = font.render(word, 0, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.


def save_data(data):
    with open(save_file, "w") as f:
        json.dump(data,f)
def main():
    clock = pg.time.Clock()
    #input_box1 = InputBox(15, 100, 70, 30)
    input_boxes = []


    done = False

    mouse_pos = []
    mouse_pos2  = []
    image_index = 0
    if load:
        with open(save_file, "r") as f:
            data = json.load(f)
            image_index = data[-1][0]+1
            print(len(data))
    else:
        data = []
    print("starting at", image_index)
    BackGround = Background(generate_link(image_index), [0, 0])
    mouse_down = False
    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                mouse_down = True
                mouse_pos = list(pg.mouse.get_pos())
                mouse_pos[0] = mouse_pos[0]-mouse_pos[0]%pixel_crop
                mouse_pos[1] = mouse_pos[1] - mouse_pos[1] % pixel_crop
                print(mouse_pos)
                input_boxes = []
            elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                mouse_down = False
                mouse_pos2 = list(pg.mouse.get_pos())
                mouse_pos2[0] = mouse_pos2[0]-mouse_pos2[0]%pixel_crop
                mouse_pos2[1] = mouse_pos2[1] - mouse_pos2[1] % pixel_crop
                [InputBox(mouse_pos[0], mouse_pos[1],mouse_pos2[0] - mouse_pos[0],
                          mouse_pos2[1] - mouse_pos[1])]
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_s:
                    data.append((image_index, None))
                    mouse_pos = []
                    mouse_pos2 = []
                    input_boxes = []
                    image_index += 1
                    BackGround = Background(generate_link(image_index), [0, 0])
                    print(data[-1])
                elif event.key == pg.K_d:
                    if(len(mouse_pos2)==2):
                        data.append((image_index, (mouse_pos,mouse_pos2)))
                        mouse_pos = []
                        mouse_pos2 = []
                        input_boxes=[]
                        image_index += 1
                        BackGround = Background(generate_link(image_index), [0, 0])
                        print(data[-1])
                elif event.key == pg.K_a:
                    data = data[:-1]
                    mouse_pos = []
                    mouse_pos2 = []
                    input_boxes=[]
                    image_index -= 1
                    BackGround = Background(generate_link(image_index), [0, 0])
                    print(data[-1])
                elif event.key == pg.K_w:
                    save_data(data)
                    print("saved")
                    done = True
                elif event.key == pg.K_q:
                    save_data(data)
                    print("saved")

        if mouse_down:
            mouse_x,mouse_y = pg.mouse.get_pos()
            mouse_x = mouse_x - mouse_x%pixel_crop
            mouse_y = mouse_y-mouse_y%pixel_crop
            input_boxes = [InputBox(mouse_pos[0], mouse_pos[1], mouse_x - mouse_pos[0],
                                    mouse_y - mouse_pos[1])]
        screen.fill((255, 255, 255))
        screen.blit(BackGround.image, BackGround.rect)
        #print text
        for box in input_boxes:
            box.draw(screen)

        pg.display.flip()
        clock.tick(30)

main()
pg.quit()
