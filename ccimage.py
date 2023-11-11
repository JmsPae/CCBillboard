from PIL import Image
import math
import random
import numpy as np
from numba import jit

def crop(filename):
    image = Image.open(filename)
    width, height = image.size
    if width / height >= 328 / 243: # width is too large, set correct height
        new_width = int(243 / height * width)
        new_height = 243
        top, bottom = 0, new_height
        left = (new_width - 328) / 2
        
        if left == math.floor(left):
            right = new_width - left
        else:
            right = new_width - math.floor(left)
            left = math.floor(left) + 1

    else: # height is too large, set correct width
        new_width = 328
        new_height = int(328 / width * height)
        left, right = 0, new_width
        top = (new_height - 243) / 2
        
        if top == math.floor(top):
            bottom = new_height - top
        else:
            bottom = new_height - math.floor(top)
            top = math.floor(top) + 1

    image = image.resize((new_width, new_height))
    image = image.crop((left, top, right, bottom))

    if image.size != (328, 243):
        raise Exception("rescale failed " + new_width + " " + new_height)

    return image.convert("RGB")

def extend(filename):
    source = Image.open(filename)
    source_width, source_height = source.size
    target = Image.new(source.mode, (328, 243), (0, 0, 0))
    
    source.crop()

    if source_width / source_height >= 328 / 243: # width is too large, pad top and bottom
        new_width = 328
        new_height = int(source_height * 328 / source_width)
        left = 0
        top = int((243 - new_height) / 2)
        
    else: # height is too large, pad left and right
        new_width = int(source_width * 243 / source_height)
        new_height = 243
        left = int((328 - new_width) / 2)
        top = 0

    source = source.resize((new_width, new_height))
    target.paste(source, (left, top))

    if target.size != (328, 243):
        raise Exception("rescale failed " + new_width + " " + new_height)

    return target.convert("RGB")

@jit(nopython=True)
def get_palette(source_map, tries):
    max_score = 0
    best_palette = np.zeros((16, 3), dtype=np.float64)
    for palette_index in range(tries):
        new_palette = np.zeros((16, 3), dtype=np.float64)
        sample_count = 0
    
        sample_attempts = 0
        thresh_test = np.zeros((164, 81), dtype=np.float64)
        while sample_count < 16:
            sample_attempts += 1
            if sample_attempts > 2000:
                raise Exception("palette failed :(")

            sample_x = math.floor(random.random() * 163)
            sample_y = math.floor(random.random() * 80)
            sample = source_map[sample_x][sample_y]

            
            if len(new_palette) > 0 and np.any(np.abs(new_palette[0:sample_count] - sample).sum(axis = 1) < 25):
                    continue
            
            thresh_test += np.where(np.abs(source_map[...,:3] - sample).sum(axis=2) < 25, 1, 0)
            new_palette[sample_count] = sample
            sample_count += 1
        
        total = np.sum(thresh_test.clip(0, 1))
        score = total / (164 * 81)
        if score > max_score:
            max_score = score
            #print(max_score * 100, palette_index, sample_attempts)
            best_palette = new_palette.copy()
            
    return best_palette

@jit(nopython=True)
def dither(image, palette):
    image = image.astype(np.float32)
    for x in range(328):
        for y in range(243):
            old = image[x, y].copy()
            diff_table = np.abs(palette - old).sum(axis = 1)
            index = diff_table.argmin()
            new = palette[index]
            error = old - new

            image[x, y] = new
            
            if x + 1 < 328:
                image[x + 1, y] += error * 0.4375
            if (y + 1 < 243) and (x + 1 < 328):
                image[x + 1, y + 1] += error * 0.0625
            if y + 1 < 243:
                image[x, y + 1] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < 243):
                image[x - 1, y + 1] += error * 0.1875
    return image

@jit(nopython=True)
def prepare(image, palette):
    char_array = np.zeros((81, 164))
    fore_array = np.zeros((81, 164))
    back_array = np.zeros((81, 164))

    for char_y in range(81):
        for char_x in range(164):
            anchor_x = char_x * 2
            anchor_y = char_y * 3

            appear_table = np.zeros(16)
            for x in range(2):
                for y in range(3):
                    pixel_x = anchor_x + x
                    pixel_y = anchor_y + y
                    nearest_index = np.abs(palette - image[pixel_x, pixel_y]).sum(axis = 1).argmin()
                    appear_table[nearest_index] += 1

            fore_index = appear_table.argmax()
            appear_table[fore_index] = 0
            back_index = appear_table.argmax()

            block = 0
            fore_color = palette[fore_index].copy()
            back_color = palette[back_index].copy()

            for y in range(3):
                for x in range(2):
                    pixel_x = anchor_x + x
                    pixel_y = anchor_y + y
                    power = pow(2, x + y * 2)
                    fore_diff = np.abs(fore_color - image[pixel_x, pixel_y]).sum()
                    back_diff = np.abs(back_color - image[pixel_x, pixel_y]).sum()

                    if fore_diff < back_diff:
                        block += power
                        image[pixel_x, pixel_y] = fore_color
                    else:
                        image[pixel_x, pixel_y] = back_color
            char_array[char_y, char_x] = block
            fore_array[char_y, char_x] = fore_index
            back_array[char_y, char_x] = back_index
    return char_array, fore_array, back_array

            
def process(image, palette_count):
    color_codes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
    img = np.array(image).swapaxes(0, 1)
    pal_img = np.array(image.resize((164, 81)), dtype=np.int32)
    palette = get_palette(pal_img.swapaxes(0, 1), palette_count)
    img = dither(img, palette).clip(0, 255)

    chars, fores, backs = prepare(img, palette)

    mask = chars >= 32
    chars = np.where(mask, 63 - chars, chars)
    final_fores = np.where(mask, backs, fores)
    final_backs = np.where(mask, fores, backs)
    
    pal_strings = ["-".join(item) for item in (palette/255).astype(str)]
    out_string = "+".join(pal_strings) + "#"

    for y in range(81):
        out_string += "+".join(["{:02d}".format(int(item)) for item in chars[y]]) + ","
        out_string += "".join([color_codes[int(item)] for item in final_fores[y]]) + ","
        out_string += "".join([color_codes[int(item)] for item in final_backs[y]]) + ","
        
    return(out_string[:-1])


def init(image):
    img = np.array(image).swapaxes(0, 1)
    pal_img = np.array(image.resize((164, 81)), dtype=np.int32)
    pal = get_palette(pal_img, 1)
    prepare(img, pal)
    dither(img, pal)
    




                    

            
            
            

            
            

        
    

    
    
    

