from typing import List, Tuple
import copy

from PIL import Image, ImageFont, ImageDraw, ImageEnhance

from gobang_utils import CHESSBOARD_SIZE


def save_history_img(history: List[Tuple[int, int, int]], path: str):
    assert path.endswith(".jpeg") or path.endswith(".gif")

    background = Image.open("imgs/chessboard.png").convert("RGB")
    draw = ImageDraw.Draw(background)
    font = ImageFont.load_default()

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    stone_colors = [BLACK, WHITE]
    text_colors = [WHITE, BLACK]

    all_imgs = [copy.deepcopy(background)]

    r = 16
    for i, (who, x, y) in enumerate(history):
        img_x = 20 + y * 40
        img_y = 20 + x * 40

        draw.ellipse(
            (img_x - r, img_y - r, img_x + r, img_y + r), fill=stone_colors[who]
        )
        msg = str(i + 1)
        w = draw.textlength(msg, font=font)
        h = font.size
        draw.text((img_x - w / 2, img_y - h / 2), msg, font=font, fill=text_colors[who])
        all_imgs.append(copy.deepcopy(background))

    if path.endswith(".jpeg"):
        background.save(path, "jpeg")
    else:
        all_imgs[0].save(
            path, save_all=True, append_images=all_imgs[1:], duration=1000, loop=0
        )


def chessboard_str(chessboard) -> str:
    s = ""
    for i in range(CHESSBOARD_SIZE):
        for j in range(CHESSBOARD_SIZE):
            if chessboard[0, i, j] > 0:
                s += "x "
            elif chessboard[1, i, j] > 0:
                s += "o "
            else:
                s += ". "
        s += "\n"
    return s
