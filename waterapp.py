
import streamlit as st
import cv2
import pandas as pd
import os
import glob
import numpy as np
import torch
import copy
from math import sqrt, ceil, floor

from fastai.vision.core import *
from fastai.vision.data import *
from fastai.vision.all import *
from fastai.vision import *
from fastai.vision.core import PILImage, PILMask
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image
from shapely.geometry import Polygon
import requests
import shutil
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def download_file(url, filename, path=None):
    #local_filename = url.split('/')[-1]
    if path:
        local_filename = path.joinpath(filename)
    else:
        local_filename = filename
    r = requests.get(url, stream=True)
    file_size = int(r.headers.get('Content-Length', 0))
    with open(local_filename, 'wb') as f:
        with tqdm.wrapattr(r.raw, "read", total=file_size) as r_raw:
            shutil.copyfileobj(r_raw, f)
    return r.status_code


def get_label_image(fpath, imgfile):
    label_file = fpath.joinpath(str(imgfile.name).replace('.JPG', '_lb.png'))
    return label_file


def get_y(img_file):
    maskfile = get_label_image(path_lab, img_file)
    return get_msk(maskfile)


def acc_camvid(inp, targ):
    targ = targ.squeeze(1)
    mask = targ != void_code
    return (inp.argmax(dim=1)[mask] == targ[mask]).float().mean()


def apply_mask(image, mask, color, num, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask[:, :, c] == num, image[:, :, c]
                                  * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image


def num2id(imgvalue):
    # value is based on processing method
    classes = ["background", "pond", "waterway", "wgrass", "eground"]
    color_code = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                  (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0)]
    ind = [0, 1, 2, 3, 4]
    ind = [x*60 for x in ind]
    try:
        clsid = ind.index(imgvalue)
        clsname = classes[clsid]
        color = color_code[clsid]
        return [clsid, clsname, color]
    except ValueError:
        pass


def id2num(idname):
    # value is based on processing method
    classes = ["background", "pond", "waterway", "wgrass", "eground"]
    color_code = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                  (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0)]
    ind = [0, 1, 2, 3, 4]
    ind = [x*60 for x in ind]
    try:
        clsid = classes.index(idname)
        color = color_code[clsid]
        return [clsid, idname, color]
    except ValueError:
        pass


def maskimage_to_apply(fraw, fmask, classname=None):
    # fraw: the original image to detect/ image data
    # fmask: output image with mask/ image data
    img = np.array(fraw)
    masked_image = img.copy()
    maskarr = np.array(fmask)
    maskinfo = {}
    maskname = []
    tmp = id2num(classname)
    masked_image = apply_mask(masked_image, maskarr, tmp[2], tmp[0]*60)
    maskinfo['classname'] = classname  # maskname
    maskinfo['data'] = masked_image
    return maskinfo  # masked_image


def mask_to_polygon(maskfile):
    maskarr = maskfile  # np.array(msk)
    contours, hierarchy = cv2.findContours(
        maskarr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    area = []
    polygons = []
    for poly in contours:
        # print(cv2.contourArea(poly))
        s = poly.tolist()
        s = [x[0] for x in s]
        try:
            polygon = Polygon(s)
            area.append(polygon.area)
            polygons.append(polygon)
        except ValueError:
            continue
#        polygon = Polygon(s)
#        area.append(polygon.area)
#        polygons.append(polygon)
    return polygons


class Tile(object):
    """Represents a single tile."""

    def __init__(self, image, number, position, coords):
        self.image = image
        self.number = number
        self.position = position
        self.coords = coords
        #self.fid = fid

    def generate_filename(
        self, directory=os.getcwd(), prefix="tile", format="png", path=True
    ):
        """Construct and return a filename for this tile."""
        filename = prefix + "_{col:02d}_{row:02d}.{ext}".format(
            col=self.position[1], row=self.position[0], ext=format.lower().replace(
                "jpeg", "jpg")
        )
        if not path:
            return filename
        return os.path.join(directory, filename)

    def save(self, filename=None, format="png"):
        if not filename:
            filename = self.generate_filename(format=format)
        self.image.save(filename, format)


def save_tiles(tiles, prefix='cropped', directory=None, format='png'):
    for tile in tiles:
        filename = tile.generate_filename(
            prefix=prefix, directory=directory, format=format)
        tile.save(filename=filename, format=format,)
    return tuple(tiles)


def silcer(image, fid, num_tiles=None, save=False, path=None):
    if not num_tiles:
        num_tiles = 150
    im_w, im_h = image.size
    columns = int(ceil(sqrt(num_tiles)))
    rows = int(ceil(num_tiles / float(columns)))
    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))
    tiles = []
    number = 1
    #print(tile_w, tile_h)
    # print(columns,rows)
    for pos_y in range(0, im_h - rows, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w):  # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            img = image.crop(area)
            position = (int(floor(pos_x / tile_w)) + 1,
                        int(floor(pos_y / tile_h)) + 1)
            coords = (pos_x, pos_y)
            tile = Tile(img, number, position, coords)
            tiles.append(tile)
            number += 1
    # cropped={'filename':fid,'crop_image':tuple(tiles),'shape':[columns,rows],'size':[im_w,im_h]}
    cropped = {'filename': fid, 'crop_image': tuple(
        tiles), 'shape': [columns, rows], 'size': [tile_w, tile_h]}
    if save is True:
        save_tiles(tiles=tiles, prefix='cropped_'+fid, directory=path)
    return cropped


def calc_columns_rows(n):
    """
    Calculate the number of columns and rows required to divide an image
    into ``n`` parts.
    Return a tuple of integers in the format (num_columns, num_rows)
    """
    num_columns = int(ceil(sqrt(n)))
    num_rows = int(ceil(n / float(num_columns)))
    return (num_columns, num_rows)


def get_combined_size(tiles):
    """Calculate combined size of tiles."""
    # TODO: Refactor calculating layout to avoid repetition.
    columns, rows = calc_columns_rows(len(tiles))
    tile_size = tiles[0].image.size
    return (tile_size[0] * columns, tile_size[1] * rows)


def imgjoin(tiles, width=0, height=0):
    """
    @param ``tiles`` - Tuple of ``Image`` instances.
    @param ``width`` - Optional, width of combined image.
    @param ``height`` - Optional, height of combined image.
    @return ``Image`` instance.
    """
    # Don't calculate size if width and height are provided
    # this allows an application that knows what the
    # combined size should be to construct an image when
    # pieces are missing.
    if width > 0 and height > 0:
        im = Image.new("RGBA", (width, height), None)
    else:
        im = Image.new("RGBA", get_combined_size(tiles), None)
    columns, rows = calc_columns_rows(len(tiles))
    for tile in tiles:
        try:
            im.paste(tile.image, tile.coords)
        except IOError:
            # do nothing, blank out the image
            continue
    return im


def rescale(prediction):
    test = prediction[0].numpy()
    test[test == 1] = 60*1
    test[test == 2] = 60*2
    test[test == 3] = 60*3
    test[test == 4] = 60*4
    return test


def object_detection_image(file, learn, target=None, num=None):
    st.title('Waterway Detection for Images')
    st.subheader("""
    This app will detect the waterway in an image and output the image with polygons.
    """)
    #file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    if file != None:
        st.write("Image dection calculating...plz be patient (~ 2-5 mins)")
        #st.write("Image Uploaded Successfully:")
        # img=PILImage.create(file)
        #st.image(img, caption = "Uploaded Image")
        img = file  # Image.open(files)
        fname = 'uploadfile'
        scale = round((5.36+6.65+7.77)/3)
        unit = scale**2/10000

        if num != None:
            slice_data = silcer(img, fname, num_tiles=num,
                                save=False)  # cropped data
        else:
            slice_data = silcer(img, fname, num_tiles=100,
                                save=False)  # cropped data

        data = slice_data['crop_image']

        if target != None:
            clsname = str(target)
        else:
            clsname = 'wgrass'  # default target name

        my_bar = st.progress(0)
        all_area = []
        for subdata in data:
            image = subdata.image  # PIL obj
            img_fastai = np.array(subdata.image)
            orgsz = img_fastai.shape
            pred = learn.predict(img_fastai)  # image tensor
            rescaled = rescale(pred)  # np array
            ind = np.unique(pred[0].numpy())
            if len(ind) >= 2:
                maskimg = Image.fromarray(rescaled.astype(np.uint8))  # PIL obj
                szm = maskimg.shape
                # PIL obj, must be (456,684,3)
                img_fastai2 = image.resize((szm[1], szm[0]))
                stacked_img = np.stack((rescaled,)*3, axis=-1)
                maskinfo = maskimage_to_apply(
                    img_fastai2, stacked_img, clsname)
                img2 = Image.fromarray(maskinfo['data'])
                img2 = img2.resize((szm[1]*2, szm[0]*2))
                draw = ImageDraw.Draw(img2)
                polys = mask_to_polygon(rescaled.astype(np.uint8))
                area = []
                for x in polys:
                    cx = x.representative_point().x
                    cy = x.representative_point().y
                    draw.text(
                        (cx*2, cy*2), '{:8.1f}'.format(x.area*unit), stroke_fill=(255, 0, 0), fill=255)
                    area.append(x.area)

                all_area.append(sum(area)*unit)
                subdata.image = img2.resize((orgsz[1], orgsz[0]))
#              img2.save(path.joinpath(imgpath,'de_'+'{0:02d}'.format(i)+'.png'))
            else:
                subdata.image = image

        pred_img = imgjoin(data)
        st.image(pred_img, caption='Proccesed Image.', width=400)
        st.write(pd.DataFrame({
            'Target Name': [str(target)],
            'Target Area (m2)': [round(sum(all_area), 2)]}))
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        my_bar.progress(100)


def main():
    new_title = '<p style="font-size: 42px;">Demo: welcome to waterway detection </p>'
#    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

#    read_me = st.markdown("""
#    This Web app is for image model demo and test usage!"""
#    )
#    st.sidebar.title("Select Activity")
#    choice  = st.sidebar.selectbox("MODE",("About","Detection(Image)","(Coming soon) Detection(Video)"))
#    st.write(f'Prepare the Model, please wait!')
    waterway_model_file = 'model_version_1.pkl'
    url = 'https://www.dropbox.com/s/1khic5wgtwzf2x7/model_v1.pkl?dl=0'
    url = url.replace('0', '1')  # dl=1 is important
    # codes = np.loadtxt(path/'classes.txt', dtype=str) #classes:["pond","waterway","wgrass","eground"]
    loadfile = download_file(url, filename=waterway_model_file)
    if loadfile == 200:
        #        print('Successful download')
        #        st.write(f'Successful download')
        st.write(f'Plz upload your image to detect.')
    else:
        print('Error')

    learn_inf = load_learner(waterway_model_file)
    save_path = '/image/'
    #font = ImageFont.truetype('arial',30)
    classes = learn_inf.dls.train.after_item.vocab

    scale = round((5.36+6.65+7.77)/3)
    unit = scale**2/10000

    file = st.file_uploader('Upload Image (One image)',
                            type=['jpg', 'png', 'jpeg'])
    if file != None:
        st.write("Image Uploaded Successfully:")
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", width=200)

        option = st.selectbox('Choose your target:',
                              ("pond", "waterway", "wgrass"))
        st.write('Your target:', option)

        if st.button('Classify'):
            #            object_detection_image(file)
            st.write("Image dection calculating...")
            object_detection_image(img, learn_inf, option)
            #pred, pred_idx, probs = learn_inf.predict(img)
            #st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
        else:
            st.write(f'Click the button to classify')


if __name__ == '__main__':
    main()
