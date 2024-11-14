# from typing import Union

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}
from fastapi import FastAPI, UploadFile, File, HTTPException, Query,Form
from fastapi.responses import FileResponse,StreamingResponse,HTMLResponse,JSONResponse
from io import BytesIO
import cv2
import uuid
import os
import io
from pydantic import BaseModel
import aiohttp
import requests
import numpy as np
from PIL import Image,ImageFilter
from datetime import datetime
import rembg
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import base64

ORIGINAL = "original/"
MASK = "masked/"
OVERLAY = "overlay_images/"
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
remove_bg = os.path.join(desktop_path, "Remove Background")
if not os.path.exists(remove_bg):
    os.makedirs(remove_bg)

ORIGINAL = os.path.join(remove_bg,ORIGINAL)
if not os.path.exists(ORIGINAL):
    os.makedirs(ORIGINAL)

MASK = os.path.join(remove_bg,MASK)
if not os.path.exists(MASK):
    os.makedirs(MASK)

OVERLAY = os.path.join(remove_bg,OVERLAY)
if not os.path.exists(OVERLAY):
    os.makedirs(OVERLAY)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"{timestamp}.png"
mask_filename = f"mask_BW_{filename}"
app = FastAPI(
    title="Image Background Removal API",
    description="An API to remove background from uploaded images",
    )

class ImageRequest(BaseModel):
    image_url: str = None

@app.post("/remove_bg/")
async def removeBg_process_use_any_one_Field(
        file: UploadFile = File(None),
        image_url: str = Form(None)):
    global mask_filename
    global filename
    if file:
        image_bytes = await file.read()
    elif image_url:
        response = requests.get(image_url)
        image_bytes = response.content
    else:
        return {"error": "No image provided"}
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    filename = f"{timestamp}.png"
    # Remove the background
    output_image = rembg.remove(image_bytes)
    # output_image = output_image.convert("RGBA")

    original_imag = Image.open(io.BytesIO(image_bytes))

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    filename = f"{timestamp}.png"
    # Save to a file
    original_imag.save(f'{ORIGINAL}{timestamp}.png')


    # return StreamingResponse(output_stream, media_type="image/png", headers={"Content-Disposition": "attachment; filename=output.png"})

    with open(f"{MASK}mask_{filename}", "wb") as overlay_file:
        overlay_file.write(output_image)

    mask_image = cv2.imread(f'{MASK}mask_{filename}', cv2.IMREAD_UNCHANGED)
    _, mask = cv2.threshold(mask_image[:, :, 3], 145, 255, cv2.THRESH_BINARY)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    mask_filename = f"mask_BW_{filename}"
    cv2.imwrite(f"{MASK}{mask_filename}", mask)

    mask = cv2.imread(f"{MASK}{mask_filename}", cv2.IMREAD_GRAYSCALE)

    image = cv2.imread(f"{ORIGINAL}{filename}", cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 3:  # If the image doesn't have an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)


    mask = cv2.bitwise_not(mask)
    overlay_mask = np.zeros_like(image)
    overlay_mask[:, :, 3] = mask
    overlay_mask[mask != 0] = [0, 0, 255, 128]
    cv2.imwrite(f"{MASK}mask_o_{filename}", overlay_mask)

    o_image = Image.open(f"{ORIGINAL}{filename}")
    color_mask_image = Image.open(f"{MASK}mask_o_{filename}")
    color_mask_image = color_mask_image.resize(o_image.size)
    merge_image = Image.alpha_composite(o_image.convert("RGBA"), color_mask_image.convert("RGBA"))
    merge_image.save(f"{OVERLAY}o_mask_{filename}")

    img_byte_arr = BytesIO()
    output_pil_image = Image.open(BytesIO(output_image))
    # output_pil_image = output_pil_image.filter(ImageFilter.UnsharpMask)

    output_pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    # Convert images to base64 for embedding in HTML
    img_base64_1 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    return JSONResponse(content={
        "remove background image": f"data:image/png;base64,{img_base64_1}",
    })
    
